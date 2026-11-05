"""SQLite persistence. The service serializes observation processing and transactions."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

PRIVATE_DATABASES = set()


def dumps(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':'))


class SQLiteStore:
    def __init__(self, path):
        self.path = Path(path)
        PRIVATE_DATABASES.add(self.path.resolve())
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self.path.chmod(0o600)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute('PRAGMA foreign_keys=ON')
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY, participant TEXT NOT NULL,
                mode TEXT NOT NULL, context_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS readings (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE, session_id TEXT NOT NULL REFERENCES sessions(session_id),
                sample_id TEXT NOT NULL, timestamp_ms REAL NOT NULL,
                received_ms REAL NOT NULL, sample_json TEXT NOT NULL, effective_json TEXT, result_json TEXT NOT NULL,
                UNIQUE(session_id, sample_id)
            );
            CREATE TABLE IF NOT EXISTS alerts (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE, session_id TEXT NOT NULL REFERENCES sessions(session_id),
                reading_id TEXT NOT NULL REFERENCES readings(id), data_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE, session_id TEXT NOT NULL REFERENCES sessions(session_id),
                kind TEXT NOT NULL, reading_id TEXT REFERENCES readings(id),
                alert_id TEXT REFERENCES alerts(id), data_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS readings_session_time ON readings(session_id,timestamp_ms);
            CREATE INDEX IF NOT EXISTS feedback_session ON feedback(session_id);
            CREATE INDEX IF NOT EXISTS alerts_session ON alerts(session_id);
        ''')
        columns = {row['name'] for row in self.connection.execute('PRAGMA table_info(readings)')}
        if 'effective_json' not in columns:
            self.connection.execute('ALTER TABLE readings ADD COLUMN effective_json TEXT')
            self.connection.commit()

    def close(self):
        self.connection.close()

    def replay_samples(self):
        for row in self.connection.execute('SELECT COALESCE(effective_json,sample_json) AS sample_json FROM readings ORDER BY sequence'):
            yield json.loads(row['sample_json'])

    def alert_histories(self):
        histories = {}
        for row in self.connection.execute('SELECT session_id,data_json FROM alerts ORDER BY sequence'):
            histories.setdefault(row['session_id'], []).append(json.loads(row['data_json']))
        # Include sessions with no actual alerts to erase alerts invented during replay.
        for row in self.connection.execute('SELECT DISTINCT session_id FROM readings'):
            histories.setdefault(row['session_id'], [])
        return histories

    def session(self, session_id):
        row = self.connection.execute('SELECT * FROM sessions WHERE session_id=?', (session_id,)).fetchone()
        return dict(row) if row else None

    def ensure_session(self, session_id, participant, mode):
        row = self.session(session_id)
        if row and (row['participant'] != participant or row['mode'] != mode):
            raise ValueError('Session participant and mode cannot change; start a new session.')
        self.connection.execute('INSERT OR IGNORE INTO sessions(session_id,participant,mode) VALUES (?,?,?)',
                                (session_id, participant, mode))

    def reading(self, session_id, *, sample_id=None, reading_id=None):
        column, value = ('sample_id', sample_id) if sample_id is not None else ('id', reading_id)
        row = self.connection.execute(f'SELECT * FROM readings WHERE session_id=? AND {column}=?',
                                      (session_id, value)).fetchone()
        return dict(row) if row else None

    def recent(self, session_id, limit):
        rows = self.connection.execute('SELECT * FROM readings WHERE session_id=? ORDER BY timestamp_ms DESC, sequence DESC LIMIT ?',
                                       (session_id, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def newer(self, session_id, after_ms, limit):
        """The oldest `limit` readings after a time, so a reader catching up never skips any."""
        rows = self.connection.execute('SELECT * FROM readings WHERE session_id=? AND timestamp_ms>? ORDER BY timestamp_ms, sequence LIMIT ?',
                                       (session_id, after_ms, limit)).fetchall()
        return [dict(row) for row in rows]

    def sessions(self, limit):
        """Saved sessions, most recently updated first."""
        rows = self.connection.execute(
            'SELECT s.session_id, s.participant, s.mode, COUNT(r.sequence) AS readings, MAX(r.timestamp_ms) AS last_ms, '
            'MAX(r.received_ms) AS received_ms FROM sessions s LEFT JOIN readings r ON r.session_id=s.session_id '
            'GROUP BY s.session_id ORDER BY received_ms DESC LIMIT ?', (limit,)).fetchall()
        return [dict(row) for row in rows]

    def cue_rows(self, session_id):
        """Results holding a delivered or would-be cue; dumps() writes these keys without spaces."""
        rows = self.connection.execute(
            """SELECT timestamp_ms, result_json FROM readings WHERE session_id=?
               AND (result_json LIKE '%"alert":{%' OR result_json LIKE '%"wouldAlert":{%') ORDER BY timestamp_ms, sequence""",
            (session_id,)).fetchall()
        return [dict(row) for row in rows]

    def latest(self, session_id):
        row = self.connection.execute('SELECT * FROM readings WHERE session_id=? ORDER BY timestamp_ms DESC, sequence DESC LIMIT 1',
                                      (session_id,)).fetchone()
        return dict(row) if row else None

    def save_reading(self, sample, result, received_ms, effective_sample=None):
        self.connection.execute('INSERT INTO readings(id,session_id,sample_id,timestamp_ms,received_ms,sample_json,effective_json,result_json) VALUES (?,?,?,?,?,?,?,?)',
                                (result['reading_id'], sample['session_id'], sample['sample_id'], sample['timestamp_ms'],
                                 received_ms, dumps(sample), dumps(effective_sample or sample), dumps(result)))
        for alert in result['alerts']:
            self.connection.execute('INSERT INTO alerts(id,session_id,reading_id,data_json) VALUES (?,?,?,?)',
                                    (alert['id'], sample['session_id'], result['reading_id'], dumps(alert)))

    def alert(self, session_id, alert_id):
        row = self.connection.execute('SELECT data_json FROM alerts WHERE session_id=? AND id=?', (session_id, alert_id)).fetchone()
        return json.loads(row['data_json']) if row else None

    def save_feedback(self, data):
        self.connection.execute('INSERT INTO feedback(id,session_id,kind,reading_id,alert_id,data_json) VALUES (?,?,?,?,?,?)',
                                (data['id'], data['session_id'], data['kind'], data.get('reading_id'), data.get('alert_id'), dumps(data)))

    def latest_rating(self, session_id):
        row = self.connection.execute(
            "SELECT data_json FROM feedback WHERE session_id=? AND kind='rating' ORDER BY sequence DESC LIMIT 1",
            (session_id,)).fetchone()
        return json.loads(row['data_json']) if row else None

    def summary(self, session_id):
        result = {'session_id': session_id}
        for table in ('readings', 'alerts'):
            result[table] = self.connection.execute(f'SELECT COUNT(*) FROM {table} WHERE session_id=?', (session_id,)).fetchone()[0]
        for kind, key in (('response', 'responses'), ('outcome', 'outcomes'), ('rating', 'ratings')):
            result[key] = self.connection.execute('SELECT COUNT(*) FROM feedback WHERE session_id=? AND kind=?', (session_id, kind)).fetchone()[0]
        return result

    def export_records(self, session_id):
        readings = [json.loads(row['result_json']) | {'sample': json.loads(row['sample_json'])}
                    for row in self.connection.execute('SELECT sample_json,result_json FROM readings WHERE session_id=? ORDER BY sequence', (session_id,))]
        alerts = [json.loads(row['data_json']) | {'reading_id': row['reading_id']}
                  for row in self.connection.execute('SELECT data_json,reading_id FROM alerts WHERE session_id=? ORDER BY sequence', (session_id,))]
        feedback = [json.loads(row['data_json']) for row in self.connection.execute('SELECT data_json FROM feedback WHERE session_id=? ORDER BY sequence', (session_id,))]
        return readings, alerts, feedback
