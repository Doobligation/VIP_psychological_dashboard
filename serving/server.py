"""Local durable feedback API. Run ``python -m serving.server`` to serve the dashboard."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

# Support both `python -m serving.server` and `python serving/server.py`.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage import SQLiteStore
from storage.sqlite_store import PRIVATE_DATABASES
from pipeline.feedback_pipeline import STATES
from pipeline import checkins as checkin_rules

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / '.local' / 'closed_loop.sqlite3'
MAX_JSON_BYTES = 1_000_000
ROUTES = {'/samples', '/latest', '/decision', '/context', '/feedback', '/summary', '/export', '/history', '/sessions', '/cues',
          '/checkins'}
MAX_HISTORY = 2000
MAX_SESSIONS = 200


def identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 200 or any(ord(c) < 32 for c in value):
        raise ValueError(f'{name} must be a nonempty string of at most 200 characters.')
    return value


def finite_json(value, depth=0):
    if depth > 40:
        raise ValueError('JSON nesting is too deep.')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Numeric values must be finite.')
    if isinstance(value, dict):
        for item in value.values(): finite_json(item, depth+1)
    elif isinstance(value, list):
        for item in value: finite_json(item, depth+1)


def iso_time(ms):
    return datetime.fromtimestamp(ms/1000, timezone.utc).isoformat().replace('+00:00', 'Z')


def _atomic_text(path, text):
    """Replace a file in one step, so Excel or a reader never sees half a file."""
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=path.parent, suffix='.tmp', delete=False) as stream:
        stream.write(text)
        temporary = stream.name
    os.replace(temporary, path)


class ClosedLoopService:
    def __init__(self, db_path=DEFAULT_DB, model_path=None, checkin_settings=None, checkin_dir=None):
        self.lock = threading.RLock()
        self.model_path = model_path
        self.store = SQLiteStore(db_path)
        self.checkin_settings = checkin_rules.load_settings(checkin_settings)
        # The guide's live_output/<worker>: one folder per participant beside the private database.
        self.checkin_dir = Path(checkin_dir) if checkin_dir else Path(db_path).parent / 'checkins'
        self.level_cache = {}
        self._restore_pipeline()
        self.causes = {key: cause.get('label', key) for state in checkin_rules.STATES
                       for key, cause in self.pipeline.rules['states'][state].get('causes', {}).items() if key != 'unspecified'}
        with self.lock:
            self._expire_checkins(time.time()*1000)

    def _restore_pipeline(self):
        from pipeline import FeedbackPipeline
        self.pipeline = FeedbackPipeline(model_path=self.model_path)
        for sample in self.store.replay_samples():
            self.pipeline.process(sample)
        for session_id, alerts in self.store.alert_histories().items():
            self.pipeline.restore_alert_history(session_id, alerts)

    def close(self):
        with self.lock: self.store.close()

    def _identity(self, payload):
        session_id = identifier(payload.get('session_id'), 'session_id')
        participant = identifier(payload.get('participant'), 'participant')
        mode = payload.get('mode')
        if mode not in ('live', 'demo', 'batch'): raise ValueError('mode must be live, demo or batch.')
        return session_id, participant, mode

    def process_sample(self, sample):
        from pipeline.ingest import validate_sample
        if not isinstance(sample, dict): raise ValueError('A JSON object is required.')
        finite_json(sample)
        # Validate the original evidence before a stale projection hides any fields.
        sample = validate_sample(sample)
        session_id, participant, mode = self._identity(sample)
        identifier(sample.get('sample_id'), 'sample_id')
        timestamp = sample.get('timestamp_ms')
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not 0 <= timestamp <= 253402300799999:
            raise ValueError('timestamp_ms must be a finite nonnegative epoch timestamp.')
        # Reject impossible future evidence rather than leaving latest permanently fresh.
        if mode in ('live', 'batch') and timestamp > time.time()*1000 + self._fresh_ms():
            raise ValueError('timestamp_ms is in the future.')
        with self.lock:
            # Close questions that ran out first, so an expired one never blocks the next.
            self._expire_checkins(time.time()*1000)
            existing = self.store.reading(session_id, sample_id=sample['sample_id'])
            if existing:
                # An idempotency key cannot silently change the session identity.
                session = self.store.session(session_id)
                if session['participant'] != participant or session['mode'] != mode:
                    raise ValueError('Session participant and mode cannot change.')
                return json.loads(existing['result_json'])
            try:
                with self.store.connection:
                    self.store.ensure_session(session_id, participant, mode)
                    received_ms = time.time()*1000
                    effective = copy.deepcopy(sample)
                    stale_on_arrival = mode == 'live' and received_ms - timestamp > self._fresh_ms()
                    if stale_on_arrival:
                        # Preserve original measurements, but replay the same ineligible observation.
                        snapshot = effective.setdefault('snapshot', {})
                        if not isinstance(snapshot, dict):
                            raise ValueError('snapshot must be an object.')
                        snapshot['fresh'] = False
                        snapshot['posture'] = {'level':'unavailable','reason':'Observation arrived stale.'}
                        # Late live EDA/temperature must not feed the frustration model either.
                        if isinstance(effective.get('raw'), dict):
                            for key in ('eda_us', 'temp_c'):
                                effective['raw'].pop(key, None)
                    result = self.pipeline.process(effective)
                    result['stale_on_arrival'] = stale_on_arrival
                    result['reading_id'] = uuid.uuid4().hex
                    # Opaque durable IDs are owned by this store, never derived from a participant.
                    mapping = {}
                    for alert in result['alerts']:
                        old = alert['id']
                        alert['id'] = uuid.uuid4().hex
                        mapping[old] = alert['id']
                    for state in result['states'].values():
                        if state.get('alert') and state['alert']['id'] in mapping:
                            state['alert']['id'] = mapping[state['alert']['id']]
                    self.store.save_reading(sample, result, received_ms, effective)
                    opened = self._consider_checkin(effective, result, received_ms)
            except Exception:
                # Failed storage must not leave an unlogged observation in decision state.
                self._restore_pipeline()
                raise
            if session_id in self.level_cache:
                self._count_levels(self.level_cache[session_id], self._level_row(effective, result))
            if opened:
                self._write_checkin_files(opened['participant'])
            return copy.deepcopy(result)

    # ------------------------------------------------------------ closed-loop check-ins (guide steps 5-8)

    def _consider_checkin(self, sample, result, now):
        """Open a Quick check for a High prediction: one open at a time, a gap between, recent moments only."""
        settings = self.checkin_settings
        states = checkin_rules.triggers(settings, sample, result)
        if not states or now - result['timestamp_ms'] > settings['maxAgeMinutes'] * checkin_rules.MINUTE_MS:
            return None
        session_id = result['session_id']
        if self.store.open_checkin(session_id):
            return None
        last = self.store.last_checkin_ms(session_id, 'prompted')
        if last is not None and now - last < settings['minGapMinutes'] * checkin_rules.MINUTE_MS:
            return None
        prediction = checkin_rules.predicted(sample, result)
        record = dict(id=uuid.uuid4().hex, kind='prompted', status='open', session_id=session_id,
                      participant=sample['participant'], mode=sample['mode'], device=checkin_rules.device_key(sample),
                      reading_id=result['reading_id'], moment_ms=result['timestamp_ms'], asked_ms=now,
                      expires_ms=now + settings['expireMinutes'] * checkin_rules.MINUTE_MS, answered_ms=None,
                      trigger=states, predicted=prediction, predictedCause=checkin_rules.predicted_cause(prediction, states),
                      answers={}, reportedCause=None, note='', features=checkin_rules.moment_features(result),
                      settingsVersion=settings['version'])
        self.store.save_checkin(record)
        return record

    def _expire_checkins(self, now):
        """A question nobody answers within expireMinutes is saved as "no answer"."""
        due = self.store.due_checkins(now)
        if not due:
            return
        with self.store.connection:
            for record in due:
                record.update(status='no-answer', closed_ms=record['expires_ms'])
                self.store.update_checkin(record)
        for participant in {record['participant'] for record in due}:
            self._write_checkin_files(participant)

    def _checkin_folder(self, participant):
        return self.checkin_dir / checkin_rules.folder_name(participant)

    def _write_checkin_files(self, participant):
        """Mirror the participant's check-ins to checkins.csv (Excel) and checkins.json (with features).

        SQLite stays the record; a failed file write is reported and retried on the next change."""
        records = self.store.checkins(participant=participant)
        little = self.checkin_settings['littleCountsAs']
        folder = self._checkin_folder(participant)
        try:
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            _atomic_text(folder / 'checkins.csv', checkin_rules.csv_text(records, little))
            _atomic_text(folder / 'checkins.json', checkin_rules.json_text(records, little))
        except OSError as exc:
            print(f'Could not write check-in files for {participant}: {exc}', file=sys.stderr)

    @staticmethod
    def _level_row(sample, result):
        row = {'source': (sample.get('raw') or {}).get('source') if isinstance(sample.get('raw'), dict) else None}
        for state in checkin_rules.STATES:
            d = result['states'].get(state) or {}
            row[f'{state}_available'], row[f'{state}_level'] = d.get('available'), d.get('level')
            row[f'{state}_prediction'] = d.get('prediction')
        return row

    @staticmethod
    def _count_levels(levels, row):
        if levels['device'] is None:
            levels['device'] = checkin_rules.device_label(row['source'] or 'polar-h10')
        for state in checkin_rules.STATES:
            prediction = row[f'{state}_prediction']
            if prediction is not None and not prediction:
                continue  # a held result between a wearable's predictions
            counts = levels[state]
            level = row[f'{state}_level'] if row[f'{state}_available'] and row[f'{state}_level'] in checkin_rules.LEVELS else 'unavailable'
            counts[level] += 1
            counts['run'] = counts['run'] + 1 if level in ('medium', 'high') else 0
            counts['longestRun'] = max(counts['longestRun'], counts['run'])

    def _levels(self, session_id):
        """Predictions by level for one session, plus the Medium-or-higher run; cached, then counted as readings arrive."""
        if session_id not in self.level_cache:
            levels = dict(device=None, **{state: dict(low=0, medium=0, high=0, unavailable=0, run=0, longestRun=0)
                                          for state in checkin_rules.STATES})
            for row in self.store.prediction_levels(session_id):
                self._count_levels(levels, dict(row))
            self.level_cache[session_id] = levels
        return copy.deepcopy(self.level_cache[session_id])

    def checkins(self, session_id):
        """The open question, this session's check-ins, agreement and predictions by level."""
        identifier(session_id, 'session_id')
        little = self.checkin_settings['littleCountsAs']
        with self.lock:
            now = time.time()*1000
            self._expire_checkins(now)
            records, everything = self.store.checkins(session_id=session_id), self.store.checkins()
            levels = self._levels(session_id)
            session = self.store.session(session_id)
        public = lambda r: {k: v for k, v in r.items() if k != 'features'} | {'agrees': checkin_rules.agreement(r, little)}
        folder = None
        if session:
            path = self._checkin_folder(session['participant'])
            try:
                folder = str(path.resolve().relative_to(ROOT))
            except ValueError:
                folder = str(path)
        settings = self.checkin_settings
        return dict(session_id=session_id, now_ms=now, open=next((public(r) for r in reversed(records) if r['status'] == 'open'), None),
                    checkins=[public(r) for r in reversed(records[-200:])], session=checkin_rules.summarize(records, little),
                    all=checkin_rules.summarize(everything, little), levels=levels, causes=self.causes, folder=folder,
                    settings={k: settings[k] for k in ('version', 'questions', 'askLevels', 'askSources', 'minGapMinutes',
                                                       'maxAgeMinutes', 'expireMinutes', 'littleCountsAs')})

    def checkin_action(self, payload):
        """Answer or dismiss an open question, or save a self-report ("How do you feel?")."""
        finite_json(payload)
        session_id = identifier(payload.get('session_id'), 'session_id')
        kind = payload.get('kind')
        little = self.checkin_settings['littleCountsAs']
        with self.lock:
            now = time.time()*1000
            self._expire_checkins(now)
            if kind in ('answer', 'dismiss'):
                record = self.store.checkin(session_id, identifier(payload.get('checkin_id'), 'checkin_id'))
                if not record:
                    raise ValueError('checkin_id does not belong to this session.')
                if record['status'] != 'open':
                    raise ValueError(f"This question is already closed ({checkin_rules.STATUS_LABELS[record['status']].lower()}).")
                if kind == 'answer':
                    answers, cause, note = checkin_rules.parse_answers(payload, self.causes)
                    record.update(status='answered', answers=answers, reportedCause=cause, note=note, answered_ms=now)
                else:
                    record.update(status='dismissed', closed_ms=now)
                with self.store.connection:
                    self.store.update_checkin(record)
            elif kind == 'self-report':
                session = self.store.session(session_id)
                if not session:
                    raise ValueError('Unknown session: readings must arrive before a self-report.')
                answers, cause, note = checkin_rules.parse_answers(payload, self.causes)
                last = self.store.last_checkin_ms(session_id, 'self-report')
                if last is not None and now - last < checkin_rules.SELF_REPORT_SPACING_MS:
                    raise ValueError('A self-report was just saved for this session; wait a moment before sending another.')
                row = self.store.latest(session_id)
                sample = json.loads(row['sample_json']) if row else {}
                linked = row is not None and abs(now - row['timestamp_ms']) <= checkin_rules.SELF_REPORT_MATCH_MS
                result = json.loads(row['result_json']) if linked else None
                device = checkin_rules.device_key(sample) if row else 'unknown'
                if linked:
                    prediction = checkin_rules.predicted(sample, result)
                else:
                    label = checkin_rules.device_label(device)
                    prediction = {s: dict(level='unavailable', cause=None, causeLabel=None, source=f'{label} · no reading yet')
                                  for s in checkin_rules.STATES}
                record = dict(id=uuid.uuid4().hex, kind='self-report', status='answered', session_id=session_id,
                              participant=session['participant'], mode=session['mode'], device=device,
                              reading_id=row['id'] if linked else None, moment_ms=now, asked_ms=now, expires_ms=None,
                              answered_ms=now, trigger=[], predicted=prediction,
                              predictedCause=checkin_rules.predicted_cause(prediction, checkin_rules.STATES),
                              answers=answers, reportedCause=cause, note=note,
                              features=checkin_rules.moment_features(result) if linked else None,
                              settingsVersion=self.checkin_settings['version'])
                if not linked:
                    record['featuresNote'] = ('No reading at this moment yet (delayed upload); retraining uses the '
                                              'session reading nearest this time.')
                with self.store.connection:
                    self.store.save_checkin(record)
            else:
                raise ValueError('kind must be answer, dismiss or self-report.')
            self._write_checkin_files(record['participant'])
            return {k: v for k, v in record.items() if k != 'features'} | {'agrees': checkin_rules.agreement(record, little)}

    def _fresh_ms(self):
        return self.pipeline.feedback['decision']['freshSeconds'] * 1000

    def latest(self, session_id):
        identifier(session_id, 'session_id')
        with self.lock:
            row = self.store.latest(session_id)
            if not row:
                return {'session_id': session_id, 'reading_id': None, 'stale': True, 'features': {},
                        'states': {state: {'state':state,'available':False,'level':'unavailable','alert':None,'reason':'No observations.'} for state in STATES}, 'alerts': []}
            result = json.loads(row['result_json'])
            now = time.time()*1000
            # Batch results are historical by design (delayed uploads); only live/demo data goes stale.
            result['stale'] = result.get('mode') != 'batch' and now - min(row['received_ms'], row['timestamp_ms']) > self._fresh_ms()
            if result['stale']:
                result['alerts'] = []
                features = result['features']
                features.update(fresh=False, fatigueEligible=False)
                for key in ('hrChangePct', 'rmssdChangePct', 'sdnnChangePct', 'edaChangePct', 'scrChangePct', 'absPitch', 'absRoll'):
                    features.pop(key, None)
                if isinstance(features.get('snapshot'), dict):
                    features['snapshot'].update(fresh=False, posture={'level':'unavailable','reason':'Latest observation is stale.'})
                    quick = features['snapshot'].get('quickFatigue')
                    if isinstance(quick, dict):
                        quick.update(available=False, score=None, hrBpm=None, held=False,
                                     reason='Latest saved observation is stale; waiting for another HR packet.')
                for state in result['states'].values():
                    state.update(available=False, level='unavailable', alert=None, reason='Latest observation is stale.', suppression='stale')
            return result

    def context(self, session_id):
        identifier(session_id, 'session_id')
        with self.lock:
            row = self.store.session(session_id)
            if not row: return {'session_id':session_id,'participant':None,'mode':None,'context':{}}
            return {'session_id':session_id,'participant':row['participant'],'mode':row['mode'],'context':json.loads(row['context_json'])}

    def set_context(self, payload):
        finite_json(payload)
        session_id, participant, mode = self._identity(payload)
        context = payload.get('context')
        if not isinstance(context, dict): raise ValueError('context must be an object.')
        with self.lock, self.store.connection:
            self.store.ensure_session(session_id, participant, mode)
            self.store.connection.execute('UPDATE sessions SET context_json=? WHERE session_id=?', (json.dumps(context, allow_nan=False), session_id))
            return self.context(session_id)

    def feedback(self, payload):
        finite_json(payload)
        session_id = identifier(payload.get('session_id'), 'session_id')
        kind = payload.get('kind')
        now = time.time()*1000
        record = {'id':uuid.uuid4().hex,'session_id':session_id,'kind':kind,'time':iso_time(now)}
        with self.lock, self.store.connection:
            if kind == 'rating':
                reading_id = identifier(payload.get('reading_id'), 'reading_id')
                row = self.store.reading(session_id, reading_id=reading_id)
                if not row: raise ValueError('reading_id does not belong to this session.')
                result = json.loads(row['result_json'])
                if result.get('mode') != 'live':
                    raise ValueError('Independent live ratings require a live reading; demo labels are excluded.')
                features = result['features']
                if (now-min(row['received_ms'],row['timestamp_ms']) > self._fresh_ms()
                        or not all(key in features for key in ('hrChangePct','rmssdChangePct','sdnnChangePct'))
                        or result['states']['fatigue'].get('suppression') == 'duplicate-or-reversed-timestamp'
                        or not result['states']['fatigue'].get('available')):
                    raise ValueError('Rating requires an eligible fresh fatigue reading.')
                observed = payload.get('observed_at_ms', row['timestamp_ms'])
                if (isinstance(observed, bool) or not isinstance(observed, (int, float))
                        or not math.isfinite(observed)
                        or not 0 <= observed-row['timestamp_ms'] <= self._fresh_ms()
                        or abs(now-observed) > self._fresh_ms()):
                    raise ValueError('observed_at_ms must be fresh and at or after the linked reading.')
                previous = self.store.latest_rating(session_id)
                spacing = self.pipeline.rules['featurePolicy']['windowSeconds'] * 1000
                if previous:
                    previous_ms = previous.get('observed_at_ms')
                    if previous_ms is None:
                        previous_ms = datetime.fromisoformat(previous['time'].replace('Z', '+00:00')).timestamp()*1000
                    if observed-previous_ms < spacing:
                        raise ValueError(f'Ratings must be at least {spacing/1000:g} seconds apart.')
                level = payload.get('level')
                if level not in ('none','low','medium','high','very-high'): raise ValueError('Unknown fatigue level.')
                task = payload.get('task', '')
                if not isinstance(task, str) or len(task) > 500: raise ValueError('task must be a string of at most 500 characters.')
                sample = json.loads(row['sample_json'])
                snapshot = sample.get('snapshot') or features.get('snapshot')
                if not isinstance(snapshot, dict): raise ValueError('Reading has no eligible measurement snapshot.')
                record.update(reading_id=reading_id, level=level, task=task, source='participant-self-report', snapshot=snapshot,
                              time=iso_time(observed), observed_at_ms=observed, received_at_ms=now)
            elif kind in ('response','outcome'):
                alert_id = identifier(payload.get('alert_id'), 'alert_id')
                if not self.store.alert(session_id, alert_id): raise ValueError('alert_id does not belong to this session.')
                key, choices = ('response', ('acknowledged','break-taken','not-helpful','dismissed')) if kind == 'response' else ('outcome', ('better','same','worse'))
                if payload.get(key) not in choices: raise ValueError(f'Unknown {key}.')
                record.update(alert_id=alert_id, **{key:payload[key]})
            else: raise ValueError('kind must be rating, response or outcome.')
            self.store.save_feedback(record)
            return record

    def history(self, session_id, limit=60, after=None):
        """Compact decisions for the most recent readings, or those after a time; the full export is far larger."""
        identifier(session_id, 'session_id')
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_HISTORY:
            raise ValueError(f'limit must be an integer from 1 to {MAX_HISTORY}.')
        keep = ('state', 'available', 'level', 'cause', 'causeLabel', 'causeMatch', 'causeAgreement', 'causeAssessed',
                'reason', 'alternatives', 'streak', 'suppression', 'alert', 'wouldAlert', 'prediction')
        with self.lock:
            rows = self.store.recent(session_id, limit) if after is None else self.store.newer(session_id, after, limit)
        out = []
        for row in rows:
            result = json.loads(row['result_json'])
            states = {}
            for name, state in result.get('states', {}).items():
                states[name] = {k: state.get(k) for k in keep}
                model = state.get('model') or {}
                if model:
                    states[name]['model'] = {k: model.get(k) for k in ('status', 'family', 'probabilities', 'accuracy', 'accuracyScope', 'reason')}
            features = result.get('features', {})
            snapshot = features.get('snapshot') if isinstance(features.get('snapshot'), dict) else {}
            compact = {k: v for k, v in features.items() if k.endswith('ChangePct')}
            compact['snapshot'] = {k: snapshot.get(k) for k in ('hr', 'rmssd', 'sdnn', 'eda', 'scr', 'baseline', 'worn',
                                                                'cleanBeatFraction', 'hrvUsable', 'edaBelowFloor')}
            out.append(dict(reading_id=result.get('reading_id'), session_id=session_id, timestamp_ms=row['timestamp_ms'],
                            mode=result.get('mode'), states=states, alerts=result.get('alerts', []), features=compact, stale=False))
        return out

    def sessions(self, limit=50):
        """Saved sessions for the dashboard's picker. Local-only, like every route here."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SESSIONS:
            raise ValueError(f'limit must be an integer from 1 to {MAX_SESSIONS}.')
        with self.lock:
            return self.store.sessions(limit)

    def cues(self, session_id):
        """Every cue in a session, oldest first: delivered, or recorded as would-be for late data."""
        identifier(session_id, 'session_id')
        with self.lock:
            rows = self.store.cue_rows(session_id)
        out = []
        for row in rows:
            for state in json.loads(row['result_json']).get('states', {}).values():
                cue, delivered = state.get('alert') or state.get('wouldAlert'), bool(state.get('alert'))
                if cue:
                    out.append(dict(time=row['timestamp_ms'], delivered=delivered,
                                    **{k: cue.get(k) for k in ('id', 'state', 'cause', 'causeLabel', 'causeMatch', 'message', 'consecutive')}))
        return out

    def summary(self, session_id):
        identifier(session_id, 'session_id')
        with self.lock: return self.store.summary(session_id)

    def export(self, session_id):
        identifier(session_id, 'session_id')
        with self.lock:
            context = self.context(session_id)
            readings, alerts, feedback = self.store.export_records(session_id)
            return {'schemaVersion':2,'sessionId':session_id,'participant':context['participant'], 'mode':context['mode'],
                    'context':context['context'], 'fatigueRatings':[r for r in feedback if r['kind']=='rating'],
                    'readings':readings,'alerts':alerts,'responses':[r for r in feedback if r['kind']=='response'],
                    'outcomes':[r for r in feedback if r['kind']=='outcome'],
                    'checkins':self.store.checkins(session_id=session_id)}


_default_service = None
_default_lock = threading.Lock()


def get_feedback_service():
    global _default_service
    with _default_lock:
        if _default_service is None: _default_service = ClosedLoopService()
        return _default_service


def private_path(path, root=ROOT):
    """Block hidden directories, runtime databases and aliases into private trees."""
    parsed = unquote(urlsplit(path).path)
    parts = Path(parsed).parts
    if any(part.startswith('.') and part not in ('.', '..') for part in parts): return True
    if any(part.lower().endswith(('.sqlite','.sqlite3','.db','.sqlite3-wal','.sqlite3-shm','.db-wal','.db-shm')) for part in parts): return True
    candidate = (Path(root) / parsed.lstrip('/')).resolve()
    if any(str(candidate) in (str(db), str(db)+'-wal', str(db)+'-shm', str(db)+'-journal') for db in PRIVATE_DATABASES):
        return True
    try: relative = candidate.relative_to(Path(root).resolve())
    except ValueError: return True
    return any(part.startswith('.') for part in relative.parts)


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate JSON keys are not allowed.')
        result[key] = value
    return result


class FeedbackAPIMixin:
    def _feedback_json(self, payload, status=200):
        body = json.dumps(payload, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _safe_host(self):
        try:
            authority = self.headers.get('Host', '')
            host = urlsplit('http://' + authority)
            if host.hostname not in ('localhost','127.0.0.1','::1') or host.username or host.password or host.path:
                return False
            origin = self.headers.get('Origin')
            if origin is None: return True
            parsed = urlsplit(origin)
            return (parsed.scheme == 'http' and parsed.netloc.lower() == authority.lower()
                    and not parsed.path and not parsed.query and not parsed.fragment)
        except ValueError: return False

    def handle_feedback_api(self):
        parsed = urlsplit(self.path)
        if parsed.path not in ROUTES: return False
        try:
            if not self._safe_host():
                self.close_connection = True
                self._feedback_json({'error':'Only same-origin loopback requests are allowed.'},403)
                return True
            method = self.command
            allowed = ('POST',) if parsed.path in ('/samples','/feedback') else ('GET','POST') if parsed.path in ('/context','/checkins') else ('GET',)
            if method not in allowed:
                self.close_connection = True
                self._feedback_json({'error':'Method not allowed.'},405)
                return True
            payload = None
            if method == 'POST':
                if self.headers.get('Transfer-Encoding'): raise ValueError('Transfer-Encoding is unsupported.')
                size = self.headers.get('Content-Length')
                if size is None or not size.isdigit(): raise ValueError('A valid Content-Length is required.')
                size = int(size)
                if size > MAX_JSON_BYTES:
                    self.close_connection = True
                    self._feedback_json({'error':'JSON body is too large.'},413)
                    return True
                body = self.rfile.read(size)
                if len(body) != size: raise ValueError('Incomplete JSON body.')
                payload = json.loads(body, object_pairs_hook=_json_pairs,
                                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Numeric values must be finite.')))
                if not isinstance(payload, dict): raise ValueError('A JSON object is required.')
                finite_json(payload)
            service = getattr(self, 'feedback_service', None) or get_feedback_service()
            if method == 'POST':
                handler = {'/samples':service.process_sample,'/feedback':service.feedback,'/context':service.set_context,
                           '/checkins':service.checkin_action}[parsed.path]
                result = handler(payload)
            else:
                query = parse_qs(parsed.query)
                limits = query.get('limit', ['60' if parsed.path == '/history' else '50'])
                if parsed.path in ('/history', '/sessions') and (len(limits) != 1 or not limits[0].isdigit()):
                    raise ValueError('limit must be a positive integer.')
                if parsed.path == '/sessions':
                    self._feedback_json(service.sessions(int(limits[0])))
                    return True
                sessions = query.get('session_id', [])
                if len(sessions) != 1: raise ValueError('Exactly one session_id is required.')
                session_id = identifier(sessions[0], 'session_id')
                if parsed.path == '/history':
                    afters = query.get('after', [])
                    if len(afters) > 1: raise ValueError('after may be given once.')
                    after = float(afters[0]) if afters else None
                    if after is not None and not math.isfinite(after): raise ValueError('after must be a finite time.')
                    result = service.history(session_id, int(limits[0]), after)
                else:
                    handler = {'/latest':service.latest,'/decision':service.latest,'/context':service.context,'/summary':service.summary,
                               '/export':service.export,'/cues':service.cues,'/checkins':service.checkins}[parsed.path]
                    result = handler(session_id)
            self._feedback_json(result)
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeDecodeError) as exc:
            self.close_connection = True
            self._feedback_json({'error':str(exc)[:240] or 'Invalid request.'},400)
        except (sqlite3.Error, OSError):
            self.close_connection = True
            self._feedback_json({'error':'Local storage is unavailable; this request was not saved.'},503)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    from http.server import ThreadingHTTPServer
    from serve_dashboard import Handler
    Handler.feedback_service = ClosedLoopService(args.db, model_path=args.model)
    print(f'Serving dashboard and feedback API at http://127.0.0.1:{args.port}/')
    try:
        with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
            server.serve_forever()
    finally:
        Handler.feedback_service.close()


if __name__ == '__main__': main()
