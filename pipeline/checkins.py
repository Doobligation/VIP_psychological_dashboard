"""Closed-loop check-ins: when to ask the worker about a High prediction, and how answers score it.

Follows the Closed-Loop Feedback guide (Nov 11, 2026), steps 5-8. The guide's "stress" is this
project's frustration (its stress labels come from the Frustration DBR) and its "mental load" is
fatigue. Settings live in config/checkins.json; storage and the HTTP routes are in serving/server.py,
and training/retrain_from_checkins.py turns answers into a candidate model (steps 9-10).
"""
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re

from .config import CONFIG_DIR

STATES = ('frustration', 'fatigue')
LEVELS = ('low', 'medium', 'high')
ANSWERS = ('yes', 'a-little', 'no')
ANSWER_LABELS = {'yes': 'Yes', 'a-little': 'A little', 'no': 'No'}
STATUS_LABELS = {'open': 'Open', 'answered': 'Answered', 'no-answer': 'No answer', 'dismissed': 'Not now'}
KIND_LABELS = {'prompted': 'Quick check', 'self-report': 'Self-report'}
# raw.source -> device. The Polar page sends no source, so a missing one is the H10.
DEVICES = {'empatica-avro': 'Empatica', 'team3-dashboard': 'Scripted demo', 'synthetic-demo': 'Polar demo',
           'synthetic-script': 'Terminal demo', 'polar-h10': 'Polar H10'}
EXTRA_CAUSES = {'other': 'Something else', 'not-sure': 'Not sure'}
# A self-report is linked to the latest reading only when that reading is this close to "now".
SELF_REPORT_MATCH_MS = 120_000
# Two self-reports closer than this are a double submission, not two answers.
SELF_REPORT_SPACING_MS = 30_000
MAX_NOTE = 500
MINUTE_MS = 60_000


def load_settings(value=None):
    """Validated check-in settings from config/checkins.json (or a dict, for tests)."""
    data = deepcopy(value) if isinstance(value, dict) else json.loads(
        Path(value or CONFIG_DIR / 'checkins.json').read_text(encoding='utf-8'))
    try:
        if data.get('schemaVersion') != 1 or not isinstance(data.get('version'), str) or not data['version'].strip():
            raise ValueError
        for key in ('minGapMinutes', 'maxAgeMinutes', 'expireMinutes'):
            value = data[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError
        if data['littleCountsAs'] not in ('yes', 'no'):
            raise ValueError
        levels = data['askLevels']
        if not isinstance(levels, dict) or not set(levels) <= set(STATES):
            raise ValueError
        if any(not isinstance(v, list) or not set(v) <= set(LEVELS) for v in levels.values()):
            raise ValueError
        if not isinstance(data['askSources'], list) or not all(isinstance(s, str) and s for s in data['askSources']):
            raise ValueError
        questions = data['questions']
        if not all(isinstance(questions.get(k), str) and questions[k].strip()
                   for k in ('frustration', 'fatigue', 'cause', 'selfFrustration', 'selfFatigue')):
            raise ValueError
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Invalid config/checkins.json: check schemaVersion, version, askLevels, askSources, '
                         'minGapMinutes, maxAgeMinutes, expireMinutes, littleCountsAs and questions.') from exc
    return data


def device_key(sample):
    raw = sample.get('raw') if isinstance(sample.get('raw'), dict) else {}
    source = raw.get('source')
    return source if isinstance(source, str) and source else 'polar-h10'


def device_label(key):
    return DEVICES.get(key, key)


def method_label(state, decision):
    if state == 'frustration' and (decision.get('model') or {}).get('status') == 'ml':
        return 'ML model'
    return 'DBR rules'


def predicted(sample, result):
    """What each state predicted at this reading, and which source made the prediction."""
    device = device_label(device_key(sample))
    out = {}
    for state in STATES:
        d = (result.get('states') or {}).get(state) or {}
        entry = dict(level=d.get('level') if d.get('available') else 'unavailable',
                     cause=d.get('cause'), causeLabel=d.get('causeLabel'),
                     source=f'{device} · {method_label(state, d)}')
        model = d.get('model') or {}
        if model.get('status') == 'ml':
            entry['probabilities'] = model.get('probabilities')
        out[state] = entry
    return out


def moment_features(result):
    """The sensor features of one reading, stored with the answer (checkins.json) for retraining."""
    features = result.get('features') or {}
    snapshot = features.get('snapshot') if isinstance(features.get('snapshot'), dict) else {}
    model = ((result.get('states') or {}).get('frustration') or {}).get('model') or {}
    return dict(changePct={k: v for k, v in features.items() if k.endswith('ChangePct')},
                model=model.get('features') if model.get('status') == 'ml' else None,
                normalization=model.get('normalization') if model.get('status') == 'ml' else None,
                snapshot={k: snapshot.get(k) for k in ('hr', 'rmssd', 'sdnn', 'eda', 'scr', 'worn', 'cleanBeatFraction')
                          if k in snapshot})


def triggers(settings, sample, result):
    """States whose prediction at this reading is one the guide asks about (step 5)."""
    if device_key(sample) not in settings['askSources']:
        return []
    out = []
    for state in STATES:
        d = (result.get('states') or {}).get(state) or {}
        # Between predictions a wearable session holds the last result; only a new prediction counts.
        if (d.get('available') and d.get('level') in settings['askLevels'].get(state, [])
                and d.get('prediction') is not False and d.get('suppression') != 'duplicate-or-reversed-timestamp'):
            out.append(state)
    return out


def predicted_cause(prediction, states):
    """The cause to fill in for "What was going on?": from the first triggering state that named one."""
    for state in states:
        p = prediction.get(state) or {}
        if p.get('cause') and p['cause'] != 'unspecified' and p.get('level') not in (None, 'unavailable', 'low'):
            return dict(key=p['cause'], label=p.get('causeLabel') or p['cause'], state=state)
    return None


def counts_as_felt(answer, little='yes'):
    return answer == 'yes' or (answer == 'a-little' and little == 'yes')


def agreement(record, little='yes'):
    """Per state: did the answer agree with the prediction? None when either side is missing."""
    out = {}
    for state in STATES:
        answer = (record.get('answers') or {}).get(state)
        level = ((record.get('predicted') or {}).get(state) or {}).get('level')
        out[state] = None if not answer or level in (None, 'unavailable') else (level == 'high') == counts_as_felt(answer, little)
    return out


def summarize(records, little='yes'):
    """Question counts, and per state and source how often a High prediction was confirmed (step 8)."""
    counts = dict(asked=0, answered=0, noAnswer=0, dismissed=0, open=0, selfReports=0)
    rows = {}
    for record in records:
        if record['kind'] == 'self-report':
            counts['selfReports'] += 1
        else:
            counts['asked'] += 1
            counts[{'answered': 'answered', 'no-answer': 'noAnswer', 'dismissed': 'dismissed', 'open': 'open'}[record['status']]] += 1
        if record['status'] != 'answered':
            continue
        for state in STATES:
            answer = (record.get('answers') or {}).get(state)
            p = (record.get('predicted') or {}).get(state) or {}
            if not answer or p.get('level') in (None, 'unavailable'):
                continue
            source = p.get('source') or 'Unknown'
            if record.get('mode') == 'demo':
                source += ' (demo)'
            row = rows.setdefault((state, source), dict(state=state, source=source, answers=0, predictedHigh=0,
                                                        yes=0, aLittle=0, no=0, notHigh=0, feltWhenNotHigh=0, agree=0))
            felt, high = counts_as_felt(answer, little), p['level'] == 'high'
            row['answers'] += 1
            row['agree'] += high == felt
            if high:
                row['predictedHigh'] += 1
                row[{'yes': 'yes', 'a-little': 'aLittle', 'no': 'no'}[answer]] += 1
            else:
                row['notHigh'] += 1
                row['feltWhenNotHigh'] += felt
    out = []
    for row in sorted(rows.values(), key=lambda r: (STATES.index(r['state']), r['source'])):
        confirmed = row['yes'] + (row['aLittle'] if little == 'yes' else 0)
        row['confirmed'] = confirmed
        row['confirmedRate'] = round(confirmed / row['predictedHigh'], 4) if row['predictedHigh'] else None
        row['agreement'] = round(row['agree'] / row['answers'], 4)
        out.append(row)
    return dict(counts=counts, agreement=out, littleCountsAs=little)


def parse_answers(payload, causes):
    """Validated (answers, reported cause, note) from a POST; both questions must be answered."""
    answers = payload.get('answers')
    if not isinstance(answers, dict) or set(answers) != set(STATES) or any(answers[s] not in ANSWERS for s in STATES):
        raise ValueError('answers must give frustration and fatigue, each yes, a-little or no.')
    cause = payload.get('cause') or None
    known = {**causes, **EXTRA_CAUSES}
    if cause is not None and cause not in known:
        raise ValueError('cause must be one of the listed causes, other or not-sure.')
    note = payload.get('note', '')
    if not isinstance(note, str) or len(note) > MAX_NOTE or any(ord(c) < 32 and c not in '\t' for c in note):
        raise ValueError(f'note must be one line of at most {MAX_NOTE} characters.')
    return ({s: answers[s] for s in STATES}, dict(key=cause, label=known[cause]) if cause else None, note.strip())


def folder_name(participant):
    """A safe per-worker folder name: the guide's live_output/<worker>."""
    name = re.sub(r'[^A-Za-z0-9_.-]', '_', participant)[:120].lstrip('.') or 'unknown'
    # Two IDs that clean to the same name must not share (and overwrite) one folder.
    return name if name == participant else f"{name}-{hashlib.sha1(participant.encode('utf-8')).hexdigest()[:8]}"


def _local(ms):
    return datetime.fromtimestamp(ms / 1000).astimezone().strftime('%Y-%m-%d %H:%M:%S') if ms is not None else ''


CSV_COLUMNS = ['id', 'type', 'status', 'session', 'participant', 'mode', 'asked_about', 'asked_at', 'answered_at',
               'trigger', 'frustration_predicted', 'frustration_source', 'fatigue_predicted', 'fatigue_source',
               'predicted_cause', 'felt_frustrated', 'felt_tired', 'what_was_going_on', 'note',
               'frustration_agrees', 'fatigue_agrees', 'asked_about_utc']


def csv_text(records, little='yes'):
    """One row per question, oldest first, for reading in Excel (times in this computer's time zone)."""
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(CSV_COLUMNS)
    yes_no = {True: 'yes', False: 'no', None: ''}
    for r in sorted(records, key=lambda r: (r['asked_ms'], r['id'])):
        p, a, agrees = r.get('predicted') or {}, r.get('answers') or {}, agreement(r, little)
        writer.writerow([
            r['id'], KIND_LABELS[r['kind']], STATUS_LABELS[r['status']], r['session_id'], r['participant'], r['mode'],
            _local(r['moment_ms']), _local(r['asked_ms']), _local(r.get('answered_ms')), ' + '.join(r.get('trigger') or []),
            (p.get('frustration') or {}).get('level', ''), (p.get('frustration') or {}).get('source', ''),
            (p.get('fatigue') or {}).get('level', ''), (p.get('fatigue') or {}).get('source', ''),
            (r.get('predictedCause') or {}).get('label', ''),
            ANSWER_LABELS.get(a.get('frustration'), ''), ANSWER_LABELS.get(a.get('fatigue'), ''),
            (r.get('reportedCause') or {}).get('label', ''), r.get('note', ''),
            yes_no[agrees['frustration']], yes_no[agrees['fatigue']],
            datetime.fromtimestamp(r['moment_ms'] / 1000, timezone.utc).isoformat().replace('+00:00', 'Z')])
    # The byte-order mark makes Excel read the file as UTF-8 (sources contain "·").
    return '﻿' + stream.getvalue()


def json_text(records, little='yes'):
    """Every question with the sensor features of its moment, for retraining and analysis."""
    rows = [dict(r, agrees=agreement(r, little)) for r in sorted(records, key=lambda r: (r['asked_ms'], r['id']))]
    return json.dumps(dict(schemaVersion=1, littleCountsAs=little, checkins=rows), indent=2, allow_nan=False) + '\n'
