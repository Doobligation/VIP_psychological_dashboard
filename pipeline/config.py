"""Load and validate versioned, editable rule and feedback settings."""
from copy import deepcopy
import json
from pathlib import Path
from .ingest import finite

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
LEGACY_FEATURE_POLICY = dict(windowSeconds=60, freshSeconds=5, minRr=30, rrMinMs=300, rrMaxMs=2000, baselineSamples=30)


def _number(value, minimum=0, integer=False):
    return finite(value) and value >= minimum and (not integer or isinstance(value, int))


def validate_feature_policy(value):
    """Validate a portable measurement protocol and return an independent copy."""
    if not isinstance(value, dict) or set(value) != set(LEGACY_FEATURE_POLICY):
        raise ValueError('Feature policy must provide exactly the supported measurement settings.')
    p = {key: deepcopy(value[key]) for key in LEGACY_FEATURE_POLICY}
    if (not all(_number(p[k], minimum=0.000001) for k in ('windowSeconds', 'freshSeconds', 'rrMinMs', 'rrMaxMs'))
            or p['freshSeconds'] >= p['windowSeconds'] or p['rrMaxMs'] <= p['rrMinMs']
            or not _number(p['minRr'], 2, True) or not _number(p['baselineSamples'], 1, True)):
        raise ValueError('Invalid feature policy.')
    return p


def _validate_source_policies(policies):
    """Measurement settings for a raw wearable source, keyed by the sample's raw.source."""
    if not isinstance(policies, dict):
        raise ValueError('sourcePolicies must be an object.')
    for policy in policies.values():
        validate_feature_policy(policy['featurePolicy'])
        if not 0 < policy['beatTolerance'] < 1 or not 0 <= policy['minCleanBeatFraction'] <= 1:
            raise ValueError('Beat cleaning settings must be fractions.')
        for key in ('beatNeighbours', 'baselineMinSeconds'):
            if not _number(policy[key], 1, True):
                raise ValueError(f'{key} must be a positive integer.')
        for key in ('edaFloorUs', 'scrFloorPerMinute', 'baselineHistorySeconds', 'baselineGapSeconds', 'predictionSeconds'):
            if not _number(policy[key], .000001):
                raise ValueError(f'{key} must be positive.')
        if not 0 <= policy['wear']['minPercent'] <= 100 or not finite(policy['wear']['minTemperatureC']):
            raise ValueError('Invalid wear detection settings.')


def _validate_rules(data):
    validate_feature_policy(data['featurePolicy'])
    _validate_source_policies(data.get('sourcePolicies', {}))
    bands = data['probabilityBands']
    if not isinstance(bands, list) or not bands or bands[0]['min'] != 0:
        raise ValueError('Score bands must start at zero.')
    previous = -1
    for band in bands:
        if (not isinstance(band['level'], str) or not _number(band['severity'], 0, True)
                or not _number(band['min']) or not previous < band['min'] <= 1
                or not _number(band['score']) or band['score'] > 1):
            raise ValueError('Invalid ascending score band.')
        previous = band['min']
    for state in data['states'].values():
        signals = state['requiredSignals']
        if (not isinstance(state['enabled'], bool) or not isinstance(signals, list)
                or not all(isinstance(s, str) and s for s in signals) or len(signals) != len(set(signals))
                or not _number(state['minEvidence'], 1, True) or not isinstance(state['rules'], list)
                or (state['enabled'] and state['minEvidence'] > len(signals))
                or not _number(state.get('closestPatternMinSignals', 0), 0, True)
                or state.get('closestPatternMinSignals', 0) > len(signals)):
            raise ValueError('Invalid state minimum evidence or signals.')
        for rule in state['rules']:
            conditions = rule['conditions']
            if (rule['match'] not in ('all', 'any') or not isinstance(conditions, list)
                    or not _number(rule['minEvidence'], 1, True) or rule['minEvidence'] > len(conditions)
                    or not _number(rule['severity'], 0, True) or not _number(rule['score']) or rule['score'] > 1
                    or not all(isinstance(rule[k], str) and rule[k] for k in ('cause', 'level', 'explanation'))):
                raise ValueError('Invalid state rule.')
            for condition in conditions:
                if (condition['operator'] not in ('>', '<', '>=', '<=', '==', '!=')
                        or not isinstance(condition['signal'], str) or not condition['signal'] or not finite(condition['value'])):
                    raise ValueError('Invalid rule condition.')


# Assignment rule: Low and Medium risk (and normal levels) never issue feedback.
NO_FEEDBACK_LEVELS = ('none', 'low', 'medium', 'good')
SCHEMA_VERSIONS = {'rules.json': 1, 'feedback_table.json': 2}


def _validate_feedback(data):
    d = data['decision']
    if (not _number(d['freshSeconds'], .000001) or not _number(d['cooldownSeconds'])
            or not _number(d['consecutiveHighPredictions'], 1, True)
            or not _number(d['warningWindowSeconds'], .000001) or not _number(d['maxWarnings'], 0, True)):
        raise ValueError('Invalid decision timing, consecutive prediction count or warning limit.')
    if not _number(d.get('sameCueRepeatSeconds', 0)):
        raise ValueError('sameCueRepeatSeconds must be nonnegative.')
    for levels in data['states'].values():
        if not isinstance(levels, dict):
            raise ValueError('Feedback state levels must be objects.')
        for level, row in levels.items():
            causes = row['causes']
            if not isinstance(row['enabled'], bool) or not isinstance(causes, dict):
                raise ValueError('Invalid feedback row.')
            if row['enabled'] and (level in NO_FEEDBACK_LEVELS or not causes):
                raise ValueError('Low and medium levels cannot issue feedback; enabled levels need cause rows.')
            for cause, text in causes.items():
                if (not isinstance(cause, str) or not cause
                        or not all(isinstance(text[k], str) and text[k].strip() for k in ('message', 'action', 'goal'))):
                    raise ValueError('Feedback cause rows need a message, action and goal.')


def load_config(value, filename):
    if value is None:
        value = CONFIG_DIR / filename
    data = deepcopy(value) if isinstance(value, dict) else json.loads(Path(value).read_text(encoding='utf-8'))
    if (not isinstance(data, dict) or data.get('schemaVersion') != SCHEMA_VERSIONS[filename]
            or not isinstance(data.get('version'), str) or not data['version'].strip()):
        raise ValueError(f'Invalid versioned {filename} configuration.')
    try:
        if not isinstance(data['states'], dict):
            raise ValueError('States must be an object.')
        (_validate_rules if filename == 'rules.json' else _validate_feedback)(data)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f'Malformed {filename} configuration.') from exc
    return data
