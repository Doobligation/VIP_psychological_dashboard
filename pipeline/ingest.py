"""Validate bounded sample identities and JSON sensor measurements."""
from copy import deepcopy
import math


def finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 256 or any(ord(c) < 32 for c in value):
        raise ValueError(f'{name} must be a nonempty string of at most 256 characters.')
    return value


def validate_sample(sample):
    if not isinstance(sample, dict):
        raise ValueError('A sample object is required.')
    for key in ('session_id', 'sample_id', 'participant'):
        identifier(sample.get(key), key)
    # batch: real device data that arrives late (e.g. Empatica cloud uploads); never delivers cues.
    if sample.get('mode') not in ('live', 'demo', 'batch'):
        raise ValueError('mode must be live, demo or batch.')
    if not finite(sample.get('timestamp_ms')) or sample['timestamp_ms'] < 0:
        raise ValueError('timestamp_ms must be a finite nonnegative number.')
    if not isinstance(sample.get('snapshot'), dict) and not isinstance(sample.get('raw'), dict):
        raise ValueError('A snapshot or raw sensor object is required.')
    numeric_fields = {'hr', 'heart_rate_bpm', 'rmssd', 'sdnn', 'eda', 'eda_us', 'edaChangePct', 'scr', 'scrChangePct', 'temp_c', 'pitch', 'roll', 'rrCount', 'windowSeconds',
                      'requiredWindowSeconds', 'baselineProgress', 'recv_epoch_time_ms', 'hrChangePct',
                      'rmssdChangePct', 'sdnnChangePct', 'absPitch', 'absRoll', 'x', 'y', 'z'}
    def check(value, depth=0):
        if depth > 20:
            raise ValueError('Sample nesting is too deep.')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Sensor values must be finite.')
        if isinstance(value, dict):
            for key, child in value.items():
                if key in numeric_fields and child is not None and not finite(child):
                    raise ValueError(f'{key} must be a finite number or null.')
                if key == 'rr_intervals_ms':
                    if not isinstance(child, list) or len(child) > 4096 or not all(finite(v) for v in child):
                        raise ValueError('rr_intervals_ms must contain at most 4096 finite numbers.')
                check(child, depth + 1)
        elif isinstance(value, list):
            if len(value) > 16384:
                raise ValueError('Sensor array is too large.')
            for child in value:
                check(child, depth + 1)
    check(sample)
    return deepcopy(sample)
