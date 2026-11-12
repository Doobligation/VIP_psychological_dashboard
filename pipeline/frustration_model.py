"""Frustration ML: shared window features, live EDA/temperature tracking and portable inference.

Training (training/train_frustration_model.py) and the live server use the same
`window_features` and `signal_scale`, so a model sees identical inputs in both places. Each
signal is scaled per wearer using only that session's earlier data, one of three ways:

  running-minmax  the session's running min/max
  baseline-z      z-score against the wearer's first minute
  rolling-z       z-score against the wearer's recent past, the current window excluded (deployed)

All are unchanged by a linear rescale of the input, so a model trained on the labelled files
(already min-max scaled) receives the same features from raw microsiemens.
"""
import json
import math
from pathlib import Path
import struct
from .ingest import finite

FEATURES = ['eda_mean', 'eda_std', 'eda_slope', 'eda_range', 'eda_peaks', 'eda_peaks_small', 'eda_absdiff',
            'eda_last', 'eda_skew', 'temp_mean', 'temp_slope', 'temp_std']
CLASSES = ['low', 'medium', 'high']
WINDOW_SECONDS = 60
MIN_WINDOW_SECONDS = 55
# Peak thresholds are fractions of the session's running EDA range (SCR-like responses).
PEAK_AMPLITUDE, SMALL_PEAK_AMPLITUDE = .05, .02
NORMALIZATIONS = ('running-minmax', 'baseline-z', 'rolling-z')
ROLLING_HISTORY_SECONDS = 1200
# z-scored signals are divided by this many standard deviations, keeping feature magnitudes
# and the peak fractions above comparable across normalizations (peaks at 0.5 and 0.2 SD).
Z_SPAN = 10
# A nearly flat reference minute cannot define a unit, so the SD has a floor of 1% of the
# session's range. That also bounds every scaled value to +/- Z_SPAN.
MIN_SD_FRACTION = .01


def _mean(v):
    return sum(v) / len(v)


def _std(v):
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _slope_per_minute(v):
    n = len(v)
    if n < 2:
        return 0.0
    xbar, ybar = (n - 1) / 2, _mean(v)
    return sum((i - xbar) * (y - ybar) for i, y in enumerate(v)) / sum((i - xbar) ** 2 for i in range(n)) * 60


def _skew(v):
    """Bias-corrected sample skewness (pandas convention); 0 for a flat signal."""
    n = len(v)
    if n < 3:
        return 0.0
    m = _mean(v)
    m2 = sum((x - m) ** 2 for x in v) / n
    if m2 <= 1e-14:
        return 0.0
    m3 = sum((x - m) ** 3 for x in v) / n
    return math.sqrt(n * (n - 1)) / (n - 2) * m3 / m2 ** 1.5


def count_peaks(v, amplitude):
    count, trough, peak = 0, v[0], None
    for x in v[1:]:
        if peak is not None and x < peak:
            count, trough, peak = count + 1, x, None
        elif peak is not None:
            peak = x
        elif x < trough:
            trough = x
        elif x - trough >= amplitude:
            peak = x
    return count


def _scaled(values, low, high):
    return [(x - low) / (high - low) for x in values] if high > low else [0.0] * len(values)


def window_features(eda, temp, eda_range, temp_range):
    """Features for one window of 1 Hz values, scaled by the session's running (min, max)."""
    e, t = _scaled(eda, *eda_range), _scaled(temp, *temp_range)
    per_minute = 60 / len(e)
    return dict(
        eda_mean=_mean(e), eda_std=_std(e), eda_slope=_slope_per_minute(e), eda_range=max(e) - min(e),
        eda_peaks=count_peaks(e, PEAK_AMPLITUDE) * per_minute,
        eda_peaks_small=count_peaks(e, SMALL_PEAK_AMPLITUDE) * per_minute,
        eda_absdiff=_mean([abs(b - a) for a, b in zip(e, e[1:])]), eda_last=_mean(e[-10:]), eda_skew=_skew(e),
        temp_mean=_mean(t), temp_slope=_slope_per_minute(t), temp_std=_std(t))


def signal_scale(values, first, low, high, normalization, window_seconds=WINDOW_SECONDS):
    """(low, high) for `window_features`. `values` are the session's readings so far, `first` its
    first minute, and low/high its running range; only the past is used, as in live serving."""
    if normalization == 'running-minmax':
        return low, high
    reference = first
    if normalization == 'rolling-z':
        # Everything comes from the recent past, including the range behind the SD floor, so a spike
        # when the band went on an hour ago, or the choice of start time, stops affecting predictions.
        recent = values[-(window_seconds + ROLLING_HISTORY_SECONDS):]
        low, high = min(recent), max(recent)
        settled = recent[:-window_seconds]
        if len(settled) >= window_seconds:
            reference = settled
    centre = _mean(reference)
    return centre, centre + Z_SPAN * max(_std(reference), MIN_SD_FRACTION * (high - low))


class SignalTracker:
    """Per-session EDA and skin-temperature history at 1 Hz with a running range.

    min_eda: a wearable's validity floor in microsiemens. Scaling is relative, so without it the
    noise of a sensor off the skin is stretched to look like a full-range signal."""
    def __init__(self, window_seconds=WINDOW_SECONDS, min_seconds=MIN_WINDOW_SECONDS, min_eda=None):
        self.window_seconds, self.min_seconds, self.min_eda = window_seconds, min_seconds, min_eda
        self.first = []         # the session's first minute of completed (eda, temp) pairs
        self.bins = {}          # second -> {'eda': [sum, n], 'temp': [sum, n]}
        self.seconds = []       # completed seconds with both signals, ascending
        self.values = {}        # completed second -> (eda, temp)
        self.range = dict(eda=[math.inf, -math.inf], temp=[math.inf, -math.inf])
        self.last_second = None
        self.retain_seconds = window_seconds + ROLLING_HISTORY_SECONDS

    def _complete(self, second):
        both = self.bins.pop(second, {})
        if len(both) < 2:
            return
        eda, temp = (both[k][0] / both[k][1] for k in ('eda', 'temp'))
        self.values[second] = (eda, temp)
        self.seconds.append(second)
        if len(self.first) < self.window_seconds:
            self.first.append((eda, temp))
        for key, value in (('eda', eda), ('temp', temp)):
            low, high = self.range[key]
            self.range[key] = [min(low, value), max(high, value)]
        while self.seconds and self.seconds[0] < second - self.retain_seconds:
            self.values.pop(self.seconds.pop(0), None)

    def add(self, timestamp_ms, eda=None, temp=None):
        second = int(timestamp_ms // 1000)
        if self.last_second is not None and second < self.last_second:
            return
        for pending in [s for s in self.bins if s < second]:
            self._complete(pending)
        self.last_second = second
        slot = self.bins.setdefault(second, {})
        for key, value in (('eda', eda), ('temp', temp)):
            if finite(value):
                total = slot.setdefault(key, [0.0, 0])
                total[0] += value
                total[1] += 1

    def features(self, now_ms, normalization='running-minmax'):
        """Features for the last full window of completed seconds, or None with a reason."""
        now = int(now_ms // 1000)
        recent = [s for s in self.seconds if now - self.window_seconds <= s < now]
        if len(recent) < self.min_seconds:
            return None, f'Needs {self.min_seconds} s of EDA and skin temperature in the last {self.window_seconds} s (has {len(recent)} s).'
        eda = [self.values[s][0] for s in recent]
        temp = [self.values[s][1] for s in recent]
        if self.min_eda is not None and _mean(eda) < self.min_eda:
            return None, f'EDA averages {_mean(eda):.3f} µS, below the {self.min_eda} µS validity floor.'
        done = [s for s in self.seconds if s < now]
        scales = [signal_scale([self.values[s][index] for s in done], [pair[index] for pair in self.first],
                               *self.range[key], normalization, self.window_seconds)
                  for index, key in enumerate(('eda', 'temp'))]
        return window_features(eda, temp, *scales), None


def _f32(value):
    """Tree libraries compare features as float32; round the same way for identical splits."""
    return struct.unpack('f', struct.pack('f', value))[0]


def _softmax(values):
    top = max(values)
    weights = [math.exp(v - top) for v in values]
    return [w / sum(weights) for w in weights]


class FrustrationModel:
    """Stdlib inference for exported logistic, sklearn tree-ensemble and XGBoost models."""
    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if (data.get('schemaVersion') != 1 or data.get('features') != FEATURES or data.get('classes') != CLASSES
                or data.get('modelType') not in ('logistic', 'tree-ensemble', 'xgboost')
                or data.get('normalization', NORMALIZATIONS[0]) not in NORMALIZATIONS):
            raise ValueError('Unsupported frustration model file.')
        self.data = data
        self.kind = data['modelType']
        self.normalization = data.get('normalization', NORMALIZATIONS[0])

    @staticmethod
    def _walk(tree, x):
        node = 0
        while tree['left'][node] != -1:
            value, threshold = _f32(x[tree['feature'][node]]), tree['threshold'][node]
            go_left = value < _f32(threshold) if tree.get('strict') else value <= threshold
            node = tree['left'][node] if go_left else tree['right'][node]
        return tree['value'][node]

    def predict_proba(self, features):
        x = [float(features[name]) for name in FEATURES]
        if not all(math.isfinite(v) for v in x):
            raise ValueError('Frustration features must be finite.')
        d = self.data
        if self.kind == 'logistic':
            z = [(v - m) / s for v, m, s in zip(x, d['mean'], d['scale'])]
            return _softmax([sum(w * v for w, v in zip(row, z)) + b for row, b in zip(d['coefficients'], d['intercepts'])])
        if self.kind == 'tree-ensemble':
            totals = [0.0] * len(CLASSES)
            for tree in d['trees']:
                leaf = self._walk(tree, x)
                total = sum(leaf)
                totals = [a + b / total for a, b in zip(totals, leaf)]
            return [v / len(d['trees']) for v in totals]
        margins = list(d['baseMargins'])
        for index, tree in enumerate(d['trees']):
            margins[index % len(CLASSES)] += self._walk(tree, x)[0]
        return _softmax(margins)

    def predict(self, features):
        probabilities = self.predict_proba(features)
        best = max(range(len(CLASSES)), key=lambda i: probabilities[i])
        return CLASSES[best], dict(zip(CLASSES, probabilities))
