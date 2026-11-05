"""Rolling raw HRV/SCR and native browser snapshot feature extraction."""
from collections import deque
from copy import deepcopy
import math
from statistics import median
from .config import LEGACY_FEATURE_POLICY, load_config, validate_feature_policy
from .ingest import finite

# Minimum trough-to-peak EDA rise counted as a skin conductance response (common 0.01 µS criterion).
SCR_MIN_AMPLITUDE_US = 0.01


def calculate_hrv(rr_intervals_ms, rr_min_ms=300, rr_max_ms=2000):
    values, differences = [], []
    previous = None
    for value in rr_intervals_ms:
        if not finite(value) or not rr_min_ms <= value <= rr_max_ms:
            previous = None
            continue
        values.append(value)
        if previous is not None:
            differences.append((value - previous) ** 2)
        previous = value
    mean = sum(values) / len(values) if values else 0
    return dict(rmssd=math.sqrt(sum(differences) / len(differences)) if differences else None,
                sdnn=math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) >= 2 else None)


def clean_beats(intervals, tolerance=.2, neighbours=5):
    """Flag each beat interval as usable: within `tolerance` of the median of the `neighbours`
    intervals on each side. Wrist PPG misses beats (one interval reads about double) and picks up
    noise peaks, and both pass a 300-2000 ms range check. Too few neighbours means not usable."""
    flags = []
    for i, value in enumerate(intervals):
        around = intervals[max(0, i - neighbours):i] + intervals[i + 1:i + 1 + neighbours]
        if len(around) < 3:
            flags.append(False)
            continue
        centre = median(around)
        flags.append(abs(value - centre) <= tolerance * centre)
    return flags


def scr_per_minute(points, min_amplitude_us=SCR_MIN_AMPLITUDE_US):
    """SCR frequency from (time_ms, eda_us) points: rises of at least the minimum amplitude
    from the preceding trough, counted once the signal turns down, per minute of coverage."""
    if len(points) < 2 or points[-1][0] <= points[0][0]:
        return None
    count, trough, peak = 0, points[0][1], None
    for _, value in points[1:]:
        if peak is not None and value < peak:
            count += 1
            trough, peak = value, None
        elif peak is not None:
            peak = value
        elif value < trough:
            trough = value
        elif value - trough >= min_amplitude_us:
            peak = value
    return count * 60000 / (points[-1][0] - points[0][0])


class FeaturePipeline:
    """One sensor session. Raw mode automatically records an initial resting baseline.

    A source policy (config/rules.json sourcePolicies) replaces that for noisy wearables: beats
    are artifact-cleaned, EDA has a validity floor, and the baseline is the wearer's own recent
    median instead of one window at the start of the session."""
    def __init__(self, rules=None, source_policy=None):
        self.source_policy = source_policy
        self.policy = source_policy['featurePolicy'] if source_policy else load_config(rules, 'rules.json')['featurePolicy']
        self.history = deque()  # (time, measured signals) for each second with a full window
        self.baseline_at = None
        self.last_seen = None
        self.hr, self.rr = [], []
        self.baseline = None
        self.start = None
        self.sequence = 0
        self.last_hr = self.last_rr = None
        self.eda, self.last_eda = [], None

    def process(self, sample):
        snapshot = sample.get('snapshot')
        if snapshot is None:
            build = self._source_snapshot if self.source_policy else self._raw_snapshot
            snapshot = build(sample['raw'], sample['timestamp_ms'])
        return self.from_snapshot(snapshot)

    def from_snapshot(self, snapshot):
        p = self.policy
        features = dict(fresh=snapshot.get('fresh') is True, fatigueEligible=False, snapshot=deepcopy(snapshot))
        baseline = snapshot.get('baseline')
        try:
            policy_matches = validate_feature_policy(snapshot.get('featurePolicy', LEGACY_FEATURE_POLICY)) == p
        except ValueError:
            policy_matches = False
        eligible = (policy_matches and features['fresh'] and snapshot.get('baselineState') == 'ready'
            and snapshot.get('requiredWindowSeconds') == p['windowSeconds']
            and finite(snapshot.get('windowSeconds')) and snapshot['windowSeconds'] >= p['windowSeconds'] - p['freshSeconds']
            and finite(snapshot.get('rrCount')) and snapshot['rrCount'] >= p['minRr']
            and isinstance(baseline, dict))
        if eligible:
            changes = {}
            for key in ('hr', 'rmssd', 'sdnn', 'eda', 'scr'):
                # SCR is derived from EDA, so both need a fresh EDA window.
                if key in ('eda', 'scr') and snapshot.get('edaFresh') is not True:
                    continue
                current, reference = snapshot.get(key), baseline.get(key)
                if not finite(current) or not finite(reference) or reference <= 0 or current < 0 or (key == 'hr' and current == 0):
                    continue
                change = (current - reference) / reference * 100
                if not finite(change):
                    continue
                changes[key + 'ChangePct'] = change
            features.update(changes, baseline=deepcopy(baseline),
                            fatigueEligible=all(key in changes for key in ('rmssdChangePct','sdnnChangePct')))
        posture = snapshot.get('posture')
        if (snapshot.get('postureCalibrated') is True and isinstance(posture, dict)
                and posture.get('level') not in (None, 'unavailable')
                and finite(posture.get('pitch')) and finite(posture.get('roll'))):
            features.update(absPitch=abs(posture['pitch']), absRoll=abs(posture['roll']))
        return features

    def _clear(self):
        self.hr, self.rr = [], []
        self.start = self.last_hr = self.last_rr = None
        self.sequence += 1
        self.eda, self.last_eda = [], None

    def _raw_snapshot(self, raw, now):
        p = self.policy
        hr = raw.get('hr', raw.get('heart_rate_bpm'))
        fresh_ms, window_ms = p['freshSeconds'] * 1000, p['windowSeconds'] * 1000
        if (any(raw.get(key) is False for key in ('contact', 'sensor_contact', 'sensor_contact_detected'))
                or not finite(hr) or hr <= 0):
            self._clear()
            return dict(fresh=False, featurePolicy=deepcopy(p), baselineState='ready' if self.baseline else 'collecting')
        if ((self.last_hr is not None and now - self.last_hr > fresh_ms)
                or (self.last_rr is not None and now - self.last_rr > fresh_ms)):
            self._clear()
        if self.start is None:
            self.start = now
        self.hr.append((now, hr))
        self.last_hr = now
        eda = raw.get('eda_us', raw.get('eda'))
        if finite(eda) and eda >= 0:
            if self.last_eda is not None and now - self.last_eda > fresh_ms:
                self.eda = []
            self.eda.append((now, eda))
            self.last_eda = now
        elif 'eda_us' in raw or 'eda' in raw:
            self.eda, self.last_eda = [], None
        self.eda = [(t,v) for t,v in self.eda if t >= now - window_ms][-4096:]
        eda_fresh = (self.last_eda is not None and now - self.last_eda <= fresh_ms
                     and len(self.eda) > 1 and self.last_eda-self.eda[0][0] >= window_ms-fresh_ms)
        eda_mean = sum(v / len(self.eda) for _,v in self.eda) if eda_fresh else None
        scr = scr_per_minute(self.eda) if eda_fresh else None
        for rr in raw.get('rr_intervals_ms', []):
            if p['rrMinMs'] <= rr <= p['rrMaxMs']:
                self.rr.append((now, rr, self.sequence))
                self.last_rr = now
            else:
                self.sequence += 1
        self.hr = [(t, v) for t, v in self.hr if t >= now - window_ms][-4096:]
        self.rr = [(t, v, seq) for t, v, seq in self.rr if t >= now - window_ms][-int(window_ms / p['rrMinMs'] + 64):]
        sequence, values = None, []
        for _, value, seq in self.rr:
            if sequence is not None and sequence != seq:
                values.append(None)
            values.append(value)
            sequence = seq
        metrics = calculate_hrv(values, p['rrMinMs'], p['rrMaxMs'])
        fresh = self.last_rr is not None and now - self.last_rr <= fresh_ms
        coverage = min(now - self.hr[0][0], self.last_rr - self.rr[0][0] if self.rr else 0)
        metrics['hr'] = sum(v / len(self.hr) for _, v in self.hr)
        eligible = (fresh and coverage >= window_ms - fresh_ms and len(self.rr) >= p['minRr'])
        if (self.baseline is None and now - self.start >= window_ms and eligible
                and len(self.hr) >= p['baselineSamples'] and metrics['rmssd'] is not None and metrics['rmssd'] > 0
                and metrics['sdnn'] is not None and metrics['sdnn'] > 0):
            self.baseline = deepcopy(metrics)
            if eda_fresh and eda_mean > 0:
                self.baseline['eda'] = eda_mean
            # A zero resting SCR rate cannot anchor a percentage change; SCR then stays unmeasured.
            if scr:
                self.baseline['scr'] = scr
        return dict(**metrics, eda=eda_mean, scr=scr, edaFresh=bool(eda_fresh), fresh=fresh, featurePolicy=deepcopy(p), baseline=deepcopy(self.baseline), baselineState='ready' if self.baseline else 'collecting',
                    rrCount=len(self.rr), windowSeconds=coverage / 1000, requiredWindowSeconds=p['windowSeconds'])

    def _rolling_baseline(self, now, measured):
        """Median of the wearer's own recent values. Windows ending inside the gap overlap the
        current window and are skipped, so a change cannot pull its own reference towards it."""
        q = self.source_policy
        if measured is not None:
            self.history.append((now, measured))
        oldest = now - (q['baselineHistorySeconds'] + q['baselineGapSeconds']) * 1000
        while self.history and self.history[0][0] < oldest:
            self.history.popleft()
        if self.baseline_at is not None and now - self.baseline_at < 10000:
            return  # recomputed every 10 s; it moves slowly by design
        self.baseline_at = now
        settled = [values for t, values in self.history if t <= now - q['baselineGapSeconds'] * 1000]
        baseline = {}
        for key in ('hr', 'rmssd', 'sdnn', 'eda', 'scr'):
            seen = [values[key] for values in settled if values[key] is not None]
            if len(seen) >= q['baselineMinSeconds']:
                baseline[key] = median(seen)
        if 'scr' in baseline:
            # Often no responses per minute at rest, which cannot anchor a percentage. One response
            # per window is the smallest rate the window can measure, so it is the reference floor.
            baseline['scr'] = max(baseline['scr'], q['scrFloorPerMinute'])
        self.baseline = baseline or None

    def _source_snapshot(self, raw, now):
        p, q = self.policy, self.source_policy
        window_ms, fresh_ms = p['windowSeconds'] * 1000, p['freshSeconds'] * 1000
        worn = raw.get('contact') is not False
        if not worn or (self.last_seen is not None and now - self.last_seen > fresh_ms):
            self._clear()
        self.last_seen = now
        if not worn:
            self._rolling_baseline(now, None)
            return dict(fresh=False, worn=False, featurePolicy=deepcopy(p), baseline=deepcopy(self.baseline),
                        baselineState='ready' if self.baseline else 'collecting')
        if self.start is None:
            self.start = now
        # A small negative EDA offset stays in the mean; the floor below decides validity.
        if finite(raw.get('eda_us')):
            self.eda.append((now, raw['eda_us']))
        for rr in raw.get('rr_intervals_ms', []):
            if p['rrMinMs'] <= rr <= p['rrMaxMs']:
                self.rr.append((now, rr, self.sequence))
            else:
                self.sequence += 1
        self.eda = [point for point in self.eda if point[0] > now - window_ms]
        self.rr = [beat for beat in self.rr if beat[0] > now - window_ms]
        coverage = min(now - self.start, window_ms)
        full = coverage >= window_ms - fresh_ms

        values = [value for _, value, _ in self.rr]
        keep = clean_beats(values, q['beatTolerance'], q['beatNeighbours'])
        clean = [value for value, kept in zip(values, keep) if kept]
        fraction = len(clean) / len(values) if values else 0
        # A rejected beat breaks successive differences, the same as a gap in the sequence.
        series, sequence = [], None
        for (_, value, seq), kept in zip(self.rr, keep):
            if sequence is not None and seq != sequence:
                series.append(None)
            series.append(value if kept else None)
            sequence = seq
        metrics = calculate_hrv(series, p['rrMinMs'], p['rrMaxMs'])
        hrv_usable = fraction >= q['minCleanBeatFraction'] and len(clean) >= p['minRr']
        if not hrv_usable:
            metrics.update(rmssd=None, sdnn=None)
        metrics['hr'] = 60000 / (sum(clean) / len(clean)) if clean else None

        enough_eda = len(self.eda) >= p['windowSeconds'] - p['freshSeconds']
        eda_mean = sum(value for _, value in self.eda) / len(self.eda) if enough_eda else None
        eda_below_floor = eda_mean is not None and eda_mean < q['edaFloorUs']
        if eda_below_floor:
            eda_mean = None
        scr = scr_per_minute(self.eda) if eda_mean is not None else None

        measured = dict(metrics, eda=eda_mean, scr=scr) if full else None
        self._rolling_baseline(now, measured)
        return dict(**metrics, eda=eda_mean, scr=scr, edaFresh=eda_mean is not None, edaBelowFloor=eda_below_floor,
                    fresh=True, worn=True, cleanBeatFraction=round(fraction, 3), hrvUsable=hrv_usable,
                    featurePolicy=deepcopy(p), baseline=deepcopy(self.baseline),
                    baselineState='ready' if self.baseline else 'collecting',
                    rrCount=len(values), windowSeconds=coverage / 1000, requiredWindowSeconds=p['windowSeconds'])
