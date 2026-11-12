#!/usr/bin/env python3
"""Feed EmbracePlus raw data from the study's AWS S3 bucket into the feedback pipeline.

Empatica uploads raw Avro files in 30-minute chunks, so this data usually arrives 30-60 minutes
late and sometimes hours late. Give each experiment session its own --since/--until and
--session-id. Each new file becomes one sample per second (mean EDA, skin temperature, and beat
intervals from Empatica's systolic peaks) posted in mode "batch": levels, causes and High
streaks are computed and saved, and any feedback live data would have triggered is
recorded as `wouldAlert` instead of being delivered. Seconds when the device was not worn are
flagged, and nothing is predicted from them.

  # What is on S3 for a date? (Enrollment names are typed per session and can change between days.)
  python3 tools/empatica_bridge.py --list --date 2026-11-01

  # Pull new files every minute and send them to the running dashboard server:
  python3 tools/empatica_bridge.py --url http://127.0.0.1:8000 --participant 11 --poll 60

  # Process a downloaded copy (same folder layout) once, without a server:
  python3 tools/empatica_bridge.py --source local --date 2024-05-16 --participant 11

  # Demo: replay a recording one reading per second as if it were live (mode "demo",
  # so High-frustration cues are delivered with a "Demonstration" prefix):
  python3 tools/empatica_bridge.py --url http://127.0.0.1:8000 --source local --date 2024-05-16 --participant 11 \
      --since 2024-05-16T23:09 --until 2024-05-16T23:39 --replay-speed 1

S3 access uses AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION from the
environment or .env, and the bucket/prefix of avro_stream_poller_with_ml.py.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import re
from statistics import median
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

S3_BUCKET = 'empatica-us-east-1-prod-data'
S3_PREFIX = 'v2/1324/1/1/participant_data/'
LOCAL_DIR = ROOT / '.local/data/participant_data'
STATE_FILE = ROOT / '.local/empatica_bridge_state.json'
RR_MIN_MS, RR_MAX_MS = 300, 2000
HR_MAX_AGE_S = 5
MINUTE_CSVS = 'digital_biomarkers/aggregated_per_minute'


# ---------------------------------------------------------------- sources

class LocalSource:
    """A downloaded copy laid out like the bucket: <date>/<enrollment>-<device>/raw_data/v6/*.avro"""
    def __init__(self, root):
        self.root = Path(root)

    def everything(self, date):
        return sorted(str(p.relative_to(self.root)) for p in (self.root / date).glob('*/**/*') if p.is_file())

    def read(self, key):
        return (self.root / key).read_bytes()

    def read_optional(self, key):
        try:
            return self.read(key)
        except OSError:
            return None


class S3Source:
    def __init__(self, bucket=S3_BUCKET, prefix=S3_PREFIX):
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / '.env')
        except ImportError:
            pass
        import boto3
        self.bucket, self.prefix = bucket, prefix
        self.s3 = boto3.client('s3', region_name=os.environ.get('AWS_REGION', 'us-east-1'))

    def everything(self, date):
        keys, token = [], None
        while True:
            args = dict(Bucket=self.bucket, Prefix=f'{self.prefix}{date}/')
            if token:
                args['ContinuationToken'] = token
            response = self.s3.list_objects_v2(**args)
            keys += [o['Key'][len(self.prefix):] for o in response.get('Contents', [])]
            if not response.get('IsTruncated'):
                return sorted(keys)
            token = response['NextContinuationToken']

    def read(self, key):
        return self.s3.get_object(Bucket=self.bucket, Key=self.prefix + key)['Body'].read()

    def read_optional(self, key):
        try:
            return self.read(key)
        except self.s3.exceptions.NoSuchKey:
            return None


def is_raw(key):
    return '/raw_data/v6/' in key and key.endswith('.avro')


def raw_keys(source, date):
    return [key for key in source.everything(date) if is_raw(key)]


def availability(source, date):
    """{enrollment: (Avro files, per-minute CSV files)} for one UTC date folder."""
    found = {}
    for key in source.everything(date):
        files = found.setdefault(enrollment_of(key), [0, 0])
        files[0] += is_raw(key)
        files[1] += f'/{MINUTE_CSVS}/' in key and key.endswith('.csv')
    return {name: tuple(files) for name, files in sorted(found.items())}


def explain_empty(source, dates, participant):
    """Say why nothing was processed; "no new files" alone reads as if the data did not exist."""
    lines = []
    for date in dates:
        found = availability(source, date)
        if participant and participant not in found:
            others = ', '.join(f'{name} ({avro} Avro)' for name, (avro, _) in found.items()) or 'nobody'
            lines.append(f'{date}: no enrollment named {participant}. On that date: {others}.')
        elif participant and not found[participant][0]:
            lines.append(f'{date}: {participant} has {found[participant][1]} per-minute CSVs but no raw Avro yet. '
                         'Avro uploads can lag by hours; keep polling. Per-minute CSVs cannot feed the 60 s features.')
        elif not found:
            lines.append(f'{date}: nothing on the source for this date.')
        else:
            lines.append(f"{date}: no new files for {participant or 'any enrollment'} (already sent, or outside --since/--until).")
    return '\n'.join(lines)


def enrollment_of(key):
    """'2024-05-16/11-3YK3K151Y5/raw_data/v6/1-1-11_1715818778.avro' -> '11'"""
    return key.split('/')[1].split('-')[0]


def file_start(key):
    match = re.search(r'_(\d+)\.avro$', key)
    return int(match[1]) if match else 0


# ---------------------------------------------------------------- conversion

def _per_second(stream):
    """(second, value) pairs from an Avro signal block with a start time and sampling rate."""
    if not stream or not stream.get('values') or not stream.get('samplingFrequency'):
        return
    t0, fs = stream['timestampStart'] / 1e6, stream['samplingFrequency']
    for i, value in enumerate(stream['values']):
        yield int(t0 + i / fs), value


def avro_to_samples(data):
    """One raw sample per second from an EmbracePlus v6 Avro file."""
    import fastavro
    seconds = {}
    beats = []
    tz = 0
    for record in fastavro.reader(io.BytesIO(data)):
        tz = record.get('timezone') or 0
        raw = record['rawData']
        for key, stream in (('eda', raw.get('eda')), ('temp', raw.get('temperature'))):
            for second, value in _per_second(stream):
                seconds.setdefault(second, {}).setdefault(key, []).append(value)
        beats += [t / 1e9 for t in (raw.get('systolicPeaks') or {}).get('peaksTimeNanos', [])]
    beats.sort()
    # Each interval belongs to the second its closing beat falls in.
    intervals = {}
    for previous, current in zip(beats, beats[1:]):
        ms = (current - previous) * 1000
        if RR_MIN_MS <= ms <= RR_MAX_MS:
            intervals.setdefault(int(current), []).append(round(ms, 1))
    recent = []
    for second in sorted(seconds):
        values = seconds[second]
        raw = {}
        # EmbracePlus EDA can carry a small negative offset; keep it. The ML features are scale-free,
        # and the DBR %-change path already ignores negative EDA on its own.
        eda = values.get('eda', [])
        if eda:
            raw['eda_us'] = sum(eda) / len(eda)
        if values.get('temp'):
            raw['temp_c'] = sum(values['temp']) / len(values['temp'])
        rr = intervals.get(second, [])
        recent = [(t, v) for t, v in recent if second - t < HR_MAX_AGE_S] + [(second, v) for v in rr]
        if recent:
            raw['hr'] = 60000 / (sum(v for _, v in recent) / len(recent))
            raw['rr_intervals_ms'] = rr
        if raw:
            yield second, raw, tz


class WearCheck:
    """Was the device worn? Empatica's per-minute flag where published, else skin temperature.

    The flag can lag the raw files, so temperature decides until it appears. Evidence for each
    threshold is in docs/TEAM3_FRUSTRATION_FATIGUE.md."""
    def __init__(self, source, policy):
        self.source, self.policy, self.tables = source, policy, {}

    def _table(self, key, second):
        """{minute: percent worn} from the wear CSV of this sample's UTC date."""
        day = datetime.fromtimestamp(second, timezone.utc).strftime('%Y-%m-%d')
        folder, stem = key.split('/')[1], Path(key).stem.rsplit('_', 1)[0]
        name = f'{day}/{folder}/{MINUTE_CSVS}/{stem}_{day}_wearing-detection.csv'
        if name not in self.tables:
            table, data = {}, self.source.read_optional(name)
            for row in csv.DictReader(io.StringIO(data.decode())) if data else ():
                value = (row.get('wearing_detection_percentage') or '').strip()
                if value:
                    table[int(row['timestamp_unix']) // 60000] = float(value)
            self.tables[name] = table
        return self.tables[name]

    def worn(self, key, second, minute_temperature):
        percent = self._table(key, second).get(second // 60)
        if percent is not None:
            return percent >= self.policy['minPercent']
        return minute_temperature is None or minute_temperature >= self.policy['minTemperatureC']


# ---------------------------------------------------------------- delivery

class Replay:
    """Paces a recording at `speed` readings per second and re-times it to start now (demo mode).

    Timestamps keep the recording's one-second spacing; the original second is kept as recorded_s."""
    def __init__(self, speed):
        if speed <= 0:
            raise ValueError('Replay speed must be positive.')
        self.speed, self.first = speed, None

    def __call__(self, second, sample):
        if self.first is None:
            self.first, self.wall, self.origin = second, time.monotonic(), int(time.time())
        offset = second - self.first
        delay = self.wall + offset / self.speed - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        sample.update(mode='demo', timestamp_ms=(self.origin + offset) * 1000)
        sample['raw']['recorded_s'] = second

def post(url, sample):
    request = urllib.request.Request(url.rstrip('/') + '/samples', data=json.dumps(sample, allow_nan=False).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def process_file(source, key, send, session_id=None, quiet=False, window=(None, None), sent_until=None, replay=None,
                 wear=None):
    """Send one file's samples in order; returns samples sent, would-be cues and seconds not worn.

    sent_until maps session -> last second sent; consecutive files overlap by about a second."""
    enrollment = enrollment_of(key)
    sent_until = {} if sent_until is None else sent_until
    sent, cues, minute, last, not_worn = 0, [], None, None, 0
    samples = list(avro_to_samples(source.read(key)))
    temperatures = {}
    for second, raw, _ in samples:
        if 'temp_c' in raw:
            temperatures.setdefault(second // 60, []).append(raw['temp_c'])
    for second, raw, tz in samples:
        if (window[0] is not None and second < window[0]) or (window[1] is not None and second > window[1]):
            continue
        day = datetime.fromtimestamp(second, timezone.utc).strftime('%Y-%m-%d')
        session = session_id or f'empatica-{enrollment}-{day}'
        if second <= sent_until.get(session, -1):
            continue
        sent_until[session] = second
        raw = dict(raw, source='empatica-avro')
        minute_temperatures = temperatures.get(second // 60)
        if wear and not wear.worn(key, second, median(minute_temperatures) if minute_temperatures else None):
            raw['contact'] = False
            not_worn += 1
        sample = dict(session_id=session, participant=f'empatica-{enrollment}',
                      mode='batch', sample_id=f'{Path(key).stem}-{second}', timestamp_ms=second * 1000, raw=raw)
        if replay:
            replay(second, sample)
        result = send(sample)
        sent += 1
        last = (second, tz, result)
        for name, state in result['states'].items():
            if state.get('wouldAlert'):
                cues.append((second, tz, name, state['wouldAlert']['message']))
                if not quiet:
                    print(f"  {_clock(second, tz)}  would have cued {name}: {state['wouldAlert']['message']}")
        if not quiet and second // 60 != minute:
            minute = second // 60
            print('  ' + _summary(second, tz, result))
    if not quiet and last:
        print('  ' + _summary(*last))
    return sent, cues, not_worn


def _clock(second, tz):
    return datetime.fromtimestamp(second + tz, timezone.utc).strftime('%Y-%m-%d %H:%M')


def _summary(second, tz, result):
    cells = []
    for name in ('frustration', 'fatigue'):
        state = result['states'].get(name, {})
        level = state.get('level', 'unavailable') if state.get('available') else 'unavailable'
        model = state.get('model', {})
        if name == 'frustration' and model.get('status') == 'ml':
            level += f" (ML high {model['probabilities']['high']:.2f})"
        cells.append(f'{name} {level}')
    return f"{_clock(second, tz)}  " + ' · '.join(cells)


def load_state(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return dict(done=[])


def run(source, dates, send, participant=None, since=None, state=None, session_id=None, quiet=False, until=None,
        replay=None, wear_policy=None):
    """Process every not-yet-sent file for the dates, oldest first. Returns per-file results."""
    state = state if state is not None else dict(done=[])
    state.setdefault('sentUntil', {})
    done = set(state['done'])
    results = []
    if wear_policy is None:
        from pipeline.config import load_config
        wear_policy = load_config(None, 'rules.json')['sourcePolicies']['empatica-avro']['wear']
    wear = WearCheck(source, wear_policy)  # rebuilt each pass: the published flag grows through the day
    for date in dates:
        keys = [k for k in raw_keys(source, date) if k not in done
                and (participant is None or enrollment_of(k) == participant)
                and (since is None or file_start(k) + 1800 >= since)
                and (until is None or file_start(k) <= until)]
        for key in sorted(keys, key=file_start):
            if not quiet:
                print(f'{key}:')
            sent, cues, not_worn = process_file(source, key, send, session_id, quiet, (since, until), state['sentUntil'],
                                                replay, wear)
            results.append(dict(key=key, samples=sent, wouldAlerts=len(cues), notWorn=not_worn))
            state['done'].append(key)
    return results


def state_path(session_id, chosen=None):
    """Sent files are remembered per session, so re-running a recording under a new session ID reprocesses it."""
    if chosen:
        return chosen
    if not session_id:
        return STATE_FILE
    return STATE_FILE.with_name(f"{STATE_FILE.stem}-{re.sub(r'[^A-Za-z0-9_.-]', '_', session_id)}.json")


def report(results):
    samples, not_worn = sum(r['samples'] for r in results), sum(r['notWorn'] for r in results)
    print(f"Processed {len(results)} file(s): {samples} readings, {sum(r['wouldAlerts'] for r in results)} would-be cue(s), "
          f"{not_worn} second(s) not worn.")
    if samples and not_worn / samples > .5:
        print(f'WARNING: the device was not worn for {not_worn / samples:.0%} of this data, so it cannot show frustration or fatigue.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', choices=('s3', 'local'), default='s3')
    parser.add_argument('--local-dir', type=Path, default=LOCAL_DIR)
    parser.add_argument('--bucket', default=S3_BUCKET)
    parser.add_argument('--prefix', default=S3_PREFIX)
    parser.add_argument('--date', action='append', help='UTC date folder (repeatable); default today and yesterday')
    parser.add_argument('--list', action='store_true', help='Show which enrollments have data on the dates, then exit')
    parser.add_argument('--participant', help='Enrollment ID, e.g. 11 for folder 11-3YK3K151Y5')
    parser.add_argument('--since', help='Session start, UTC, e.g. 2024-05-16T22:39 (earlier data is skipped)')
    parser.add_argument('--until', help='Session end, UTC (later data is skipped)')
    parser.add_argument('--session-id', help='Override the session (default empatica-<enrollment>-<UTC date>)')
    parser.add_argument('--url', help='Loopback dashboard server, e.g. http://127.0.0.1:8000; default: process locally')
    parser.add_argument('--poll', type=float, default=0, help='Seconds between checks for new files; 0 runs once')
    parser.add_argument('--state', type=Path, help='Remembers files already sent to the server (default: one file per --session-id)')
    parser.add_argument('--replay-speed', type=float, default=0,
                        help='Demo: replay at this many readings per second (1 = real time) in a new demo session')
    args = parser.parse_args(argv)
    if args.url and not re.match(r'^http://(127\.0\.0\.1|localhost)(:\d+)?/?$', args.url):
        parser.error('--url must be the local dashboard server.')
    source = LocalSource(args.local_dir) if args.source == 'local' else S3Source(args.bucket, args.prefix)
    if args.list:
        for date in args.date or [(datetime.now(timezone.utc) - timedelta(days=d)).strftime('%Y-%m-%d') for d in (1, 0)]:
            print(date)
            for name, (avro, csvs) in availability(source, date).items():
                print(f'  {name}: {avro} raw Avro, {csvs} per-minute CSV')
        return
    utc = lambda text: datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp() if text else None
    since, until = utc(args.since), utc(args.until)
    replay = None
    args.state = state_path(args.session_id, args.state)
    if args.replay_speed:
        if args.poll:
            parser.error('--replay-speed replays recorded files once; do not combine it with --poll.')
        replay = Replay(args.replay_speed)
        args.session_id = args.session_id or f"replay-{args.participant or 'empatica'}-{datetime.now():%H%M%S}"
        print(f'Replaying at {args.replay_speed:g} reading(s) per second into demo session {args.session_id}')
    if args.url and replay:
        send, state = (lambda sample: post(args.url, sample)), dict(done=[])  # a replay never marks files as sent
        print(f"Follow it: {args.url.rstrip('/')}/serving/dashboard.html?session={args.session_id}", flush=True)
    elif args.url:
        send = lambda sample: post(args.url, sample)
        state = load_state(args.state)
    else:
        from pipeline import FeedbackPipeline
        pipeline = FeedbackPipeline()
        send, state = pipeline.process, dict(done=[])
    try:
        while True:
            dates = args.date or [(datetime.now(timezone.utc) - timedelta(days=d)).strftime('%Y-%m-%d') for d in (1, 0)]
            results = run(source, dates, send, args.participant, since, state, args.session_id, until=until, replay=replay)
            if args.url and not replay:
                args.state.parent.mkdir(parents=True, exist_ok=True)
                args.state.write_text(json.dumps(state))
            if args.url and results and not replay:
                print(f"Follow it: {args.url.rstrip('/')}/serving/dashboard.html?session="
                      + (args.session_id or f"empatica-{args.participant or '<enrollment>'}-<UTC date>"))
            if results:
                report(results)
            else:
                print(explain_empty(source, dates, args.participant))
            if not args.poll:
                break
            time.sleep(args.poll)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
