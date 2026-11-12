"""Empatica S3/Avro bridge, batch (late data) mode, and late-live-EDA handling."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import time
import unittest

from pipeline import FeedbackPipeline
from pipeline.frustration_model import CLASSES, FEATURES

ROOT = Path(__file__).resolve().parents[1]
HAVE_AVRO = importlib.util.find_spec('fastavro') is not None
SCHEMA = {
    'type': 'record', 'name': 'AvroData', 'fields': [
        {'name': 'timezone', 'type': 'int'},
        {'name': 'rawData', 'type': {'type': 'record', 'name': 'RawData', 'fields': [
            {'name': 'eda', 'type': {'type': 'record', 'name': 'Eda', 'fields': [
                {'name': 'timestampStart', 'type': 'long'}, {'name': 'samplingFrequency', 'type': 'float'},
                {'name': 'values', 'type': {'type': 'array', 'items': 'float'}}]}},
            {'name': 'temperature', 'type': {'type': 'record', 'name': 'Temperature', 'fields': [
                {'name': 'timestampStart', 'type': 'long'}, {'name': 'samplingFrequency', 'type': 'float'},
                {'name': 'values', 'type': {'type': 'array', 'items': 'float'}}]}},
            {'name': 'systolicPeaks', 'type': {'type': 'record', 'name': 'SystolicPeaks', 'fields': [
                {'name': 'peaksTimeNanos', 'type': {'type': 'array', 'items': 'long'}}]}}]}}]}


def write_avro(path, start_s, seconds, eda=lambda t: 2 + .1 * math.sin(t / 5), beat_ms=800):
    import fastavro
    path.parent.mkdir(parents=True, exist_ok=True)
    beats = [int(start_s * 1e9 + i * beat_ms * 1e6) for i in range(int(seconds * 1000 / beat_ms))]
    record = dict(timezone=-14400, rawData=dict(
        eda=dict(timestampStart=int(start_s * 1e6), samplingFrequency=4.0, values=[eda(start_s + i / 4) for i in range(seconds * 4)]),
        temperature=dict(timestampStart=int(start_s * 1e6), samplingFrequency=1.0, values=[33.0 + i / 1000 for i in range(seconds)]),
        systolicPeaks=dict(peaksTimeNanos=beats)))
    with path.open('wb') as stream:
        fastavro.writer(stream, fastavro.parse_schema(SCHEMA), [record])


def stub_rules(directory, prefer='high'):
    path = Path(directory) / 'stub.json'
    path.write_text(json.dumps(dict(
        schemaVersion=1, modelType='logistic', features=FEATURES, classes=CLASSES, family='logistic',
        mean=[0.0] * len(FEATURES), scale=[1.0] * len(FEATURES), coefficients=[[0.0] * len(FEATURES) for _ in CLASSES],
        intercepts=[5.0 if c == prefer else 0.0 for c in CLASSES])))
    rules = json.loads((ROOT / 'config/rules.json').read_text())
    rules['states']['frustration']['modelFile'] = str(path)
    return rules


@unittest.skipUnless(HAVE_AVRO, 'fastavro not installed')
class ConversionTests(unittest.TestCase):
    def test_one_sample_per_second_with_eda_mean_temperature_and_beat_intervals(self):
        from tools.empatica_bridge import avro_to_samples
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'f.avro'
            write_avro(path, 1_700_000_000, 10, eda=lambda t: -0.05 if t < 1_700_000_002 else 2.0)
            samples = list(avro_to_samples(path.read_bytes()))
        self.assertEqual([s for s, _, _ in samples], list(range(1_700_000_000, 1_700_000_010)))
        raw = dict((s, r) for s, r, _ in samples)
        self.assertAlmostEqual(raw[1_700_000_000]['eda_us'], -0.05, places=5)  # offsets are kept
        self.assertAlmostEqual(raw[1_700_000_005]['eda_us'], 2.0)
        self.assertAlmostEqual(raw[1_700_000_005]['temp_c'], 33.005, places=4)
        self.assertEqual(raw[1_700_000_005]['rr_intervals_ms'], [800.0])
        self.assertAlmostEqual(raw[1_700_000_005]['hr'], 75.0)
        self.assertEqual(samples[0][2], -14400)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'slow.avro'
            write_avro(path, 1_700_000_000, 4, beat_ms=1500)
            slow = dict((s, r) for s, r, _ in avro_to_samples(path.read_bytes()))
        self.assertNotIn('hr', slow[1_700_000_000])  # the first interval closes at 1.5 s
        self.assertAlmostEqual(slow[1_700_000_001]['hr'], 40.0)


@unittest.skipUnless(HAVE_AVRO, 'fastavro not installed')
class RunTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        start = 1_715_890_000
        folder = self.root / '2024-05-16/11-3YK3K151Y5/raw_data/v6'
        write_avro(folder / f'1-1-11_{start}.avro', start, 120)
        write_avro(folder / f'1-1-11_{start + 119}.avro', start + 119, 120)  # one-second overlap
        write_avro(self.root / f'2024-05-16/7-3YK3K151Y5/raw_data/v6/1-1-7_{start}.avro', start, 60)
        self.start = start

    def tearDown(self):
        self.dir.cleanup()

    def test_files_are_filtered_ordered_deduplicated_and_never_reprocessed(self):
        from tools.empatica_bridge import LocalSource, run
        sent = []
        state = dict(done=[])
        results = run(LocalSource(self.root), ['2024-05-16'], lambda s: sent.append(s) or self._ok(s),
                      participant='11', state=state, quiet=True)
        self.assertEqual(len(results), 2)
        seconds = [s['timestamp_ms'] // 1000 for s in sent]
        self.assertEqual(seconds, sorted(set(seconds)))
        self.assertEqual(len(seconds), 239)
        self.assertTrue(all(s['mode'] == 'batch' and s['participant'] == 'empatica-11' for s in sent))
        self.assertEqual(run(LocalSource(self.root), ['2024-05-16'], self._ok, participant='11', state=state, quiet=True), [])

    def test_session_window_limits_samples(self):
        from tools.empatica_bridge import LocalSource, run
        sent = []
        run(LocalSource(self.root), ['2024-05-16'], lambda s: sent.append(s) or self._ok(s), participant='11',
            since=self.start + 30, until=self.start + 89, session_id='exp-1', quiet=True)
        self.assertEqual([s['timestamp_ms'] // 1000 for s in sent], list(range(self.start + 30, self.start + 90)))
        self.assertTrue(all(s['session_id'] == 'exp-1' for s in sent))

    def test_replay_paces_and_retimes_a_recording_as_demo_data(self):
        from tools.empatica_bridge import LocalSource, Replay, run
        sent, began = [], time.time()
        run(LocalSource(self.root), ['2024-05-16'], lambda s: sent.append(s) or self._ok(s), participant='11',
            until=self.start + 9, session_id='replay', quiet=True, replay=Replay(20))
        elapsed = time.time() - began
        self.assertEqual(len(sent), 10)
        self.assertGreaterEqual(elapsed, 9 / 20 - .05)  # paced at 20 readings per second
        self.assertTrue(all(s['mode'] == 'demo' for s in sent))
        stamps = [s['timestamp_ms'] for s in sent]
        self.assertEqual([b - a for a, b in zip(stamps, stamps[1:])], [1000] * 9)
        self.assertLess(abs(stamps[0] / 1000 - began), 2)
        self.assertEqual([s['raw']['recorded_s'] for s in sent], list(range(self.start, self.start + 10)))

    @staticmethod
    def _ok(sample):
        return dict(states={})

    def wear_csv(self, day, rows, enrollment='11'):
        folder = self.root / f'{day}/{enrollment}-3YK3K151Y5/digital_biomarkers/aggregated_per_minute'
        folder.mkdir(parents=True, exist_ok=True)
        lines = ['timestamp_unix,timestamp_iso,participant_full_id,wearing_detection_percentage,missing_value_reason']
        lines += [f'{minute * 60000},t,p,{percent},' for minute, percent in rows]
        (folder / f'1-1-{enrollment}_{day}_wearing-detection.csv').write_text('\n'.join(lines))

    def test_published_wear_flag_wins_and_temperature_decides_when_it_is_absent(self):
        from tools.empatica_bridge import LocalSource, WearCheck
        day = '2024-05-16'
        minute = self.start // 60
        self.wear_csv(day, [(minute, 0), (minute + 1, 100), (minute + 2, '')])
        key = f'{day}/11-3YK3K151Y5/raw_data/v6/1-1-11_{self.start}.avro'
        check = WearCheck(LocalSource(self.root), dict(minPercent=50, minTemperatureC=27))
        at = lambda m: (minute + m) * 60
        self.assertFalse(check.worn(key, at(0), 33.0))   # flagged off, despite a warm reading
        self.assertTrue(check.worn(key, at(1), 22.0))    # flagged on, despite a cold reading
        self.assertFalse(check.worn(key, at(2), 22.0))   # blank row: room temperature means off the wrist
        self.assertTrue(check.worn(key, at(2), 31.0))
        self.assertTrue(check.worn(key, at(2), None))    # no evidence either way

    def test_unworn_recording_is_flagged_and_produces_no_cues(self):
        from tools.empatica_bridge import LocalSource, run
        minute = self.start // 60
        self.wear_csv('2024-05-16', [(minute + i, 0) for i in range(10)])
        with tempfile.TemporaryDirectory() as d:
            pipe = FeedbackPipeline(rules=stub_rules(d, 'high'))
            outs, sent = [], []
            def send(sample):
                sent.append(sample)
                outs.append(pipe.process(sample))
                return outs[-1]
            results = run(LocalSource(self.root), ['2024-05-16'], send, participant='11', quiet=True)
        self.assertTrue(all(s['raw']['contact'] is False for s in sent))
        self.assertEqual(sum(r['notWorn'] for r in results), len(sent))
        self.assertEqual(sum(r['wouldAlerts'] for r in results), 0)
        self.assertFalse(any(o['states']['frustration']['available'] for o in outs))

    def test_sent_files_are_remembered_per_session_so_a_recording_can_be_rerun(self):
        from tools.empatica_bridge import STATE_FILE, state_path
        self.assertEqual(state_path(None), STATE_FILE)
        self.assertNotEqual(state_path('retest-1'), state_path('retest-2'))
        self.assertEqual(state_path('a/b c').name, f'{STATE_FILE.stem}-a_b_c.json')
        self.assertEqual(state_path('x', Path('/tmp/chosen.json')), Path('/tmp/chosen.json'))

    def test_empty_result_names_who_is_really_there(self):
        from tools.empatica_bridge import LocalSource, availability, explain_empty
        source = LocalSource(self.root)
        self.assertEqual(availability(source, '2024-05-16')['11'], (2, 0))
        # An enrollment name can change by a letter between days, which reads like missing data.
        message = explain_empty(source, ['2024-05-16'], '12')
        self.assertIn('no enrollment named 12', message)
        self.assertIn('11 (2 Avro)', message)
        self.wear_csv('2024-05-17', [(1, 100)], enrollment='11')
        self.assertIn('no raw Avro yet', explain_empty(source, ['2024-05-17'], '11'))
        self.assertIn('nothing on the source', explain_empty(source, ['2024-05-18'], None))
        self.assertIn('already sent', explain_empty(source, ['2024-05-16'], '11'))

    def test_batch_mode_records_cues_without_delivering_and_keeps_cooldown(self):
        from tools.empatica_bridge import LocalSource, run
        with tempfile.TemporaryDirectory() as d:
            pipe = FeedbackPipeline(rules=stub_rules(d, 'high'))
            outs = []
            run(LocalSource(self.root), ['2024-05-16'], lambda s: outs.append(pipe.process(s)) or outs[-1],
                participant='11', quiet=True)
        frustration = [o['states']['frustration'] for o in outs]
        cues = [i for i, f in enumerate(frustration) if f.get('wouldAlert')]
        self.assertFalse(any(o['alerts'] for o in outs))
        self.assertGreaterEqual(len(cues), 1)
        first = frustration[cues[0]]
        self.assertEqual((first['suppression'], first['wouldAlert']['consecutive']), ('retrospective', 3))
        cooldown = json.loads((ROOT / 'config/feedback_table.json').read_text())['decision']['cooldownSeconds']
        self.assertTrue(all(b - a >= cooldown for a, b in zip(cues, cues[1:])))


class ServerTests(unittest.TestCase):
    def batch(self, count, session='picker', start=None):
        from pipeline.frustration_model import FEATURES  # noqa: F401
        first = start if start is not None else int(time.time()) - 7200
        results = []
        for i in range(count):
            results.append(self.service.process_sample(dict(
                session_id=session, participant='empatica-x', mode='batch', sample_id=str(i), timestamp_ms=(first + i) * 1000,
                raw=dict(source='empatica-avro', eda_us=2 + math.sin(i / 7) / 10, temp_c=33.0, hr=75.0, rr_intervals_ms=[800.0]))))
        return first, results

    def test_sessions_lists_every_saved_session_newest_first(self):
        self.assertEqual(self.service.sessions(), [])
        self.batch(3, 'older')
        self.batch(2, 'newer')
        rows = self.service.sessions()
        self.assertEqual([r['session_id'] for r in rows], ['newer', 'older'])
        self.assertEqual((rows[0]['readings'], rows[0]['mode']), (2, 'batch'))
        with self.assertRaises(ValueError):
            self.service.sessions(0)

    def test_history_after_a_time_returns_the_oldest_newer_rows_so_nothing_is_skipped(self):
        first, _ = self.batch(10)
        page = self.service.history('picker', 3, after=(first + 2) * 1000)
        self.assertEqual([row['timestamp_ms'] // 1000 for row in page], [first + 3, first + 4, first + 5])
        self.assertEqual(self.service.history('picker', 3)[-1]['timestamp_ms'] // 1000, first + 9)

    def test_cues_lists_would_be_cues_from_late_data(self):
        with tempfile.TemporaryDirectory() as d:
            from serving.server import ClosedLoopService
            self.service.close()
            self.service = ClosedLoopService(Path(self.temp.name) / 'cues.sqlite3')
            from pipeline import FeedbackPipeline
            self.service.pipeline = FeedbackPipeline(rules=stub_rules(d, 'high'))
            _, results = self.batch(200)
            expected = sum(bool(r['states']['frustration'].get('wouldAlert')) for r in results)
            cues = self.service.cues('picker')
        self.assertEqual(expected, 1)
        self.assertEqual(len(cues), 1)
        self.assertEqual((cues[0]['state'], cues[0]['delivered'], cues[0]['consecutive']), ('frustration', False, 3))
        self.assertTrue(cues[0]['message'])

    def setUp(self):
        from serving.server import ClosedLoopService
        self.temp = tempfile.TemporaryDirectory()
        self.service = ClosedLoopService(Path(self.temp.name) / 'db.sqlite3')

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def sample(self, i, t, mode, session='s'):
        return dict(session_id=session, participant='p', mode=mode, sample_id=str(i), timestamp_ms=t,
                    raw=dict(eda_us=2 + math.sin(i / 7) / 10, temp_c=33.0))

    def test_batch_data_is_accepted_late_and_is_current_while_arriving(self):
        old = int(time.time() * 1000) - 3_600_000
        for i in range(5):
            result = self.service.process_sample(self.sample(i, old + i * 1000, 'batch'))
        self.assertFalse(result['stale_on_arrival'])
        self.assertFalse(self.service.latest('s')['stale'])
        self.assertEqual(self.service.summary('s')['alerts'], 0)
        with self.assertRaises(ValueError):
            self.service.process_sample(self.sample(9, int(time.time() * 1000) + 60_000, 'batch', 'future'))

    def test_late_live_eda_never_reaches_the_frustration_model(self):
        old = int(time.time() * 1000) - 600_000
        for i in range(70):
            result = self.service.process_sample(self.sample(i, old + i * 1000, 'live'))
        model = result['states']['frustration']['model']
        self.assertTrue(result['stale_on_arrival'])
        self.assertIn(model['status'], ('waiting', 'disabled'))
        self.assertNotIn('ml', model['status'])
        self.assertEqual(self.service.export('s')['readings'][0]['sample']['raw']['eda_us'], self.sample(0, 0, 'live')['raw']['eda_us'])


if __name__ == '__main__':
    unittest.main()
