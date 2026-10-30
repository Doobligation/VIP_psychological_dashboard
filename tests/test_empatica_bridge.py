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
