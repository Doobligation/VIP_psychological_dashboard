"""Frustration ML: shared features, live signal tracking, portable export and pipeline use."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

from pipeline import FeedbackPipeline
from pipeline.frustration_model import CLASSES, FEATURES, FrustrationModel, SignalTracker, count_peaks, window_features

ROOT = Path(__file__).resolve().parents[1]
HAVE_ML = all(importlib.util.find_spec(m) for m in ('numpy', 'pandas', 'sklearn'))


def stub_model(directory, prefer='high'):
    """A logistic model whose intercepts force one class, for deterministic pipeline checks."""
    path = Path(directory) / 'stub.json'
    path.write_text(json.dumps(dict(
        schemaVersion=1, modelType='logistic', features=FEATURES, classes=CLASSES, family='logistic',
        mean=[0.0] * len(FEATURES), scale=[1.0] * len(FEATURES),
        coefficients=[[0.0] * len(FEATURES) for _ in CLASSES],
        intercepts=[5.0 if c == prefer else 0.0 for c in CLASSES],
        evaluation=dict(metrics=dict(windowAccuracy=.58)))))
    return path


def rules_with(model_file):
    rules = json.loads((ROOT / 'config/rules.json').read_text())
    rules['states']['frustration']['modelFile'] = str(model_file)
    return rules


def eda_sample(t, eda=2.0, temp=33.0, session='ml'):
    raw = dict(eda_us=eda)
    if temp is not None:
        raw['temp_c'] = temp
    return dict(session_id=session, participant='p', mode='demo', sample_id=str(t), timestamp_ms=t, raw=raw)


class FeatureTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_ML, 'numpy/pandas not installed')
    def test_window_features_match_numpy_and_pandas(self):
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(1)
        eda, temp = rng.normal(5, 1, 60).cumsum().tolist(), rng.normal(33, .1, 60).tolist()
        f = window_features(eda, temp, (min(eda) - 1, max(eda) + 2), (min(temp), max(temp)))
        e = (np.array(eda) - (min(eda) - 1)) / (max(eda) + 2 - (min(eda) - 1))
        self.assertAlmostEqual(f['eda_std'], e.std())
        self.assertAlmostEqual(f['eda_slope'], np.polyfit(np.arange(60), e, 1)[0] * 60)
        self.assertAlmostEqual(f['eda_skew'], pd.Series(e).skew())
        self.assertAlmostEqual(f['eda_absdiff'], np.abs(np.diff(e)).mean())
        self.assertEqual(set(f), set(FEATURES))

    def test_features_are_scale_free_and_flat_signals_are_finite(self):
        eda = [2 + math.sin(i / 5) for i in range(60)]
        temp = [33 + i / 600 for i in range(60)]
        microsiemens = window_features(eda, temp, (1, 3), (33, 33.1))
        rescaled = window_features([x * 7 + 4 for x in eda], temp, (11, 25), (33, 33.1))
        for key in FEATURES:
            self.assertAlmostEqual(microsiemens[key], rescaled[key])
        flat = window_features([1.0] * 60, [1.0] * 60, (1.0, 1.0), (1.0, 1.0))
        self.assertTrue(all(math.isfinite(v) for v in flat.values()))
        self.assertEqual(count_peaks([0, .1, 0, .1, 0], .05), 2)


class TrackerTests(unittest.TestCase):
    def test_waits_for_a_full_window_and_resamples_to_one_hertz(self):
        tracker = SignalTracker()
        for t in range(0, 30000, 250):
            tracker.add(t, eda=2 + (t % 2000) / 1e4, temp=33)
        features, reason = tracker.features(30000)
        self.assertIsNone(features)
        self.assertIn('55 s', reason)
        for t in range(30000, 61000, 250):
            tracker.add(t, eda=2 + (t % 2000) / 1e4, temp=33)
        features, _ = tracker.features(61000)
        self.assertIsNotNone(features)
        self.assertEqual(len(tracker.values), 60)  # the current second is still open

    def test_missing_temperature_and_reversed_time_never_fill_the_window(self):
        tracker = SignalTracker()
        for t in range(0, 70000, 1000):
            tracker.add(t, eda=2.0)
        self.assertIsNone(tracker.features(70000)[0])
        tracker.add(5000, eda=9.0, temp=40.0)
        self.assertNotIn(5, tracker.values)


class ModelFileTests(unittest.TestCase):
    def test_shipped_model_loads_and_returns_probabilities(self):
        path = ROOT / json.loads((ROOT / 'config/rules.json').read_text())['states']['frustration']['modelFile']
        if not path.exists():
            self.skipTest('model not trained yet')
        model = FrustrationModel(path)
        eda = [2 + math.sin(i / 3) / 10 for i in range(60)]
        level, probabilities = model.predict(window_features(eda, [33.0 + i / 100 for i in range(60)], (1.8, 2.2), (33, 34)))
        self.assertIn(level, CLASSES)
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        self.assertEqual(model.data['evaluation']['scope'], 'new people')
        self.assertEqual(model.normalization, 'rolling-z')

    def test_rejects_models_with_other_features(self):
        with tempfile.TemporaryDirectory() as d:
            path = stub_model(d)
            data = json.loads(path.read_text())
            data['features'] = FEATURES[:-1]
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                FrustrationModel(path)

    @unittest.skipUnless(HAVE_ML, 'scikit-learn not installed')
    def test_exported_models_reproduce_library_probabilities(self):
        import numpy as np
        from sklearn.ensemble import ExtraTreesClassifier
        from training.train_frustration_model import calibrate_xgboost, export, make_model
        rng = np.random.default_rng(0)
        X = rng.normal(size=(150, len(FEATURES)))
        y = (X[:, 0] + X[:, 9] > 0).astype(int) + (X[:, 1] > .8)
        models = [('logistic', make_model('logistic', dict(C=1.0)).fit(X, y)),
                  ('extra-trees', ExtraTreesClassifier(15, max_depth=3, random_state=0).fit(X, y))]
        if importlib.util.find_spec('xgboost'):
            models.append(('xgboost', make_model('xgboost', dict(max_depth=2)).fit(X, y)))
        for family, model in models:
            exported = dict(export(family, model), schemaVersion=1, features=FEATURES, classes=CLASSES)
            if family == 'xgboost':
                calibrate_xgboost(exported, model, X)
            portable = FrustrationModel.__new__(FrustrationModel)
            portable.data, portable.kind = exported, exported['modelType']
            mine = np.array([portable.predict_proba(dict(zip(FEATURES, row))) for row in X])
            self.assertLess(np.abs(mine - model.predict_proba(X)).max(), 1e-5, family)


@unittest.skipUnless(HAVE_ML, 'numpy/pandas not installed')
class TrainingDataTests(unittest.TestCase):
    def test_loader_skips_exported_index_columns_and_reads_labels_in_any_order(self):
        from training.train_frustration_model import build_windows, load_sessions
        with tempfile.TemporaryDirectory() as d:
            rows = '\n'.join(f'2024-01-01 00:00:{i % 60:02d},{i},{0.5 + 0.4 * math.sin(i / 7):.4f},{0.5 + 0.3 * math.cos(i / 40):.4f}' for i in range(200))
            (Path(d) / 'P4_Low_Passive.csv').write_text('timestamp_local,123_eda,0.5_eda,temp_temp\n' + rows)
            (Path(d) / 'P5_Active_High.csv').write_text('timestamp_local,Unnamed: 0_eda,Unnamed: 2_eda,temperature_temp\n' + rows)
            sessions = load_sessions(d)
            self.assertEqual({(s['participant'], s['label'], s['condition']) for s in sessions},
                             {('P4', 'low', 'passive'), ('P5', 'high', 'active')})
            self.assertTrue(all(abs(s['eda'][1] - s['eda'][0]) < 1 for s in sessions))
            frame = build_windows(sessions)
            self.assertEqual(len(frame), 2 * 5)
            self.assertEqual(set(frame.columns) >= set(FEATURES), True)


class PipelineTests(unittest.TestCase):
    def test_streamed_eda_drives_ml_level_and_high_feedback_needs_three_consecutive(self):
        with tempfile.TemporaryDirectory() as d:
            pipe = FeedbackPipeline(rules=rules_with(stub_model(d, 'high')))
            outs = [pipe.process(eda_sample(t, eda=2 + math.sin(t / 5000) / 10, temp=33 + t / 1e6))
                    for t in range(0, 64000, 1000)]
        waiting = outs[30]['states']['frustration']
        self.assertEqual(waiting['model']['status'], 'waiting')
        ml = [o['states']['frustration'] for o in outs if o['states']['frustration']['model']['status'] == 'ml']
        self.assertGreaterEqual(len(ml), 3)
        self.assertEqual(ml[0]['level'], 'high')
        self.assertEqual(ml[0]['cause'], 'unspecified')
        self.assertEqual([m['streak']['count'] for m in ml[:3]], [1, 2, 3])
        self.assertIsNone(ml[1]['alert'])
        self.assertEqual(ml[2]['alert']['cause'], 'unspecified')
        self.assertIn('reassess', ml[2]['alert']['message'])

    def test_low_ml_prediction_gives_no_feedback_and_missing_model_falls_back_to_rules(self):
        with tempfile.TemporaryDirectory() as d:
            pipe = FeedbackPipeline(rules=rules_with(stub_model(d, 'low')))
            outs = [pipe.process(eda_sample(t, eda=2 + t / 1e5)) for t in range(0, 64000, 1000)]
            self.assertEqual(outs[-1]['states']['frustration']['level'], 'low')
            self.assertFalse(any(o['alerts'] for o in outs))
            missing = FeedbackPipeline(rules=rules_with(Path(d) / 'absent.json'))
            result = missing.process(eda_sample(1000))
        self.assertEqual(result['states']['frustration']['model']['status'], 'disabled')
        self.assertIn('unavailable', result['states']['frustration']['model']['reason'])
        no_model = copy.deepcopy(rules_with('x'))
        no_model['states']['frustration'].pop('modelFile')
        self.assertIn('No frustration model', FeedbackPipeline(rules=no_model).process(eda_sample(1000))['states']['frustration']['model']['reason'])


if __name__ == '__main__':
    unittest.main()
