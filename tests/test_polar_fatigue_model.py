"""Synthetic fixtures test software contracts, not physiological accuracy."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from train_polar_fatigue import features_from_snapshot, load_sessions, train_model, bootstrap_samples

ROOT = Path(__file__).resolve().parents[1]


def snapshot(hr=88, rmssd=32, sdnn=24):
    return dict(hr=hr, rmssd=rmssd, sdnn=sdnn,
                baseline=dict(hr=80, rmssd=40, sdnn=30), baselineState='ready',
                fresh=True, rrCount=60, windowSeconds=59, requiredWindowSeconds=60)


def session(participant='P01', session_id='session-1'):
    return dict(schemaVersion=2, mode='live', participant=participant, sessionId=session_id,
                fatigueRatings=[dict(time=f'2026-10-21T12:{i:02d}:00Z',
                                     source='participant-self-report', level='low' if i < 5 else 'high',
                                     snapshot=snapshot(80 + i * 3, 40 - i * 2, 30 - i))
                                for i in range(10)])


class PolarModelTests(unittest.TestCase):
    def test_features_are_baseline_percentages_not_rule_labels(self):
        value = snapshot()
        value['fatigue'] = dict(level='very-high', changes=dict(hr=999, rmssd=999, sdnn=999))
        np.testing.assert_allclose(features_from_snapshot(value), [10, -20, -20])

    def test_features_reject_ineligible_and_malformed_snapshots(self):
        for key, value in [('fresh', False), ('fresh', 1), ('windowSeconds', 54.9),
                           ('rrCount', 29), ('hr', None), ('rmssd', float('nan')),
                           ('sdnn', -1), ('baseline', dict(hr=80, rmssd=0, sdnn=30)),
                           ('requiredWindowSeconds', 30), ('baselineState', 'collecting')]:
            with self.subTest(key=key, value=value):
                row = snapshot()
                row[key] = value
                with self.assertRaises(ValueError):
                    features_from_snapshot(row)

    def test_only_independent_live_labels_can_train(self):
        original = session()
        for mutate in [lambda s: s.update(mode='demo'), lambda s: s.update(participant=''),
                       lambda s: s['fatigueRatings'][0].update(source='rule'),
                       lambda s: s['fatigueRatings'][0].update(level='healthy'),
                       lambda s: s['fatigueRatings'][0]['snapshot'].update(fresh=False)]:
            altered = copy.deepcopy(original)
            mutate(altered)
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                load_sessions([altered])
        x, y, groups = load_sessions([original])
        self.assertEqual(x.shape, (10, 3))
        self.assertEqual(list(y[:5]), ['low'] * 5)
        self.assertEqual(set(groups), {'P01'})

    def test_duplicate_sessions_and_overlapping_ratings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_sessions([session(), session()])
        s = session()
        s['fatigueRatings'][1]['time'] = '2026-10-21T12:00:59Z'
        with self.assertRaisesRegex(ValueError, '60 seconds'):
            load_sessions([s])

    def test_personal_model_is_unvalidated_and_binary_logits_are_exported_correctly(self):
        x, y, groups = load_sessions([session()])
        artifact, pipeline = train_model(x, y, groups, source='participant-self-report')
        self.assertEqual(artifact['classes'], ['low', 'high'])
        self.assertEqual(artifact['evaluation']['status'], 'unavailable')
        self.assertIn('participant', artifact['evaluation']['reason'])
        self.assertEqual(artifact['training']['participants'], 1)
        z = (x - artifact['mean']) / artifact['scale']
        logits = z @ np.asarray(artifact['coefficients']).T + artifact['intercepts']
        predictions = np.asarray(artifact['classes'])[np.argmax(logits, axis=1)]
        np.testing.assert_array_equal(predictions, pipeline.predict(x))

    def test_grouped_evaluation_keeps_each_participant_out_of_training(self):
        data = [session(f'P{i}', f'session-{i}') for i in range(3)]
        x, y, groups = load_sessions(data)
        artifact, _ = train_model(x, y, groups, source='participant-self-report')
        evaluation = artifact['evaluation']
        self.assertEqual(evaluation['method'], 'leave-one-participant-out')
        self.assertEqual(evaluation['status'], 'available')
        self.assertEqual(len(evaluation['folds']), 3)
        for fold in evaluation['folds']:
            self.assertEqual(fold['trainingParticipants'], 2)
            self.assertEqual(fold['testSamples'], 10)
        self.assertEqual(sum(map(sum, evaluation['confusionMatrix'])), 30)

    def test_incomplete_fold_class_coverage_is_not_reported_as_validated(self):
        a, b = session('A', 'a'), session('B', 'b')
        for row in a['fatigueRatings']:
            row['level'] = 'low'
        for row in b['fatigueRatings']:
            row['level'] = 'high'
        x, y, groups = load_sessions([a, b])
        artifact, _ = train_model(x, y, groups, source='participant-self-report')
        self.assertEqual(artifact['evaluation']['status'], 'unavailable')
        self.assertIn('class', artifact['evaluation']['reason'])

    def test_training_needs_varied_labels_and_rejects_nonfinite_features(self):
        x, y, groups = load_sessions([session()])
        for labels in [['low'] * len(y), ['high'] + ['low'] * (len(y) - 1)]:
            with self.assertRaises(ValueError):
                train_model(x, np.asarray(labels), groups, source='participant-self-report')
        x[0, 0] = np.inf
        with self.assertRaises(ValueError):
            train_model(x, y, groups, source='participant-self-report')

    def test_bootstrap_has_explicit_synthetic_provenance_and_no_accuracy_claim(self):
        x, y, groups = bootstrap_samples()
        artifact, _ = train_model(x, y, groups, source='synthetic-bootstrap')
        self.assertEqual(artifact['training']['source'], 'synthetic-bootstrap')
        self.assertEqual(artifact['training']['participants'], 0)
        self.assertEqual(len(artifact['classes']), 5)
        self.assertEqual(artifact['evaluation']['status'], 'not-applicable')
        self.assertNotIn('accuracy', artifact['evaluation'])
        json.dumps(artifact, allow_nan=False)

    def test_python_artifact_predictions_match_real_browser_module(self):
        script = """
import { validateModel, predictFatigue } from './sensor/fatigue-model.js';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const {artifact, snapshots} = JSON.parse(input);
const model = validateModel(artifact);
process.stdout.write(JSON.stringify(snapshots.map(s => predictFatigue(model, s, {mode: 'demo'}).level)));
"""
        constant = session()
        for rating in constant['fatigueRatings']:
            rating['snapshot'] = snapshot(80, 40, 30)
        for data in [session(), constant]:
            with self.subTest(constant=data is constant):
                x, y, groups = load_sessions([data])
                artifact, pipeline = train_model(x, y, groups, source='participant-self-report')
                result = subprocess.run(['node', '--input-type=module', '-e', script], cwd=ROOT,
                                        input=json.dumps(dict(artifact=artifact, snapshots=[r['snapshot'] for r in data['fatigueRatings']])),
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), pipeline.predict(x).tolist())

    def test_cli_preserves_input_and_existing_artifact_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'session.json'
            target = Path(directory) / 'model.json'
            source.write_text(json.dumps(dict(mode='demo')))
            target.write_text('previous model')
            result = subprocess.run([sys.executable, str(ROOT / 'train_polar_fatigue.py'),
                                     '--sessions', str(source), '--output', str(target)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('Traceback', result.stderr)
            self.assertEqual(target.read_text(), 'previous model')
            source.write_text(json.dumps(session()))
            result = subprocess.run([sys.executable, str(ROOT / 'train_polar_fatigue.py'),
                                     '--sessions', str(source), '--output', str(source)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('input', result.stderr)
            self.assertEqual(json.loads(source.read_text())['mode'], 'live')


if __name__ == '__main__':
    unittest.main()
