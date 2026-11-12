"""Retraining from check-in answers (guide steps 9-10) on synthetic study data: gates, labels, safe switch."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import pandas as pd

from pipeline.frustration_model import CLASSES, FEATURES, FrustrationModel
from storage import SQLiteStore
from training import retrain_from_checkins as retrain
from training import train_frustration_model as study

LABELS = ('low', 'medium', 'high')


def synthetic_study(folder, people=6, seconds=300, seed=0):
    """Labelled sessions whose EDA gets livelier with the label, like the real P<n>_<condition>_<level>.csv files."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds)
    for person in range(1, people + 1):
        for condition, label in (('Active', LABELS[person % 3]), ('Passive', LABELS[(person + 1) % 3])):
            level = LABELS.index(label)
            eda = 2 + 0.05 * level * t / 60 + (0.1 + 0.2 * level) * np.sin(t / (9 - 2 * level)) + rng.normal(0, .02, seconds)
            temp = 33 + 0.3 * np.sin(t / 50) - 0.1 * level * t / seconds + rng.normal(0, .01, seconds)
            pd.DataFrame(dict(eda=eda, temp=temp)).to_csv(Path(folder) / f'P{person}_{condition}_{label.capitalize()}.csv', index=False)


def deploy_current(folder, model_path):
    """The 'current' model: the deployed family and scaling, trained on the study only."""
    sessions = study.load_sessions(folder)
    frame = study.build_windows(sessions, normalization='rolling-z')
    X, y = frame[FEATURES].to_numpy(float), frame['label'].map(CLASSES.index).to_numpy()
    weights = 1 / frame.groupby('session')['session'].transform('count').to_numpy(float)
    params = dict(C=1.0, normalization='rolling-z')
    model = study.fit('logistic', params, X, y, weights)
    meta = study.model_meta(frame, sessions, 'logistic', params, study.HOP_SECONDS,
                            dict(method='test', scope='new people', metrics=dict(windowAccuracy=.5)))
    exported, _ = study.export_checked('logistic', model, X, meta)
    Path(model_path).write_text(json.dumps(exported))
    return frame


def seed_answers(db, frame, workers=3, per_worker=10, mode='batch', normalization='rolling-z', answer=None):
    """Answered Quick checks whose moments carry ML features taken from study windows of known label."""
    store = SQLiteStore(db)
    now = time.time() * 1000
    rows = frame[frame.label.isin(['high', 'low'])]
    with store.connection:
        for w in range(workers):
            session = f'{mode}-session-{w}-{uuid.uuid4().hex[:6]}'
            store.ensure_session(session, f'worker-{w}', mode)
            for i, (_, row) in enumerate(rows.sample(per_worker, random_state=w, replace=len(rows) < per_worker).iterrows()):
                reading, ts = uuid.uuid4().hex, now - 3_600_000 + i * 60_000
                model = dict(status='ml', features={k: float(row[k]) for k in FEATURES}, normalization=normalization)
                result = dict(reading_id=reading, alerts=[], features={},
                              states=dict(frustration=dict(available=True, level='high', model=model),
                                          fatigue=dict(available=True, level='low')))
                store.save_reading(dict(session_id=session, sample_id=str(i), timestamp_ms=ts), result, ts)
                felt = answer or ('yes' if row['label'] == 'high' else 'no')
                store.save_checkin(dict(
                    id=uuid.uuid4().hex, kind='prompted', status='answered', session_id=session, participant=f'worker-{w}',
                    mode=mode, reading_id=reading, moment_ms=ts, asked_ms=ts, expires_ms=ts + 600_000, answered_ms=ts,
                    trigger=['frustration'], answers=dict(frustration=felt, fatigue='no'),
                    predicted=dict(frustration=dict(level='high', source='Empatica · ML model'),
                                   fatigue=dict(level='low', source='Empatica · DBR rules'))))
    store.close()


class RetrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.data, self.db, self.model = root / 'study', root / 'db.sqlite3', root / 'model.json'
        self.backups, self.report = root / 'backups', root / 'report.json'
        self.data.mkdir()
        synthetic_study(self.data)
        self.frame = deploy_current(self.data, self.model)
        self.original = self.model.read_bytes()

    def tearDown(self):
        self.temp.cleanup()

    def run_retrain(self, *extra):
        args = ['--db', str(self.db), '--model', str(self.model), '--data', str(self.data),
                '--backup-dir', str(self.backups), '--report', str(self.report), *extra]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            report = retrain.main(args)
        return report, out.getvalue()

    def test_too_few_answers_trains_nothing_and_keeps_the_model(self):
        seed_answers(self.db, self.frame, workers=1, per_worker=5)
        report, out = self.run_retrain('--apply')
        self.assertIsNone(report['evaluation'])
        self.assertFalse(report['verdict']['switch'])
        self.assertIn('Not enough answers yet', out)
        self.assertEqual(self.model.read_bytes(), self.original)
        self.assertTrue(self.report.exists())

    def test_both_felt_and_no_are_required(self):
        seed_answers(self.db, self.frame, workers=3, per_worker=10, answer='yes')
        report, _ = self.run_retrain()
        self.assertEqual(report['answers']['usable'], 30)
        self.assertIsNone(report['evaluation'])

    def test_demo_answers_and_other_scalings_are_left_out(self):
        seed_answers(self.db, self.frame, workers=2, per_worker=4, mode='demo')
        seed_answers(self.db, self.frame, workers=1, per_worker=3, normalization='baseline-z')
        report, _ = self.run_retrain()
        skipped = report['answers']['skipped']
        self.assertEqual(sum(n for reason, n in skipped.items() if reason.startswith('demo')), 8)
        self.assertEqual(sum(n for reason, n in skipped.items() if 'baseline-z' in reason), 3)
        self.assertEqual(report['answers']['usable'], 0)
        practice, _ = self.run_retrain('--include-demo')
        self.assertEqual(practice['answers']['usable'], 8)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.run_retrain('--include-demo', '--apply')

    def test_a_little_follows_the_little_option(self):
        seed_answers(self.db, self.frame, workers=1, per_worker=4, answer='a-little')
        felt, _ = self.run_retrain()
        notfelt, _ = self.run_retrain('--little', 'no')
        self.assertEqual((felt['answers']['high'], notfelt['answers']['low']), (4, 4))

    def test_full_run_reports_both_tests_and_force_switches_with_a_backup(self):
        seed_answers(self.db, self.frame)
        report, out = self.run_retrain()
        evaluation = report['evaluation']
        self.assertEqual((evaluation['answers']['answers'], evaluation['answers']['workers']), (30, 3))
        self.assertEqual(evaluation['study']['participants'], 6)
        for scope in ('answers', 'study'):
            for model in ('current', 'candidate'):
                self.assertTrue(0 <= evaluation[scope][model] <= 1)
        self.assertEqual(report['verdict'], retrain.verdict(evaluation, .02))
        self.assertIn('Verdict:', out)
        self.assertEqual(self.model.read_bytes(), self.original, 'reporting alone must not switch the model')

        applied, _ = self.run_retrain('--apply', '--force')
        self.assertTrue(applied['applied'])
        backup = Path(applied['backup'])
        self.assertEqual(backup.read_bytes(), self.original)
        deployed = FrustrationModel(self.model)
        self.assertEqual(deployed.data['training']['checkins']['answers'], 30)
        self.assertEqual(deployed.normalization, 'rolling-z')
        self.assertEqual(deployed.data['evaluation']['scope'], 'new people')

    def test_apply_respects_a_failing_verdict(self):
        seed_answers(self.db, self.frame)
        with patch.object(retrain, 'verdict', return_value=dict(switch=False, reasons=['Answers: worse.'])):
            report, out = self.run_retrain('--apply')
        self.assertFalse(report['applied'])
        self.assertIn('Not applied', out)
        self.assertEqual(self.model.read_bytes(), self.original)
        self.assertFalse(self.backups.exists())


class VerdictTests(unittest.TestCase):
    def evaluation(self, answers, study_current, study_candidate, workers=3):
        return dict(answers=dict(current=answers[0], candidate=answers[1], workers=workers),
                    study=dict(current=study_current, candidate=study_candidate))

    def test_switch_needs_as_good_on_answers_and_at_most_the_allowed_study_loss(self):
        self.assertTrue(retrain.verdict(self.evaluation((.6, .6), .6, .58), .02)['switch'])
        self.assertTrue(retrain.verdict(self.evaluation((.6, .7), .6, .65), .02)['switch'])
        self.assertFalse(retrain.verdict(self.evaluation((.6, .59), .6, .7), .02)['switch'])
        self.assertFalse(retrain.verdict(self.evaluation((.6, .7), .6, .579), .02)['switch'])
        lone = retrain.verdict(self.evaluation((.6, .6), .6, .6, workers=1), .02)
        self.assertTrue(any('Only one worker' in reason for reason in lone['reasons']))


if __name__ == '__main__':
    unittest.main()
