"""Fixtures verify software contracts only, never physiological performance."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sixty_second_rules():
    # Archived training fixtures use the original measurement protocol.
    from pipeline.config import LEGACY_FEATURE_POLICY
    rules = json.loads((ROOT / 'config/rules.json').read_text())
    rules['featurePolicy'] = dict(LEGACY_FEATURE_POLICY)
    return rules


def sessions(count=3):
    result = []
    for person in range(count):
        ratings = []
        for i in range(12):
            high = i >= 6
            ratings.append(dict(time=f'2026-10-21T12:{i:02d}:00Z', source='participant-self-report',
                                level='high' if high else 'none',
                                snapshot=dict(hr=110 + person if high else 80 + person,
                                              rmssd=20 if high else 40, sdnn=15 if high else 30,
                                              baseline=dict(hr=80, rmssd=40, sdnn=30),
                                              fresh=True, baselineState='ready', rrCount=60,
                                              windowSeconds=60, requiredWindowSeconds=60)))
        result.append(dict(schemaVersion=2, mode='live', participant=f'P{person}',
                           sessionId=f'session-{person}', fatigueRatings=ratings))
    return result


def policy_sessions(window=120):
    data = sessions()
    policy = dict(windowSeconds=window, freshSeconds=5, minRr=60, rrMinMs=300,
                  rrMaxMs=2000, baselineSamples=30)
    for person in data:
        for index, rating in enumerate(person['fatigueRatings']):
            rating['time'] = f'2026-10-21T12:{index * 2:02d}:00Z'
            rating['snapshot'].update(featurePolicy=copy.deepcopy(policy),
                                      requiredWindowSeconds=window, windowSeconds=window - 5)
    return data, policy


class FeaturePolicyTests(unittest.TestCase):
    def test_120_second_policy_accepts_eligible_snapshot_and_window_spacing(self):
        from train_polar_fatigue import features_from_snapshot, load_sessions
        data, policy = policy_sessions()
        x, y, people = load_sessions(data)
        self.assertEqual(x.shape, (36, 3))
        np.testing.assert_allclose(features_from_snapshot(data[0]['fatigueRatings'][0]['snapshot']), [0, 0, 0])
        for changes in [dict(windowSeconds=114.9), dict(rrCount=59), dict(requiredWindowSeconds=60)]:
            snapshot = {**data[0]['fatigueRatings'][0]['snapshot'], **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                features_from_snapshot(snapshot)
        data[0]['fatigueRatings'][1]['time'] = '2026-10-21T12:01:59Z'
        with self.assertRaisesRegex(ValueError, '120 seconds'):
            load_sessions(data)

    def test_mixed_policies_and_invalid_explicit_policies_are_rejected(self):
        from train_polar_fatigue import features_from_snapshot, load_sessions
        data = sessions()
        data[-1]['fatigueRatings'][-1]['snapshot']['featurePolicy'] = dict(
            windowSeconds=60, freshSeconds=5, minRr=31, rrMinMs=300, rrMaxMs=2000, baselineSamples=30)
        with self.assertRaisesRegex(ValueError, 'same feature policy|Mixed feature polic'):
            load_sessions(data)
        original = sessions()[0]['fatigueRatings'][0]['snapshot']
        policy = dict(windowSeconds=60, freshSeconds=5, minRr=30, rrMinMs=300, rrMaxMs=2000, baselineSamples=30)
        for invalid in [None, {}, {**policy, 'freshSeconds': 60}, {**policy, 'minRr': True},
                        {**policy, 'rrMaxMs': 200}, {**policy, 'baselineSamples': 0}]:
            with self.subTest(policy=invalid), self.assertRaises(ValueError):
                features_from_snapshot({**original, 'featurePolicy': invalid})

    def test_comparison_requires_data_policy_to_match_selected_rules(self):
        from training.train_model import compare_sessions
        data, policy = policy_sessions()
        with self.assertRaisesRegex(ValueError, 'feature policy|featurePolicy'):
            compare_sessions(data)
        rules = json.loads((ROOT / 'config/rules.json').read_text())
        rules['featurePolicy'] = policy
        result = compare_sessions(data, rules)
        self.assertEqual(result['featurePolicy'], policy)
        legacy_rules = json.loads((ROOT / 'config/rules.json').read_text())
        legacy_rules['featurePolicy']['windowSeconds'] = 120
        with self.assertRaisesRegex(ValueError, 'feature policy|featurePolicy'):
            compare_sessions(sessions(), legacy_rules)

    def test_portable_artifact_declares_its_feature_protocol(self):
        from train_polar_fatigue import load_sessions, train_model
        data, policy = policy_sessions()
        x, y, groups = load_sessions(data)
        artifact, _ = train_model(x, y, groups, source='participant-self-report', feature_policy=policy)
        self.assertEqual(artifact['featurePolicy'], policy)
        legacy_x, legacy_y, legacy_groups = load_sessions(sessions())
        legacy, _ = train_model(legacy_x, legacy_y, legacy_groups, source='participant-self-report')
        self.assertEqual(legacy['featurePolicy']['windowSeconds'], 60)
        self.assertEqual(legacy['featurePolicy']['minRr'], 30)


class TrainingTests(unittest.TestCase):
    def test_tied_comparison_preserves_deployed_model_and_writes_report(self):
        from training.train_model import train_sessions
        with tempfile.TemporaryDirectory() as directory:
            output, report = Path(directory) / 'model.json', Path(directory) / 'report.json'
            output.write_text('existing approved model')
            result = train_sessions(sessions(), output, report, rules=sixty_second_rules())
            self.assertFalse(result['approved'])
            self.assertEqual(output.read_text(), 'existing approved model')
            self.assertEqual(json.loads(report.read_text()), result)
            self.assertEqual(set(result['metrics']), {'rules', 'ml', 'blend'})
            self.assertEqual(result['blendWeight'], .5)
            self.assertEqual(len(result['folds']), 3)
            self.assertEqual(result['coverage']['unseenLevels'], ['low', 'medium', 'very-high'])

    def test_winning_blend_writes_portable_approved_candidate(self):
        from training.train_model import train_sessions
        # Intentionally constructed rule/label disagreement tests the approval
        # branch only. These numbers have no physiological interpretation.
        rng = np.random.default_rng(0)
        data = sessions()
        for person in data:
            for i, row in enumerate(person['fatigueRatings']):
                high = i % 2 == 0
                values = rng.uniform([26, -90, -90], [100, -41, -36])
                if not high:
                    values[i % 3] = [12, -30, -25][i % 3]
                row['level'] = 'high' if high else 'medium'
                row['snapshot'].update(hr=80 * (1 + values[0] / 100),
                                       rmssd=40 * (1 + values[1] / 100),
                                       sdnn=30 * (1 + values[2] / 100))
        with tempfile.TemporaryDirectory() as directory:
            output, report = Path(directory) / 'model.json', Path(directory) / 'report.json'
            result = train_sessions(data, output, report, ROOT / 'tests/fixtures/legacy_polar_rules.json')
            self.assertTrue(result['approved'])
            self.assertGreater(result['metrics']['blend']['macroF1'], result['metrics']['rules']['macroF1'])
            self.assertGreater(result['metrics']['blend']['macroF1'], result['metrics']['ml']['macroF1'])
            artifact = json.loads(output.read_text())
            self.assertEqual(artifact['evaluation']['status'], 'available')
            self.assertEqual(artifact['evaluation']['macroF1'], result['metrics']['blend']['macroF1'])
            script = "import { validateModel } from './sensor/fatigue-model.js'; let input=''; for await (const part of process.stdin) input += part; process.stdout.write(validateModel(JSON.parse(input)).modelId);"
            browser = subprocess.run(['node', '--input-type=module', '-e', script], cwd=ROOT,
                                     input=json.dumps(artifact), capture_output=True, text=True)
            self.assertEqual(browser.returncode, 0, browser.stderr)
            self.assertEqual(browser.stdout, artifact['modelId'])
            self.assertEqual(artifact['deployment']['metrics'], result['metrics'])
            self.assertEqual(artifact['deployment']['blendWeight'], .5)
            self.assertEqual(artifact['deployment']['ruleConfigVersion'], result['ruleConfigVersion'])
            self.assertTrue(artifact['deployment']['approved'])
            self.assertEqual(artifact['training']['source'], 'participant-self-report')
            self.assertEqual(artifact['training']['participants'], 3)
            self.assertEqual(len(artifact['coefficients']), 2)
            self.assertEqual(len(artifact['mean']), 3)
            self.assertEqual(len(artifact['scale']), 3)
            # Exercise the real deployment reader, including binary JSON logits.
            from pipeline.model import ModelBlend
            from pipeline.rule_engine import RuleEngine
            from train_polar_fatigue import load_sessions
            rules = json.loads((ROOT / 'tests/fixtures/legacy_polar_rules.json').read_text())
            deployed = ModelBlend(output, rules)
            self.assertIsNone(deployed.reason)
            x, _, _ = load_sessions(data)
            features = dict(zip(['hrChangePct', 'rmssdChangePct', 'sdnnChangePct'], x[0]))
            rule = RuleEngine(rules).evaluate(features)
            decision = deployed.apply(rule, features, 'live')
            self.assertEqual(decision['model']['status'], 'approved-blend')
            self.assertEqual(decision['score_kind'], 'blend-score')
            self.assertAlmostEqual(decision['score'], .5 * rule['score'] + .5 * decision['mlScore'])

            # A second protocol must survive promotion into its own declared model.
            dynamic, policy = policy_sessions()
            for generated_person, dynamic_person in zip(data, dynamic):
                for generated, target in zip(generated_person['fatigueRatings'], dynamic_person['fatigueRatings']):
                    target['level'] = generated['level']
                    for key in ('hr', 'rmssd', 'sdnn'):
                        target['snapshot'][key] = generated['snapshot'][key]
            rules['featurePolicy'] = policy
            dynamic_report = train_sessions(dynamic, output, report, rules)
            self.assertTrue(dynamic_report['approved'])
            dynamic_artifact = json.loads(output.read_text())
            self.assertEqual(dynamic_artifact['featurePolicy'], policy)
            self.assertIsNone(ModelBlend(output, rules).reason)

    def test_actual_sqlite_exports_load_as_independent_training_sessions(self):
        from datetime import datetime
        from serving.server import ClosedLoopService
        from train_polar_fatigue import load_sessions
        data = sessions()
        policy = json.loads((ROOT / 'config/rules.json').read_text())['featurePolicy']
        for person in data:
            for rating in person['fatigueRatings']:
                rating['snapshot'].update(featurePolicy=dict(policy), windowSeconds=policy['windowSeconds'],
                    requiredWindowSeconds=policy['windowSeconds'], rrCount=policy['minRr'])
        expected_x, expected_y, expected_people = load_sessions(data)
        with tempfile.TemporaryDirectory() as directory:
            service = ClosedLoopService(Path(directory) / 'records.sqlite3')
            try:
                for person in data:
                    for index, rating in enumerate(person['fatigueRatings']):
                        epoch = datetime.fromisoformat(rating['time'].replace('Z', '+00:00')).timestamp()
                        observation = dict(session_id=person['sessionId'], participant=person['participant'],
                                           mode='live', sample_id=str(index), timestamp_ms=int(epoch * 1000),
                                           snapshot=rating['snapshot'])
                        with patch('serving.server.time.time', return_value=epoch):
                            reading = service.process_sample(observation)
                            service.feedback(dict(session_id=person['sessionId'], kind='rating',
                                                  reading_id=reading['reading_id'], level=rating['level']))
                exported = [service.export(person['sessionId']) for person in data]
            finally:
                service.close()
        actual_x, actual_y, actual_people = load_sessions(exported)
        np.testing.assert_allclose(actual_x, expected_x)
        np.testing.assert_array_equal(actual_y, expected_y)
        np.testing.assert_array_equal(actual_people, expected_people)
        self.assertTrue(all(len(person['fatigueRatings']) == 12 for person in exported))

    def test_insufficient_people_demo_and_missing_fold_classes_rejected(self):
        from training.train_model import compare_sessions
        with self.assertRaisesRegex(ValueError, 'three|3'):
            compare_sessions(sessions(2))
        data = sessions()
        data[0]['mode'] = 'demo'
        with self.assertRaisesRegex(ValueError, 'live'):
            compare_sessions(data)
        data = sessions()
        for i, session in enumerate(data):
            for rating in session['fatigueRatings']:
                rating['level'] = 'high' if i == 0 else 'none'
        with self.assertRaisesRegex(ValueError, 'class'):
            compare_sessions(data)

    def test_oversized_numeric_measurement_is_rejected_without_overflow(self):
        from training.train_model import compare_sessions
        data = sessions()
        data[0]['fatigueRatings'][0]['snapshot']['hr'] = 10 ** 400
        with self.assertRaisesRegex(ValueError, 'finite'):
            compare_sessions(data)

    def test_every_scaler_excludes_the_held_out_person(self):
        from training import train_model as module
        from train_polar_fatigue import load_sessions, new_pipeline
        data = sessions()
        x, _, groups = load_sessions(data)
        fitted = []
        def factory():
            model = new_pipeline()
            fitted.append(model)
            return model
        with patch.object(module, 'new_pipeline', side_effect=factory):
            result = module.compare_sessions(data, rules=sixty_second_rules())
        self.assertEqual(len(fitted), 3)
        for fold, model in zip(result['folds'], fitted):
            held = fold['heldOutParticipant']
            self.assertNotIn(held, fold['trainingParticipants'])
            np.testing.assert_allclose(model.steps[0][1].mean_, x[groups != held].mean(axis=0))

    def test_held_out_blend_uses_rules_outside_training_feature_range(self):
        from training.train_model import compare_sessions
        result = compare_sessions(sessions(), ROOT / 'tests/fixtures/legacy_polar_rules.json')
        self.assertEqual([fold['outOfRangeFallbackSamples'] for fold in result['folds']], [6, 0, 6])
        # P2's six high-labeled rows exceed the other people's HR range.
        # Deployable inference retains the very-high rule result on those rows.
        self.assertEqual(result['metrics']['blend']['confusionMatrix'][3][4], 6)

    def test_promotion_gate_is_strict_and_rejects_nonfinite_metrics(self):
        from training.train_model import should_promote
        def metrics(rules, ml, blend):
            return {name: dict(macroF1=value, accuracy=.5)
                    for name, value in zip(['rules', 'ml', 'blend'], [rules, ml, blend])}
        self.assertTrue(should_promote(metrics(.4, .5, .6)))
        for values in [(.6, .5, .6), (.4, .6, .6), (.8, .5, .6), (.4, .5, float('nan'))]:
            self.assertFalse(should_promote(metrics(*values)))

    def test_cli_cannot_overwrite_input_or_report_with_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'session.json'
            source.write_text(json.dumps(sessions()[0]))
            original = source.read_bytes()
            for tail in [['--output', str(source), '--report', str(Path(directory) / 'report.json')],
                         ['--output', str(Path(directory) / 'm.json'), '--report', str(source)]]:
                result = subprocess.run([sys.executable, str(ROOT / 'training/train_model.py'),
                                         '--sessions', str(source), *tail], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('overwrite', result.stderr)
                self.assertNotIn('Traceback', result.stderr)
            self.assertEqual(source.read_bytes(), original)


class FeedbackImportTests(unittest.TestCase):
    HEADER = 'state,level,cause,message,action,goal,reference,enabled\n'

    def test_template_round_trip_preserves_decision_defaults(self):
        from training.build_feedback_table import build_feedback_table
        original = json.loads((ROOT / 'config/feedback_table.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'feedback.json'
            result = build_feedback_table(ROOT / 'config/feedback_template.csv', output)
            self.assertEqual(result['decision'], original['decision'])
            self.assertEqual(result['states'], original['states'])
            self.assertEqual(json.loads(output.read_text()), result)

    def test_invalid_cells_and_duplicate_keys_never_replace_existing_config(self):
        from training.build_feedback_table import build_feedback_table
        template = (ROOT / 'config/feedback_template.csv').read_text().splitlines(keepends=True)
        valid = ''.join(template[1:])
        high = 'fatigue,high,fatigue-related-stress,Rest,Rest,Recover,Ref,true\n'
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'feedback.csv', Path(directory) / 'feedback.json'
            target.write_text('preserved')
            for rows in [valid.replace('fatigue,medium,any,,,,,false', 'fatigue,medium,any,Rest,Rest,Recover,Ref,true'),
                         valid.replace('frustration,low,any,,,,,false', 'frustration,low,any,Rest,Rest,Recover,Ref,true'),
                         valid.replace('fatigue,medium,any,,,,,false', 'fatigue,medium,any,,,,,maybe'),
                         valid + high.replace(',true', ',true,extra'),
                         valid + high + high,
                         valid + 'fatigue,high,other-cause,Rest,Rest,Recover,Ref,false\n',
                         valid + 'fatigue,high,other-cause,,Rest,Recover,Ref,true\n',
                         valid.replace('posture,critical,chest-tilt', 'posture,unknown,chest-tilt'),
                         ''.join(line for line in template[1:] if not line.startswith('fatigue,low'))]:
                source.write_text(self.HEADER + rows)
                with self.subTest(rows=rows[-80:]), self.assertRaises(ValueError):
                    build_feedback_table(source, target)
                self.assertEqual(target.read_text(), 'preserved')

    def test_import_cannot_enable_normal_posture_feedback(self):
        from training.build_feedback_table import build_feedback_table
        text = (ROOT / 'config/feedback_template.csv').read_text()
        text = text.replace('posture,good,any,,,,,false', 'posture,good,any,Rest,Rest,Recover,Ref,true')
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'input.csv', Path(directory) / 'output.json'
            source.write_text(text)
            with self.assertRaises(ValueError):
                build_feedback_table(source, target)
            self.assertFalse(target.exists())

    @unittest.skipUnless(importlib.util.find_spec('openpyxl'), 'optional openpyxl is not installed')
    def test_normalized_xlsx_accepts_boolean_cells(self):
        import csv
        import openpyxl
        from training.build_feedback_table import build_feedback_table
        original = json.loads((ROOT / 'config/feedback_table.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'table.xlsx'
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            with (ROOT / 'config/feedback_template.csv').open(newline='') as stream:
                for index, row in enumerate(csv.reader(stream)):
                    sheet.append(row if index == 0 else row[:-1] + [row[-1] == 'true'])
            workbook.save(source)
            result = build_feedback_table(source, Path(directory) / 'out.json')
            self.assertEqual(result['states'], original['states'])

    @unittest.skipIf(importlib.util.find_spec('openpyxl'), 'dependency installed; XLSX round-trip covers import')
    def test_xlsx_missing_dependency_has_actionable_error(self):
        from training.build_feedback_table import build_feedback_table
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'openpyxl'):
                build_feedback_table(Path(directory) / 'input.xlsx', Path(directory) / 'output.json')



class DemoTests(unittest.TestCase):
    def test_demo_records_its_configured_feature_protocol(self):
        from tools.demo_stream import demo_samples
        policy = dict(windowSeconds=120, freshSeconds=5, minRr=90, rrMinMs=300,
                      rrMaxMs=2000, baselineSamples=30)
        sample = next(demo_samples(feature_policy=policy))
        self.assertEqual(sample['snapshot']['featurePolicy'], policy)
        self.assertEqual(sample['snapshot']['requiredWindowSeconds'], 120)
        self.assertEqual(sample['snapshot']['windowSeconds'], 120)
        self.assertGreaterEqual(sample['snapshot']['rrCount'], 90)

    def test_demo_observes_rest_frustration_fatigue_and_recovery_with_explicit_provenance(self):
        from tools.demo_stream import demo_samples, run_demo
        samples = list(demo_samples())
        self.assertTrue(samples)
        self.assertTrue(all(sample['mode'] == 'demo' for sample in samples))
        self.assertTrue(all(sample['raw']['source'] == 'synthetic-script' for sample in samples))
        results = list(run_demo())
        self.assertEqual(len(results), len(samples))
        for state in ('frustration', 'fatigue'):
            levels = [result['states'][state]['level'] for result in results]
            self.assertEqual(levels[0], 'low')
            self.assertEqual(levels[-1], 'low')
            self.assertEqual({'low', 'medium', 'high'}, set(levels))
        self.assertEqual(sorted(a['state'] for r in results for a in r['alerts']), ['fatigue', 'frustration'])


if __name__ == '__main__':
    unittest.main()
