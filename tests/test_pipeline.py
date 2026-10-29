"""Behavioral checks for the configured wearable processing pathway.

One test per threshold, per timing rule, and an end-to-end run of the scripted demo.
"""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from pipeline import FeedbackPipeline, RuleEngine, DecisionEngine, FeaturePipeline, calculate_hrv
from pipeline.features import scr_per_minute

ROOT = Path(__file__).resolve().parents[1]
FRUSTRATION_HIGH = {
    'rework-repeated-task-failure': (35, 45, 17, -27, -27),
    'schedule-pressure': (27, 27, 22, -32, -32),
    'equipment-tool-problems': (35, 30, 17, -27, -27),
    'poor-communication-unclear-instructions': (35, 30, 17, -32, -32),
    'material-shortage-waiting-time': (5, 5, 5, -27, -27),
}
SIGNALS = ('edaChangePct', 'scrChangePct', 'hrChangePct', 'rmssdChangePct', 'sdnnChangePct')


def config(name):
    return json.loads((ROOT / 'config' / name).read_text())


def changes(values):
    return dict(zip(SIGNALS, values))


def snapshot(level='medium', posture=0, policy=None, eda=None):
    """Fatigue-shaped HR/HRV snapshot; EDA/SCR only when given (as % change)."""
    policy = copy.deepcopy(policy or config('rules.json')['featurePolicy'])
    hr, rmssd, sdnn = {'low': (0, 0, 0), 'medium': (11, -26, -21), 'high': (5, -36, -31)}[level]
    if level == 'high' and eda is None:
        eda = 0
    result = dict(hr=100 + hr, rmssd=100 + rmssd, sdnn=100 + sdnn,
                  baseline=dict(hr=100, rmssd=100, sdnn=100), baselineState='ready',
                  fresh=True, rrCount=policy['minRr'], windowSeconds=policy['windowSeconds'],
                  requiredWindowSeconds=policy['windowSeconds'], featurePolicy=policy,
                  postureCalibrated=True, posture=dict(level='good', pitch=posture, roll=0))
    if eda is not None:
        result.update(eda=100 + eda, scr=100 + eda, edaFresh=True)
        result['baseline'].update(eda=100, scr=100)
    return result


def sample(t, level='medium', session='one', policy=None, **kwargs):
    return dict(session_id=session, participant='person', mode='live', sample_id=str(t),
                timestamp_ms=t, snapshot=snapshot(level, policy=policy), **kwargs)


def single_cause_rules(cause):
    rules = config('rules.json')
    state = rules['states']['frustration']
    state['rules'] = [r for r in state['rules'] if r['cause'] == cause]
    return rules


def passing(rule):
    """Values just inside every condition; two-sided bands use their midpoint."""
    bounds = {}
    for c in rule['conditions']:
        bounds.setdefault(c['signal'], []).append(c)
    values = {}
    for signal, conditions in bounds.items():
        if len(conditions) == 2:
            values[signal] = sum(c['value'] for c in conditions) / 2
        else:
            c = conditions[0]
            values[signal] = c['value'] + (0.5 if c['operator'] in ('>', '>=') else -0.5)
    return values


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.rules = config('rules.json')
        self.engine = RuleEngine(self.rules)

    def test_risk_levels_are_exactly_low_medium_high(self):
        for name in ('frustration', 'fatigue'):
            state = self.rules['states'][name]
            self.assertEqual({r['level'] for r in state['rules']}, {'low', 'medium', 'high'})
            self.assertEqual(state['defaultLevel'], 'low')

    def test_required_hrv_evidence_and_newest_fatigue_levels(self):
        for missing in ({}, {'hrChangePct': 50}, {'hrChangePct': 50, 'rmssdChangePct': -50}):
            self.assertFalse(self.engine.evaluate(missing)['available'])
        self.assertEqual(self.engine.evaluate(dict(rmssdChangePct=0, sdnnChangePct=0))['level'], 'low')
        self.assertEqual(self.engine.evaluate(dict(rmssdChangePct=-16, sdnnChangePct=-16))['level'], 'low')
        self.assertEqual(self.engine.evaluate(dict(rmssdChangePct=-26, sdnnChangePct=-21))['level'], 'medium')
        self.assertEqual(self.engine.evaluate(dict(rmssdChangePct=-36, sdnnChangePct=-31,hrChangePct=0,edaChangePct=0))['level'], 'high')

    def test_every_frustration_threshold_is_strict(self):
        """One check per condition: just inside passes, the exact boundary drops the level."""
        for rule in self.rules['states']['frustration']['rules']:
            engine = RuleEngine(single_cause_rules(rule['cause']))
            features = passing(rule)
            above = [r['severity'] for r in self.rules['states']['frustration']['rules']
                     if r['cause'] == rule['cause'] and r['severity'] > rule['severity']]
            for index, condition in enumerate(rule['conditions']):
                with self.subTest(cause=rule['cause'], level=rule['level'], condition=condition):
                    result = engine.evaluate(features, 'frustration')
                    if not above:
                        self.assertEqual(result['level'], rule['level'])
                    self.assertGreaterEqual(result['severity'], rule['severity'])
                    at_boundary = engine.evaluate(dict(features, **{condition['signal']: condition['value']}), 'frustration')
                    if rule['level'] == 'low':
                        # Below the Low pattern is still Low risk, but no cause pattern matches.
                        self.assertEqual((at_boundary['level'], at_boundary['cause']), ('low', None))
                    else:
                        self.assertLess(at_boundary['severity'], rule['severity'])

    def test_frustration_selects_the_most_specific_cause(self):
        labels = {k: v['label'] for k, v in self.rules['states']['frustration']['causes'].items()}
        for cause, values in FRUSTRATION_HIGH.items():
            with self.subTest(cause=cause):
                result = self.engine.evaluate(changes(values), 'frustration')
                self.assertEqual((result['level'], result['cause']), ('high', cause))
                self.assertEqual(result['causeLabel'], labels[cause])
        overlap = self.engine.evaluate(changes(FRUSTRATION_HIGH['poor-communication-unclear-instructions']), 'frustration')
        self.assertEqual(overlap['alternatives'], ['Equipment / Tool Problems'])

    def test_frustration_never_treats_missing_eda_or_scr_as_normal(self):
        polar_only = dict(hrChangePct=17, rmssdChangePct=-27, sdnnChangePct=-27)
        result = self.engine.evaluate(polar_only, 'frustration')
        self.assertFalse(result['available'])
        self.assertEqual(result['missingSignals'], ['edaChangePct', 'scrChangePct'])
        self.assertIn('EDA, SCR', result['reason'])
        self.assertEqual(self.engine.evaluate(changes((0, 0, 0, 0, 0)), 'frustration')['level'], 'low')

    def test_config_changes_take_effect_and_posture_requires_both_angles(self):
        features = dict(absPitch=20, absRoll=10)
        self.assertEqual(self.engine.evaluate(features, 'posture')['level'], 'good')
        self.assertEqual(self.engine.evaluate(dict(absPitch=20.1, absRoll=0), 'posture')['level'], 'poor')
        self.assertEqual(self.engine.evaluate(dict(absPitch=0, absRoll=20.1), 'posture')['level'], 'critical')
        self.assertFalse(self.engine.evaluate({'absPitch': 70}, 'posture')['available'])
        self.assertFalse(self.engine.evaluate({}, 'stress')['available'])
        changed = copy.deepcopy(self.rules)
        changed['states']['posture']['rules'][-1]['conditions'][0]['value'] = 5
        self.assertEqual(RuleEngine(changed).evaluate(dict(absPitch=10, absRoll=0), 'posture')['level'], 'poor')


class FeatureTests(unittest.TestCase):
    def test_hrv_breaks_pairs_at_artifacts(self):
        hrv = calculate_hrv([800, 900, 100, 1000, 1100])
        self.assertAlmostEqual(hrv['rmssd'], 100)
        self.assertAlmostEqual(hrv['sdnn'], math.sqrt(50000 / 3))
        self.assertIsNone(calculate_hrv([800])['rmssd'])

    def test_scr_counts_eda_rises_of_at_least_the_minimum_amplitude(self):
        wave = [(t * 1000, 2.0 + (0.05 if t % 2 else 0)) for t in range(7)]
        self.assertAlmostEqual(scr_per_minute(wave), 3 * 60 / 6)
        self.assertEqual(scr_per_minute([(t * 1000, 2.0 + 0.005 * (t % 2)) for t in range(7)]), 0)
        self.assertIsNone(scr_per_minute([(0, 2.0)]))

    def test_snapshot_scr_change_requires_fresh_eda_and_nonzero_baseline(self):
        s = snapshot('low', eda=40)
        features = FeaturePipeline().from_snapshot(s)
        self.assertAlmostEqual(features['scrChangePct'], 40)
        for patch in (dict(edaFresh=False), dict(baseline=dict(s['baseline'], scr=0))):
            self.assertNotIn('scrChangePct', FeaturePipeline().from_snapshot(dict(s, **patch)))

    def test_native_snapshot_ineligible_and_stale_do_not_supply_features(self):
        pipe = FeedbackPipeline()
        ready = pipe.process(sample(1000))
        self.assertEqual(ready['states']['fatigue']['level'], 'medium')
        self.assertAlmostEqual(ready['features']['hrChangePct'], 11)
        p = config('rules.json')['featurePolicy']
        for field, value in [('fresh', False), ('baselineState', 'collecting'), ('rrCount', p['minRr']-1),
                             ('windowSeconds', p['windowSeconds']-p['freshSeconds']-1)]:
            s = sample(2000 + len(field))
            s['snapshot'][field] = value
            result = FeedbackPipeline().process(s)
            self.assertNotIn('hrChangePct', result['features'])
            self.assertFalse(result['states']['fatigue']['available'])
        s = sample(9000)
        s['snapshot']['posture'] = dict(level='unavailable', pitch=50, roll=0)
        self.assertFalse(pipe.process(s)['states']['posture']['available'])

    def test_extreme_finite_hr_does_not_create_nonfinite_output(self):
        pipe = FeedbackPipeline()
        for t in (0, 1000):
            result = pipe.process(dict(session_id='extreme', participant='p', mode='live', sample_id=str(t),
                timestamp_ms=t, raw=dict(hr=1e308, rr_intervals_ms=[800, 900])))
        json.dumps(result, allow_nan=False)

    def test_snapshot_policy_mismatch_rejects_fatigue_but_retains_posture(self):
        rules = config('rules.json')
        policy = rules['featurePolicy']
        current = snapshot()
        current['featurePolicy'] = dict(policy, rrMaxMs=1900)
        features = FeaturePipeline(rules).from_snapshot(current)
        self.assertFalse(features['fatigueEligible'])
        self.assertNotIn('hrChangePct', features)
        self.assertEqual(features['absPitch'], 0)
        current['featurePolicy'] = dict(policy)
        self.assertTrue(FeaturePipeline(rules).from_snapshot(current)['fatigueEligible'])
        current.pop('featurePolicy')
        # Undeclared snapshots retain the legacy 60s policy and cannot silently
        # enter the temporary 5s protocol.
        self.assertFalse(FeaturePipeline(rules).from_snapshot(current)['fatigueEligible'])
        changed = copy.deepcopy(rules)
        changed['featurePolicy']['rrMaxMs'] = 1900
        self.assertFalse(FeaturePipeline(changed).from_snapshot(current)['fatigueEligible'])

    def test_dynamic_raw_snapshot_retains_its_feature_protocol(self):
        rules = config('rules.json')
        rules['featurePolicy']['windowSeconds'] = 30
        pipe = FeedbackPipeline(rules=rules)
        for second in range(31):
            result = pipe.process(dict(session_id='dynamic', participant='p', mode='live', sample_id=str(second),
                timestamp_ms=second * 1000, raw=dict(hr=80, rr_intervals_ms=[750, 800])))
        self.assertTrue(result['features']['fatigueEligible'])
        self.assertEqual(result['features']['snapshot']['featurePolicy'], rules['featurePolicy'])

    def test_raw_baseline_and_contact_loss(self):
        pipe = FeedbackPipeline()
        result = None
        for second in range(61):
            result = pipe.process(dict(session_id='raw', participant='p', mode='live', sample_id=str(second),
                timestamp_ms=second * 1000, raw=dict(hr=80, rr_intervals_ms=[750, 800])))
        self.assertTrue(result['states']['fatigue']['available'])
        self.assertEqual(result['states']['fatigue']['level'], 'low')
        self.assertAlmostEqual(result['features']['baseline']['hr'], 80)
        lost = pipe.process(dict(session_id='raw', participant='p', mode='live', sample_id='lost',
                                 timestamp_ms=61000, raw=dict(hr=90, rr_intervals_ms=[700, 750], contact=False)))
        self.assertFalse(lost['states']['fatigue']['available'])

    def test_raw_eda_stream_yields_scr_and_frustration(self):
        pipe = FeedbackPipeline()
        for t in range(0, 12000, 250):
            # Rest: one 0.05 µS response every 2 s. Then faster, larger responses on a raised level.
            calm = t < 6000
            eda = (2.0 if calm else 2.8) + (0.05 if calm and t % 2000 == 1000 else 0.1 if not calm and t % 1000 == 500 else 0)
            rr = [800, 840] if calm else [690, 700]
            result = pipe.process(dict(session_id='eda', participant='p', mode='live', sample_id=str(t), timestamp_ms=t,
                                       raw=dict(hr=75 if calm else 90, rr_intervals_ms=rr, eda_us=eda)))
        self.assertGreater(result['features']['baseline']['scr'], 0)
        self.assertIn('scrChangePct', result['features'])
        self.assertTrue(result['states']['frustration']['available'])
        self.assertEqual(result['states']['frustration']['level'], 'high')


class DecisionTests(unittest.TestCase):
    """Timing rules: Medium never alerts; High alerts after three consecutive Highs."""
    def setUp(self):
        self.engine = RuleEngine()
        self.feedback = config('feedback_table.json')
        self.decision = DecisionEngine(self.feedback)

    def result(self, level='high'):
        if level == 'unavailable':
            return self.engine.evaluate({})
        return self.engine.evaluate(changes(
            {'low': (0, 0, 0, 0, 0), 'medium': (0, 0, 11, -26, -21), 'high': (0, 0, 5, -36, -31)}[level]))

    def run_levels(self, levels, start=1000):
        return [self.decision.update(self.result(level), start + i * 1000) for i, level in enumerate(levels)]

    def test_low_and_medium_never_issue_feedback(self):
        for out in self.run_levels(['medium'] * 20 + ['low'] * 20):
            self.assertIsNone(out['alert'])
            self.assertEqual(out['suppression'], 'no-feedback-for-level')
            self.assertEqual(out['streak']['count'], 0)

    def test_third_consecutive_high_prediction_issues_feedback(self):
        first, second, third = self.run_levels(['high'] * 3)
        for out, count in ((first, 1), (second, 2)):
            self.assertIsNone(out['alert'])
            self.assertEqual(out['suppression'], 'awaiting-consecutive-high')
            self.assertEqual(out['streak'], dict(count=count, required=3, level='high'))
        expected = self.feedback['states']['fatigue']['high']['causes']['fatigue-related-stress']
        self.assertEqual(third['alert']['message'], expected['message'])
        self.assertEqual(third['alert']['action'], expected['action'])
        self.assertEqual(third['alert']['consecutive'], 3)

    def test_low_or_medium_prediction_resets_the_count(self):
        for reset in ('low', 'medium'):
            with self.subTest(reset=reset):
                self.decision = DecisionEngine(self.feedback)
                outs = self.run_levels(['high', 'high', reset, 'high', 'high', 'high'])
                self.assertEqual([o['streak']['count'] for o in outs], [1, 2, 0, 1, 2, 3])
                self.assertEqual([bool(o['alert']) for o in outs], [False] * 5 + [True])

    def test_unavailable_prediction_resets_the_count(self):
        outs = self.run_levels(['high', 'high', 'unavailable', 'high', 'high', 'high'])
        self.assertEqual(outs[2]['suppression'], 'insufficient-evidence')
        self.assertEqual([o['streak']['count'] for o in outs], [1, 2, 0, 1, 2, 3])
        self.assertTrue(outs[-1]['alert'])

    def test_gap_between_predictions_resets_the_count(self):
        fresh_ms = self.feedback['decision']['freshSeconds'] * 1000
        self.run_levels(['high', 'high'])
        late = self.decision.update(self.result(), 2000 + fresh_ms + 1)
        self.assertEqual(late['streak']['count'], 1)
        self.assertIsNone(late['alert'])

    def test_duplicate_or_reversed_predictions_do_not_count(self):
        self.decision.update(self.result(), 1000)
        self.assertEqual(self.decision.update(self.result(), 1000)['suppression'], 'duplicate-or-reversed-timestamp')
        self.assertEqual(self.decision.update(self.result(), 500)['suppression'], 'duplicate-or-reversed-timestamp')
        self.assertEqual(self.decision.update(self.result(), 2000)['streak']['count'], 2)
        self.assertTrue(self.decision.update(self.result(), 3000)['alert'])

    def test_count_restarts_after_feedback_and_cooldown_limits_repeats(self):
        cooldown_ms = self.feedback['decision']['cooldownSeconds'] * 1000
        outs = self.run_levels(['high'] * 6)
        self.assertEqual([o['streak']['count'] for o in outs], [1, 2, 3, 1, 2, 3])
        self.assertTrue(outs[2]['alert'])
        self.assertEqual(outs[5]['suppression'], 'cooldown')
        later = self.run_levels(['high'] * 3, start=3000 + cooldown_ms)
        self.assertTrue(later[-1]['alert'])

    def test_warning_cap_limits_alerts_across_states(self):
        feedback = copy.deepcopy(self.feedback)
        feedback['decision'].update(cooldownSeconds=0, maxWarnings=1)
        self.decision = DecisionEngine(feedback)
        outs = self.run_levels(['high'] * 6)
        self.assertTrue(outs[2]['alert'])
        self.assertEqual(outs[5]['suppression'], 'warning-cap')

    def test_consecutive_count_comes_from_config(self):
        feedback = copy.deepcopy(self.feedback)
        feedback['decision']['consecutiveHighPredictions'] = 5
        self.decision = DecisionEngine(feedback)
        outs = self.run_levels(['high'] * 5)
        self.assertEqual([bool(o['alert']) for o in outs], [False] * 4 + [True])


class PipelineTests(unittest.TestCase):
    def test_session_isolation_reset_and_retries(self):
        pipe = FeedbackPipeline()
        for t in (0, 1000):
            pipe.process(sample(t, 'high'))
        self.assertFalse(pipe.process(sample(2000, 'high', session='two'))['alerts'])
        result = pipe.process(sample(2000, 'high'))
        self.assertEqual([a['state'] for a in result['alerts']], ['fatigue'])
        self.assertFalse(pipe.process(sample(2000, 'high'))['alerts'])
        reversed_result = pipe.process(sample(1500, 'high'))
        self.assertFalse(reversed_result['alerts'])
        self.assertFalse(reversed_result['states']['fatigue']['available'])
        self.assertNotIn('hrChangePct', reversed_result['features'])
        pipe.reset('one')
        self.assertFalse(pipe.process(sample(3000, 'high'))['alerts'])

    def test_replay_restores_actual_alert_history_after_config_changes(self):
        feedback = config('feedback_table.json')
        feedback['decision']['consecutiveHighPredictions'] = 1
        pipe = FeedbackPipeline(feedback=feedback)
        for t in range(0, 5000, 1000):
            pipe.process(sample(t, 'high'))
        # Replay with the new config invented an alert at 0s; the actual old config alerted at 4s.
        pipe.restore_alert_history('one', [dict(state='fatigue', time=4000)])
        self.assertEqual(pipe.process(sample(5000, 'high'))['states']['fatigue']['suppression'], 'cooldown')
        # No persisted alerts means replay's hypothetical alerts must not consume limits.
        pipe.restore_alert_history('one', [])
        self.assertEqual(len(pipe.process(sample(6000, 'high'))['alerts']), 1)

    def test_validation_rejects_nonfinite_boolean_and_bad_ids(self):
        for changes_ in [dict(timestamp_ms=True), dict(timestamp_ms=float('nan')), dict(timestamp_ms=10**1000), dict(session_id=''),
                         dict(sample_id='x' * 257), dict(mode='other')]:
            s = sample(1000)
            s.update(changes_)
            with self.assertRaises(ValueError):
                FeedbackPipeline().process(s)
        s = sample(1000)
        s['snapshot']['hr'] = '100'
        with self.assertRaises(ValueError):
            FeedbackPipeline().process(s)

    def test_runtime_config_rejects_invalid_policies_and_feedback(self):
        for field, value in [('windowSeconds', 0), ('freshSeconds', 60), ('baselineSamples', True)]:
            rules = config('rules.json')
            rules['featurePolicy'][field] = value
            with self.assertRaises(ValueError):
                FeedbackPipeline(rules=rules)
        for field, value in [('cooldownSeconds', -1), ('maxWarnings', True),
                             ('consecutiveHighPredictions', 0), ('consecutiveHighPredictions', 2.5)]:
            feedback = config('feedback_table.json')
            feedback['decision'][field] = value
            with self.assertRaises(ValueError):
                FeedbackPipeline(feedback=feedback)
        for state, level in (('fatigue', 'low'), ('fatigue', 'medium'), ('frustration', 'medium')):
            feedback = config('feedback_table.json')
            feedback['states'][state][level].update(enabled=True, causes=dict(any=dict(message='m', action='a', goal='g')))
            with self.subTest(state=state, level=level), self.assertRaises(ValueError):
                FeedbackPipeline(feedback=feedback)
        feedback = config('feedback_table.json')
        feedback['states']['fatigue']['high']['causes'] = {}
        with self.assertRaises(ValueError):
            FeedbackPipeline(feedback=feedback)

    def test_shipped_feedback_covers_every_high_cause(self):
        rules, feedback = config('rules.json'), config('feedback_table.json')
        for name in ('frustration', 'fatigue'):
            for rule in rules['states'][name]['rules']:
                row = feedback['states'][name][rule['level']]
                self.assertEqual(row['enabled'], rule['level'] == 'high')
                if row['enabled']:
                    self.assertIn(rule['cause'], row['causes'])

    def test_feature_policy_validator_returns_independent_normalized_policy(self):
        from pipeline.config import validate_feature_policy
        original = config('rules.json')['featurePolicy']
        original_count = original['minRr']
        normalized = validate_feature_policy(original)
        self.assertEqual(normalized, original)
        normalized['minRr'] = 2
        self.assertEqual(original['minRr'], original_count)
        for bad in (None, {}, dict(original, freshSeconds=60), dict(original, baselineSamples=True)):
            with self.assertRaises(ValueError):
                validate_feature_policy(bad)

    def test_model_import_fallback_live_gate_and_score_blending(self):
        rules = json.loads((ROOT / 'tests/fixtures/legacy_polar_rules.json').read_text())
        model = dict(schemaVersion=1, modelType='multinomial-logistic-regression',
            features=['hrChangePct', 'rmssdChangePct', 'sdnnChangePct'], classes=['none', 'very-high'],
            mean=[0, 0, 0], scale=[1, 1, 1], coefficients=[[0, 0, 0], [0, 0, 0]],
            intercepts=[-100, 100], featureMin=[-100, -100, -100], featureMax=[100, 100, 100],
            training=dict(source='synthetic-bootstrap', participants=0),
            deployment=dict(approved=True, ruleConfigVersion=rules['version'], blendWeight=.5,
                metrics=dict(rules=dict(macroF1=.5, accuracy=.5), ml=dict(macroF1=.6, accuracy=.6),
                             blend=dict(macroF1=.7, accuracy=.7))))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            path.write_text(json.dumps(model))
            blocked = FeedbackPipeline(rules=rules, model_path=path).process(sample(1000, 'low', policy=rules['featurePolicy']))
            self.assertEqual(blocked['states']['fatigue']['level'], 'none')
            self.assertIn('rule-only', blocked['states']['fatigue']['model']['status'])
            model['training'] = dict(source='participant-self-report', participants=3)
            path.write_text(json.dumps(model))
            accepted = FeedbackPipeline(rules=rules, model_path=path).process(sample(1000, 'low', policy=rules['featurePolicy']))
            self.assertEqual(accepted['states']['fatigue']['level'], 'medium')
            self.assertAlmostEqual(accepted['states']['fatigue']['score'], .45)
            model['featurePolicy'] = dict(rules['featurePolicy'], rrMaxMs=1900)
            path.write_text(json.dumps(model))
            mismatch = FeedbackPipeline(rules=rules, model_path=path).process(sample(1000, 'low', policy=rules['featurePolicy']))
            self.assertEqual(mismatch['states']['fatigue']['level'], 'none')
            self.assertIn('policy', mismatch['states']['fatigue']['model']['reason'].lower())
            changed_rules = copy.deepcopy(rules)
            changed_rules['featurePolicy']['rrMaxMs'] = 1900
            s = sample(1000, 'low', policy=rules['featurePolicy'])
            s['snapshot']['featurePolicy'] = dict(changed_rules['featurePolicy'])
            approved = FeedbackPipeline(rules=changed_rules, model_path=path).process(s)
            self.assertEqual(approved['states']['fatigue']['model']['status'], 'approved-blend')
            model.pop('featurePolicy')
            path.write_text(json.dumps(model))
            legacy = FeedbackPipeline(rules=changed_rules, model_path=path).process(s)
            self.assertEqual(legacy['states']['fatigue']['model']['status'], 'rule-only')
            model['deployment']['metrics']['blend']['macroF1'] = .6
            path.write_text(json.dumps(model))
            self.assertEqual(FeedbackPipeline(rules=rules, model_path=path).process(sample(1000, 'low', policy=rules['featurePolicy']))['states']['fatigue']['level'], 'none')
            path.write_text('{}')
            fallback = FeedbackPipeline(rules=rules, model_path=path).process(sample(1000, policy=rules['featurePolicy']))
            self.assertEqual(fallback['states']['fatigue']['level'], 'medium')
            self.assertTrue(fallback['states']['fatigue']['model']['reason'])


class EndToEndTests(unittest.TestCase):
    def test_scripted_demo_shows_medium_silence_reset_and_consecutive_high_feedback(self):
        from tools.demo_stream import run_demo
        results = list(run_demo())
        frustration = [(r['states']['frustration']['level'], r['states']['frustration']['streak']['count']) for r in results]
        fatigue = [(r['states']['fatigue']['level'], r['states']['fatigue']['streak']['count']) for r in results]
        alerts = [(i, a['state'], a['cause']) for i, r in enumerate(results) for a in r['alerts']]
        self.assertEqual(frustration[4:14], [('medium', 0)] * 4 + [('high', 1), ('high', 2), ('medium', 0),
                                                                   ('high', 1), ('high', 2), ('high', 3)])
        self.assertEqual(fatigue[17:24], [('medium', 0)] * 4 + [('high', 1), ('high', 2), ('high', 3)])
        self.assertEqual(alerts, [(13, 'frustration', 'rework-repeated-task-failure'),
                                  (23, 'fatigue', 'fatigue-related-stress')])
        self.assertTrue(all(r['states'][s]['level'] == 'low' for r in results[-3:] for s in ('frustration', 'fatigue')))


if __name__ == '__main__':
    unittest.main()
