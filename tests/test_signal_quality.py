"""Empatica signal quality, wear gating, prediction cadence, repeat limits and cause matching.

The patterns are the ones found in real EmbracePlus data: missed beats reading about double,
an unworn device at room temperature, EDA near zero, and no skin responses at rest."""
import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from pipeline import FeedbackPipeline
from pipeline.config import load_config
from pipeline.decision import DecisionEngine
from pipeline.features import FeaturePipeline, calculate_hrv, clean_beats
from pipeline.frustration_model import (CLASSES, FEATURES, MIN_SD_FRACTION, NORMALIZATIONS, Z_SPAN, FrustrationModel,
                                        SignalTracker, signal_scale, window_features)
from pipeline.rule_engine import RuleEngine

ROOT = Path(__file__).resolve().parents[1]
RULES = load_config(None, 'rules.json')
POLICY = RULES['sourcePolicies']['empatica-avro']
SOURCE = 'empatica-avro'


def stub_rules(directory, prefer='high', normalization=None):
    path = Path(directory) / 'stub.json'
    path.write_text(json.dumps(dict(
        schemaVersion=1, modelType='logistic', features=FEATURES, classes=CLASSES, family='logistic',
        mean=[0.0] * len(FEATURES), scale=[1.0] * len(FEATURES), coefficients=[[0.0] * len(FEATURES) for _ in CLASSES],
        intercepts=[5.0 if c == prefer else 0.0 for c in CLASSES], **({'normalization': normalization} if normalization else {}))))
    rules = copy.deepcopy(RULES)
    rules['states']['frustration']['modelFile'] = str(path)
    return rules


def raw_sample(second, session='s', eda=2.0, temp=33.0, rr=(800.0,), contact=None, mode='batch'):
    raw = dict(source=SOURCE, eda_us=eda, temp_c=temp, hr=75.0, rr_intervals_ms=list(rr))
    if contact is not None:
        raw['contact'] = contact
    return dict(session_id=session, participant='p', mode=mode, sample_id=str(second), timestamp_ms=second * 1000, raw=raw)


class BeatCleaningTests(unittest.TestCase):
    def test_missed_beats_are_rejected_and_normal_variation_is_kept(self):
        beats = [800, 810, 790, 805, 1600, 795, 800, 815, 785, 800, 400, 810, 795]
        kept = clean_beats(beats)
        self.assertEqual([b for b, k in zip(beats, kept) if not k], [1600, 400])

    def test_too_few_neighbours_is_not_usable(self):
        self.assertEqual(clean_beats([800, 810]), [False, False])

    def test_real_shape_missed_beats_stop_inflating_rmssd(self):
        # One missed beat in ten passes the 300-2000 ms gate and dominates a gate-only RMSSD.
        rng = random.Random(1)
        beats = [1600.0 if i % 10 == 5 else 800.0 + rng.uniform(-20, 20) for i in range(80)]
        self.assertGreater(calculate_hrv(beats)['rmssd'], 200)
        pipe = FeaturePipeline(source_policy=POLICY)
        for second, beat in enumerate(beats):
            snapshot = pipe._source_snapshot(dict(source=SOURCE, eda_us=2.0, temp_c=33.0, rr_intervals_ms=[beat]), second * 1000)
        self.assertLess(snapshot['rmssd'], 40)
        self.assertAlmostEqual(snapshot['hr'], 75, delta=2)  # missed beats no longer drag HR down
        self.assertAlmostEqual(snapshot['cleanBeatFraction'], .9, delta=.03)

    def test_rejected_beats_break_successive_differences(self):
        # Removing the 1600 must not create a difference between its two neighbours.
        pipe = FeaturePipeline(source_policy=POLICY)
        pattern = [800.0, 800.0, 800.0, 800.0, 1600.0, 600.0, 600.0, 600.0, 600.0, 600.0]
        for second, beat in enumerate(pattern * 8):
            snapshot = pipe._source_snapshot(dict(source=SOURCE, rr_intervals_ms=[beat]), second * 1000)
        self.assertTrue(snapshot['rmssd'] is None or snapshot['rmssd'] < 200)

    def test_too_few_clean_beats_reports_no_hrv_rather_than_a_noisy_value(self):
        rng = random.Random(2)
        pipe = FeaturePipeline(source_policy=POLICY)
        for second in range(70):
            snapshot = pipe._source_snapshot(dict(source=SOURCE, eda_us=2.0, rr_intervals_ms=[rng.uniform(320, 1900)]), second * 1000)
        self.assertLess(snapshot['cleanBeatFraction'], POLICY['minCleanBeatFraction'])
        self.assertFalse(snapshot['hrvUsable'])
        self.assertIsNone(snapshot['rmssd'])
        self.assertIsNone(snapshot['sdnn'])


class BaselineTests(unittest.TestCase):
    def feed(self, seconds, start=0, pipe=None, eda=lambda t: 1.0, rr=lambda t: 800.0 + 10 * math.sin(t)):
        pipe = pipe or FeaturePipeline(source_policy=POLICY)
        out = None
        for t in range(start, start + seconds):
            out = pipe.process(raw_sample(t, eda=eda(t), rr=(rr(t),)))
        return pipe, out

    def test_waits_for_recent_history_then_reports_near_zero_change_at_rest(self):
        pipe, early = self.feed(150)
        self.assertNotEqual(early['snapshot']['baselineState'], 'ready')
        _, later = self.feed(150, start=150, pipe=pipe)
        self.assertEqual(later['snapshot']['baselineState'], 'ready')
        self.assertAlmostEqual(later['hrChangePct'], 0, delta=2)
        self.assertAlmostEqual(later['edaChangePct'], 0, delta=.01)

    def test_a_noisy_first_window_cannot_set_the_baseline(self):
        # The saved session whose first 5 seconds gave a 144 bpm baseline, read as -41% ever after.
        pipe, _ = self.feed(30, rr=lambda t: 420.0 + 5 * math.sin(t))
        _, out = self.feed(600, start=30, pipe=pipe)
        self.assertAlmostEqual(out['snapshot']['baseline']['hr'], 75, delta=2)
        self.assertAlmostEqual(out['hrChangePct'], 0, delta=2)

    def test_slow_eda_drift_stops_reading_as_a_huge_change(self):
        # EDA rising through a long wear: from a first-minute baseline this reads over +1000%.
        _, out = self.feed(2400, eda=lambda t: .1 + t * .001)
        self.assertLess(abs(out['edaChangePct']), 40)

    def test_a_sudden_rise_shows_and_does_not_chase_its_own_reference(self):
        step = 600
        _, out = self.feed(step + 90, eda=lambda t: 1.0 if t < step else 2.0)
        self.assertGreater(out['edaChangePct'], 80)

    def test_zero_resting_scr_still_gives_a_usable_change(self):
        # At rest many windows contain no response; that used to leave SCR, and so every cause, unmeasured.
        _, out = self.feed(400)
        self.assertEqual(out['snapshot']['baseline']['scr'], POLICY['scrFloorPerMinute'])
        self.assertEqual(out['scrChangePct'], -100)
        bumps = lambda t: 1.0 + (.2 if t > 400 and t % 15 < 3 else 0)
        _, out = self.feed(460, eda=bumps)
        self.assertGreater(out['scrChangePct'], 40)

    def test_eda_below_the_floor_is_reported_as_unmeasured_not_as_a_percentage(self):
        _, out = self.feed(400, eda=lambda t: .02 + .002 * math.sin(t))
        self.assertTrue(out['snapshot']['edaBelowFloor'])
        self.assertIsNone(out['snapshot']['eda'])
        self.assertNotIn('edaChangePct', out)
        self.assertNotIn('scrChangePct', out)
        self.assertIn('hrChangePct', out)

    def test_other_sources_keep_their_original_measurement_protocol(self):
        pipe = FeaturePipeline()
        self.assertEqual(pipe.policy, RULES['featurePolicy'])
        self.assertIsNone(pipe.source_policy)


class WearAndCadenceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def stream(self, pipe, seconds, start=0, **kwargs):
        return [pipe.process(raw_sample(t, eda=2 + .1 * math.sin(t / 5), **kwargs)) for t in range(start, start + seconds)]

    def test_unworn_device_predicts_nothing_and_cues_nothing_even_if_the_model_says_high(self):
        pipe = FeedbackPipeline(rules=stub_rules(self.dir.name))
        outs = self.stream(pipe, 900, contact=False, temp=22.0)
        for out in outs:
            frustration = out['states']['frustration']
            self.assertFalse(frustration['available'])
            self.assertIn('not worn', frustration['reason'])
            self.assertIsNone(frustration.get('wouldAlert'))
        self.assertEqual(outs[-1]['features']['snapshot']['worn'], False)

    def test_taking_the_device_off_stops_high_at_once_and_breaks_the_run(self):
        pipe = FeedbackPipeline(rules=stub_rules(self.dir.name))
        self.stream(pipe, 100)
        off = pipe.process(raw_sample(100, contact=False, temp=22.0))
        self.assertFalse(off['states']['frustration']['available'])
        self.assertEqual(off['states']['frustration']['streak']['count'], 0)

    def test_unworn_readings_do_not_stretch_the_models_scaling(self):
        pipe = FeedbackPipeline(rules=stub_rules(self.dir.name))
        self.stream(pipe, 100)
        before = copy.deepcopy(pipe.sessions['s']['signals'].range)
        self.stream(pipe, 10, start=100, contact=False, temp=-5.0)
        self.assertEqual(pipe.sessions['s']['signals'].range, before)

    def test_the_model_waits_with_a_reason_when_eda_is_below_the_floor(self):
        pipe = FeedbackPipeline(rules=stub_rules(self.dir.name))
        for t in range(100):
            out = pipe.process(raw_sample(t, eda=.02))
        model = out['states']['frustration']['model']
        self.assertEqual(model['status'], 'waiting')
        self.assertIn('validity floor', model['reason'])

    def test_one_prediction_every_30_seconds_so_three_highs_span_a_minute(self):
        pipe = FeedbackPipeline(rules=stub_rules(self.dir.name))
        outs = self.stream(pipe, 200)
        ticks = [t for t, out in enumerate(outs) if out['states']['frustration']['prediction']]
        self.assertEqual([b - a for a, b in zip(ticks, ticks[1:])], [POLICY['predictionSeconds']] * (len(ticks) - 1))
        cue = next(t for t, out in enumerate(outs) if out['states']['frustration'].get('wouldAlert'))
        self.assertEqual(cue, ticks[2])
        self.assertEqual(cue - ticks[0], 2 * POLICY['predictionSeconds'])

    def test_held_seconds_never_repeat_a_cue(self):
        pipe = FeedbackPipeline(rules=stub_rules(self.dir.name))
        outs = self.stream(pipe, 200)
        cues = [t for t, out in enumerate(outs) if out['states']['frustration'].get('wouldAlert')]
        self.assertEqual(len(cues), 1)
        self.assertFalse(any(out['alerts'] for out in outs))

    def test_a_sustained_high_gives_one_identical_cue_not_a_flood(self):
        # The review saw 22+ identical cues in one session.
        pipe = FeedbackPipeline(rules=stub_rules(self.dir.name))
        outs = self.stream(pipe, 1400)
        cues = sum(bool(out['states']['frustration'].get('wouldAlert')) for out in outs)
        self.assertLessEqual(cues, 2)

    def test_data_gap_between_predictions_breaks_the_run(self):
        pipe = FeedbackPipeline(rules=stub_rules(self.dir.name))
        self.stream(pipe, 100)
        outs = self.stream(pipe, 80, start=160)
        counted = [out['states']['frustration'] for out in outs if out['states']['frustration']['prediction']]
        self.assertEqual(counted[0]['streak']['count'], 0)  # still refilling the window after the hole


class RepeatLimitTests(unittest.TestCase):
    def setUp(self):
        self.feedback = load_config(None, 'feedback_table.json')
        self.engine = RuleEngine()
        self.high = self.engine.evaluate(dict(rmssdChangePct=-36, sdnnChangePct=-31, hrChangePct=5, edaChangePct=0), 'fatigue')

    def run_high(self, decision, start, count=3):
        return [decision.update(self.high, start + i * 1000) for i in range(count)]

    def test_identical_cue_waits_for_the_repeat_window_even_past_the_cooldown(self):
        decision = DecisionEngine(self.feedback)
        self.assertTrue(self.run_high(decision, 1000)[-1]['alert'])
        cooldown = self.feedback['decision']['cooldownSeconds'] * 1000
        soon = self.run_high(decision, 4000 + cooldown)[-1]
        self.assertIsNone(soon['alert'])
        self.assertEqual(soon['suppression'], 'same-cue-repeat')
        repeat = self.feedback['decision']['sameCueRepeatSeconds'] * 1000
        self.assertTrue(self.run_high(decision, 1000 + repeat + 5000)[-1]['alert'])

    def test_a_different_message_is_not_a_repeat(self):
        decision = DecisionEngine(self.feedback)
        self.run_high(decision, 1000)
        other = dict(self.high, state='frustration', cause='schedule-pressure', causeLabel='Schedule Pressure')
        outs = [decision.update(other, 2000 + i * 1000) for i in range(3)]
        self.assertTrue(outs[-1]['alert'])

    def test_repeat_history_survives_a_server_restart(self):
        decision = DecisionEngine(self.feedback)
        first = self.run_high(decision, 1000)[-1]['alert']
        restored = DecisionEngine(self.feedback)
        restored.restore_alert_history([first])
        cooldown = self.feedback['decision']['cooldownSeconds'] * 1000
        self.assertEqual(self.run_high(restored, 4000 + cooldown)[-1]['suppression'], 'same-cue-repeat')


class ClosestCauseTests(unittest.TestCase):
    def setUp(self):
        self.engine = RuleEngine()

    def test_four_of_five_signals_name_the_closest_pattern(self):
        closest = self.engine.closest_cause(dict(edaChangePct=35, scrChangePct=45, hrChangePct=20, rmssdChangePct=-30, sdnnChangePct=50))
        self.assertEqual(closest['agreeing'], 4)
        self.assertEqual(closest['total'], 5)
        self.assertEqual(closest['cause'], 'rework-repeated-task-failure')

    def test_three_of_five_is_not_enough_to_name_a_cause(self):
        self.assertIsNone(self.engine.closest_cause(dict(edaChangePct=35, scrChangePct=45, hrChangePct=20, rmssdChangePct=20, sdnnChangePct=50)))

    def test_causes_identical_at_low_are_separated_by_their_stronger_levels(self):
        # Heart rate far up and HRV far down with little else is the Schedule Pressure shape (HR ↑↑, HRV ↓↓).
        schedule = self.engine.closest_cause(dict(edaChangePct=11, scrChangePct=-100, hrChangePct=25, rmssdChangePct=-35, sdnnChangePct=-35))
        self.assertEqual(schedule['cause'], 'schedule-pressure')

    def test_ml_high_reports_closest_exact_or_why_not(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = FeedbackPipeline(rules=stub_rules(directory))
            tracker = SignalTracker()
            for t in range(70):
                tracker.add(t * 1000, 2 + math.sin(t / 5) / 10, 33.0)
            base = self.engine.evaluate(dict(edaChangePct=35, scrChangePct=45, hrChangePct=20, rmssdChangePct=-30, sdnnChangePct=50), 'frustration')
            named = pipe._frustration(base, tracker, 70000)
            self.assertEqual((named['causeMatch'], named['cause']), ('closest', 'rework-repeated-task-failure'))
            self.assertIn('4 of 5', named['reason'])
            flat = self.engine.evaluate(dict(edaChangePct=0, scrChangePct=0, hrChangePct=0, rmssdChangePct=0, sdnnChangePct=0), 'frustration')
            none = pipe._frustration(flat, tracker, 70000)
            self.assertEqual((none['causeMatch'], none['cause']), (None, 'unspecified'))
            self.assertIn('too few agree', none['reason'])
            missing = pipe._frustration(self.engine.evaluate({}, 'frustration'), tracker, 70000)
            self.assertIn('Cause not assessed', missing['reason'])


class NormalizationTests(unittest.TestCase):
    def session(self, seconds=400, seed=3):
        rng = np.random.default_rng(seed)
        eda = np.cumsum(rng.normal(0, .02, seconds)) + 2
        temp = 33 + np.cumsum(rng.normal(0, .005, seconds))
        return eda, temp

    def scaled_features(self, eda, temp, end, normalization):
        history = [eda[:end].tolist(), temp[:end].tolist()]
        scales = [signal_scale(values, values[:60], min(values), max(values), normalization) for values in history]
        return window_features(eda[end - 60:end].tolist(), temp[end - 60:end].tolist(), *scales)

    def test_every_normalization_is_unchanged_by_a_linear_rescale(self):
        # Training reads min-max scaled files; live reads raw microsiemens. Features must agree.
        eda, temp = self.session()
        for normalization in NORMALIZATIONS:
            a = self.scaled_features(eda, temp, 300, normalization)
            b = self.scaled_features(eda * 7 + 3, temp * 2 - 5, 300, normalization)
            for name in FEATURES:
                self.assertAlmostEqual(a[name], b[name], places=7, msg=f'{normalization} {name}')

    def test_z_scaling_has_a_bound_when_the_reference_minute_is_flat(self):
        values = [1.0] * 60 + [1.0 + i / 10 for i in range(60)]
        low, high = signal_scale(values, values[:60], min(values), max(values), 'baseline-z')
        self.assertEqual(low, 1.0)
        self.assertAlmostEqual(high - low, Z_SPAN * MIN_SD_FRACTION * (max(values) - min(values)))
        self.assertLessEqual((max(values) - low) / (high - low), Z_SPAN)

    def test_rolling_reference_excludes_the_current_window(self):
        values = [1.0 + .01 * (i % 2) for i in range(300)] + [50.0] * 60
        low, _ = signal_scale(values, values[:60], min(values), max(values), 'rolling-z')
        self.assertLess(low, 1.1)

    def test_rolling_predictions_do_not_depend_on_when_the_session_started(self):
        # Same moment, one session 25 minutes older with a donning spike at its start. Both have
        # more than the full rolling history by then, so nothing older should matter.
        rng = np.random.default_rng(5)
        eda = 1 + np.cumsum(rng.normal(0, .002, 3000)) + .01 * np.sin(np.arange(3000) / 9)
        eda[:30] = 30.0
        temp = 33 + np.cumsum(rng.normal(0, .001, 3000))
        full, later = SignalTracker(), SignalTracker()
        start = 1500
        for t in range(3000):
            full.add(t * 1000, float(eda[t]), float(temp[t]))
            if t >= start:
                later.add(t * 1000, float(eda[t]), float(temp[t]))
        a, _ = full.features(2999 * 1000, 'rolling-z')
        b, _ = later.features(2999 * 1000, 'rolling-z')
        for name in FEATURES:
            self.assertAlmostEqual(a[name], b[name], places=9, msg=name)

    def test_live_tracker_matches_training_windows_for_every_normalization(self):
        from training.train_frustration_model import build_windows
        eda, temp = self.session()
        session = dict(name='P1_Active_High', participant='P1', label='high', condition='active', eda=eda, temp=temp)
        for normalization in NORMALIZATIONS:
            frame = build_windows([session], normalization=normalization)
            tracker = SignalTracker()
            for t in range(len(eda)):
                tracker.add(t * 1000, float(eda[t]), float(temp[t]))
                # A window ending at second e is complete once time reaches e.
                end = t
                row = frame.iloc[(end - 60) // 30] if end >= 60 and (end - 60) % 30 == 0 else None
                if row is not None:
                    live, _ = tracker.features(end * 1000, normalization)
                    for name in FEATURES:
                        self.assertAlmostEqual(live[name], row[name], places=9, msg=f'{normalization} {name} at {end}')

    def test_model_file_declares_its_normalization_and_old_files_still_load(self):
        with tempfile.TemporaryDirectory() as directory:
            rules = stub_rules(directory, normalization='rolling-z')
            self.assertEqual(FrustrationModel(rules['states']['frustration']['modelFile']).normalization, 'rolling-z')
            rules = stub_rules(directory)
            self.assertEqual(FrustrationModel(rules['states']['frustration']['modelFile']).normalization, 'running-minmax')
            bad = stub_rules(directory, normalization='magic')
            with self.assertRaises(ValueError):
                FrustrationModel(bad['states']['frustration']['modelFile'])


class ConfigTests(unittest.TestCase):
    def test_invalid_source_policies_are_rejected(self):
        for change in (dict(beatTolerance=0), dict(minCleanBeatFraction=2), dict(edaFloorUs=0), dict(predictionSeconds=-1),
                       dict(baselineMinSeconds=1.5), dict(wear=dict(minPercent=500, minTemperatureC=27))):
            rules = copy.deepcopy(RULES)
            rules['sourcePolicies'][SOURCE].update(change)
            with self.assertRaises(ValueError, msg=str(change)):
                load_config(rules, 'rules.json')

    def test_invalid_repeat_setting_is_rejected(self):
        feedback = load_config(None, 'feedback_table.json')
        feedback['decision']['sameCueRepeatSeconds'] = -1
        with self.assertRaises(ValueError):
            load_config(feedback, 'feedback_table.json')


if __name__ == '__main__':
    unittest.main()
