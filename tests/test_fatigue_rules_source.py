import unittest
from pipeline import FeedbackPipeline, RuleEngine, FeaturePipeline
from pipeline.config import load_config


def snapshot():
    policy = load_config(None, 'rules.json')['featurePolicy']
    return dict(hr=100, rmssd=50, sdnn=50, eda=2.2, edaFresh=True,
                baseline=dict(hr=100, rmssd=100, sdnn=100, eda=2), baselineState='ready',
                fresh=True, rrCount=policy['minRr'], windowSeconds=policy['windowSeconds'],
                requiredWindowSeconds=policy['windowSeconds'], featurePolicy=policy)


class NewFatigueRulesTests(unittest.TestCase):
    def test_temporary_default_captures_baseline_at_five_seconds(self):
        pipe = FeaturePipeline()
        self.assertEqual(pipe.policy['windowSeconds'], 5)
        for second in range(6):
            result = pipe.process(dict(timestamp_ms=second*1000,
                raw=dict(hr=75, rr_intervals_ms=[840 if second%2 else 760])))
            self.assertEqual(result['fatigueEligible'], second == 5)

    def test_latest_source_thresholds_and_strict_boundaries(self):
        engine = RuleEngine()
        # Below the Low pattern is still Low risk, but without a matched cause.
        for rmssd, sdnn, level, cause in [(-15, -16, 'low', None), (-16, -15, 'low', None),
                                          (-16, -16, 'low', 'fatigue-related-stress'),
                                          (-25, -21, 'low', 'fatigue-related-stress'), (-26, -20, 'low', 'fatigue-related-stress'),
                                          (-26, -21, 'medium', 'fatigue-related-stress')]:
            result = engine.evaluate(dict(rmssdChangePct=rmssd, sdnnChangePct=sdnn))
            self.assertEqual((result['level'], result['cause']), (level, cause))
        high = dict(hrChangePct=9, edaChangePct=14, rmssdChangePct=-36, sdnnChangePct=-31)
        self.assertEqual(engine.evaluate(high)['level'], 'high')
        for key, boundary in [('hrChangePct',10), ('edaChangePct',15), ('rmssdChangePct',-35), ('sdnnChangePct',-30)]:
            self.assertEqual(engine.evaluate(dict(high, **{key:boundary}))['level'], 'medium')

    def test_missing_optional_high_inputs_are_explicit_not_zero_filled(self):
        result = RuleEngine().evaluate(dict(hrChangePct=0, rmssdChangePct=-50, sdnnChangePct=-50))
        self.assertEqual(result['level'], 'medium')
        self.assertEqual(result['unassessedLevels'], [dict(level='high',missingSignals=['edaChangePct'])])
        self.assertFalse(RuleEngine().evaluate(dict(rmssdChangePct=-50))['available'])

    def test_features_include_only_fresh_eda_with_nonzero_baseline(self):
        s = snapshot()
        features = FeaturePipeline().from_snapshot(s)
        self.assertAlmostEqual(features['edaChangePct'],10)
        for changes in [dict(edaFresh=False), dict(eda=None), dict(baseline=dict(s['baseline'],eda=0))]:
            self.assertNotIn('edaChangePct',FeaturePipeline().from_snapshot(dict(s,**changes)))
        self.assertEqual(RuleEngine().evaluate(FeaturePipeline().from_snapshot(dict(s,hr=None)))['level'],'medium')

    def test_fatigue_pipeline_is_rules_only_even_with_a_model_path(self):
        pipe = FeedbackPipeline(model_path='models/polar_fatigue_model.json')
        for mode in ('live','demo'):
            result = pipe.process(dict(session_id=mode,participant='test',mode=mode,sample_id='1',timestamp_ms=1000,snapshot=snapshot()))
            fatigue = result['states']['fatigue']
            self.assertEqual(fatigue['level'],'high')
            self.assertEqual(fatigue['model']['status'],'disabled')
            self.assertNotIn('mlScore',fatigue)

    def test_raw_eda_capture_is_optional_and_requires_fresh_coverage(self):
        pipe = FeedbackPipeline()
        for t in range(0,131000,1000):
            result = pipe.process(dict(session_id='raw', participant='p', mode='live',sample_id=str(t),timestamp_ms=t,
                raw=dict(hr=100,eda_us=2 if t<=60000 else 2.2,rr_intervals_ms=[700,900] if t<=60000 else [780,820])))
        self.assertEqual(result['states']['fatigue']['level'],'high')
        self.assertAlmostEqual(result['features']['edaChangePct'],10)
        for t in range(131000,137000,1000):
            result = pipe.process(dict(session_id='raw', participant='p', mode='live',sample_id=str(t),timestamp_ms=t,
                raw=dict(hr=100,rr_intervals_ms=[780,820])))
        self.assertEqual(result['states']['fatigue']['level'],'medium')
        self.assertNotIn('edaChangePct',result['features'])


if __name__ == '__main__':
    unittest.main()
