import test from 'node:test';
import assert from 'node:assert/strict';
import { getConfiguration, configureRules, evaluateRule } from '../sensor/rule-config.js';
import { FatigueSession } from '../sensor/fatigue.js';
import { recordFatigueRating } from '../sensor/fatigue-model.js';

test('browser rule evaluation honors config, strict boundaries and missing evidence', () => {
  const original = structuredClone(getConfiguration());
  assert.equal(evaluateRule('fatigue', { hrChangePct: 10, rmssdChangePct: -25, sdnnChangePct: -21 }).level, 'low');
  assert.equal(evaluateRule('fatigue', { hrChangePct: 40 }).level, 'unavailable');
  const next = structuredClone(original);
  next.rules.states.fatigue.rules.find(r => r.level === 'medium').conditions[0].value = -24;
  configureRules(next.rules, next.feedback);
  assert.equal(evaluateRule('fatigue', { hrChangePct: 10, rmssdChangePct: -25, sdnnChangePct: -21 }).level, 'medium');
  configureRules(original.rules, original.feedback);
});

test('independent rating eligibility follows configured window duration', () => {
  const original = getConfiguration(), next = getConfiguration();
  next.rules.featurePolicy.windowSeconds = 120;
  configureRules(next.rules, next.feedback);
  try {
    const session = new FatigueSession(); session.startBaseline(0);
    for (let t = 0; t <= 120000; t += 1000) session.heartRate({ recv_epoch_time_ms: t, heart_rate_bpm: 75, rr_intervals_ms: [760, 840] });
    const args = { snapshot: session.snapshot(120000), mode: 'live', connected: true, participant: 'P01', level: 'low', now: 120000 };
    assert.equal(recordFatigueRating(args).accepted, true);
    assert.equal(recordFatigueRating({ ...args, lastRatingAt: 60000 }).accepted, false);
  } finally { configureRules(original.rules, original.feedback); }
});

test('baseline sample count and distinct evidence are enforced', () => {
  const original = getConfiguration();
  const malformed = structuredClone(original);
  malformed.rules.states.fatigue.requiredSignals = ['hrChangePct', 'hrChangePct', 'hrChangePct'];
  assert.throws(() => configureRules(malformed.rules, malformed.feedback));
  const session = new FatigueSession({ baselineSamples: 100 });
  session.startBaseline(0);
  for (let t = 0; t <= 60000; t += 1000) session.heartRate({ recv_epoch_time_ms: t, heart_rate_bpm: 75, rr_intervals_ms: [760, 840] });
  assert.notEqual(session.snapshot(60000).baselineState, 'ready');
});
