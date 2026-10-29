import test from 'node:test';
import assert from 'node:assert/strict';
import { validateModel, modelFeatures, predictFatigue, recordFatigueRating } from '../sensor/fatigue-model.js';

// Archived model fixtures retain their original 60-second measurement protocol.
import { configureRules, getConfiguration } from '../sensor/rule-config.js';
const archived = getConfiguration();
archived.rules.featurePolicy = { windowSeconds: 60, freshSeconds: 5, minRr: 30, rrMinMs: 300, rrMaxMs: 2000, baselineSamples: 30 };
configureRules(archived.rules, archived.feedback);

const snapshot = () => ({ hr: 88, rmssd: 32, sdnn: 24,
  baseline: { hr: 80, rmssd: 40, sdnn: 30 }, baselineState: 'ready', fresh: true,
  rrCount: 60, windowSeconds: 59, requiredWindowSeconds: 60,
  fatigue: { level: 'very-high', changes: { hr: 999 } } });

const model = () => ({ schemaVersion: 1, modelType: 'multinomial-logistic-regression', modelId: 'test-model',
  features: ['hrChangePct', 'rmssdChangePct', 'sdnnChangePct'], classes: ['low', 'high'],
  mean: [0, 0, 0], scale: [10, 20, 20], coefficients: [[0, 0, 0], [1, -1, -1]], intercepts: [0, -2],
  featureMin: [-20, -95, -95], featureMax: [120, 100, 100],
  training: { source: 'synthetic-bootstrap', scope: 'synthetic-starter', samples: 10, participants: 0, classCounts: { low: 5, high: 5 } },
  evaluation: { status: 'not-applicable', reason: 'Synthetic fixture.' } });

test('ML features use measured baseline changes, ignoring rule predictions', () => {
  assert.deepEqual(modelFeatures(snapshot()), [10, -20, -20]);
  const fitted = validateModel(model());
  const prediction = predictFatigue(fitted, snapshot(), { mode: 'demo' });
  // z=[1,-1,-1]; logits=[0,1], so high wins.
  assert.equal(prediction.level, 'high');
  assert.equal(prediction.source, 'synthetic-bootstrap');
  assert.equal(prediction.modelId, 'test-model');
  assert.equal(prediction.probability, undefined);
  const low = snapshot(); low.hr = 80; low.rmssd = 40; low.sdnn = 30;
  assert.equal(predictFatigue(fitted, low, { mode: 'demo' }).level, 'low');
});

test('malformed artifacts are rejected before they replace a working model', () => {
  for (const mutate of [m => m.scale[0] = 0, m => m.coefficients[0].pop(),
    m => m.mean[0] = null, m => m.intercepts[0] = Infinity,
    m => m.classes[1] = 'healthy', m => m.features.reverse(),
    m => m.training.source = 'worker-validated', m => m.training.participants = 10,
    m => m.featureMin[0] = 121, m => m.modelType = 'pickle',
    m => m.training.classCounts.low = 2, m => m.evaluation.status = 'available']) {
    const malformed = model(); mutate(malformed);
    assert.throws(() => validateModel(malformed));
  }
  assert.throws(() => validateModel(null));
});

test('tied logits use sklearn alphabetical class order, including constant-feature models', () => {
  const tied = model();
  tied.coefficients = [[0, 0, 0], [0, 0, 0]];
  tied.intercepts = [0, 0];
  assert.equal(predictFatigue(validateModel(tied), snapshot(), { mode: 'demo' }).level, 'high');
});

test('stale, incomplete, nonfinite and out-of-range windows never yield an ML level', () => {
  const fitted = validateModel(model());
  for (const change of [{ fresh: false }, { fresh: 1 }, { baseline: null }, { baselineState: 'collecting' },
    { rrCount: 29 }, { windowSeconds: 54.9 }, { requiredWindowSeconds: 30 },
    { hr: null }, { hr: NaN }, { rmssd: -1 }, { hr: 300 }]) {
    assert.equal(predictFatigue(fitted, { ...snapshot(), ...change }, { mode: 'demo' }).level, 'unavailable');
  }
  assert.equal(predictFatigue(null, snapshot()).level, 'unavailable');
});

test('independent live ratings store an immutable snapshot and require eligible measurements', () => {
  const state = snapshot();
  const args = { snapshot: state, mode: 'live', connected: true, participant: 'P01', level: 'low',
    task: ' drywall ', now: 1800000000000, lastRatingAt: null };
  const saved = recordFatigueRating(args);
  assert.equal(saved.accepted, true);
  assert.equal(saved.rating.level, 'low'); // Rule fixture said very-high.
  assert.equal(saved.rating.source, 'participant-self-report');
  assert.equal(saved.rating.task, 'drywall');
  state.baseline.hr = 900;
  assert.equal(saved.rating.snapshot.baseline.hr, 80);
  for (const change of [{ mode: 'demo' }, { connected: false }, { participant: ' ' }, { level: '' },
    { level: 'healthy' }, { now: NaN }, { snapshot: { ...snapshot(), fresh: false } },
    { lastRatingAt: args.now - 59999 }, { lastRatingAt: args.now + 1 }]) {
    assert.equal(recordFatigueRating({ ...args, ...change }).accepted, false);
  }
  assert.equal(recordFatigueRating({ ...args, lastRatingAt: args.now - 60000 }).accepted, true);
});

test('synthetic model cannot produce live estimates', () => {
  assert.equal(predictFatigue(validateModel(model()), snapshot()).level, 'unavailable');
});
