// Portable inference for artifacts produced by train_polar_fatigue.py.
// Synthetic starter estimates approximate assignment rules; they are not validated fatigue measurements.
import { featurePolicy } from './rule-config.js';
const DEFAULT_POLICY = { windowSeconds: 60, freshSeconds: 5, minRr: 30, rrMinMs: 300, rrMaxMs: 2000, baselineSamples: 30 };
const FEATURES = ['hrChangePct', 'rmssdChangePct', 'sdnnChangePct'];
export const FATIGUE_CLASSES = ['none', 'low', 'medium', 'high', 'very-high'];
const finiteVector = (value, length) => Array.isArray(value) && value.length === length && value.every(Number.isFinite);
const fail = message => { throw new Error(message); };

export function validateModel(model) {
  const bad = () => fail('Unsupported or malformed fatigue model. Load a JSON artifact from train_polar_fatigue.py.');
  if (!model || model.schemaVersion !== 1 || model.modelType !== 'multinomial-logistic-regression'
    || typeof model.modelId !== 'string' || !model.modelId || model.modelId.length > 100
    || JSON.stringify(model.features) !== JSON.stringify(FEATURES)) bad();
  const { classes, training, evaluation } = model;
  if (!Array.isArray(classes) || classes.length < 2 || classes.length > 5
    || JSON.stringify(classes) !== JSON.stringify(FATIGUE_CLASSES.filter(level => classes.includes(level)))) bad();
  for (const key of ['mean', 'scale', 'featureMin', 'featureMax']) if (!finiteVector(model[key], 3)) bad();
  if (model.scale.some(value => value <= 0) || model.featureMin.some((value, i) => value > model.featureMax[i])) bad();
  if (!Array.isArray(model.coefficients) || model.coefficients.length !== classes.length
    || !model.coefficients.every(row => finiteVector(row, 3)) || !finiteVector(model.intercepts, classes.length)) bad();
  if (!training || !['synthetic-bootstrap', 'participant-self-report'].includes(training.source)
    || !Number.isInteger(training.samples) || training.samples < 10
    || !Number.isInteger(training.participants) || !training.classCounts
    || classes.some(level => !Number.isInteger(training.classCounts[level]) || training.classCounts[level] < 3)
    || Object.keys(training.classCounts).length !== classes.length
    || classes.reduce((sum, level) => sum + training.classCounts[level], 0) !== training.samples) bad();
  if (!evaluation) bad();
  if (training.source === 'synthetic-bootstrap') {
    if (training.participants !== 0 || training.scope !== 'synthetic-starter'
      || evaluation.status !== 'not-applicable' || 'accuracy' in evaluation || 'macroF1' in evaluation) bad();
  } else {
    if (training.participants < 1 || !['personal', 'study'].includes(training.scope)
      || !['available', 'unavailable'].includes(evaluation.status)) bad();
    if (evaluation.status === 'available' && (training.participants < 2
      || evaluation.method !== 'leave-one-participant-out'
      || ![evaluation.accuracy, evaluation.macroF1].every(value => Number.isFinite(value) && value >= 0 && value <= 1))) bad();
  }
  return structuredClone(model);
}

export function modelFeatures(snapshot, policy = featurePolicy()) {
  if (!snapshot || snapshot.fresh !== true || snapshot.baselineState !== 'ready'
    || snapshot.requiredWindowSeconds !== policy.windowSeconds || !Number.isFinite(snapshot.windowSeconds)
    || snapshot.windowSeconds < policy.windowSeconds - policy.freshSeconds
    || !Number.isFinite(snapshot.rrCount) || snapshot.rrCount < policy.minRr || !snapshot.baseline)
    fail(`Capture a resting baseline and wait for a complete, fresh ${policy.windowSeconds}-second HR/RR window.`);
  return ['hr', 'rmssd', 'sdnn'].map(key => {
    const value = snapshot[key], baseline = snapshot.baseline[key];
    if (!Number.isFinite(value) || !Number.isFinite(baseline) || baseline <= 0 || value < 0 || (key === 'hr' && value === 0))
      fail('Valid HR, RMSSD and SDNN with a nonzero resting baseline are required.');
    const change = (value - baseline) / baseline * 100;
    if (!Number.isFinite(change)) fail('Measurement changes are outside the supported numeric range.');
    return change;
  });
}

export function predictFatigue(model, snapshot, { mode = 'live' } = {}) {
  const details = { modelId: model?.modelId ?? null, source: model?.training.source ?? null };
  try {
    if (!model) fail('No model loaded. The threshold result remains available.');
    if (mode !== 'demo') fail('Live feedback uses the local rule pipeline. A model blend is enabled there only after held-out-participant validation.');
    const features = modelFeatures(snapshot, model.featurePolicy || DEFAULT_POLICY);
    if (features.some((value, i) => value < model.featureMin[i] - 1e-9 || value > model.featureMax[i] + 1e-9))
      fail('Measurements are outside this model’s training range. Record your fatigue rating to help expand the training data.');
    const normalized = features.map((value, i) => (value - model.mean[i]) / model.scale[i]);
    const logits = model.coefficients.map((row, i) => row.reduce((sum, weight, j) => sum + weight * normalized[j], model.intercepts[i]));
    if (!logits.every(Number.isFinite)) fail('Model could not evaluate these measurements.');
    // sklearn orders string classes alphabetically and picks the first tied
    // maximum. The artifact uses severity order, so preserve that tie rule here.
    const index = logits.reduce((best, value, i) => value > logits[best]
      || (value === logits[best] && model.classes[i] < model.classes[best]) ? i : best, 0);
    return { ...details, level: model.classes[index], features,
      reason: model.training.source === 'synthetic-bootstrap'
        ? 'Synthetic starter: learned from assignment threshold examples, not worker measurements.'
        : `Learned from participant self-ratings (${model.training.participants} participant${model.training.participants === 1 ? '' : 's'}). Not externally validated.` };
  } catch (error) {
    return { ...details, level: 'unavailable', reason: error.message };
  }
}

export function recordFatigueRating({ snapshot, mode, connected, participant, level, task = '', now, lastRatingAt }) {
  try {
    if (mode !== 'live' || !connected) fail('Ratings require a live Polar connection; demo data is excluded.');
    if (typeof participant !== 'string' || !participant.trim()) fail('Enter a participant ID before connecting.');
    if (!FATIGUE_CLASSES.includes(level)) fail('Choose how fatigued you feel before recording.');
    if (!Number.isFinite(now)) fail('A valid observation time is required.');
    if (lastRatingAt != null && (!Number.isFinite(lastRatingAt) || now - lastRatingAt < featurePolicy().windowSeconds * 1000))
      fail(`Wait at least ${featurePolicy().windowSeconds} seconds between ratings so measurement windows do not overlap.`);
    modelFeatures(snapshot);
    return { accepted: true, rating: { time: new Date(now).toISOString(), source: 'participant-self-report',
      level, task: String(task).trim().slice(0, 120), snapshot: structuredClone(snapshot) } };
  } catch (error) {
    return { accepted: false, reason: error.message };
  }
}
