import { validateModel, predictFatigue, modelFeatures, recordFatigueRating } from './fatigue-model.js';
import { featurePolicy } from './rule-config.js';

const levels = { unavailable: 'Unavailable', none: 'No fatigue pattern estimated', low: 'Estimated low fatigue',
  medium: 'Estimated medium fatigue', high: 'Estimated high fatigue', 'very-high': 'Estimated very high fatigue' };

export function mountFatigueModelPanel(root, { context, onEvent }) {
  root.innerHTML = `
    <div id="polar-model-card" class="study-result">
      <h3>Fatigue model · <span id="polar-model-level">Unavailable</span></h3>
      <p id="polar-model-status" class="study-small" role="status">Loading model…</p>
      <p id="polar-model-context" class="study-small"></p>
      <p id="polar-model-reason"></p>
      <details><summary>Load a retrained model</summary>
        <div class="study-controls"><label>Load trained model JSON
          <input id="polar-model-file" type="file" accept=".json,application/json"></label></div>
        <p id="polar-model-import-status" class="study-small" role="status"></p>
        <p class="study-small">Imported models are previewed in demo only. Live deployment requires the training comparison gate. Export the session to retain its model and readings.
          <a href="docs/POLAR_FATIGUE_ML.md">Training instructions</a>.</p>
      </details>
    </div>
    <div class="study-result">
      <h3>Record your fatigue</h3>
      <p class="study-small">Choose how you feel independently of the displayed estimates. These ratings and the current
        measurement window can train a model for your study. Record at least one full measurement window apart.</p>
      <div class="study-controls">
        <label>How fatigued do you feel?
          <select id="polar-rating-level"><option value="">Choose a rating</option>
            <option value="none">No fatigue felt</option><option value="low">Low</option>
            <option value="medium">Medium</option><option value="high">High</option>
            <option value="very-high">Very high</option></select></label>
        <label>Current task <input id="polar-rating-task" type="text" maxlength="120" placeholder="e.g. drywall installation"></label>
        <button id="polar-rating-save" type="button" disabled>Record fatigue rating</button>
      </div>
      <p id="polar-rating-status" class="study-small" role="status"></p>
    </div>`;
  const el = id => root.querySelector(`#polar-${id}`);
  const text = (id, value) => { if (el(id).textContent !== value) el(id).textContent = value; };
  let model = null, loadError = '', ratingMessage = '', disposed = false, request = 0;
  let importedModel = false;
  let ratings = [], models = new Map();

  function setModel(value) {
    const next = validateModel(value);
    model = next;
    models.set(next.modelId, next);
    loadError = '';
    onEvent('model-loaded', { modelId: next.modelId, source: next.training.source });
  }

  // Only a successful local import supersedes the default fetch. A malformed
  // import must not discard a usable default that is still on its way.
  fetch('./models/polar_fatigue_model.json', { cache: 'no-store' }).then(async response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const value = await response.json();
    if (!disposed && !importedModel) setModel(value);
  }).catch(error => {
    if (!disposed && !importedModel) loadError = `Could not load the default model (${error.message}). Load a model JSON to continue.`;
  }).finally(() => { if (!disposed) render(); });

  el('model-file').addEventListener('change', async () => {
    const file = el('model-file').files?.[0];
    if (!file) return;
    const currentRequest = ++request;
    try {
      if (file.size > 1000000) throw new Error('Model JSON must be smaller than 1 MB.');
      const value = JSON.parse(await file.text());
      if (disposed || request !== currentRequest) return;
      setModel(value);
      importedModel = true;
      text('model-import-status', `Loaded ${file.name}.`);
    } catch (error) {
      if (disposed || request !== currentRequest) return;
      text('model-import-status', error.message);
    }
    el('model-file').value = '';
    render();
  });

  el('rating-level').addEventListener('change', () => render());
  el('rating-save').addEventListener('click', () => {
    const ctx = context();
    const result = recordFatigueRating({ ...ctx, level: el('rating-level').value,
      task: el('rating-task').value, lastRatingAt: ratings.length ? Date.parse(ratings.at(-1).time) : null });
    if (result.accepted) {
      ratings.push(result.rating);
      el('rating-level').value = '';
      ratingMessage = 'Rating saved. Export the session before leaving this page.';
      onEvent('fatigue-rating', { level: result.rating.level, source: result.rating.source, rating: result.rating });
    } else ratingMessage = result.reason;
    render();
  });

  function render(snapshot) {
    const ctx = context();
    snapshot ??= ctx.snapshot;
    const result = predictFatigue(model, snapshot, { mode: ctx.mode });
    text('model-level', levels[result.level]);
    text('model-reason', result.reason);
    text('model-context', ctx.mode === 'demo' ? 'DEMO · Synthetic sensor data; ratings disabled.'
      : 'LIVE · Rules first. Approved model blends appear in the local feedback loop below.');
    text('model-status', model ? `${model.training.source === 'synthetic-bootstrap' ? 'Synthetic starter'
      : `${model.training.participants === 1 ? 'Personal' : 'Study'} model · ${model.training.samples} self-ratings`} · ${model.modelId}`
      : loadError || 'Loading model…');
    root.querySelector('#polar-model-card').dataset.level = result.level;
    let reason = '';
    if (ctx.mode !== 'live' || !ctx.connected) reason = 'Connect a live sensor to record ratings.';
    else if (!ctx.participant.trim()) reason = 'Enter a participant ID before connecting.';
    else {
      try { modelFeatures(snapshot); } catch (error) { reason = error.message; }
    }
    const wait = ratings.length ? Math.ceil((Date.parse(ratings.at(-1).time) + featurePolicy().windowSeconds * 1000 - ctx.now) / 1000) : 0;
    if (!reason && wait > 0) reason = `Next rating available in ${wait}s.`;
    el('rating-save').disabled = Boolean(reason) || !el('rating-level').value;
    text('rating-status', `${ratings.length} rating${ratings.length === 1 ? '' : 's'} recorded. ${reason || ratingMessage || 'Choose a rating when you are ready.'}`);
    return result;
  }

  return { render,
    reset() {
      ratings = []; models = new Map(model ? [[model.modelId, model]] : []); ratingMessage = '';
      el('rating-level').value = ''; el('rating-task').value = '';
    },
    exportData: () => ({ fatigueRatings: structuredClone(ratings), models: [...models.values()].map(value => structuredClone(value)) }),
    dispose() { disposed = true; request++; },
  };
}
