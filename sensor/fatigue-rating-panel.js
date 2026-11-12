import { modelFeatures, recordFatigueRating } from './fatigue-model.js';
import { featurePolicy } from './rule-config.js';

export function mountFatigueRatingPanel(root, { context, onEvent }) {
  root.innerHTML = `
    <div class="study-result">
      <h3>Record your fatigue</h3>
      <p class="study-small">Choose how you feel independently of the displayed rule results. These ratings and the current
        measurement window let you evaluate the rules. Record at least one full measurement window apart.</p>
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
  let ratings = [], ratingMessage = '';
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
  }
  return { render, reset() { ratings = []; ratingMessage = ''; el('rating-level').value = ''; el('rating-task').value = ''; },
    exportData: () => ({ fatigueRatings: structuredClone(ratings) }), dispose() {} };
}
