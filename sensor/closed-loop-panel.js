import { ClosedLoopClient } from './closed-loop-client.js';
import { getConfiguration } from './rule-config.js';

export function mountClosedLoopPanel(root, { speechEnabled = () => false, isActive = () => true } = {}) {
  root.innerHTML = `<div class="study-result">
    <h3>Local feedback loop</h3>
    <p id="loop-storage" class="study-small" role="status">Start a live or demo session to save readings locally.</p>
    <p id="loop-decision" role="status" aria-live="polite">Waiting for observations.</p>
    <p id="loop-model" class="study-small"></p>
    <p id="loop-evidence" class="study-small"></p>
    <p id="loop-alert"></p><p id="loop-goal" class="study-small"></p>
    <div class="study-controls" id="loop-responses">
      <button type="button" data-response="acknowledged">Acknowledge feedback</button>
      <button type="button" data-response="break-taken">I took a break</button>
      <button type="button" data-response="not-helpful">Feedback wasn’t helpful</button>
    </div>
    <div class="study-controls" id="loop-outcomes">
      <span>After the feedback, I feel:</span>
      <button type="button" data-outcome="better">Better</button>
      <button type="button" data-outcome="same">The same</button>
      <button type="button" data-outcome="worse">Worse</button>
    </div>
    <p id="loop-feedback-status" role="status" class="study-small"></p>
    <a id="loop-export" hidden download="polar-saved-session.json">Download saved session</a>
    <p class="study-small">Saved on this computer. Responses and outcomes are linked to the last feedback cue.
      Only High risk issues feedback, after consecutive High predictions; Low and Medium never do.</p>
  </div>`;
  const el = id => root.querySelector(`#loop-${id}`);
  const text = (id, value) => { if (el(id).textContent !== value) el(id).textContent = value; };
  // Every saved response passes through render, so per-state cues persist after the alerting prediction.
  let lastAlert = null, lastAlerts = {}, seen = new Set(), busy = false, previousSession = null;
  try { previousSession = localStorage.getItem('polar-last-saved-session'); } catch { /* storage may be disabled */ }
  function exportLink(id) {
    el('export').hidden = !id;
    if (id) el('export').href = `/export?session_id=${encodeURIComponent(id)}`;
  }
  exportLink(previousSession);
  const client = new ClosedLoopClient({ onChange: render });
  function render() {
    const current = client.latest;
    const freshMs = getConfiguration().feedback.decision.freshSeconds * 1000;
    const stale = !isActive() || !current || current.stale || Date.now() - client.receivedAt > freshMs
      || (client.session?.mode === 'live' && Date.now() - current.timestamp_ms > freshMs);
    if (client.session) text('storage', `${client.saved} readings saved · ${client.queue.length} pending · ${client.dropped} dropped${client.error ? ` · ${client.error}` : ''}`);
    if (client.saved) {
      exportLink(client.session.session_id);
      try { localStorage.setItem('polar-last-saved-session', client.session.session_id); } catch { /* optional convenience */ }
    }
    const states = current?.states || {};
    text('decision', stale ? 'Waiting for fresh, saved observations.' : Object.entries(states).map(([name, value]) => {
      if (!value.available) return `${name}: unavailable`;
      const statuses = { 'awaiting-consecutive-high': 'waiting for consecutive High predictions', cooldown: 'waiting between cues',
        'warning-cap': 'warning limit reached', 'no-feedback-for-level': 'no feedback at this level',
        'no-feedback-for-cause': 'no feedback configured for this cause',
        'insufficient-evidence': 'more signal data needed', 'duplicate-or-reversed-timestamp': 'waiting for a new reading' };
      const streak = !value.streak?.level ? '' : value.streak.count > value.streak.required
        ? ` (High ×${value.streak.count})` : ` (${value.streak.count}/${value.streak.required} consecutive)`;
      return `${name}: ${value.level} · ${statuses[value.suppression] || 'feedback issued'}${streak}`;
    }).join(' · '));
    text('model', 'Rule-based frustration and fatigue assessment from the DBR masterfile; no ML model is used yet.');
    const missingReason = current?.features?.snapshot?.fatigue?.reason || states.fatigue?.reason;
    text('evidence', !stale && states.fatigue?.available === false ? missingReason || ''
      : !stale && states.fatigue?.unassessedLevels?.length ? states.fatigue.reason : '');
    for (const alert of current?.alerts || []) {
      if (seen.has(alert.id)) continue;
      seen.add(alert.id);
      if (stale) continue;
      lastAlert = alert;
      lastAlerts[alert.state] = alert;
      text('alert', `Last feedback · ${alert.state}${alert.causeLabel ? ` (${alert.causeLabel})` : ''} · ${alert.message} ${alert.action}`);
      text('goal', `Goal: ${alert.goal}`);
      if (speechEnabled() && globalThis.speechSynthesis && globalThis.SpeechSynthesisUtterance) {
        const prefix = client.session.mode === 'demo' ? 'Demonstration. ' : '';
        globalThis.speechSynthesis.speak(new SpeechSynthesisUtterance(prefix + alert.message));
      }
    }
    for (const button of root.querySelectorAll('button')) button.disabled = !lastAlert || busy;
  }
  root.addEventListener('click', async event => {
    const button = event.target.closest('button');
    if (!button || button.disabled || !lastAlert) return;
    const kind = button.dataset.response ? 'response' : 'outcome';
    const epoch = client.epoch;
    busy = true; render();
    try {
      await client.feedback({ kind, alert_id: lastAlert.id, [kind]: button.dataset[kind] });
      if (epoch === client.epoch) text('feedback-status', kind === 'response' ? 'Response saved.' : 'Outcome saved.');
    } catch (error) { if (epoch === client.epoch) text('feedback-status', `Not saved: ${error.message}`); }
    finally { if (epoch === client.epoch) { busy = false; render(); } }
  });
  render();
  return {
    client, render,
    lastAlert: state => lastAlerts[state] || null,
    start(session) { lastAlert = null; lastAlerts = {}; seen = new Set(); busy = false;
      text('alert', ''); text('goal', ''); text('feedback-status', ''); client.start(session); },
    sample: (...args) => client.sample(...args),
    async rating(rating) {
      const epoch = client.epoch;
      try {
        if (!client.latest?.reading_id || client.error
          || Date.now() - client.latest.timestamp_ms > getConfiguration().feedback.decision.freshSeconds * 1000)
          throw new Error('No fresh saved reading. The rating remains in your browser session export.');
        await client.feedback({ kind: 'rating', reading_id: client.latest.reading_id, level: rating.level,
          task: rating.task, observed_at_ms: Date.parse(rating.time) });
        if (epoch === client.epoch) text('feedback-status', 'Fatigue rating saved to the local database.');
      } catch (error) { if (epoch === client.epoch) text('feedback-status', `Rating not saved to database: ${error.message}`); }
    },
    dispose() { client.dispose(); },
  };
}
