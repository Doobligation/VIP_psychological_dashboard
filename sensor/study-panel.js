import { FatigueSession } from './fatigue.js';
import { mountClosedLoopPanel } from './closed-loop-panel.js';
import { getConfiguration } from './rule-config.js';
import { mountFatigueRatingPanel } from './fatigue-rating-panel.js';

const labels = { unavailable: 'Unavailable', low: 'Low fatigue', medium: 'Medium fatigue', high: 'High fatigue',
  good: 'Within posture thresholds', poor: 'Poor posture', critical: 'Critical posture' };
const riskLabel = (kind, level) => level === 'unavailable' || !level ? 'Unavailable'
  : `${level[0].toUpperCase()}${level.slice(1)} ${kind}`;
// Server decision-engine outcomes, shown as the reason a cue was or was not issued.
const decisionLabels = { 'no-feedback-for-level': 'No feedback at this level.',
  'awaiting-consecutive-high': 'Waiting for consecutive High predictions',
  cooldown: 'Feedback already given recently; waiting for the cooldown.', 'warning-cap': 'Warning limit reached; feedback suppressed.',
  'insufficient-evidence': 'More signal data needed.', 'no-feedback-for-cause': 'No feedback configured for this cause.',
  'duplicate-or-reversed-timestamp': 'Waiting for a new reading.', stale: 'Waiting for fresh, saved observations.' };
const number = (value, suffix = '') => Number.isFinite(value) ? `${value.toFixed(1)}${suffix}` : '—';

export function mountStudyPanel(root, controller) {
  root.innerHTML = `
    <style>
      #polar-panel .study { border-top: 1px solid #334155; margin-top: 18px; padding-top: 18px; }
      #polar-panel .study h3 { margin: 0 0 8px; }
      #polar-panel .study p { line-height: 1.5; }
      #polar-panel .study a { color: #7dd3fc; }
      #polar-panel .study-small { color: #b5c2d5; font-size: 13px; }
      #polar-panel .study-controls { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; margin: 12px 0; }
      #polar-panel .study-controls input[type=text] { background: #0f172a; color: #e2e8f0;
        border: 1px solid #64748b; border-radius: 6px; padding: 8px; max-width: 100%; box-sizing: border-box; }
      #polar-panel .study-controls select { background: #0f172a; color: #e2e8f0; padding: 8px; border: 1px solid #64748b; border-radius: 6px; max-width: 100%; }
      #polar-panel .study-controls label { max-width: 100%; }
      #polar-panel .study-controls input[type=file] { max-width: 100%; }
      #polar-panel .study-grid { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 12px; }
      #polar-panel .study-metric, #polar-panel .study-result { background: #0f172a; border: 1px solid #334155;
        padding: 14px; border-radius: 10px; min-width: 0; overflow-wrap: anywhere; }
      #polar-panel .study-metric strong { display: block; font-size: 23px; margin: 6px 0; }
      #polar-panel .study-result { margin-top: 12px; border-left: 4px solid #64748b; }
      #polar-panel .study-result[data-level=medium], #polar-panel .study-result[data-level=poor] { border-left-color: #fbbf24; }
      #polar-panel .study-result[data-level=high], #polar-panel .study-result[data-level=very-high],
      #polar-panel .study-result[data-level=critical] { border-left-color: #fb7185; }
      #polar-panel .study-result[data-level=good], #polar-panel .study-result[data-level=low] { border-left-color: #38bdf8; }
      #polar-panel #polar-quick-card { border-left-color: #38bdf8; margin-bottom: 16px; }
      #polar-panel #polar-quick-score { display: block; font-size: 36px; margin: 8px 0; }
      #polar-panel .study table { width: 100%; border-collapse: collapse; font-size: 13px; }
      #polar-panel .study th, #polar-panel .study td { padding: 8px 4px; text-align: left; border-bottom: 1px solid #334155; }
      #polar-panel .study details { margin-top: 16px; }
      #polar-panel .study summary { cursor: pointer; color: #7dd3fc; }
      @media(max-width: 600px) { #polar-panel .study-grid { grid-template-columns: 1fr; } }
    </style>
    <section class="study" aria-label="Group 1 fatigue and posture study">
      <h3>Team 3 · Frustration &amp; fatigue</h3>
      <p class="study-small">Applies the DBR masterfile rules to changes from your resting baseline and predicts Low, Medium or High risk.
        Only High risk gives feedback, after consecutive High predictions. Record how you feel to evaluate the rules.</p>
      <p class="study-small"><a href="serving/dashboard.html">Open the Team 3 frustration &amp; fatigue dashboard</a></p>
      <p id="polar-study-mode" role="status">LIVE · Connect a sensor to begin.</p>
      <div class="study-controls">
        <label>Participant ID <input id="polar-participant" type="text" maxlength="80" placeholder="e.g. P01"></label>
        <button id="polar-demo" type="button">Start demo</button>
        <button id="polar-export" type="button" disabled>Export session JSON</button>
      </div>
      <p class="study-small">Participant ID applies only to this Polar session. Local storage status appears below. Export browser-only ratings before leaving.
        The log retains the latest 10,000 observations in this browser.</p>
      <div id="polar-quick-card" class="study-result" aria-label="Quick test fatigue score">
        <h3>Fatigue score · quick test</h3>
        <strong id="polar-quick-score" role="status">—</strong>
        <p id="polar-quick-input" class="study-small"></p>
        <p id="polar-quick-status" class="study-small"></p>
        <p class="study-small">HR-only test scale, not a measured fatigue percentage. No baseline or RR intervals required.
          The full fatigue rules and feedback appear below.</p>
        <details><summary>How this test score works</summary>
          <p class="study-small">Maps the reported heart rate from 60–140 bpm onto 0–100, capped at either end.
            These are arbitrary testing settings. Contact warnings are bypassed for this score and shown above.
            The last nonzero HR can be held for up to 10 seconds with its age displayed.</p>
        </details>
      </div>
      <div class="study-controls">
        <button id="polar-baseline" type="button" disabled>Capture resting baseline</button>
        <button id="polar-calibrate" type="button" disabled>Calibrate upright posture</button>
        <label><input id="polar-speech" type="checkbox"> Enable spoken feedback</label>
      </div>
      <p id="polar-baseline-status" role="status"></p>
      <p id="polar-calibration-status" class="study-small"></p>
      <div class="study-grid">
        <div class="study-metric">Heart rate <strong id="polar-hr">—</strong><span id="polar-hr-change" class="study-small">Baseline needed</span></div>
        <div class="study-metric">RMSSD <strong id="polar-rmssd">—</strong><span id="polar-rmssd-change" class="study-small">Baseline needed</span></div>
        <div class="study-metric">SDNN <strong id="polar-sdnn">—</strong><span id="polar-sdnn-change" class="study-small">Baseline needed</span></div>
        <div class="study-metric">EDA <strong id="polar-eda">—</strong><span id="polar-eda-change" class="study-small">Separate EDA sensor required</span></div>
      </div>
      <p id="polar-window-status" class="study-small"></p>
      <p id="polar-heart-signal" class="study-small" role="status"></p>
      <p id="polar-capture-help" class="study-small"></p>
      <p id="polar-fatigue-method" class="study-small">Rule-based frustration and fatigue assessment · No ML model or model JSON is required.</p>
      <p id="polar-eda-note" class="study-small"></p>
      <div id="polar-ratings"></div>
      <div id="polar-loop"></div>
      <div id="polar-frustration-card" class="study-result">
        <p class="study-small">Frustration rules · EDA, SCR, HR, RMSSD and SDNN required</p>
        <h3 id="polar-frustration-level">Unavailable</h3>
        <p id="polar-frustration-cause"></p>
        <p id="polar-frustration-decision" class="study-small"></p>
        <p id="polar-frustration-feedback" role="status" aria-live="polite"></p>
        <p class="study-small">The Polar H10 measures HR/RR only. Frustration also needs EDA and SCR from a wrist EDA sensor;
          without them it stays unavailable rather than assumed normal.</p>
      </div>
      <div id="polar-fatigue-card" class="study-result">
        <p class="study-small">Full fatigue rules · baseline and HRV required</p>
        <h3 id="polar-fatigue-level">Unavailable</h3>
        <p id="polar-fatigue-cause"></p>
        <p id="polar-fatigue-decision" class="study-small"></p>
        <p id="polar-feedback" role="status" aria-live="polite"></p>
        <p id="polar-feedback-action" class="study-small"></p>
      </div>
      <div id="polar-posture-card" class="study-result">
        <h3 id="polar-posture-level">Unavailable</h3>
        <p id="polar-posture-angles"></p>
        <p id="polar-posture-cause" class="study-small"></p>
        <p id="polar-posture-decision" class="study-small"></p>
        <p id="polar-posture-feedback" role="status" aria-live="polite"></p>
      </div>
      <details><summary>What triggers feedback?</summary>
        <p class="study-small">Every condition listed in a row must hold; the highest matching row wins. Changes are percentages of the resting baseline.
          Low and Medium never give feedback. High gives feedback after <span id="polar-consecutive">3</span> consecutive High predictions;
          a Low, Medium or missing prediction resets the count.</p>
        <table><thead><tr><th>Level</th><th>Conditions (% baseline change)</th><th>Feedback</th></tr></thead>
          <tbody id="polar-rules-table"></tbody></table>
        <p id="polar-posture-rules" class="study-small"></p>
        <p class="study-small">Live feedback also observes cooldowns and a maximum warning count. Missing signals never mean normal.
          Timing defaults and heuristic scores need study review. Chest tilt cannot infer twisting, heat or sleep loss.
          <a href="docs/TEAM3_FRUSTRATION_FATIGUE.md">Read the Team 3 implementation guide</a>.</p>
      </details>
    </section>`;
  const el = id => root.querySelector(`#polar-${id}`);
  const text = (id, value) => { if (el(id).textContent !== value) el(id).textContent = value; };
  let loop = null;
  let demo = null;
  let demoTimer = null;
  let demoNow = null;
  let observations = [];
  let events = [];
  let mode = 'live';
  let participant = '';
  let lastLogged = null;
  let calibrationMessage = '';
  let startedAt = null;
  let sessionId = null;
  let lastGeneration = controller.generation;
  let savedBaseline = null;
  let droppedObservations = 0;
  const session = () => demo || controller.fatigue;
  const clock = () => demo ? demoNow : Date.now();
  const event = (type, details = {}) => {
    events.push({ time: new Date(clock()).toISOString(), type, ...details });
    if (type === 'fatigue-rating') void loop?.rating(details.rating);
  };
  const cancelSpeech = () => globalThis.speechSynthesis?.cancel();
  const ratings = mountFatigueRatingPanel(el('ratings'), {
    context: () => ({ snapshot: session().snapshot(clock()), mode: demo ? 'demo' : 'live', connected: controller.state === 'connected',
      participant, now: clock() }),
    onEvent: event,
  });

  loop = mountClosedLoopPanel(el('loop'), {
    speechEnabled: () => (Boolean(demo) || controller.state === 'connected') && el('speech').checked && !el('speech').disabled,
    isActive: () => Boolean(demo) || controller.state === 'connected',
  });

  function newLog(nextMode) {
    mode = nextMode;
    participant = el('participant').value.trim();
    observations = []; events = []; lastLogged = null; savedBaseline = null; droppedObservations = 0;
    startedAt = new Date().toISOString();
    sessionId = crypto.randomUUID();
    ratings.reset();
    loop.start({ session_id: sessionId, participant: participant || `${mode === 'demo' ? 'demo' : 'anonymous'}-${sessionId}`, mode });
    calibrationMessage = '';
    event('session-start', { mode });
  }

  function stopDemo() {
    clearInterval(demoTimer);
    demoTimer = null;
    if (demo) event('demo-stopped');
    demo = null;
    controller.fatigue.reset();
    cancelSpeech();
  }

  function demoFrame(hr, rr, pitch = 0, eda = 2) {
    demo.electrodermal({ eda_us: eda, recv_epoch_time_ms: demoNow });
    demo.heartRate({ heart_rate_bpm: hr, rr_intervals_ms: rr, recv_epoch_time_ms: demoNow });
    demo.accelerometer({ samples: [Math.sin(pitch * Math.PI / 180) * 1000, 0, Math.cos(pitch * Math.PI / 180) * 1000],
      epoch_timestamps_ms: [demoNow] });
  }

  el('demo').addEventListener('click', () => {
    if (demo) { stopDemo(); render(); return; }
    controller.disconnect();
    demo = new FatigueSession();
    demoNow = Date.now() - 60000;
    newLog('demo');
    demo.startBaseline(demoNow);
    for (let i = 0; i <= 60; i++) { demoFrame(75, [760, 840]); if (i < 60) demoNow += 1000; }
    demo.calibratePosture(demoNow);
    event('synthetic-baseline', { baseline: demo.snapshot(demoNow).baseline });
    let elapsed = 0;
    demoTimer = setInterval(() => {
      demoNow += 1000;
      elapsed++;
      if (elapsed <= 40) demoFrame(79, [730, 786], 30, 2.1);
      else if (elapsed <= 70) demoFrame(79, [744, 776], 50, 2.1);
      else if (elapsed <= 100) demoFrame(79, [744, 776], 0, 2.1);
      else demoFrame(75, [760, 840], 0);
      render();
      if (elapsed >= 140) { clearInterval(demoTimer); demoTimer = null; }
    }, 100);
    render();
  });
  el('baseline').addEventListener('click', () => {
    cancelSpeech();
    session().startBaseline(clock());
    event('baseline-start');
    render();
  });
  el('calibrate').addEventListener('click', () => {
    const ok = session().calibratePosture(clock());
    calibrationMessage = ok ? 'Upright reference captured. Keep the sensor in this position on the strap.'
      : 'No fresh accelerometer data. Wait for samples, stand upright and try again.';
    event('posture-calibration', { accepted: ok });
    render();
  });
  if (!globalThis.speechSynthesis || !globalThis.SpeechSynthesisUtterance) {
    el('speech').disabled = true;
    el('speech').parentElement.title = 'Speech is unavailable in this browser; text feedback remains available.';
  }
  el('speech').addEventListener('change', () => {
    cancelSpeech();
    event('speech-setting', { enabled: el('speech').checked });
  });
  el('export').addEventListener('click', () => {
    const config = getConfiguration();
    const result = { schemaVersion: 2, sessionId, participant, mode, startedAt, exportedAt: new Date().toISOString(),
      description: mode === 'demo' ? 'Synthetic demonstration at 20x time; not participant measurements.'
        : 'Experimental Polar H10 study with rule-based fatigue/posture results and independent participant ratings.',
      baseline: savedBaseline, policy: { ...config.rules.featurePolicy,
        rulesVersion: config.rules.version, feedbackVersion: config.feedback.version,
        decision: config.feedback.decision },
      localStorage: { saved: loop.client.saved, pending: loop.client.queue.length, dropped: loop.client.dropped },
      droppedObservations, events, observations, ...ratings.exportData() };
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `polar-${mode}-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  function decisionText(decision) {
    if (!decision) return 'Waiting for the local feedback loop.';
    if (decision.alert) return `Feedback issued after ${decision.alert.consecutive} consecutive High predictions.`;
    const status = decisionLabels[decision.suppression] || 'Waiting for the local feedback loop.';
    const { count, required, level } = decision.streak || {};
    const base = status.replace(/\.$/, '');
    return !level ? status : count > required ? `${base} (High ×${count}).` : `${base} (${count}/${required}).`;
  }
  const feedbackLine = kind => {
    const alert = loop.lastAlert(kind);
    return alert ? `Feedback: ${alert.message}` : 'No feedback issued yet.';
  };

  function render() {
    const active = Boolean(demo) || controller.wanted || controller.busy;
    const now = clock();
    const s = session().snapshot(now);
    ratings.render(s);
    loop.render();
    const quick = s.quickFatigue;
    text('quick-score', quick.available ? `${quick.score} / 100` : '—');
    text('quick-input', quick.available
      ? `${demo ? 'SIMULATED · ' : ''}Reported HR ${number(quick.hrBpm, ' bpm')} · ${quick.held ? 'Holding last reading' : 'Latest reading'} · ${number(quick.ageSeconds, 's old')}`
      : 'Waiting for HR data.');
    text('quick-status', `${quick.reason}${quick.contactWarning ? ' Contact warning bypassed for this quick test; signal quality is uncertain.' : ''}`);
    if (!s.postureCalibrated && calibrationMessage.startsWith('Upright reference captured')) calibrationMessage = '';
    text('study-mode', demo ? 'DEMO · Synthetic data · 10× playback · No live measurements'
      : `LIVE · ${controller.state === 'connected' ? 'Polar session; resting baseline required after every connection.' : 'Connect a sensor to begin.'}`);
    text('demo', demo ? 'Stop demo' : 'Start demo');
    el('participant').disabled = active;
    el('baseline').disabled = Boolean(demo) || controller.state !== 'connected';
    el('calibrate').disabled = Boolean(demo) || controller.state !== 'connected';
    el('export').disabled = observations.length === 0;
    text('baseline-status', s.baselineStatus + (s.baselineProgress !== null ? ` ${Math.floor(s.baselineProgress * s.requiredWindowSeconds)} / ${s.requiredWindowSeconds}s.` : ''));
    text('calibration-status', calibrationMessage || (s.postureCalibrated ? 'Upright reference captured for this session.'
      : 'Stand upright, then calibrate posture once acceleration is streaming.') + (controller.accMessage && !demo ? ` ${controller.accMessage}` : ''));
    for (const key of ['hr', 'rmssd', 'sdnn', 'eda']) {
      text(key, number(s[key], key === 'hr' ? ' bpm' : key === 'eda' ? ' µS' : ' ms'));
      text(`${key}-change`, key === 'eda' && (!s.edaFresh || !s.baseline?.eda) ? 'Fresh EDA + EDA baseline required' : s.fatigue.changes ? `${number(s.fatigue.changes[key], '%')} vs baseline ${number(s.baseline[key])}` : 'Baseline + complete fresh window needed');
    }
    text('eda-note', demo ? 'DEMO · EDA is simulated alongside HR/RR to demonstrate all three fatigue levels.' : 'The high fatigue rule requires EDA. The Polar H10 does not measure EDA; without a separate fresh EDA baseline, only low and medium rules can be assessed.');
    text('window-status', `${s.rrCount} valid RR intervals · ${Math.floor(s.windowSeconds)}s of a ${s.requiredWindowSeconds}s window · ${s.fresh ? 'HR/RR current' : 'Waiting for fresh HR and RR'}`);
    text('capture-help', `Saved readings count received packets, including packets that cannot support fatigue calculations. Capture a resting baseline with ${s.requiredWindowSeconds}s of continuous HR/RR and at least ${s.featurePolicy.minRr} valid RR intervals.`);
    const packet = s.rawHeartRate;
    text('heart-signal', `${packet ? `Last HR packet: ${number(packet.heart_rate_bpm, ' bpm')} · ${packet.rr_intervals_ms?.length || 0} RR intervals. ` : ''}${s.heartSignal.reason}`);
    for (const kind of ['fatigue', 'posture']) {
      el(`${kind}-card`).dataset.level = s[kind].level;
      text(`${kind}-level`, labels[s[kind].level]);
      text(`${kind}-cause`, s[kind].reason);
    }
    // Feedback lines show the server decision engine's latest saved outcome.
    const latest = loop.client.latest;
    const decisions = active && latest && !latest.stale ? latest.states : {};
    const frustration = decisions.frustration;
    el('frustration-card').dataset.level = frustration?.level || 'unavailable';
    text('frustration-level', riskLabel('frustration', frustration?.level));
    text('frustration-cause', frustration?.available
      ? `${frustration.causeLabel ? `Likely cause: ${frustration.causeLabel}. ` : ''}${frustration.reason}`
      : frustration?.reason || 'Waiting for saved observations.');
    text('frustration-decision', decisionText(frustration));
    text('frustration-feedback', feedbackLine('frustration'));
    text('fatigue-decision', decisionText(decisions.fatigue));
    text('feedback', feedbackLine('fatigue'));
    text('feedback-action', loop.lastAlert('fatigue')?.action || 'Heat, sleep loss and work duration need separate context.');
    text('posture-angles', `Upright-relative pitch ${number(s.posture.pitch, '°')} · roll ${number(s.posture.roll, '°')}`);
    text('posture-decision', decisionText(decisions.posture));
    text('posture-feedback', feedbackLine('posture'));
    // Keep an ended demo downloadable, but never advertise an invalidated live
    // baseline as current. Historical observations retain their own baselines.
    if (demo || mode === 'live') savedBaseline = s.baseline;
    if (active && (lastLogged === null || now - lastLogged >= 1000)) {
      observations.push({ time: new Date(now).toISOString(), generation: controller.generation, ...s });
      if (observations.length > 10000) { observations.shift(); droppedObservations++; }
      lastLogged = now;
      if (demo) loop.sample(s, { source: 'synthetic-demo' }, now);
    }
    for (const alert of latest?.alerts || []) {
      if (!events.some(e => e.type === 'feedback' && e.alertId === alert.id))
        event('feedback', { alertId: alert.id, kind: alert.state, level: alert.level, cause: alert.cause, message: alert.message, action: alert.action });
    }
    if (controller.generation !== lastGeneration) {
      lastGeneration = controller.generation;
      if (startedAt && !demo) event('connection-change', { generation: lastGeneration, state: controller.state });
    }
  }

  return {
    render,
    refreshRules() {
      const { rules, feedback } = getConfiguration();
      const required = feedback.decision.consecutiveHighPredictions;
      text('baseline', `Capture resting baseline (${rules.featurePolicy.windowSeconds}s)`);
      text('consecutive', String(required));
      el('rules-table').replaceChildren();
      for (const rule of [...rules.states.fatigue.rules].sort((a, b) => a.severity - b.severity)) {
        const row = document.createElement('tr');
        for (const value of [rule.level, rule.conditions.map(c => `${c.signal} ${c.operator} ${c.value}%`).join(' AND '),
          feedback.states.fatigue[rule.level]?.enabled ? `After ${required} consecutive` : 'None']) {
          const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
        }
        el('rules-table').append(row);
      }
      text('posture-rules', rules.states.posture.rules.map(r => `${r.level}: ${r.conditions.map(c => `${c.signal} ${c.operator} ${c.value}°`).join(' OR ')}`).join(' · '));
    },
    recordSample(frame) {
      if (mode === 'live' && sessionId && !demo) {
        const now = frame.recv_epoch_time_ms ?? Date.now();
        loop.sample(controller.fatigue.snapshot(now), frame, now);
      }
    },
    beginLive() { stopDemo(); newLog('live'); },
    isDemo: () => Boolean(demo),
    pause() { stopDemo(); cancelSpeech(); },
    dispose() { loop.dispose(); ratings.dispose(); stopDemo(); cancelSpeech(); },
  };
}
