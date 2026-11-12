import { evaluateRule, featurePolicy } from './rule-config.js';
import { QuickFatigueScore } from './quick-fatigue.js';
// Group 1's experimental, baseline-relative rules. These are transparent
// decision rules, not a validated fatigue probability or a medical diagnosis.
const positive = (value) => Number.isFinite(value) && value > 0;
const numericArray = (value) => Array.isArray(value) || (ArrayBuffer.isView(value) && !(value instanceof DataView));
const validRr = (value) => Number.isFinite(value) && value >= featurePolicy().rrMinMs && value <= featurePolicy().rrMaxMs;
const freshAt = (time, now, freshMs) => time !== null && now >= time && now - time <= freshMs;
const unavailable = (reason) => ({ level: 'unavailable', changes: null, reason });

/** RR plausibility filtering is an artifact heuristic, not ectopic correction.
 * Invalid entries break the sequence, so RMSSD never bridges an excluded beat.
 * SDNN uses the sample denominator n - 1; insufficient values/pairs return null.
 */
export function calculateHrv(rrMs) {
  const values = [];
  let previous = null;
  let squares = 0;
  let pairs = 0;
  for (const rr of Array.isArray(rrMs) ? rrMs : []) {
    if (!validRr(rr)) { previous = null; continue; }
    values.push(rr);
    if (previous !== null) { squares += (rr - previous) ** 2; pairs++; }
    previous = rr;
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    rmssd: pairs ? Math.sqrt(squares / pairs) : null,
    sdnn: values.length >= 2 ? Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1)) : null,
  };
}

export function classifyFatigue(current, baseline) {
  if (!baseline || !['rmssd', 'sdnn'].every((key) => positive(baseline[key])))
    return unavailable('Collect a valid personal baseline with nonzero HRV before comparing fatigue.');
  if (!current || !['rmssd', 'sdnn'].every((key) => Number.isFinite(current[key]) && current[key] >= 0))
    return unavailable('Fresh RMSSD and SDNN are required.');
  const changes = {};
  for (const key of ['hr', 'rmssd', 'sdnn', 'eda']) {
    if (key === 'eda' && current.edaFresh !== true) continue;
    if (positive(baseline[key]) && Number.isFinite(current[key]) && current[key] >= 0 && (key !== 'hr' || current[key] > 0)) {
      const value = (current[key] - baseline[key]) / baseline[key] * 100;
      if (Number.isFinite(value)) changes[key] = value;
    }
  }
  const result = evaluateRule('fatigue', Object.fromEntries(Object.entries(changes).map(([key,value]) => [`${key}ChangePct`,value])));
  return { ...result, changes };
}

export function postureAngles(vector) {
  const { x, y, z } = vector ?? {};
  if (![x, y, z].every(Number.isFinite)) return null;
  const magnitude = Math.hypot(x, y, z);
  if (!positive(magnitude)) return null;
  return { pitch: Math.atan2(x, Math.hypot(y, z)) * 180 / Math.PI,
    roll: Math.atan2(y, Math.hypot(x, z)) * 180 / Math.PI, magnitude };
}

export function classifyPosture(angles, reference) {
  if (![angles, reference].every((value) => value && Number.isFinite(value.pitch) && Number.isFinite(value.roll)))
    return { level: 'unavailable', pitch: null, roll: null, reason: 'Calibrate a neutral posture using fresh accelerometer data.' };
  const pitch = angles.pitch - reference.pitch;
  const roll = angles.roll - reference.roll;
  return { ...evaluateRule('posture', { absPitch: Math.abs(pitch), absRoll: Math.abs(roll) }), pitch, roll };
}

// Feedback decisions (consecutive High predictions, cooldowns) are made by the local
// server's decision engine from saved observations; this session only measures and classifies.

/**
 * HR uses the mean of observations in the rolling window. RR observations are
 * timestamped by receipt; this does not reconstruct precise beat timestamps.
 * Baseline needs windowMs of recording and windowMs-freshMs retained coverage
 * (allowing for notification boundaries), >=minRr intervals, and nonzero HRV.
 * Snapshot extras: baselineState/status, postureCalibrated, windowSeconds,
 * requiredWindowSeconds, fresh (both HR and RR), and the RR artifact bounds.
 */
export class FatigueSession {
  constructor({ windowMs = featurePolicy().windowSeconds * 1000, minRr = featurePolicy().minRr, freshMs = featurePolicy().freshSeconds * 1000,
    baselineSamples = featurePolicy().baselineSamples } = {}) {
    if (!positive(windowMs) || !positive(freshMs) || freshMs >= windowMs || !Number.isInteger(minRr) || minRr < 2)
      throw new RangeError('Use a positive window, a shorter freshness interval, and at least two RR intervals.');
    if (!Number.isInteger(baselineSamples) || baselineSamples < 2) throw new RangeError('At least two baseline observations are required.');
    Object.assign(this, { windowMs, minRr, freshMs, baselineSamples });
    // A fixed upper bound also protects against malformed high-volume packets.
    this.maxRr = Math.ceil(windowMs / featurePolicy().rrMinMs) + 64;
    this.reset();
  }

  reset() {
    this.quickScore = new QuickFatigueScore();
    this._clearWindow();
    this.baseline = null;
    this.baselineStart = null;
    this.baselineState = 'idle';
    this.baselineStatus = 'Start a resting baseline to enable fatigue comparison.';
    this.reference = null;
    this.angles = null;
    this.lastAcc = null;
    this.lastAccSeen = null;
    this.lastPacket = null;
    this.rawHeartRate = null;
    this.packetProblem = null;
  }

  _clearWindow() {
    this.hrWindow = [];
    this.rrWindow = [];
    this.sequence = 0;
    this.lastHr = null;
    this.lastRr = null;
    this.edaWindow = [];
    this.lastEda = null;
  }

  startBaseline(now = Date.now()) {
    if (!Number.isFinite(now)) return false;
    this._clearWindow();
    this.baseline = null;
    this.baselineStart = now;
    this.baselineState = 'collecting';
    this.baselineStatus = 'Recording a resting baseline. Keep HR and RR streaming continuously.';
    return true;
  }

  _failBaseline(reason) {
    if (this.baselineState !== 'collecting') return;
    this.baselineState = 'failed';
    this.baselineStart = null;
    this.baselineStatus = `${reason} Restart the resting baseline when the signal is stable.`;
  }

  _checkBaselineFreshness(now) {
    if (this.baselineState === 'collecting' &&
        (!freshAt(this.lastHr ?? this.baselineStart, now, this.freshMs) || !freshAt(this.lastRr ?? this.baselineStart, now, this.freshMs)))
      this._failBaseline('HR or RR data stopped during calibration.');
  }

  _prune(now) {
    this.hrWindow = this.hrWindow.filter(({ t }) => t >= now - this.windowMs).slice(-4096);
    this.rrWindow = this.rrWindow.filter(({ t }) => t >= now - this.windowMs).slice(-this.maxRr);
    this.edaWindow = this.edaWindow.filter(({ t }) => t >= now - this.windowMs).slice(-4096);
  }

  // Optional external EDA stream, in microSiemens; the H10 itself has no EDA sensor.
  electrodermal(frame) {
    const now = frame?.recv_epoch_time_ms;
    if (!Number.isFinite(now) || (this.lastEda !== null && now <= this.lastEda)) return;
    if (!Number.isFinite(frame.eda_us) || frame.eda_us < 0) { this.edaWindow = []; this.lastEda = now; return; }
    if (this.lastEda !== null && now - this.lastEda > this.freshMs) this.edaWindow = [];
    this.edaWindow.push({ t: now, value: frame.eda_us }); this.lastEda = now;
    this._prune(now);
  }

  _edaMetrics(now) {
    const edaFresh = freshAt(this.lastEda, now, this.freshMs) && this.edaWindow.length > 1
      && this.lastEda - this.edaWindow[0].t >= this.windowMs - this.freshMs;
    return { edaFresh, eda: edaFresh ? this.edaWindow.reduce((sum, x) => sum + x.value / this.edaWindow.length, 0) : null };
  }

  heartRate(frame) {
    const now = frame?.recv_epoch_time_ms ?? Date.now();
    if (!Number.isFinite(now) || (this.lastPacket !== null && now <= this.lastPacket) ||
        (this.baselineState === 'collecting' && now < this.baselineStart)) return;
    this.lastPacket = now;
    // Keep the received packet available for troubleshooting and exports even
    // when it cannot be used for HRV. Never replace rejected measurements by 0.
    this.rawHeartRate = structuredClone(frame);
    this.quickScore.observe({ ...frame, recv_epoch_time_ms: now });
    this.packetProblem = [frame?.contact, frame?.sensor_contact, frame?.sensor_contact_detected].includes(false)
      ? { status: 'contact-lost', reason: 'The sensor reports poor or missing skin contact; this HR/RR packet is excluded from fatigue calculations.' }
      : !positive(frame?.heart_rate_bpm)
        ? { status: 'invalid-hr', reason: frame?.heart_rate_bpm === 0
          ? 'The sensor reports 0 bpm; fatigue needs a nonzero heart rate and valid RR intervals.'
          : 'The sensor reports an invalid heart rate; fatigue needs a nonzero heart rate and valid RR intervals.' }
        : null;
    this._checkBaselineFreshness(now);
    if (this.packetProblem) {
      this._failBaseline(this.packetProblem.reason);
      this._clearWindow();
      return;
    }
    if ((this.lastHr !== null && now - this.lastHr > this.freshMs) || (this.lastRr !== null && now - this.lastRr > this.freshMs))
      this._clearWindow();
    this.hrWindow.push({ t: now, value: frame.heart_rate_bpm });
    this.lastHr = now;
    for (const value of Array.isArray(frame.rr_intervals_ms) ? frame.rr_intervals_ms : []) {
      if (!validRr(value)) { this.sequence++; continue; }
      this.rrWindow.push({ t: now, value, sequence: this.sequence });
      this.lastRr = now;
    }
    this._prune(now);
    this._checkBaselineFreshness(now);
    const metrics = this._metrics(now);
    if (this.baselineState === 'collecting' && now - this.baselineStart >= this.windowMs) {
      if (metrics.eligible && this.hrWindow.length >= this.baselineSamples && positive(metrics.rmssd) && positive(metrics.sdnn)) {
        this.baseline = { hr: metrics.hr, rmssd: metrics.rmssd, sdnn: metrics.sdnn };
        if (metrics.edaFresh && positive(metrics.eda)) this.baseline.eda = metrics.eda;
        this.baselineState = 'ready';
        this.baselineStart = null;
        this.baselineStatus = 'Resting baseline ready for this sensor session.';
      } else this._failBaseline('The baseline needs enough valid RR intervals and nonzero HRV.');
    }
  }

  _metrics(now) {
    this._prune(now);
    const fresh = freshAt(this.lastHr, now, this.freshMs) && freshAt(this.lastRr, now, this.freshMs);
    const rr = [];
    let sequence = null;
    for (const entry of this.rrWindow) {
      if (sequence !== null && entry.sequence !== sequence) rr.push(null);
      rr.push(entry.value);
      sequence = entry.sequence;
    }
    const hrv = calculateHrv(rr);
    const hr = freshAt(this.lastHr, now, this.freshMs) && this.hrWindow.length
      ? this.hrWindow.reduce((sum, entry) => sum + entry.value, 0) / this.hrWindow.length : null;
    const coverage = Math.min(
      this.hrWindow.length ? this.lastHr - this.hrWindow[0].t : 0,
      this.rrWindow.length ? this.lastRr - this.rrWindow[0].t : 0,
    );
    return { hr, ...hrv, ...this._edaMetrics(now), fresh, coverage, eligible: fresh && coverage >= this.windowMs - this.freshMs &&
      this.rrWindow.length >= this.minRr && hrv.rmssd !== null && hrv.sdnn !== null };
  }

  _heartSignal(now) {
    if (this.lastPacket === null) return { status: 'waiting', reason: 'Waiting for heart-rate packets. The EXG waveform is a separate stream.' };
    if (!freshAt(this.lastPacket, now, this.freshMs))
      return { status: 'stale', reason: 'Heart-rate packets stopped; waiting for fresh HR and RR intervals.' };
    if (this.packetProblem) return { ...this.packetProblem };
    if (!freshAt(this.lastRr, now, this.freshMs))
      return { status: 'missing-rr', reason: 'Heart-rate packets are arriving, but no current valid RR intervals are available for HRV.' };
    return { status: 'receiving', reason: 'Receiving current heart rate and valid RR intervals.' };
  }

  _fatigue(metrics, now) {
    const signal = this._heartSignal(now);
    if (signal.status !== 'receiving') return unavailable(signal.reason);
    if (!this.baseline) return unavailable(this.baselineStatus);
    if (!metrics.fresh) return unavailable('Fresh HR and RR intervals are both required; waiting for the sensor.');
    if (!metrics.eligible) return unavailable(`Collecting a complete rolling window with at least ${this.minRr} valid RR intervals.`);
    return classifyFatigue(metrics, this.baseline);
  }

  accelerometer(frame) {
    const samples = frame?.samples;
    const times = frame?.epoch_timestamps_ms;
    if (!numericArray(samples) || !numericArray(times) || samples.length !== times.length * 3) return;
    for (let i = 0; i < times.length; i++) {
      const now = times[i];
      if (!Number.isFinite(now) || (this.lastAccSeen !== null && now <= this.lastAccSeen)) continue;
      this.lastAccSeen = now;
      const angles = postureAngles({ x: samples[3 * i], y: samples[3 * i + 1], z: samples[3 * i + 2] });
      if (!angles) { this.angles = null; continue; }
      this.angles = angles;
      this.lastAcc = now;
    }
  }

  calibratePosture(now = Date.now()) {
    if (!this.angles || !freshAt(this.lastAcc, now, this.freshMs)) {
      this.reference = null;
      return false;
    }
    this.reference = { pitch: this.angles.pitch, roll: this.angles.roll };
    return true;
  }

  snapshot(now = Date.now()) {
    this._checkBaselineFreshness(now);
    const metrics = this._metrics(now);
    const fatigue = this._fatigue(metrics, now);
    let posture = classifyPosture(this.angles, this.reference);
    if (!freshAt(this.lastAcc, now, this.freshMs))
      posture = { level: 'unavailable', pitch: null, roll: null, reason: 'Fresh accelerometer data is required for posture.' };
    return {
      rawHeartRate: this.rawHeartRate ? structuredClone(this.rawHeartRate) : null,
      quickFatigue: this.quickScore.snapshot(now),
      heartSignal: this._heartSignal(now),
      hr: metrics.hr,
      eda: metrics.eda,
      edaFresh: metrics.edaFresh,
      rmssd: freshAt(this.lastRr, now, this.freshMs) ? metrics.rmssd : null,
      sdnn: freshAt(this.lastRr, now, this.freshMs) ? metrics.sdnn : null,
      rrCount: this.rrWindow.length,
      baseline: this.baseline ? { ...this.baseline } : null,
      baselineState: this.baselineState,
      baselineStatus: this.baselineStatus,
      baselineProgress: this.baselineState === 'collecting'
        ? Math.max(0, Math.min(1, (Math.min(this.lastHr ?? this.baselineStart, this.lastRr ?? this.baselineStart) - this.baselineStart) / this.windowMs)) : null,
      fatigue, posture,
      postureCalibrated: this.reference !== null,
      windowSeconds: metrics.coverage / 1000,
      requiredWindowSeconds: this.windowMs / 1000,
      fresh: metrics.fresh,
      featurePolicy: { ...featurePolicy(), windowSeconds: this.windowMs / 1000, freshSeconds: this.freshMs / 1000,
        minRr: this.minRr, baselineSamples: this.baselineSamples },
      rrFilter: { minMs: featurePolicy().rrMinMs, maxMs: featurePolicy().rrMaxMs },
    };
  }
}
