import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateHrv, classifyFatigue, postureAngles, classifyPosture, FatigueSession } from '../sensor/fatigue.js';

// Keep the longer-window regression scenarios explicit while the live default is 5s.
const standardWindow = { windowMs: 60000, freshMs: 5000, minRr: 30, baselineSamples: 30 };
const baseline = { hr: 100, rmssd: 100, sdnn: 100 };
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} ≈ ${expected}`);
function heart(session, t, hr = 100, rr = [700, 900], eda) {
  if (eda !== undefined) session.electrodermal({ eda_us: eda, recv_epoch_time_ms: t });
  session.heartRate({ heart_rate_bpm: hr, rr_intervals_ms: rr, recv_epoch_time_ms: t });
}
function acc(session, t, pitch = 0, roll = 0) {
  // Pure pitch or pure roll fixtures avoid solving a simultaneous orientation.
  session.accelerometer({ samples: [Math.sin(pitch * Math.PI / 180) * 1000,
    Math.sin(roll * Math.PI / 180) * 1000,
    Math.cos((pitch || roll) * Math.PI / 180) * 1000], epoch_timestamps_ms: [t], recv_epoch_time_ms: t });
}
function calibrate(session, start = 0, eda) {
  session.startBaseline(start);
  for (let t = start; t <= start + session.windowMs; t += 1000) heart(session, t, 100, [700, 900], eda);
  return start + session.windowMs;
}

test('HRV uses milliseconds, successive differences and sample SDNN', () => {
  const result = calculateHrv([800, 1000, 800]);
  near(result.rmssd, 200);
  near(result.sdnn, 115.47005383792515);
  assert.deepEqual(calculateHrv([]), { rmssd: null, sdnn: null });
  assert.deepEqual(calculateHrv([800]), { rmssd: null, sdnn: null });
  assert.deepEqual(calculateHrv([800, 800]), { rmssd: 0, sdnn: 0 });
});

test('RR artifacts are excluded without joining successive differences across them', () => {
  const result = calculateHrv([800, 1000, 100, 1400, 1400, NaN, Infinity, 3000]);
  near(result.rmssd, Math.sqrt(20000));
  near(result.sdnn, 300);
  assert.deepEqual(calculateHrv([299, 2001, -1, null]), { rmssd: null, sdnn: null });
  assert.equal(calculateHrv([300, 2000]).rmssd, 1700);
  assert.equal(calculateHrv([800, null, 1000]).rmssd, null);
});

test('fatigue evaluates each available baseline change under the newest source', () => {
  assert.equal(classifyFatigue({ hr: 100, rmssd: 84, sdnn: 84 }, baseline).level, 'low');
  assert.equal(classifyFatigue({ hr: 100, rmssd: 74, sdnn: 79 }, baseline).level, 'medium');
  assert.equal(classifyFatigue({ hr: 105, rmssd: 64, sdnn: 69, eda:2.2, edaFresh:true }, { ...baseline, eda:2 }).level, 'high');
  assert.equal(classifyFatigue({ hr: 150, rmssd: 20, sdnn: 90 }, baseline).level, 'low');
  assert.deepEqual(classifyFatigue({ hr: 100, rmssd: 59, sdnn: 64 }, baseline).changes,
    { hr: 0, rmssd: -41, sdnn: -36 });
});

test('missing, nonfinite and zero baselines are unavailable, never healthy', () => {
  for (const value of [null, undefined, NaN, Infinity, 0, -1]) {
    for (const key of ['rmssd', 'sdnn']) {
      const result = classifyFatigue(baseline, { ...baseline, [key]: value });
      assert.equal(result.level, 'unavailable');
      assert.equal(result.changes, null);
      assert.ok(result.reason);
    }
  }
  assert.equal(classifyFatigue({ hr: 130, rmssd: null, sdnn: 60 }, baseline).level, 'unavailable');
  assert.equal(classifyFatigue({ hr: 130, rmssd: 0, sdnn: 0 }, baseline).level, 'medium');
});

test('posture estimates gravity tilt and classifies relative calibrated angles only', () => {
  const angles = postureAngles({ x: 500, y: 0, z: Math.sqrt(750000) });
  near(angles.pitch, 30);
  near(angles.roll, 0);
  near(angles.magnitude, 1000);
  assert.equal(postureAngles({ x: 0, y: 0, z: 0 }), null);
  assert.equal(postureAngles({ x: Infinity, y: 0, z: 1000 }), null);
  assert.equal(classifyPosture(angles, null).level, 'unavailable');
  assert.equal(classifyPosture({ pitch: 40, roll: 10 }, { pitch: 20, roll: 0 }).level, 'good');
  assert.equal(classifyPosture({ pitch: -20.1, roll: 0 }, { pitch: 0, roll: 0 }).level, 'poor');
  assert.equal(classifyPosture({ pitch: 45, roll: 20 }, { pitch: 0, roll: 0 }).level, 'poor');
  assert.equal(classifyPosture({ pitch: 0, roll: -20.1 }, { pitch: 0, roll: 0 }).level, 'critical');
  assert.equal(classifyPosture({ pitch: 45.1, roll: 0 }, { pitch: 0, roll: 0 }).level, 'critical');
  assert.equal('yaw' in angles, false);
});

test('absent acceleration is unavailable without throwing', () => {
  assert.equal(postureAngles(null), null);
  assert.equal(postureAngles(), null);
});

test('real Polar typed-array accelerometer frames can calibrate and update posture', () => {
  const session = new FatigueSession(standardWindow);
  session.accelerometer({ type: 'ACC', samples: new Int16Array([0, 0, 1000, 0, 0, 1000]),
    epoch_timestamps_ms: new Float64Array([1000, 1020]), recv_epoch_time_ms: 1020 });
  assert.equal(session.calibratePosture(1020), true);
  session.accelerometer({ type: 'ACC', samples: new Int16Array([707, 0, 707]),
    epoch_timestamps_ms: new Float64Array([1040]), recv_epoch_time_ms: 1040 });
  assert.equal(session.snapshot(1040).posture.level, 'poor');
  near(session.snapshot(1040).posture.pitch, 45);
});

test('baseline is explicit and requires a complete observation period', () => {
  const session = new FatigueSession(standardWindow);
  for (let t = 0; t <= 70000; t += 1000) heart(session, t);
  assert.equal(session.snapshot(70000).baseline, null);
  assert.equal(session.snapshot(70000).fatigue.level, 'unavailable');
  session.startBaseline(71000);
  assert.equal(session.snapshot(71000).rrCount, 0);
  for (let t = 71000; t < 131000; t += 1000) heart(session, t);
  const collecting = session.snapshot(130000);
  assert.equal(collecting.baseline, null);
  assert.ok(collecting.baselineProgress > 0.9 && collecting.baselineProgress < 1);
  heart(session, 131000);
  const ready = session.snapshot(131000);
  assert.equal(ready.baselineState, 'ready');
  near(ready.baseline.hr, 100);
  near(ready.baseline.rmssd, 200);
  assert.equal(ready.fatigue.level, 'low');
});

test('baseline rejects zero variance and insufficient RR with a restart explanation', () => {
  for (const rr of [[800, 800], []]) {
    const session = new FatigueSession(standardWindow);
    session.startBaseline(0);
    for (let t = 0; t <= 60000; t += 1000) heart(session, t, 100, rr);
    const state = session.snapshot(60000);
    assert.equal(state.baseline, null);
    assert.equal(state.baselineState, 'failed');
    assert.match(state.baselineStatus, /restart/i);
    assert.equal(state.fatigue.level, 'unavailable');
  }
});

test('a full baseline period with fewer than 30 RR intervals remains unavailable', () => {
  const session = new FatigueSession(standardWindow);
  session.startBaseline(0);
  for (let t = 0; t <= 60000; t += 1000) heart(session, t, 100,
    t % 4000 === 0 ? [t % 8000 === 0 ? 700 : 900] : []);
  const state = session.snapshot(60000);
  assert.equal(state.rrCount, 16);
  assert.equal(state.baselineState, 'failed');
  assert.equal(state.fatigue.level, 'unavailable');
});

test('HR without RR stops baseline calibration and cannot revive stale eligibility', () => {
  const session = new FatigueSession(standardWindow);
  session.startBaseline(0);
  for (let t = 0; t <= 30000; t += 1000) heart(session, t);
  for (let t = 31000; t <= 60000; t += 1000) heart(session, t, 100, []);
  assert.equal(session.snapshot(60000).baselineState, 'failed');
  assert.equal(session.snapshot(60000).baselineProgress, null);
  session.startBaseline(61000);
  heart(session, 61000);
  assert.equal(session.snapshot(67000).baselineState, 'failed');
});

test('a fresh HRV window needs enough elapsed coverage even with many intervals', () => {
  const session = new FatigueSession(standardWindow);
  calibrate(session);
  heart(session, 70000, 130, Array(60).fill([770, 830]).flat());
  const state = session.snapshot(70000);
  assert.equal(state.fatigue.level, 'unavailable');
  assert.ok(state.rrCount >= 30);
});

test('stale HR and RR make fatigue unavailable until a new complete window exists', () => {
  const session = new FatigueSession(standardWindow);
  calibrate(session);
  assert.equal(session.snapshot(65001).fatigue.level, 'unavailable');
  heart(session, 66000);
  assert.equal(session.snapshot(66000).fatigue.level, 'unavailable');
  for (let t = 67000; t <= 126000; t += 1000) heart(session, t);
  assert.equal(session.snapshot(126000).fatigue.level, 'low');
  for (let t = 127000; t <= 132000; t += 1000) heart(session, t, 100, []);
  assert.equal(session.snapshot(132000).fatigue.level, 'unavailable');
});

test('invalid HR or contact loss immediately invalidates fatigue and baseline collection', () => {
  for (const patch of [{ heart_rate_bpm: NaN }, { sensor_contact: false }, { contact: false }]) {
    const session = new FatigueSession(standardWindow);
    calibrate(session);
    session.heartRate({ heart_rate_bpm: 100, rr_intervals_ms: [700, 900], recv_epoch_time_ms: 61000, ...patch });
    assert.equal(session.snapshot(61000).fatigue.level, 'unavailable');
    session.startBaseline(62000);
    session.heartRate({ heart_rate_bpm: 100, rr_intervals_ms: [700, 900], recv_epoch_time_ms: 62000, ...patch });
    assert.equal(session.snapshot(62000).baselineState, 'failed');
  }
});

test('rolling storage is bounded and duplicate or older HR packets cannot refresh readings', () => {
  const session = new FatigueSession(standardWindow);
  calibrate(session);
  for (let t = 61000; t <= 300000; t += 1000) heart(session, t);
  const state = session.snapshot(300000);
  assert.ok(state.rrCount <= 130);
  heart(session, 300000, 200, [300, 2000]);
  heart(session, 299999, 200, [300, 2000]);
  assert.deepEqual(session.snapshot(300000), state);
  assert.equal(session.snapshot(305001).fatigue.level, 'unavailable');
});

test('the browser session measures and classifies but never decides feedback timing', () => {
  const session = new FatigueSession({ windowMs: 10000, minRr: 10, freshMs: 2000, baselineSamples: 10 });
  calibrate(session, 0, 2);
  for (let t = 11000; t <= 60000; t += 1000) heart(session, t, 105, [740, 860], 2.2);
  const state = session.snapshot(60000);
  assert.equal(state.fatigue.level, 'high');
  assert.equal('feedback' in state, false);
  assert.equal('feedback' in state.posture, false);
});

test('posture requires fresh calibration and uses signed relative offsets', () => {
  const session = new FatigueSession(standardWindow);
  assert.equal(session.calibratePosture(0), false);
  acc(session, 1000, 10);
  assert.equal(session.snapshot(1000).posture.level, 'unavailable');
  assert.equal(session.calibratePosture(1000), true);
  near(session.snapshot(1000).posture.pitch, 0);
  acc(session, 2000, 40);
  assert.equal(session.snapshot(2000).posture.level, 'poor');
  near(session.snapshot(2000).posture.pitch, 30);
  acc(session, 33000, 60);
  assert.equal(session.snapshot(33000).posture.level, 'critical');
  assert.equal(session.snapshot(38001).posture.level, 'unavailable');
});

test('invalid ACC observations clear posture and cannot be replaced by older packets', () => {
  const session = new FatigueSession(standardWindow);
  acc(session, 1000);
  session.calibratePosture(1000);
  session.accelerometer({ samples: [NaN, 0, 1000], epoch_timestamps_ms: [3000], recv_epoch_time_ms: 3000 });
  assert.equal(session.snapshot(3000).posture.level, 'unavailable');
  acc(session, 2000, 50);
  assert.equal(session.snapshot(3000).posture.level, 'unavailable');
  acc(session, 4000, 50);
  assert.equal(session.snapshot(4000).posture.level, 'critical');
});

test('reset clears personal baselines and posture references', () => {
  const session = new FatigueSession(standardWindow);
  calibrate(session);
  acc(session, 60000);
  session.calibratePosture(60000);
  session.reset();
  const state = session.snapshot(60000);
  assert.equal(state.baseline, null);
  assert.equal(state.rrCount, 0);
  assert.equal(state.hr, null);
  assert.equal(state.fatigue.level, 'unavailable');
  assert.equal(state.posture.level, 'unavailable');
});
