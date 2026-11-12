import assert from 'node:assert/strict';
import { test } from 'node:test';
import { FatigueSession } from '../sensor/fatigue.js';

test('zero HR packets remain visible while fatigue explains the missing input', () => {
  const session = new FatigueSession();
  session.startBaseline(1000);
  session.heartRate({ heart_rate_bpm: 0, rr_intervals_ms: [], contact: true, recv_epoch_time_ms: 1000 });
  const snapshot = session.snapshot(1000);
  assert.equal(snapshot.hr, null);
  assert.equal(snapshot.rawHeartRate.heart_rate_bpm, 0);
  assert.deepEqual(snapshot.rawHeartRate.rr_intervals_ms, []);
  assert.equal(snapshot.heartSignal.status, 'invalid-hr');
  assert.match(snapshot.fatigue.reason, /0 bpm/);
  assert.match(snapshot.baselineStatus, /0 bpm/);
  assert.equal(snapshot.baselineState, 'failed');
});

test('contact rejection preserves the packet without treating it as usable HRV', () => {
  const session = new FatigueSession();
  const frame = { heart_rate_bpm: 105, rr_intervals_ms: [572, 600], contact: false, recv_epoch_time_ms: 1000 };
  session.heartRate(frame);
  frame.rr_intervals_ms[0] = 999;
  const snapshot = session.snapshot(1000);
  assert.equal(snapshot.heartSignal.status, 'contact-lost');
  assert.match(snapshot.fatigue.reason, /contact/i);
  assert.equal(snapshot.hr, null);
  assert.equal(snapshot.rrCount, 0);
  assert.deepEqual(snapshot.rawHeartRate.rr_intervals_ms, [572, 600]);
  snapshot.rawHeartRate.rr_intervals_ms[0] = 111;
  assert.equal(session.snapshot(1000).rawHeartRate.rr_intervals_ms[0], 572);
});

test('missing RR, waiting baseline, stale packets and a reset have distinct status', () => {
  const session = new FatigueSession();
  session.heartRate({ heart_rate_bpm: 75, rr_intervals_ms: [], contact: true, recv_epoch_time_ms: 1000 });
  assert.equal(session.snapshot(1000).hr, 75);
  assert.equal(session.snapshot(1000).heartSignal.status, 'missing-rr');
  assert.match(session.snapshot(1000).fatigue.reason, /RR intervals/);
  session.heartRate({ heart_rate_bpm: 75, rr_intervals_ms: [760, 840], contact: true, recv_epoch_time_ms: 2000 });
  assert.equal(session.snapshot(2000).heartSignal.status, 'receiving');
  assert.match(session.snapshot(2000).fatigue.reason, /baseline/);
  assert.equal(session.snapshot(7001).heartSignal.status, 'stale');
  assert.match(session.snapshot(7001).fatigue.reason, /stopped/);
  session.reset();
  assert.equal(session.snapshot(8000).rawHeartRate, null);
  assert.equal(session.snapshot(8000).heartSignal.status, 'waiting');
});
