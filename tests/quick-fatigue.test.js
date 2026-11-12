import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FatigueSession } from '../sensor/fatigue.js';

test('quick test score uses actual HR immediately despite a contact warning and missing RR', () => {
  const session = new FatigueSession();
  session.heartRate({ heart_rate_bpm: 111, rr_intervals_ms: [], contact: false, recv_epoch_time_ms: 1000 });
  const s = session.snapshot(1000);
  assert.equal(s.quickFatigue.score, 64);
  assert.equal(s.quickFatigue.source, 'hr-only-debug');
  assert.equal(s.quickFatigue.hrBpm, 111);
  assert.equal(s.quickFatigue.contactWarning, true);
  assert.equal(s.quickFatigue.provisional, true);
  assert.equal(s.baseline, null);
  assert.equal(s.fatigue.level, 'unavailable');
});

test('zero HR never creates a score, and a held last reading expires and resets', () => {
  const session = new FatigueSession();
  session.heartRate({ heart_rate_bpm: 0, rr_intervals_ms: [], recv_epoch_time_ms: 1000 });
  assert.equal(session.snapshot(1000).quickFatigue.score, null);
  session.heartRate({ heart_rate_bpm: 100, rr_intervals_ms: [], contact: true, recv_epoch_time_ms: 2000 });
  assert.equal(session.snapshot(2000).quickFatigue.score, 50);
  session.heartRate({ heart_rate_bpm: 0, rr_intervals_ms: [], recv_epoch_time_ms: 3000 });
  const held = session.snapshot(3000).quickFatigue;
  assert.equal(held.score, 50);
  assert.equal(held.held, true);
  assert.equal(held.ageSeconds, 1);
  assert.equal(held.measuredAtMs, 2000);
  assert.equal(session.snapshot(12001).quickFatigue.score, null);
  session.reset();
  assert.equal(session.snapshot(13000).quickFatigue.score, null);
});

test('quick test score is bounded and duplicate or older packets cannot change it', () => {
  for (const [hr, expected] of [[50, 0], [60, 0], [140, 100], [180, 100]]) {
    const session = new FatigueSession();
    session.heartRate({ heart_rate_bpm: hr, rr_intervals_ms: [], recv_epoch_time_ms: 1000 });
    assert.equal(session.snapshot(1000).quickFatigue.score, expected);
    session.heartRate({ heart_rate_bpm: 80, rr_intervals_ms: [], recv_epoch_time_ms: 1000 });
    session.heartRate({ heart_rate_bpm: 80, rr_intervals_ms: [], recv_epoch_time_ms: 999 });
    assert.equal(session.snapshot(1000).quickFatigue.score, expected);
  }
});
