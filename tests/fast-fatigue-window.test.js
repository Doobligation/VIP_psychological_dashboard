import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FatigueSession } from '../sensor/fatigue.js';

test('temporary default captures a real baseline in five seconds', () => {
  const session = new FatigueSession();
  assert.equal(session.windowMs, 5000);
  session.startBaseline(0);
  for (let second = 0; second <= 5; second++) {
    session.heartRate({ heart_rate_bpm: 75, rr_intervals_ms: [second % 2 ? 840 : 760], recv_epoch_time_ms: second * 1000 });
    const s = session.snapshot(second * 1000);
    assert.equal(s.baselineState, second < 5 ? 'collecting' : 'ready');
  }
  assert.equal(session.snapshot(5000).fatigue.level, 'low');
  assert.equal(session.snapshot(5000).requiredWindowSeconds, 5);
});

test('five-second setting still rejects zero HR and missing RR', () => {
  for (const [hr, rr] of [[0, []], [75, []]]) {
    const session = new FatigueSession(); session.startBaseline(0);
    for (let second = 0; second <= 5; second++)
      session.heartRate({ heart_rate_bpm: hr, rr_intervals_ms: rr, recv_epoch_time_ms: second * 1000 });
    assert.equal(session.snapshot(5000).fatigue.level, 'unavailable');
    assert.equal(session.snapshot(5000).baseline, null);
  }
});
