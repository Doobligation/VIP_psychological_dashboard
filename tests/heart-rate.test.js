import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as driver from '../sensor/driver.js';

test('HR measurement decodes RR units, 16-bit HR, energy and contact flags', () => {
  assert.equal(typeof driver.decodeHeartRate, 'function');
  const value = new DataView(new Uint8Array([31, 44, 1, 123, 0, 0, 4, 0, 2]).buffer);
  assert.deepEqual(driver.decodeHeartRate(value, 1234), {
    heart_rate_bpm: 300, rr_intervals_ms: [1000, 500], recv_epoch_time_ms: 1234, contact: true,
  });
});

test('HR packets without RR never invent intervals and contact loss is explicit', () => {
  assert.equal(typeof driver.decodeHeartRate, 'function');
  assert.deepEqual(driver.decodeHeartRate(new DataView(new Uint8Array([4, 80]).buffer), 99), {
    heart_rate_bpm: 80, rr_intervals_ms: [], recv_epoch_time_ms: 99, contact: false,
  });
});

test('truncated HR payloads are rejected instead of turning energy or bytes into RR', () => {
  assert.equal(typeof driver.decodeHeartRate, 'function');
  for (const bytes of [[], [1, 70], [8, 70], [16, 70, 0], [24, 70, 0]]) {
    assert.throws(() => driver.decodeHeartRate(new DataView(new Uint8Array(bytes).buffer), 1), /Truncated/);
  }
});
