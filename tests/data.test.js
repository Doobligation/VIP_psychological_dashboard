import assert from 'node:assert/strict';
import { test } from 'node:test';
import { PolarData } from '../sensor/data.js';

function frame(samples, start = Date.now()) {
  return {
    prev_sample_timestamp_ms: start - 100,
    samples,
    epoch_timestamps_ms: samples.map((_, i) => start + i * 1000 / 130),
  };
}

test('RMS waits for a complete 26-sample window and preserves microvolt units', () => {
  const data = new PolarData();
  assert.equal(data.fresh(), false);
  const start = Date.now();
  data.waveform(frame(Array(25).fill(0), start));
  assert.equal(data.rms.length, 0);
  data.waveform(frame([0], start + 25 * 1000 / 130));
  assert.equal(data.rms.length, 1);
  assert.equal(data.rms[0].values[0], 0);

  // A 100 µV amplitude sinusoid above the cutoff has RMS ≈ 100 / sqrt(2).
  data.reset();
  data.waveform(frame(Array.from({ length: 1300 }, (_, i) => 100 * Math.sin(2 * Math.PI * 40 * i / 130))));
  assert.ok(Math.abs(data.rms.at(-1).values[0] - 70.7107) < 0.1);
});

test('highpass filter suppresses DC and low-frequency components', () => {
  for (const samples of [Array(1300).fill(100), Array.from({ length: 1300 }, (_, i) => 100 * Math.sin(2 * Math.PI * 5 * i / 130))]) {
    const data = new PolarData();
    data.waveform(frame(samples));
    assert.ok(data.rms.at(-1).values[0] < 0.01);
  }
});

test('malformed, nonfinite and duplicate samples do not refresh data', () => {
  const data = new PolarData();
  const good = frame(Array(26).fill(10));
  data.waveform({ ...good, prev_sample_timestamp_ms: 0 });
  data.waveform({ ...good, epoch_timestamps_ms: [] });
  data.waveform(frame([NaN, Infinity]));
  assert.equal(data.fresh(), false);
  data.waveform(good);
  const snapshot = structuredClone(data.rms);
  data.waveform(good);
  assert.deepEqual(data.rms, snapshot);
});

test('stale sessions restart the filter and require a fresh full window', (t) => {
  t.mock.timers.enable({ apis: ['Date'], now: 100000 });
  const data = new PolarData();
  data.waveform(frame(Array(26).fill(100)));
  t.mock.timers.tick(5001);
  assert.equal(data.fresh(), false);
  data.waveform(frame([0]));
  assert.equal(data.fresh(), true);
  assert.deepEqual(data.rms, []);
  data.waveform(frame(Array(25).fill(0), Date.now() + 10));
  assert.equal(data.rms.length, 1);
  assert.equal(data.rms[0].values[0], 0);
});

test('display buffer is bounded and reset clears all readings', () => {
  const data = new PolarData();
  data.waveform(frame(Array(3000).fill(0)));
  assert.equal(data.rms.length, 1950);
  data.reset();
  assert.deepEqual(data.rms, []);
  assert.equal(data.fresh(), false);
});
