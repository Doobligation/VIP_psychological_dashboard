import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { test } from 'node:test';
import { PolarController } from '../sensor/controller.js';

// Only the hardware boundary is substituted. The controller and data pipeline run normally.
function harness(t, options = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const device = new EventTarget();
  device.name = 'Polar H10 test';
  device.gatt = {
    connected: false,
    disconnect() {
      if (!this.connected) return;
      this.connected = false;
      device.dispatchEvent(new Event('gattserverdisconnected'));
    },
  };
  const scopes = [];
  const controller = new PolarController({
    bluetooth: { requestDevice: async () => device },
    retryBaseMs: 10,
    timeoutMs: 100,
    makeDriver: () => {
      const scope = {
        disposed: false,
        dispose() { this.disposed = true; },
        driver: {
          addEventListener(type, handler) { assert.equal(type, 'ECG'); scope.onFrame = handler; },
          async init() { device.gatt.connected = true; },
          async startECG(rate) { assert.equal(rate, 130); return { error: 'SUCCESS' }; },
        },
      };
      scopes.push(scope);
      return scope;
    },
    ...options,
  });
  t.after(() => controller.disconnect());
  return { controller, device, scopes };
}

test('unsupported browser and cancelled chooser leave connection usable', async (t) => {
  const { controller } = harness(t, { bluetooth: null });
  await controller.connect();
  assert.equal(controller.state, 'error');
  assert.equal(controller.busy, false);
  controller.bluetooth = { requestDevice: async () => { throw new DOMException('Cancelled', 'NotFoundError'); } };
  await controller.connect();
  assert.match(controller.message, /No device selected/);
  assert.equal(controller.wanted, false);
  assert.equal(controller.busy, false);
});

test('disconnect clears readings and ignores notifications from a disposed session', async (t) => {
  const { controller, scopes, device } = harness(t);
  await controller.connect();
  assert.equal(controller.state, 'connected');
  const frame = { prev_sample_timestamp_ms: 1, samples: Array(26).fill(0), epoch_timestamps_ms: Array.from({ length: 26 }, (_, i) => Date.now() + i) };
  scopes[0].onFrame(frame);
  assert.equal(controller.data.rms.length, 1);
  controller.disconnect();
  scopes[0].onFrame(frame);
  assert.equal(controller.state, 'idle');
  assert.deepEqual(controller.data.rms, []);
  assert.equal(scopes[0].disposed, true);
  assert.equal(device.gatt.connected, false);
});

test('unexpected disconnection reconnects without reopening the chooser', async (t) => {
  const { controller, device, scopes } = harness(t);
  await controller.connect();
  controller.bluetooth.requestDevice = () => { throw new Error('Chooser must require a new user click'); };
  device.gatt.disconnect();
  assert.equal(controller.state, 'reconnecting');
  assert.equal(scopes[0].disposed, true);
  t.mock.timers.tick(10);
  await setImmediate();
  assert.equal(controller.state, 'connected');
  assert.equal(scopes.length, 2);
});

test('manual disconnect cancels a scheduled reconnect', async (t) => {
  const { controller, device, scopes } = harness(t);
  await controller.connect();
  device.gatt.disconnect();
  controller.disconnect();
  t.mock.timers.tick(1000);
  await setImmediate();
  assert.equal(scopes.length, 1);
  assert.equal(controller.state, 'idle');
});

test('connection timeout releases the session and exhausts bounded retries', async (t) => {
  let attempts = 0;
  let disposed = 0;
  const { controller } = harness(t, {
    maxRetries: 1,
    makeDriver: () => {
      attempts++;
      return { dispose: () => disposed++, driver: { addEventListener() {}, init: () => new Promise(() => {}) } };
    },
  });
  const opening = controller.connect();
  await setImmediate();
  t.mock.timers.tick(100);
  await opening;
  assert.equal(controller.state, 'reconnecting');
  t.mock.timers.tick(10);
  await setImmediate();
  t.mock.timers.tick(100);
  await setImmediate();
  assert.equal(controller.state, 'error');
  assert.match(controller.message, /exhausted/);
  assert.equal(controller.busy, false);
  assert.equal(controller.wanted, false);
  assert.equal(attempts, 2);
  assert.equal(disposed, 2);
});

test('driver construction failures are reported and release the busy state', async (t) => {
  const { controller } = harness(t, { maxRetries: 0, makeDriver: () => { throw new Error('GATT unavailable'); } });
  await assert.doesNotReject(controller.connect());
  assert.equal(controller.state, 'error');
  assert.equal(controller.busy, false);
  assert.equal(controller.wanted, false);
  assert.match(controller.message, /GATT unavailable/);
});
