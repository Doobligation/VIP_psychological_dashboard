import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createDriver } from '../sensor/driver.js';

function delayedDevice() {
  const pending = Promise.withResolvers();
  let first = true;
  const device = new EventTarget();
  device.gatt = {
    connected: false,
    async connect() {
      if (first) {
        first = false;
        await pending.promise;
      }
      this.connected = true;
      return this;
    },
    disconnect() { this.connected = false; },
  };
  return { device, pending };
}

test('a late connection from a disposed driver cannot disconnect a newer session', async () => {
  const { device, pending } = delayedDevice();
  const oldScope = createDriver(device);
  const oldConnect = oldScope.driver.device.gatt.connect();
  const cancelled = assert.rejects(oldConnect, /cancelled/);
  oldScope.dispose();
  const newScope = createDriver(device);
  await newScope.driver.device.gatt.connect();
  pending.resolve();
  await cancelled;
  assert.equal(device.gatt.connected, true);
  newScope.dispose();
  device.gatt.disconnect();
});

test('a late connection is closed when there is no newer active session', async () => {
  const { device, pending } = delayedDevice();
  const scope = createDriver(device);
  const cancelled = assert.rejects(scope.driver.device.gatt.connect(), /cancelled/);
  scope.dispose();
  pending.resolve();
  await cancelled;
  assert.equal(device.gatt.connected, false);
});

test('disposing the newest session still allows orphaned late connections to close', async () => {
  const { device, pending } = delayedDevice();
  const oldScope = createDriver(device);
  const cancelled = assert.rejects(oldScope.driver.device.gatt.connect(), /cancelled/);
  oldScope.dispose();
  const newScope = createDriver(device);
  await newScope.driver.device.gatt.connect();
  newScope.dispose();
  device.gatt.disconnect();
  pending.resolve();
  await cancelled;
  assert.equal(device.gatt.connected, false);
});
