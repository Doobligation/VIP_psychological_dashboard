import { test, expect } from '@playwright/test';

// Substitute browser Bluetooth transport, while running the shipped bundle,
// Polar protocol decoder, filter, controller, canvas and dashboard unchanged.
function installBluetooth() {
  const characteristics = new Map();
  let stream;
  let frameIndex = 0;
  class Characteristic extends EventTarget {
    async startNotifications() { return this; }
    async readValue() { return new DataView(new ArrayBuffer(1)); }
    async writeValueWithResponse(buffer) {
      const command = new Uint8Array(buffer);
      if (command[0] !== 2 || command[1] !== 0 || new DataView(buffer).getUint16(8, true) !== 130) {
        throw new Error('Expected EXG start at 130 Hz');
      }
      this.value = new DataView(new Uint8Array([240, 2, 0, 0, 0]).buffer);
      this.dispatchEvent(new Event('characteristicvaluechanged'));
      frameIndex = 0;
      clearInterval(stream);
      stream = setInterval(() => {
        const data = characteristics.get('fb005c82-02e7-f387-1cad-8acd2d8df0c8');
        const packet = new DataView(new ArrayBuffer(10 + 26 * 3));
        packet.setBigUint64(1, 1000000000n + BigInt(frameIndex++) * 200000000n, true);
        for (let i = 0; i < 26; i++) {
          const sample = Math.round(100 * Math.sin(2 * Math.PI * 40 * i / 130));
          for (let b = 0; b < 3; b++) packet.setUint8(10 + 3 * i + b, sample >> (8 * b) & 255);
        }
        data.value = packet;
        data.dispatchEvent(new Event('characteristicvaluechanged'));
      }, 200);
    }
  }
  const device = new EventTarget();
  device.name = 'Polar H10 simulated';
  device.gatt = {
    connected: false,
    async connect() { this.connected = true; return this; },
    disconnect() {
      clearInterval(stream);
      if (this.connected) {
        this.connected = false;
        device.dispatchEvent(new Event('gattserverdisconnected'));
      }
    },
    async getPrimaryService() {
      return { async getCharacteristic(id) {
        if (!characteristics.has(id)) characteristics.set(id, new Characteristic());
        return characteristics.get(id);
      } };
    },
  };
  window.polarTest = { device, stop: () => clearInterval(stream) };
  Object.defineProperty(navigator, 'bluetooth', {
    configurable: true,
    value: { requestDevice: async () => device },
  });
}

test('shipped bundle receives EXG, survives reconnect and clears stale readings', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(installBluetooth);
  await page.goto('/psychosocial_dashboard.html');
  await expect(page.locator('#workerList .worker-card').first()).toBeVisible();
  await page.getByRole('button', { name: 'Connect Polar H10' }).click();
  await expect(page.locator('#polar-status')).toContainText('Receiving EXG');
  await expect(page.locator('#polar-rms-value')).toHaveText(/\d+\.\d/);
  await expect(page.locator('#polar-panel canvas')).toHaveCount(1);
  await page.screenshot({ path: test.info().outputPath('polar-streaming.png') });
  await page.getByLabel('Monitoring scenario').selectOption('Mental Demand');
  await expect(page.locator('#polar-status')).toContainText('Receiving EXG');
  await page.evaluate(() => window.polarTest.device.gatt.disconnect());
  await expect(page.locator('#polar-status')).toContainText('Retrying');
  await expect(page.locator('#polar-status')).toContainText('Receiving EXG');
  await page.evaluate(() => window.polarTest.stop());
  await expect(page.locator('#polar-rms-value')).toHaveText('—', { timeout: 7000 });
  await expect(page.locator('#polar-status')).toContainText('stale');
  await page.getByRole('button', { name: 'Disconnect', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Connect Polar H10' })).toBeEnabled();
  await expect(page.locator('#polar-rms-value')).toHaveText('—');
  expect(errors).toEqual([]);
});

test('unsupported Bluetooth leaves historical dashboard usable on a narrow screen', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.addInitScript(() => Object.defineProperty(navigator, 'bluetooth', { value: undefined }));
  await page.goto('/psychosocial_dashboard.html');
  await page.getByRole('button', { name: 'Connect Polar H10' }).click();
  await expect(page.locator('#polar-status')).toContainText('Web Bluetooth requires');
  await page.getByLabel('Monitoring scenario').selectOption('Mental Demand');
  await expect(page.locator('#workerList .worker-card').first()).toBeVisible();
  await expect(page.getByRole('button', { name: 'Connect Polar H10' })).toBeEnabled();
  const panel = await page.locator('#polar-panel').boundingBox();
  expect(panel.x).toBeGreaterThanOrEqual(0);
  expect(panel.x + panel.width).toBeLessThanOrEqual(390);
  await page.screenshot({ path: test.info().outputPath('polar-mobile.png') });
});

test('alternate historical page loads the same sensor panel', async ({ page }) => {
  await page.goto('/psychosocial_dashboard_v2.html');
  await expect(page.getByRole('heading', { name: 'EXG · 200 ms RMS' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Connect Polar H10' })).toBeEnabled();
});
