import { test, expect } from '@playwright/test';

// Substitute browser Bluetooth transport, while running the shipped bundle,
// Polar protocol decoder, filter, controller, canvas and dashboard unchanged.
function installBluetooth() {
  const characteristics = new Map();
  let stream;
  let hrStream;
  let frameIndex = 0;
  function notify(characteristic) {
    const event = new Event('characteristicvaluechanged');
    // Keep PMD receive time on the same clock as Date during virtual-time tests.
    Object.defineProperty(event, 'timeStamp', { value: Date.now() - performance.timeOrigin });
    characteristic.dispatchEvent(event);
  }
  class Characteristic extends EventTarget {
    async startNotifications() { return this; }
    async readValue() { return new DataView(new ArrayBuffer(1)); }
    async writeValueWithResponse(buffer) {
      const command = new Uint8Array(buffer);
      if (command[0] === 2 && command[1] === 2) {
        this.value = new DataView(new Uint8Array([240, 2, 2, 0, 0]).buffer);
        this.dispatchEvent(new Event('characteristicvaluechanged'));
        return;
      }
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
        notify(data);
      }, 200);
    }
  }
  const device = new EventTarget();
  function sendHeartRate(hr, intervals, contact = true) {
    const characteristic = characteristics.get('00002a37-0000-1000-8000-00805f9b34fb');
    const packet = new DataView(new ArrayBuffer(2 + intervals.length * 2));
    packet.setUint8(0, (intervals.length ? 16 : 0) | (contact ? 6 : 4));
    packet.setUint8(1, hr);
    intervals.forEach((rr, index) => packet.setUint16(2 + index * 2, Math.round(rr * 1024 / 1000), true));
    characteristic.value = packet;
    characteristic.dispatchEvent(new Event('characteristicvaluechanged'));
  }
  device.name = 'Polar H10 simulated';
  device.gatt = {
    connected: false,
    async connect() { this.connected = true; return this; },
    disconnect() {
      clearInterval(stream);
      clearInterval(hrStream);
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
  window.polarTest = { device, stop: () => clearInterval(stream),
    acceleration() {
      const characteristic = characteristics.get('fb005c82-02e7-f387-1cad-8acd2d8df0c8');
      const packet = new DataView(new ArrayBuffer(16));
      packet.setUint8(0, 2);
      packet.setUint8(9, 1);
      packet.setBigUint64(1, 1000000000n + BigInt(frameIndex - 1) * 200000000n, true);
      packet.setInt16(14, 1000, true);
      characteristic.value = packet;
      notify(characteristic);
    },
    heartRate(hr = 75, rr = [760, 840], contact = true) {
      clearInterval(hrStream);
      hrStream = setInterval(() => sendHeartRate(hr, rr, contact), 1000);
    },
  };
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
  await expect(page.locator('#polar-fatigue-level')).toHaveText('Unavailable');
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

test('Group 1 demo evaluates rules, exports labeled data and resets before live use', async ({ page }) => {
  await page.addInitScript(() => {
    window.spokenCues = [];
    Object.defineProperty(window, 'speechSynthesis', { value: {
      cancel() {}, speak(utterance) { window.spokenCues.push(utterance.text); },
    } });
  });
  await page.goto('/psychosocial_dashboard.html');
  await page.locator('#polar-participant').fill('Group1-test');
  await page.getByLabel('Enable spoken feedback').check();
  await page.getByRole('button', { name: 'Start demo', exact: true }).click();
  await expect(page.locator('#polar-study-mode')).toContainText('DEMO');
  await expect(page.locator('#polar-baseline-status')).toContainText(/ready/i);
  await expect(page.locator('#polar-participant')).toBeDisabled();
  await expect(page.locator('#polar-fatigue-level')).toHaveText('Medium fatigue', { timeout: 15000 });
  // Medium never gives feedback; High does after consecutive High predictions.
  await expect(page.locator('#polar-fatigue-decision')).toContainText('No feedback at this level');
  await expect(page.locator('#polar-feedback')).toHaveText('No feedback issued yet.');
  expect(await page.evaluate(() => window.spokenCues)).toEqual([]);
  await expect(page.locator('#polar-fatigue-level')).toHaveText('High fatigue', { timeout: 15000 });
  await expect(page.locator('#polar-feedback')).toContainText('recovery break', { timeout: 15000 });
  const cues = await page.evaluate(() => window.spokenCues);
  expect(cues.filter(cue => cue.includes('recovery break'))).toHaveLength(1);
  expect(cues.every(cue => cue.startsWith('Demonstration. '))).toBe(true);
  await page.locator('#polar-study').screenshot({ path: test.info().outputPath('group1-demo.png') });
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export session JSON' }).click();
  const download = await downloadPromise;
  const stream = await download.createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  const log = JSON.parse(Buffer.concat(chunks).toString());
  expect(log.mode).toBe('demo');
  expect(log.participant).toBe('Group1-test');
  expect(log.observations.length).toBeGreaterThan(0);
  expect(log.baseline.hr).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'Stop demo', exact: true }).click();
  await expect(page.locator('#polar-fatigue-level')).toHaveText('Unavailable');
  await expect(page.locator('#polar-participant')).toBeEnabled();
  await expect(page.locator('#polar-baseline-status')).toContainText(/baseline/i);
});

test('live HR/RR packets capture a baseline, missing RR suppresses classification, reconnect clears it', async ({ page }) => {
  await page.clock.install();
  await page.addInitScript(installBluetooth);
  await page.goto('/psychosocial_dashboard.html');
  await page.getByRole('button', { name: 'Connect Polar H10' }).click();
  await page.clock.runFor(2000);
  await expect(page.locator('#polar-status')).toContainText('Receiving EXG');
  await page.evaluate(() => window.polarTest.heartRate());
  await page.getByRole('button', { name: 'Capture resting baseline (5s)', exact: true }).click();
  await page.clock.runFor(7000);
  await expect(page.locator('#polar-baseline-status')).toContainText('ready');
  await expect(page.locator('#polar-fatigue-level')).toHaveText('Low fatigue');
  await expect(page.locator('#polar-hr')).toHaveText('75.0 bpm');
  await page.evaluate(() => window.polarTest.heartRate(100, []));
  await page.clock.runFor(7000);
  await expect(page.locator('#polar-fatigue-level')).toHaveText('Unavailable');
  await expect(page.locator('#polar-feedback')).toHaveText('No feedback issued yet.');
  await page.evaluate(() => window.polarTest.acceleration());
  await page.getByRole('button', { name: 'Calibrate upright posture' }).click();
  await expect(page.locator('#polar-calibration-status')).toContainText('Upright reference captured');
  await page.getByRole('button', { name: 'Capture resting baseline (5s)', exact: true }).click();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export session JSON' }).click();
  const stream = await (await downloadPromise).createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  expect(JSON.parse(Buffer.concat(chunks).toString()).baseline).toBeNull();
  await page.evaluate(() => window.polarTest.device.gatt.disconnect());
  await page.clock.runFor(3000);
  await expect(page.locator('#polar-baseline-status')).toContainText('Start a resting baseline');
  await expect(page.locator('#polar-fatigue-level')).toHaveText('Unavailable');
  await expect(page.locator('#polar-calibration-status')).not.toContainText('Upright reference captured');
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
  await page.locator('#polar-study').screenshot({ path: test.info().outputPath('group1-mobile.png') });
});

test('alternate historical page loads the same sensor panel', async ({ page }) => {
  await page.goto('/psychosocial_dashboard_v2.html');
  await expect(page.getByRole('heading', { name: 'EXG · 200 ms RMS' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Connect Polar H10' })).toBeEnabled();
});

test('quick fatigue score appears immediately from HR-only contact-warning packets and expires', async ({ page }) => {
  await page.clock.install();
  await page.addInitScript(installBluetooth);
  await page.goto('/psychosocial_dashboard.html');
  await page.getByRole('button', { name: 'Connect Polar H10' }).click();
  await page.clock.runFor(2000);
  await page.evaluate(() => window.polarTest.heartRate(111, [], false));
  await page.clock.runFor(1500);
  await expect(page.locator('#polar-quick-score')).toHaveText('64 / 100');
  await expect(page.locator('#polar-quick-status')).toContainText('Contact warning bypassed');
  await expect(page.locator('#polar-quick-card')).toContainText('not a measured fatigue percentage');
  await expect(page.locator('#polar-baseline-status')).toContainText('Start a resting baseline');
  await expect(page.locator('#polar-fatigue-level')).toHaveText('Unavailable');
  await page.locator('#polar-quick-card').screenshot({ path: test.info().outputPath('quick-fatigue.png') });
  const pending = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export session JSON' }).click();
  const chunks = [];
  for await (const chunk of await (await pending).createReadStream()) chunks.push(chunk);
  const log = JSON.parse(Buffer.concat(chunks).toString());
  const row = log.observations.find(s => s.quickFatigue?.score === 64);
  expect(row.quickFatigue.source).toBe('hr-only-debug');
  expect(row.quickFatigue.provisional).toBe(true);
  expect(row.fatigue.level).toBe('unavailable');
  await page.evaluate(() => window.polarTest.heartRate(0, []));
  await page.clock.runFor(3000);
  await expect(page.locator('#polar-quick-score')).toHaveText('64 / 100');
  await expect(page.locator('#polar-quick-input')).toContainText('Holding last reading');
  await page.clock.runFor(11000);
  await expect(page.locator('#polar-quick-score')).toHaveText('—');
  await expect(page.locator('#polar-quick-status')).toContainText('No recent nonzero heart rate');
});

test('incoming HR diagnostics and exported raw packets explain unavailable fatigue', async ({ page }) => {
  await page.clock.install();
  await page.addInitScript(installBluetooth);
  await page.goto('/psychosocial_dashboard.html');
  await page.getByRole('button', { name: 'Connect Polar H10' }).click();
  await page.clock.runFor(2000);
  await page.evaluate(() => window.polarTest.heartRate(0, []));
  await page.clock.runFor(3000);
  await expect(page.locator('#polar-heart-signal')).toContainText('0 bpm');
  await expect(page.locator('#polar-fatigue-cause')).toContainText('0 bpm');
  await expect(page.locator('#polar-hr')).toHaveText('—');
  await expect(page.locator('#loop-evidence')).toContainText('0 bpm');
  await page.evaluate(() => window.polarTest.heartRate(105, [572, 600], false));
  await page.clock.runFor(3000);
  await expect(page.locator('#polar-heart-signal')).toContainText('105.0 bpm');
  await expect(page.locator('#polar-fatigue-cause')).toContainText('contact');
  const pending = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export session JSON' }).click();
  const chunks = [];
  for await (const chunk of await (await pending).createReadStream()) chunks.push(chunk);
  const log = JSON.parse(Buffer.concat(chunks).toString());
  expect(log.observations.some(s => s.rawHeartRate?.heart_rate_bpm === 0)).toBe(true);
  const rejected = log.observations.find(s => s.rawHeartRate?.contact === false);
  expect(rejected.rawHeartRate.rr_intervals_ms).toHaveLength(2);
  expect(rejected.heartSignal.status).toBe('contact-lost');
  expect(rejected.hr).toBeNull();
  await page.evaluate(() => window.polarTest.heartRate());
  await page.getByRole('button', { name: 'Capture resting baseline (5s)', exact: true }).click();
  await page.clock.runFor(7000);
  await expect(page.locator('#polar-baseline-status')).toContainText('baseline ready');
  await expect(page.locator('#polar-fatigue-level')).toHaveText('Low fatigue');
});

test('rule-only fatigue panel records independent ratings and exports their exact measurements', async ({ page }) => {
  await page.clock.install();
  await page.addInitScript(installBluetooth);
  await page.goto('/psychosocial_dashboard.html');
  await expect(page.locator('#polar-fatigue-method')).toContainText('No ML model');
  await expect(page.locator('#polar-model-file')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Record fatigue rating', exact: true })).toBeDisabled();
  await page.locator('#polar-participant').fill('P01');
  await page.getByRole('button', { name: 'Connect Polar H10' }).click();
  await page.clock.runFor(2000);
  await page.evaluate(() => window.polarTest.heartRate());
  await page.getByRole('button', { name: 'Capture resting baseline (5s)', exact: true }).click();
  await page.clock.runFor(7000);
  await expect(page.locator('#polar-model-file')).toHaveCount(0);

  await page.getByLabel('How fatigued do you feel?').selectOption('high');
  await page.getByLabel('Current task').fill('Drywall installation');
  await page.getByRole('button', { name: 'Record fatigue rating', exact: true }).click();
  await expect(page.locator('#polar-rating-status')).toContainText('1 rating');
  await expect(page.getByLabel('How fatigued do you feel?')).toHaveValue('');
  await expect(page.getByRole('button', { name: 'Record fatigue rating', exact: true })).toBeDisabled();
  await page.locator('#polar-ratings').screenshot({ path: test.info().outputPath('polar-model.png') });
  await page.clock.runFor(60000);
  await page.getByLabel('How fatigued do you feel?').selectOption('low');
  await page.getByRole('button', { name: 'Record fatigue rating', exact: true }).click();
  const pending = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export session JSON' }).click();
  const stream = await (await pending).createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  const log = JSON.parse(Buffer.concat(chunks).toString());
  expect(log.schemaVersion).toBe(2);
  expect(log.sessionId).toBeTruthy();
  expect(log.fatigueRatings).toHaveLength(2);
  expect(log.fatigueRatings[0].level).toBe('high');
  expect(log.fatigueRatings[0].snapshot.fatigue.level).toBe('low');
  expect(log.fatigueRatings[0].snapshot.hr).toBe(75);
  expect(log.fatigueRatings[0].source).toBe('participant-self-report');
  expect(log.fatigueRatings[0].task).toBe('Drywall installation');
  expect(log.models).toBeUndefined();
  expect(log.observations.at(-1).mlPrediction).toBeUndefined();
  await page.evaluate(() => window.polarTest.heartRate(100, []));
  await page.clock.runFor(7000);
  await expect(page.locator('#polar-model-file')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Record fatigue rating', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Disconnect', exact: true }).click();
  await expect(page.locator('#polar-model-file')).toHaveCount(0);
});

test('fatigue uses the newest three rules without loading a model', async ({ page }) => {
  const modelRequests = [];
  page.on('request', request => { if (request.url().includes('/models/')) modelRequests.push(request.url()); });
  await page.goto('/psychosocial_dashboard.html');
  await expect(page.locator('#polar-fatigue-method')).toContainText('Rule-based');
  await expect(page.getByLabel('Load trained model JSON')).toHaveCount(0);
  await expect(page.locator('#polar-rules-table tr')).toHaveCount(3);
  await expect(page.locator('#polar-rules-table')).toContainText('hrChangePct < 10%');
  await expect(page.locator('#polar-rules-table')).toContainText('edaChangePct < 15%');
  await expect(page.locator('#polar-eda-note')).toContainText('does not measure EDA');
  await page.getByRole('button', { name: 'Start demo', exact: true }).click();
  await expect(page.locator('#polar-fatigue-level')).toHaveText('High fatigue', { timeout: 20000 });
  await expect(page.locator('#polar-eda-note')).toContainText('simulated');
  expect(modelRequests).toEqual([]);
});

test('restored page retains rule-only controls after pagehide', async ({ page }) => {
  await page.goto('/psychosocial_dashboard.html');
  await page.evaluate(() => {
    window.dispatchEvent(new PageTransitionEvent('pagehide', { persisted: true }));
    window.dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true }));
  });
  await expect(page.locator('#polar-fatigue-method')).toContainText('Rule-based');
  await expect(page.getByLabel('Load trained model JSON')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Start demo', exact: true })).toBeEnabled();
});

test('demo saves readings, feedback responses and outcomes through the actual local API', async ({ page, request }) => {
  await page.goto('/psychosocial_dashboard.html');
  // Empty participant IDs are assigned a synthetic session identity for demos.
  await page.getByRole('button', { name: 'Start demo', exact: true }).click();
  await expect(page.locator('#loop-storage')).toContainText(/[1-9]\d* readings saved/, { timeout: 10000 });
  await expect(page.locator('#loop-alert')).toContainText('Last feedback', { timeout: 15000 });
  await page.getByRole('button', { name: 'Acknowledge feedback' }).click();
  await expect(page.locator('#loop-feedback-status')).toHaveText('Response saved.');
  await page.getByRole('button', { name: 'Better', exact: true }).click();
  await expect(page.locator('#loop-feedback-status')).toHaveText('Outcome saved.');
  await page.locator('#polar-loop').screenshot({ path: test.info().outputPath('closed-loop.png') });
  const href = await page.locator('#loop-export').getAttribute('href');
  const response = await request.get(href);
  expect(response.ok()).toBe(true);
  const log = await response.json();
  expect(log.mode).toBe('demo');
  expect(log.participant).toMatch(/^demo-/);
  expect(log.readings.length).toBeGreaterThan(0);
  expect(log.alerts.length).toBeGreaterThan(0);
  expect(log.responses).toHaveLength(1);
  expect(log.outcomes).toHaveLength(1);
  expect(log.fatigueRatings).toHaveLength(0);
  expect(log.responses[0].alert_id).toBe(log.outcomes[0].alert_id);
  await page.getByRole('button', { name: 'Stop demo', exact: true }).click();
  await expect(page.locator('#loop-decision')).toContainText('Waiting for fresh');
  await expect(page.locator('#polar-study-mode')).toContainText('LIVE');
  await page.reload();
  await expect(page.locator('#loop-export')).toHaveAttribute('href', href);
});

test('failed sample delivery stays visible and never claims a saved session', async ({ page }) => {
  await page.route('**/samples', route => route.fulfill({ status: 503, contentType: 'application/json', body: '{"error":"test storage unavailable"}' }));
  await page.goto('/psychosocial_dashboard.html');
  await page.getByRole('button', { name: 'Start demo', exact: true }).click();
  await expect(page.locator('#loop-storage')).toContainText('0 readings saved');
  await expect(page.locator('#loop-storage')).toContainText('test storage unavailable');
  await expect(page.locator('#loop-export')).toBeHidden();
  await expect(page.getByRole('button', { name: 'Acknowledge feedback' })).toBeDisabled();
});
