import { createDriver, SERVICES } from './driver.js';
import { PolarData } from './data.js';

export class PolarController {
  constructor({ bluetooth = globalThis.navigator?.bluetooth, makeDriver = createDriver,
    data = new PolarData(), onChange = () => {}, retryBaseMs = 1000,
    maxRetries = 5, timeoutMs = 20000 } = {}) {
    Object.assign(this, { bluetooth, makeDriver, data, onChange, retryBaseMs, maxRetries, timeoutMs });
    this.state = 'idle';
    this.message = 'No Polar H10 connected.';
    this.generation = 0;
    this.attempt = 0;
    this.wanted = false;
    this.busy = false;
    this.lost = () => {
      if (!this.wanted || this.busy) return;
      this.release();
      this.data.reset();
      this.scheduleRetry('Sensor disconnected.');
    };
  }

  status(state, message) {
    this.state = state;
    this.message = message;
    this.onChange(this);
  }

  async connect() {
    if (this.busy || this.wanted) return;
    if (!this.bluetooth) return this.status('error', 'Web Bluetooth requires Chrome or Edge on a supported platform, using localhost or HTTPS.');
    this.busy = true;
    const generation = ++this.generation;
    this.status('connecting', 'Select a Polar H10 in the Bluetooth chooser.');
    try {
      // Reused from polar-h10-visualizer/src/index.ts.
      const device = await this.bluetooth.requestDevice({ filters: [{ namePrefix: 'Polar H10' }], optionalServices: SERVICES });
      if (generation !== this.generation) return;
      this.device = device;
      this.device.addEventListener('gattserverdisconnected', this.lost);
      this.wanted = true;
      this.attempt = 0;
    } catch (error) {
      if (generation === this.generation) this.status('error', error.name === 'NotFoundError' ? 'No device selected. Connect to try again.' : `Bluetooth selection failed: ${error.message}`);
    } finally {
      this.busy = false;
    }
    if (generation === this.generation && this.wanted) await this.open(generation);
  }

  async open(generation) {
    if (this.busy || !this.wanted || generation !== this.generation) return;
    this.busy = true;
    this.data.reset();
    this.status(this.attempt ? 'reconnecting' : 'connecting', 'Initializing Polar H10…');
    let failure;
    let timer;
    const scope = this.makeDriver(this.device, () => {
      this.packetErrors = (this.packetErrors || 0) + 1;
    });
    this.scope = scope;
    const current = () => this.wanted && generation === this.generation && this.scope === scope;
    const check = () => {
      if (!current() || !this.device.gatt?.connected) throw new Error('Sensor disconnected or connection cancelled');
    };
    try {
      const setup = async () => {
        const p = scope.driver;
        p.addEventListener('ECG', (frame) => {
          if (current()) {
            this.data.waveform(frame);
            if (this.data.fresh()) this.attempt = 0;
          }
        });
        await p.init(100);
        check();
        const reply = await p.startECG(130);
        if (!['SUCCESS', 'ALREADY IN STATE'].includes(reply?.error))
          throw new Error(reply?.error || 'No EXG start acknowledgement');
        check();
      };
      await Promise.race([setup(), new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('Connection timed out')), this.timeoutMs);
      })]);
      if (current()) this.status('connected', `Connected to ${this.device.name || 'Polar H10'}. Waiting for samples.`);
    } catch (error) {
      failure = error;
      if (current()) {
        this.release();
        this.device.gatt?.disconnect();
      }
    } finally {
      clearTimeout(timer);
      this.busy = false;
    }
    if (failure && generation === this.generation && this.wanted) this.scheduleRetry(failure.message);
  }

  scheduleRetry(reason) {
    if (this.retryTimer || !this.wanted) return;
    if (this.attempt >= this.maxRetries) {
      this.disconnect();
      this.status('error', `${reason} Reconnection attempts exhausted. Check battery/range and OS pairing (firmware 4+), then Connect again.`);
      return;
    }
    const delay = Math.min(this.retryBaseMs * 2 ** this.attempt, 16000);
    this.attempt++;
    this.status('reconnecting', `${reason} Retrying ${this.attempt}/${this.maxRetries} in ${(delay / 1000).toFixed(1)}s.`);
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      void this.open(this.generation);
    }, delay);
  }

  release() {
    this.scope?.dispose();
    this.scope = null;
  }

  disconnect() {
    this.wanted = false;
    ++this.generation;
    clearTimeout(this.retryTimer);
    this.retryTimer = null;
    this.release();
    this.device?.removeEventListener('gattserverdisconnected', this.lost);
    this.device?.gatt?.disconnect();
    this.data.reset();
    this.status('idle', 'Disconnected. Connect to start a new sensor session.');
  }
}
