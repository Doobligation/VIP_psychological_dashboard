// Serial, idempotent delivery. Queue limits and failures remain visible to the wearer.
export class ClosedLoopClient {
  constructor({ fetchImpl = (...args) => globalThis.fetch(...args), onChange = () => {}, maxQueue = 256, retryMs = 2000 } = {}) {
    Object.assign(this, { fetchImpl, onChange, maxQueue, retryMs });
    this.epoch = 0; this.queue = []; this.latest = null; this.saved = 0; this.dropped = 0;
    this.error = ''; this.session = null; this.counter = 0; this.inFlight = null;
  }
  start(session) {
    // Keep earlier sessions' pending packets: their identities remain immutable.
    // Only the selected session's accepted responses may update its current decision.
    this.epoch++;
    this.session = { ...session }; this.latest = null;
    this.saved = 0; this.counter = 0;
    this.onChange(this);
    void this.flush();
  }
  sample(snapshot, raw, timestamp_ms) {
    if (!this.session) return null;
    const sample = structuredClone({ ...this.session, sample_id: `${this.session.session_id}-${++this.counter}`,
      timestamp_ms, snapshot, raw });
    if (this.queue.length >= this.maxQueue) {
      const index = this.queue.findIndex(row => row !== this.inFlight);
      if (index < 0) { this.dropped++; this.onChange(this); return null; }
      this.queue.splice(index, 1); this.dropped++;
    }
    this.queue.push(sample); this.onChange(this); void this.flush(); return sample.sample_id;
  }
  async flush() {
    if (this.inFlight || !this.queue.length || this.retry) return;
    const sample = this.queue[0]; this.inFlight = sample;
    this.abort = new AbortController();
    const abort = this.abort;
    const timeout = setTimeout(() => abort.abort(), 10000);
    try {
      const response = await this.fetchImpl('/samples', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sample), signal: this.abort.signal });
      const value = await response.json();
      if (this.inFlight !== sample) return;
      if (!response.ok) {
        if (response.status >= 400 && response.status < 500) { this.queue = this.queue.filter(row => row !== sample); this.dropped++; }
        throw new Error(value.error || `HTTP ${response.status}`);
      }
      this.queue = this.queue.filter(row => row !== sample); this.error = '';
      if (sample.session_id === this.session?.session_id) {
        this.latest = value; this.saved++; this.receivedAt = Date.now();
      }
    } catch (error) {
      if (this.inFlight !== sample) return;
      this.error = `Local storage unavailable: ${error.message}`;
      this.retry = setTimeout(() => { this.retry = null; void this.flush(); }, this.retryMs);
    } finally {
      clearTimeout(timeout);
      if (this.inFlight === sample) { this.inFlight = null; this.onChange(this); if (!this.retry) void this.flush(); }
    }
  }
  async feedback(fields) {
    if (!this.session) throw new Error('Start a session first.');
    const response = await this.fetchImpl('/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: this.session.session_id, ...fields }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    return result;
  }
  dispose() { this.epoch++; clearTimeout(this.retry); this.retry = null; this.abort?.abort(); this.inFlight = null; this.queue = []; this.session = null; }
}
