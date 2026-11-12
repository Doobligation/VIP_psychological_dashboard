import test from 'node:test';
import assert from 'node:assert/strict';
import { ClosedLoopClient } from '../sensor/closed-loop-client.js';

const tick = () => new Promise(resolve => setImmediate(resolve));
test('retry keeps sample identity and bounds pending readings without dropping the active request', async () => {
  const requests = []; let release;
  const client = new ClosedLoopClient({ maxQueue: 2, retryMs: 10,
    fetchImpl: async (url, options) => { requests.push(JSON.parse(options.body));
      if (requests.length === 1) await new Promise(resolve => { release = resolve; });
      return { ok: true, json: async () => ({ reading_id: requests.at(-1).sample_id, alerts: [] }) }; } });
  client.start({ session_id: 's', participant: 'P01', mode: 'live' });
  client.sample({ fresh: true }, {}, 1);
  await tick();
  client.sample({}, {}, 2); client.sample({}, {}, 3);
  assert.equal(client.dropped, 1);
  release(); await tick(); await tick();
  assert.deepEqual(requests.map(r => r.timestamp_ms), [1, 3]);
  assert.equal(client.saved, 2);
  client.dispose();
});

test('switching sessions preserves pending readings and ignores their decisions in the new session', async () => {
  let release;
  const client = new ClosedLoopClient({ fetchImpl: async () => {
    await new Promise(resolve => { release = resolve; });
    return { ok: true, json: async () => ({ reading_id: 'old', alerts: [{ id: 'old-alert' }] }) };
  } });
  client.start({ session_id: 'old', participant: 'P01', mode: 'live' });
  client.sample({}, {}, 1); await tick();
  client.start({ session_id: 'new', participant: 'P02', mode: 'live' });
  assert.equal(client.queue.length, 1);
  release(); await tick();
  assert.equal(client.queue.length, 0); assert.equal(client.saved, 0); assert.equal(client.latest, null);
  client.dispose();
});

test('failed delivery retries exactly once per attempt and reset ignores obsolete results', async () => {
  const ids = []; let fail = true;
  const client = new ClosedLoopClient({ retryMs: 1, fetchImpl: async (url, options) => {
    const sample = JSON.parse(options.body); ids.push(sample.sample_id);
    if (fail) { fail = false; throw new Error('offline'); }
    return { ok: true, json: async () => ({ reading_id: 'saved', alerts: [] }) };
  } });
  client.start({ session_id: 's', participant: 'P01', mode: 'live' }); client.sample({}, {}, 1);
  await new Promise(resolve => setTimeout(resolve, 30));
  assert.equal(ids.length, 2); assert.equal(ids[0], ids[1]); assert.equal(client.saved, 1);
  client.start({ session_id: 'next', participant: 'P02', mode: 'demo' });
  assert.equal(client.latest, null); assert.equal(client.saved, 0);
  client.dispose();
});
