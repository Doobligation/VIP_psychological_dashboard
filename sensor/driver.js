import { PolarH10, SERVICES } from 'polar-h10';

export { SERVICES };

const deviceOwners = new WeakMap();

// Bluetooth Heart Rate Measurement: RR is in 1/1024 seconds on the wire.
// The upstream decoder does not skip the optional Energy Expended field.
export function decodeHeartRate(value, receivedAt = Date.now()) {
  const requireBytes = (offset, size) => {
    if (!value || offset + size > value.byteLength) throw new Error('Truncated heart-rate measurement');
  };
  requireBytes(0, 1);
  const flags = value.getUint8(0);
  let offset = 1;
  requireBytes(offset, flags & 1 ? 2 : 1);
  const heart_rate_bpm = flags & 1 ? value.getUint16(offset, true) : value.getUint8(offset);
  offset += flags & 1 ? 2 : 1;
  if (flags & 8) { requireBytes(offset, 2); offset += 2; }
  const rr_intervals_ms = [];
  if (flags & 16) {
    while (offset < value.byteLength) {
      requireBytes(offset, 2);
      rr_intervals_ms.push(value.getUint16(offset, true) * 1000 / 1024);
      offset += 2;
    }
  }
  return { heart_rate_bpm, rr_intervals_ms, recv_epoch_time_ms: receivedAt,
    contact: flags & 4 ? Boolean(flags & 2) : null };
}

// Upstream registers bound native listeners without a dispose API. Track them so
// reconnecting does not leave old drivers subscribed to the same characteristics.
export function createDriver(device, onPacketError = () => {}) {
  const owner = { active: true };
  deviceOwners.set(device, owner);
  const listeners = [];
  const proxies = new WeakMap();
  const wrap = (target) => {
    if (proxies.has(target)) return proxies.get(target);
    const proxy = new Proxy(target, {
      get(object, key) {
        if (key === 'gatt') return wrap(object.gatt);
        if (key === 'addEventListener') return (type, callback, options) => {
          if (!owner.active) return;
          const guarded = (event) => {
            if (!owner.active) return;
            try { callback(event); } catch (error) { onPacketError(error); }
          };
          listeners.push({ object, type, callback, guarded });
          object.addEventListener(type, guarded, options);
        };
        if (key === 'removeEventListener') return (type, callback) => {
          for (const entry of listeners) {
            if (entry.object === object && entry.type === type && entry.callback === callback)
              object.removeEventListener(type, entry.guarded);
          }
        };
        if (['connect', 'getPrimaryService', 'getCharacteristic'].includes(key)) {
          return async (...args) => {
            if (!owner.active) throw new Error('Connection cancelled');
            const result = await object[key](...args);
            if (!owner.active) {
              // A timed-out operation may settle after a retry already owns the
              // same GATT server. Only close it when no active session owns it.
              if (!deviceOwners.get(device)?.active) device.gatt?.disconnect();
              throw new Error('Connection cancelled');
            }
            return wrap(result);
          };
        }
        const value = Reflect.get(object, key, object);
        if (typeof value !== 'function') return value;
        return (...args) => {
          if (!owner.active) throw new Error('Connection cancelled');
          return value.apply(object, args);
        };
      },
    });
    proxies.set(target, proxy);
    return proxy;
  };
  const driver = new PolarH10(wrap(device), false);
  driver.HearRateHandle = (event) => {
    const measurement = decodeHeartRate(event.target.value);
    for (const handler of driver.heartRateHandleList) handler(measurement);
  };
  return {
    driver,
    dispose() {
      owner.active = false;
      for (const { object, type, guarded } of listeners) object.removeEventListener(type, guarded);
      listeners.length = 0;
      driver.heartRateHandleList.length = 0;
      for (const handlers of Object.values(driver.dataHandleDict)) handlers.length = 0;
    },
  };
}
