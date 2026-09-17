import { PolarH10, SERVICES } from 'polar-h10';

export { SERVICES };

// Upstream registers bound native listeners without a dispose API. Track them so
// reconnecting does not leave old drivers subscribed to the same characteristics.
export function createDriver(device, onPacketError = () => {}) {
  let active = true;
  const listeners = [];
  const proxies = new WeakMap();
  const wrap = (target) => {
    if (proxies.has(target)) return proxies.get(target);
    const proxy = new Proxy(target, {
      get(object, key) {
        if (key === 'gatt') return wrap(object.gatt);
        if (key === 'addEventListener') return (type, callback, options) => {
          if (!active) return;
          const guarded = (event) => {
            if (!active) return;
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
            if (!active) throw new Error('Connection cancelled');
            const result = await object[key](...args);
            if (!active) {
              device.gatt?.disconnect();
              throw new Error('Connection cancelled');
            }
            return wrap(result);
          };
        }
        const value = Reflect.get(object, key, object);
        if (typeof value !== 'function') return value;
        return (...args) => {
          if (!active) throw new Error('Connection cancelled');
          return value.apply(object, args);
        };
      },
    });
    proxies.set(target, proxy);
    return proxy;
  };
  const driver = new PolarH10(wrap(device), false);
  return {
    driver,
    dispose() {
      active = false;
      for (const { object, type, guarded } of listeners) object.removeEventListener(type, guarded);
      listeners.length = 0;
      driver.heartRateHandleList.length = 0;
      for (const handlers of Object.values(driver.dataHandleDict)) handlers.length = 0;
    },
  };
}
