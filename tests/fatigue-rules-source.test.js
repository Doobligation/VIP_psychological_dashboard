import test from 'node:test';
import assert from 'node:assert/strict';
import { evaluateRule } from '../sensor/rule-config.js';
import { classifyFatigue, FatigueSession } from '../sensor/fatigue.js';

test('newest fatigue section uses HRV for low/medium and low HR plus EDA for high', () => {
  for (const [features, level] of [
    [{ rmssdChangePct: -15, sdnnChangePct: -16 }, 'low'],
    [{ rmssdChangePct: -16, sdnnChangePct: -15 }, 'low'],
    [{ rmssdChangePct: -16, sdnnChangePct: -16 }, 'low'],
    [{ rmssdChangePct: -25, sdnnChangePct: -21 }, 'low'],
    [{ rmssdChangePct: -26, sdnnChangePct: -20 }, 'low'],
    [{ rmssdChangePct: -26, sdnnChangePct: -21 }, 'medium'],
    [{ rmssdChangePct: -36, sdnnChangePct: -31, hrChangePct: 9, edaChangePct: 14 }, 'high'],
    [{ rmssdChangePct: -35, sdnnChangePct: -31, hrChangePct: 9, edaChangePct: 14 }, 'medium'],
    [{ rmssdChangePct: -36, sdnnChangePct: -30, hrChangePct: 9, edaChangePct: 14 }, 'medium'],
    [{ rmssdChangePct: -36, sdnnChangePct: -31, hrChangePct: 10, edaChangePct: 14 }, 'medium'],
    [{ rmssdChangePct: -36, sdnnChangePct: -31, hrChangePct: 9, edaChangePct: 15 }, 'medium'],
  ]) assert.equal(evaluateRule('fatigue', features).level, level, JSON.stringify(features));
});

test('a separately measured EDA stream needs its own full window and baseline', () => {
  const session = new FatigueSession(); session.startBaseline(0);
  for (let t=0; t<=60000; t+=1000) {
    session.electrodermal({ eda_us: 2, recv_epoch_time_ms: t });
    session.heartRate({ heart_rate_bpm:100, rr_intervals_ms:[700,900], recv_epoch_time_ms:t });
  }
  for (let t=61000; t<=130000; t+=1000) {
    session.electrodermal({ eda_us:2.2, recv_epoch_time_ms:t });
    session.heartRate({ heart_rate_bpm:100, rr_intervals_ms:[780,820], recv_epoch_time_ms:t });
  }
  assert.equal(session.snapshot(130000).fatigue.level,'high');
  session.heartRate({ heart_rate_bpm:100, rr_intervals_ms:[780,820], recv_epoch_time_ms:135001 });
  assert.notEqual(session.snapshot(135001).fatigue.level,'high');
});

test('missing EDA reports the matched medium rule and the unassessed high rule', () => {
  const result = evaluateRule('fatigue', { rmssdChangePct: -50, sdnnChangePct: -50, hrChangePct: 0 });
  assert.equal(result.level, 'medium');
  assert.deepEqual(result.unassessedLevels, [{ level: 'high', missingSignals: ['edaChangePct'] }]);
  assert.match(result.reason, /EDA/);
  assert.equal(evaluateRule('fatigue', { rmssdChangePct: -50 }).level, 'unavailable');
});

test('measured EDA is baseline-relative and cannot be substituted when stale or zero-baseline', () => {
  const baseline = { hr: 100, rmssd: 100, sdnn: 100, eda: 2 };
  const current = { hr: 100, rmssd: 50, sdnn: 50, eda: 2.2, edaFresh: true };
  assert.equal(classifyFatigue(current, baseline).level, 'high');
  assert.equal(classifyFatigue({ ...current, edaFresh: false }, baseline).level, 'medium');
  assert.equal(classifyFatigue(current, { ...baseline, eda: 0 }).level, 'medium');
});
