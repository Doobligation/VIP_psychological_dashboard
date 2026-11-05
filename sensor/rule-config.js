import defaultRules from '../config/rules.json' with { type: 'json' };
import defaultFeedback from '../config/feedback_table.json' with { type: 'json' };

let active = { rules: structuredClone(defaultRules), feedback: structuredClone(defaultFeedback) };
const operators = { '>': (a, b) => a > b, '<': (a, b) => a < b,
  '>=': (a, b) => a >= b, '<=': (a, b) => a <= b, '==': (a, b) => a === b, '!=': (a, b) => a !== b };
export const getConfiguration = () => structuredClone(active);
export const featurePolicy = () => active.rules.featurePolicy;
// Assignment rule: only High risk speaks, after this many consecutive High predictions.
export const consecutiveHighRequired = () => active.feedback.decision.consecutiveHighPredictions;

export function configureRules(rules, feedback) {
  if (rules?.schemaVersion !== 1 || feedback?.schemaVersion !== 2 || !rules.version || !feedback.version
    || !rules.featurePolicy || !feedback.decision || !Number.isInteger(feedback.decision.consecutiveHighPredictions)
    || feedback.decision.consecutiveHighPredictions < 1) throw new Error('Invalid rule configuration.');
  for (const state of ['frustration', 'fatigue', 'posture']) {
    const config = rules.states?.[state];
    if (!config || !Array.isArray(config.rules) || !Array.isArray(config.requiredSignals)
      || new Set(config.requiredSignals).size !== config.requiredSignals.length
      || !Number.isInteger(config.minEvidence) || config.minEvidence < 1 || config.minEvidence > config.requiredSignals.length || !feedback.states?.[state])
      throw new Error(`Invalid ${state} configuration.`);
    for (const rule of config.rules) {
      // A signal may appear twice in one rule to express a two-sided band (e.g. "slight" increases).
      if (!['all', 'any'].includes(rule.match) || !Array.isArray(rule.conditions)
        || !Number.isInteger(rule.minEvidence) || rule.minEvidence < 1 || rule.minEvidence > rule.conditions.length || !Number.isFinite(rule.severity)
        || rule.conditions.some(c => !operators[c.operator] || !Number.isFinite(c.value) || typeof c.signal !== 'string'))
        throw new Error(`Invalid ${state} rule.`);
    }
    for (const [level, row] of Object.entries(feedback.states[state])) {
      if (typeof row.enabled !== 'boolean' || !row.causes || typeof row.causes !== 'object'
        || (row.enabled && (['none', 'low', 'medium', 'good'].includes(level) || !Object.keys(row.causes).length))
        || Object.values(row.causes).some(text => !['message', 'action', 'goal'].every(k => typeof text[k] === 'string' && text[k].trim())))
        throw new Error('Invalid feedback row.');
    }
  }
  const p = rules.featurePolicy;
  if (![p.windowSeconds, p.freshSeconds, p.rrMinMs, p.rrMaxMs].every(n => Number.isFinite(n) && n > 0)
    || p.freshSeconds >= p.windowSeconds || p.rrMaxMs <= p.rrMinMs || !Number.isInteger(p.minRr) || p.minRr < 2
    || !Number.isInteger(p.baselineSamples) || p.baselineSamples < 2)
    throw new Error('Invalid feature policy.');
  active = structuredClone({ rules, feedback });
}

export async function loadRuleConfiguration() {
  const values = await Promise.all(['rules.json', 'feedback_table.json'].map(async file => {
    const response = await fetch(`./config/${file}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Configuration unavailable (HTTP ${response.status}).`);
    return response.json();
  }));
  configureRules(...values);
}

export function evaluateRule(state, features) {
  const config = active.rules.states[state];
  const unavailable = { level: 'unavailable', reason: 'Required signals or rules are unavailable.' };
  if (!config?.enabled || config.requiredSignals.filter(k => Number.isFinite(features[k])).length < config.minEvidence) return unavailable;
  const incomplete = [];
  const matches = [...config.rules].sort((a, b) => b.severity - a.severity).filter(rule => {
    const evidence = rule.conditions.filter(c => Number.isFinite(features[c.signal]));
    const checks = evidence.map(c => operators[c.operator](features[c.signal], c.value));
    const missingSignals = rule.conditions.filter(c => !Number.isFinite(features[c.signal])).map(c => c.signal);
    if (missingSignals.length && rule.match === 'all' && checks.every(Boolean)) incomplete.push({ level: rule.level, severity: rule.severity, missingSignals });
    if (evidence.length < rule.minEvidence) return false;
    return rule.match === 'all' ? checks.every(Boolean) : checks.some(Boolean);
  });
  const match = matches[0];
  const unassessedLevels = incomplete.filter(r => r.severity > (match?.severity || 0)).map(({ level, missingSignals }) => ({ level, missingSignals }));
  const names = { edaChangePct: 'EDA', scrChangePct: 'SCR', hrChangePct: 'HR', rmssdChangePct: 'RMSSD', sdnnChangePct: 'SDNN' };
  const missing = unassessedLevels.map(r => ` ${r.level[0].toUpperCase() + r.level.slice(1)} rule cannot be assessed: missing ${r.missingSignals.map(k => names[k] || k).join(', ')} baseline changes.`).join('');
  return { level: match?.level || config.defaultLevel, cause: match?.cause || null, unassessedLevels,
    reason: (match?.explanation || 'The complete current window does not meet the configured thresholds.') + missing };
}
