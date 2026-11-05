"""Generic minimum-evidence rules with explicit heuristic scores."""
import operator
from .config import load_config
from .ingest import finite

OPERATORS = {'>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le,
             '==': operator.eq, '!=': operator.ne}
SIGNAL_NAMES = {'edaChangePct': 'EDA', 'scrChangePct': 'SCR', 'hrChangePct': 'HR',
                'rmssdChangePct': 'RMSSD', 'sdnnChangePct': 'SDNN', 'absPitch': 'pitch', 'absRoll': 'roll'}


def specificity(rule):
    """Stricter thresholds describe a more specific cause pattern."""
    return sum(abs(c['value']) for c in rule['conditions'])


def signal_names(keys):
    return ', '.join(SIGNAL_NAMES.get(k, k) for k in keys)


class RuleEngine:
    def __init__(self, rules=None):
        self.rules = load_config(rules, 'rules.json')
        self.bands = self.rules['probabilityBands']
        for state in self.rules['states'].values():
            for rule in state['rules']:
                if rule['match'] not in ('all', 'any') or any(c['operator'] not in OPERATORS for c in rule['conditions']):
                    raise ValueError('Unknown rule match or condition operator.')

    def closest_cause(self, evidence, state='frustration'):
        """The DBR cause pattern nearest the measured signals, when no pattern matches exactly.

        Each signal counts for a cause if it meets that cause's thresholds at any level. Causes
        rank by how many signals agree, then by the strictest thresholds met (the engine's usual
        "most specific" tie-break), because several causes share the same Low thresholds and
        differ only at Medium and High. Returns None below the configured minimum."""
        policy = self.rules['states'].get(state, {})
        needed = policy.get('closestPatternMinSignals')
        if not needed:
            return None
        best = None
        for order, cause in enumerate(policy.get('causes', {})):
            strength = {}
            for rule in policy['rules']:
                if rule['cause'] != cause:
                    continue
                conditions = {}
                for condition in rule['conditions']:
                    conditions.setdefault(condition['signal'], []).append(condition)
                for signal, parts in conditions.items():
                    if signal in evidence and all(OPERATORS[c['operator']](evidence[signal], c['value']) for c in parts):
                        strength[signal] = max(strength.get(signal, 0), sum(abs(c['value']) for c in parts))
            rank = (len(strength), sum(strength.values()), -order)
            if strength and (best is None or rank > best[0]):
                best = (rank, cause, sorted(strength))
        if best is None or best[0][0] < needed:
            return None
        return dict(cause=best[1], label=policy['causes'][best[1]].get('label', best[1]), agreeing=best[0][0],
                    total=len(policy['requiredSignals']), signals=best[2])

    def evaluate(self, features, state='fatigue'):
        policy = self.rules['states'].get(state)
        evidence = {key: value for key, value in features.items() if finite(value)}
        result = dict(state=state, available=False, level='unavailable', severity=None, cause=None, causeLabel=None,
                      score=None, score_kind='rule-score', evidence={}, alternatives=[],
                      reason='State thresholds are not configured.',
                      ruleConfigVersion=self.rules['version'], scoreSemantics=self.rules.get('scoreSemantics'))
        if not policy or not policy['enabled']:
            return result
        result['evidence'] = {key: evidence[key] for key in policy['requiredSignals'] if key in evidence}
        if len(result['evidence']) < policy['minEvidence']:
            missing = [key for key in policy['requiredSignals'] if key not in evidence]
            result['missingSignals'] = missing
            result['reason'] = f'Insufficient fresh evidence; missing {signal_names(missing)} baseline changes.'
            return result
        labels = {cause: value.get('label', cause) for cause, value in policy.get('causes', {}).items()}
        matches, incomplete = [], []
        for order, rule in enumerate(policy['rules']):
            missing = [c['signal'] for c in rule['conditions'] if c['signal'] not in evidence]
            checks = [OPERATORS[c['operator']](evidence[c['signal']], c['value'])
                      for c in rule['conditions'] if c['signal'] in evidence]
            if missing and rule['match'] == 'all' and all(checks):
                incomplete.append(dict(level=rule['level'], severity=rule['severity'], missingSignals=missing))
            if len(checks) >= rule['minEvidence'] and (all(checks) if rule['match'] == 'all' else any(checks)):
                matches.append((rule['severity'], specificity(rule), -order, rule))
        if matches:
            matches.sort(key=lambda m: m[:3], reverse=True)
            rule = matches[0][3]
            result.update(available=True, **{key: rule[key] for key in ('level', 'severity', 'cause', 'score')},
                          causeLabel=labels.get(rule['cause'], rule['cause']), reason=rule['explanation'])
            # Other causes whose patterns also match at this level stay visible, not hidden.
            result['alternatives'] = list(dict.fromkeys(
                labels.get(m[3]['cause'], m[3]['cause']) for m in matches[1:]
                if m[0] == rule['severity'] and m[3]['cause'] != rule['cause']))
        else:
            band = next((b for b in self.bands if b['level'] == policy['defaultLevel']), self.bands[0])
            result.update(available=True, level=policy['defaultLevel'], severity=band['severity'], score=band['score'],
                          reason='The complete current evidence is below the configured thresholds.')
        result['unassessedLevels'] = [{k: r[k] for k in ('level', 'missingSignals')}
                                      for r in incomplete if r['severity'] > result['severity']]
        for row in result['unassessedLevels']:
            result['reason'] += (f" {row['level'].capitalize()} rule cannot be assessed: "
                                 f"missing {signal_names(row['missingSignals'])} baseline changes.")
        return result
