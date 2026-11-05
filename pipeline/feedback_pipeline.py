"""Session-isolated configured feature, rule, model and feedback processing."""
from collections import OrderedDict
from copy import deepcopy
from .config import CONFIG_DIR, load_config
from .ingest import finite, validate_sample, identifier
from .features import FeaturePipeline
from .rule_engine import RuleEngine
from .model import ModelBlend
from .decision import DecisionEngine
from .frustration_model import FrustrationModel, SignalTracker

STATES = ('frustration', 'fatigue', 'posture')
NOT_WORN = 'Device not worn (Empatica wear detection or skin temperature), so nothing is predicted from this data.'


class FeedbackPipeline:
    def __init__(self, rules=None, feedback=None, model_path=None):
        self.rules = load_config(rules, 'rules.json')
        self.feedback = load_config(feedback, 'feedback_table.json')
        self.rule_engine = RuleEngine(self.rules)
        self.rule_only = self.rules['states']['fatigue'].get('method') == 'rules-only'
        self.model = None if self.rule_only else ModelBlend(model_path, self.rules)
        self.sessions = {}
        self.frustration_model, self.frustration_model_reason = None, 'No frustration model is configured; rules decide the level.'
        model_file = self.rules['states'].get('frustration', {}).get('modelFile')
        if model_file:
            try:
                self.frustration_model = FrustrationModel(CONFIG_DIR.parent / model_file)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.frustration_model_reason = f'Frustration model unavailable ({exc}); rules decide the level.'

    def _frustration(self, rule, signals, now):
        """DBR architecture: ML gives the frustration level; the rule engine names the cause."""
        model = self.frustration_model
        if model is None:
            rule['model'] = dict(status='disabled', reason=self.frustration_model_reason)
            return rule
        features, waiting = signals.features(now, model.normalization)
        if features is None:
            rule['model'] = dict(status='waiting', reason=f'{waiting} Rules decide the level meanwhile.')
            return rule
        level, probabilities = model.predict(features)
        band = next(b for b in self.rules['probabilityBands'] if b['level'] == level)
        labels = {k: v.get('label', k) for k, v in self.rules['states']['frustration'].get('causes', {}).items()}
        cause = rule['cause'] if rule['available'] and rule['cause'] else 'unspecified'
        match, closest = ('exact' if cause != 'unspecified' else None), None
        if match:
            cause_note = f"Rule pattern: {rule['reason']}"
        elif rule['available']:
            closest = self.rule_engine.closest_cause(rule['evidence'])
            if closest:
                cause, match = closest['cause'], 'closest'
                cause_note = (f"Closest DBR pattern: {closest['label']}, {closest['agreeing']} of {closest['total']} signals "
                              'agree (not an exact match).')
            else:
                cause_note = 'All five signals were measured, but too few agree with any DBR cause pattern.'
        else:
            cause_note = f"Cause not assessed: {rule['reason']}"
        evaluation = model.data.get('evaluation', {})
        out = dict(rule, available=True, level=level, severity=band['severity'], score=probabilities[level],
                   score_kind='ml-probability', cause=cause, causeLabel=labels.get(cause, cause),
                   alternatives=rule.get('alternatives', []) if cause == rule['cause'] else [],
                   causeAssessed=rule['available'], causeMatch=match, causeAgreement=closest,
                   reason=f"ML model predicts {level} frustration ({probabilities[level]:.0%}). {cause_note}")
        out['model'] = dict(status='ml', family=model.data.get('family'), probabilities=probabilities,
                            ruleLevel=rule['level'], features=features,
                            accuracy=evaluation.get('metrics', {}).get('windowAccuracy'),
                            accuracyScope=evaluation.get('scope', 'unseen participants'))
        return out

    def reset(self, session_id):
        identifier(session_id, 'session_id')
        self.sessions.pop(session_id, None)

    def restore_alert_history(self, session_id, alerts):
        """After replay, retain actual persisted cooldown/cap history across config edits."""
        identifier(session_id, 'session_id')
        if session_id not in self.sessions:
            if alerts:
                raise ValueError('Replay session observations before restoring its alert history.')
            return
        self.sessions[session_id]['decisions'].restore_alert_history(alerts)

    def _new_session(self, sample):
        raw = sample.get('raw') if isinstance(sample.get('raw'), dict) else {}
        policy = self.rules.get('sourcePolicies', {}).get(raw.get('source'))
        return dict(features=FeaturePipeline(self.rules, policy), decisions=DecisionEngine(self.feedback),
                    participant=sample['participant'], mode=sample['mode'], last=None, result=None, seen=OrderedDict(),
                    signals=SignalTracker(min_eda=policy['edaFloorUs'] if policy else None),
                    policy=policy, predicted=None, held={})

    def _decide(self, state, features, sample, now, worn):
        """Evaluate every state, at the source's prediction cadence when it has one.

        A rolling window advanced one second overlaps the last by 98%, so at 1 Hz "3 consecutive
        Highs" would mean 3 seconds of nearly the same data. Between predictions the last result
        is held; losing wear contact is reported at once and breaks the run."""
        policy = state['policy']
        cadence = policy['predictionSeconds'] * 1000 if policy else None
        due = cadence is None or state['predicted'] is None or now - state['predicted'] >= cadence
        if not due and worn and state['held']:
            decisions = deepcopy(state['held'])
            for decision in decisions.values():
                decision.pop('wouldAlert', None)
                decision.update(alert=None, prediction=False)
            return decisions
        max_gap = cadence + self.feedback['decision']['freshSeconds'] * 1000 if cadence else None
        decisions = {}
        for name in STATES:
            rule = self.rule_engine.evaluate(features, name)
            if name == 'frustration':
                # The window still holds a minute of earlier data, which no longer describes the wearer.
                rule = self._frustration(rule, state['signals'], now) if worn else dict(rule, model=dict(status='waiting', reason=NOT_WORN))
            elif name == 'fatigue':
                if self.rule_only:
                    rule['model'] = dict(status='disabled', reason='Rule-based fatigue assessment from the fatigue document; no ML model is used.')
                else:
                    rule = self.model.apply(rule, features, sample['mode'])
            if not worn:
                rule['reason'] = NOT_WORN
            decisions[name] = state['decisions'].update(rule, now, deliver=sample['mode'] != 'batch', max_gap_ms=max_gap)
        if policy:
            # The wait for the first full window is not a prediction, whatever it reports.
            counted = due and (state['predicted'] is not None or any(d['available'] for d in decisions.values()))
            if counted:
                state['predicted'] = now
            for decision in decisions.values():
                decision['prediction'] = counted
            state['held'] = deepcopy(decisions)
        return decisions

    def process(self, sample):
        sample = validate_sample(sample)
        session_id, sample_id, now = sample['session_id'], sample['sample_id'], sample['timestamp_ms']
        state = self.sessions.get(session_id)
        if state is None:
            state = self.sessions[session_id] = self._new_session(sample)
        if state['participant'] != sample['participant'] or state['mode'] != sample['mode']:
            raise ValueError('Participant and mode cannot change within a session.')
        if sample_id in state['seen'] or (state['last'] is not None and now <= state['last']):
            out = dict(session_id=session_id, sample_id=sample_id, timestamp_ms=now, mode=sample['mode'],
                       features=dict(fresh=False, fatigueEligible=False), alerts=[], states={})
            for name in STATES:
                decision = self.rule_engine.evaluate({}, name)
                decision.update(alert=None, suppression='duplicate-or-reversed-timestamp',
                                reason='Duplicate or reversed observation does not supply new evidence.')
                out['states'][name] = decision
            return out
        features = state['features'].process(sample)
        raw = sample.get('raw') if isinstance(sample.get('raw'), dict) else {}
        eda, temp = raw.get('eda_us'), raw.get('temp_c')
        # Data from a device off the wrist must not reach the model or stretch its scaling.
        worn = raw.get('contact') is not False
        if worn and (finite(eda) or finite(temp)):
            state['signals'].add(now, eda, temp)
        decisions = self._decide(state, features, sample, now, worn)
        result = dict(session_id=session_id, sample_id=sample_id, timestamp_ms=now, mode=sample['mode'],
                      features=features, states=decisions, alerts=[d['alert'] for d in decisions.values() if d['alert']])
        state['last'] = now
        state['result'] = deepcopy(result)
        state['seen'][sample_id] = now
        if len(state['seen']) > 4096:
            state['seen'].popitem(last=False)
        return result
