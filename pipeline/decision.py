"""Consecutive-High feedback decisions with cooldown and warning limits.

Assignment rule: Low and Medium predictions never issue feedback and reset the count;
High feedback is issued only after N consecutive High predictions (N from config).
"""
from copy import deepcopy
import uuid
from .config import load_config
from .ingest import finite


class DecisionEngine:
    """One session's decisions; state cooldowns persist through recovery and missing data."""
    def __init__(self, feedback=None):
        self.feedback = load_config(feedback, 'feedback_table.json')
        self.settings = self.feedback['decision']
        self.states = {}
        self.warnings = []

    def _state(self, name):
        return self.states.setdefault(name, dict(last=None, level=None, count=0, lastAlert=None))

    def restore_alert_history(self, alerts):
        """Replace replay's hypothetical alerts with actual durable episode history."""
        validated = []
        for alert in alerts:
            when, state_name = alert.get('time'), alert.get('state')
            if not finite(when) or when < 0 or state_name not in self.feedback['states']:
                raise ValueError('Persisted alert must have a configured state and finite millisecond time.')
            validated.append((when, state_name))
        for state in self.states.values():
            state['lastAlert'] = None
        self.warnings = sorted(when for when, _ in validated)
        for when, state_name in sorted(validated):
            self._state(state_name)['lastAlert'] = when

    def update(self, result, timestamp_ms):
        if not finite(timestamp_ms) or timestamp_ms < 0:
            raise ValueError('Decision timestamp must be finite and nonnegative.')
        required = self.settings['consecutiveHighPredictions']
        out = deepcopy(result)
        state = self._state(result['state'])
        out.update(alert=None, suppression=None, feedbackConfigVersion=self.feedback['version'],
                   streak=dict(count=state['count'], required=required, level=state['level']))
        if state['last'] is not None and timestamp_ms <= state['last']:
            out['suppression'] = 'duplicate-or-reversed-timestamp'
            return out
        gap = state['last'] is not None and timestamp_ms - state['last'] > self.settings['freshSeconds'] * 1000
        state['last'] = timestamp_ms
        row = self.feedback['states'].get(result['state'], {}).get(result['level'])
        if not result['available'] or gap or not row or not row['enabled']:
            # Missing evidence, a data gap, or a Low/Medium prediction breaks the run.
            state.update(level=None, count=0)
        if not result['available']:
            out['suppression'] = 'insufficient-evidence'
        elif not row or not row['enabled']:
            out['suppression'] = 'no-feedback-for-level'
        if out['suppression']:
            out['streak'].update(count=0, level=None)
            return out
        if state['level'] != result['level']:
            state.update(level=result['level'], count=0)
        state['count'] += 1
        out['streak'].update(count=state['count'], level=result['level'])
        self.warnings = [t for t in self.warnings if timestamp_ms - t < self.settings['warningWindowSeconds'] * 1000]
        text = row['causes'].get(result['cause'])
        if state['count'] < required:
            out['suppression'] = 'awaiting-consecutive-high'
        elif text is None:
            out['suppression'] = 'no-feedback-for-cause'
        elif state['lastAlert'] is not None and timestamp_ms - state['lastAlert'] < self.settings['cooldownSeconds'] * 1000:
            out['suppression'] = 'cooldown'
        elif len(self.warnings) >= self.settings['maxWarnings']:
            out['suppression'] = 'warning-cap'
        else:
            out['alert'] = dict(id=uuid.uuid4().hex, state=result['state'], cause=result['cause'],
                                causeLabel=result.get('causeLabel'), level=result['level'],
                                message=text['message'], action=text['action'], goal=text['goal'],
                                reference=text.get('reference', ''), consecutive=state['count'], time=timestamp_ms)
            state.update(lastAlert=timestamp_ms, count=0)
            self.warnings.append(timestamp_ms)
        return out
