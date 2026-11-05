"""Portable logistic JSON inference and the explicit held-out-person deployment gate."""
import json
import math
from pathlib import Path
from .ingest import finite
from .config import LEGACY_FEATURE_POLICY, validate_feature_policy

FEATURES = ['hrChangePct', 'rmssdChangePct', 'sdnnChangePct']
CLASSES = ['none', 'low', 'medium', 'high', 'very-high']


class ModelBlend:
    def __init__(self, model_path, rules):
        self.rules = rules
        self.model = None
        self.model_policy = None
        self.reason = 'No approved model configured; live decisions use rules.'
        if model_path is None:
            return
        try:
            path = Path(model_path)
            if path.stat().st_size > 5_000_000:
                raise ValueError('Model JSON is too large.')
            model = json.loads(path.read_text(encoding='utf-8'))
            if (model.get('schemaVersion') != 1 or model.get('modelType') != 'multinomial-logistic-regression'
                    or model.get('features') != FEATURES):
                raise ValueError('Unsupported portable model schema or features.')
            classes = model['classes']
            if (not isinstance(classes, list) or not 2 <= len(classes) <= 5
                    or classes != [c for c in CLASSES if c in classes]):
                raise ValueError('Malformed model classes.')
            def vector(value, length):
                return isinstance(value, list) and len(value) == length and all(finite(v) for v in value)
            if any(not vector(model[key], 3) for key in ('mean', 'scale', 'featureMin', 'featureMax')):
                raise ValueError('Malformed preprocessing vectors.')
            if (any(s <= 0 for s in model['scale'])
                    or any(low > high for low, high in zip(model['featureMin'], model['featureMax']))):
                raise ValueError('Invalid preprocessing scale or training range.')
            if (not isinstance(model['coefficients'], list) or len(model['coefficients']) != len(classes)
                    or any(not vector(row, 3) for row in model['coefficients'])
                    or not vector(model['intercepts'], len(classes))):
                raise ValueError('Malformed logistic coefficients.')
            if model['training']['source'] not in ('participant-self-report', 'synthetic-bootstrap'):
                raise ValueError('Unknown model training provenance.')
            self.model_policy = validate_feature_policy(model.get('featurePolicy', LEGACY_FEATURE_POLICY))
            self.model = model
            self.reason = self._gate_reason()
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self.model = None
            self.reason = f'Model import failed; rules retained: {exc}'

    def _gate_reason(self):
        model = self.model
        if self.model_policy != self.rules['featurePolicy']:
            return 'Model feature policy differs from the active measurement protocol; rules retained.'
        if model['training']['source'] != 'participant-self-report':
            return 'Synthetic model is demo-only; live decisions use rules.'
        participants = model['training'].get('participants')
        if not isinstance(participants, int) or isinstance(participants, bool) or participants < 3:
            return 'Live blend requires held-out-person evaluation with at least three participants.'
        deployment = model.get('deployment', {})
        if deployment.get('approved') is not True or deployment.get('ruleConfigVersion') != self.rules['version']:
            return 'Model has no approved deployment for the current rule configuration.'
        if deployment.get('blendWeight') != .5:
            return 'Only the predeclared blend weight 0.5 is supported.'
        metrics = deployment.get('metrics', {})
        try:
            for method in ('rules', 'ml', 'blend'):
                if not all(finite(metrics[method][k]) and 0 <= metrics[method][k] <= 1 for k in ('macroF1', 'accuracy')):
                    raise ValueError()
            if not metrics['blend']['macroF1'] > max(metrics['rules']['macroF1'], metrics['ml']['macroF1']):
                return 'Held-out-person blend did not beat both rules and ML.'
        except (ValueError, KeyError, TypeError):
            return 'Missing or malformed held-out-person comparison metrics.'
        return None

    def apply(self, result, features, mode):
        result = dict(result)
        result['model'] = dict(status='rule-only', reason=self.reason, modelId=self.model.get('modelId') if self.model else None)
        if self.model is None or not result['available'] or self.model_policy != self.rules['featurePolicy']:
            return result
        # A synthetic artifact is allowed only in an explicitly marked demo.
        demo = mode == 'demo' and self.model['training']['source'] == 'synthetic-bootstrap'
        if self.reason and not demo:
            return result
        try:
            x = [features[key] for key in FEATURES]
            if any(v < low - 1e-9 or v > high + 1e-9 for v, low, high in zip(x, self.model['featureMin'], self.model['featureMax'])):
                raise ValueError('Current features are outside the model training range; rules retained.')
            normalized = [(v - mean) / scale for v, mean, scale in zip(x, self.model['mean'], self.model['scale'])]
            logits = [sum(w * v for w, v in zip(row, normalized)) + intercept
                      for row, intercept in zip(self.model['coefficients'], self.model['intercepts'])]
            if not all(finite(v) for v in logits):
                raise ValueError('Nonfinite model inference; rules retained.')
            weights = [math.exp(v - max(logits)) for v in logits]
            probabilities = [w / sum(weights) for w in weights]
            scores = {b['level']: b['score'] for b in self.rules['probabilityBands']}
            ml_score = sum(p * scores[level] for p, level in zip(probabilities, self.model['classes']))
            score = .5 * result['score'] + .5 * ml_score
            band = max((b for b in self.rules['probabilityBands'] if score >= b['min']), key=lambda b: b['min'])
            result.update(ruleScore=result['score'], mlScore=ml_score, score=score, score_kind='blend-score',
                          level=band['level'], severity=band['severity'], cause='cardiovascular-strain' if band['severity'] else None,
                          reason='Heuristic rule score blended with class-probability-weighted model severity; not a calibrated fatigue probability.')
            result['model'].update(status='demo-blend' if demo else 'approved-blend', reason='Synthetic demo only.' if demo else 'Approved held-out-person blend.',
                                   blendWeight=.5, probabilities=dict(zip(self.model['classes'], probabilities)))
        except (KeyError, ValueError, OverflowError) as exc:
            result['model']['reason'] = str(exc)
        return result
