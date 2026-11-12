#!/usr/bin/env python3
"""Compare real-label models on held-out people; write only a winning blend candidate.

Fixtures and synthetic demos cannot establish physiological performance. An approved
output is a candidate artifact; this command never changes a server's model setting.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut

from train_polar_fatigue import CLASSES, FEATURES, load_sessions, new_pipeline, train_model as fit_portable

BLEND_WEIGHT = .5
DEFAULT_RULES = ROOT / 'config/rules.json'


def atomic_json(path, value, indent=2):
    """Serialize before touching the destination, then replace on the same filesystem.

    indent=None writes compact JSON (large generated model files)."""
    path = Path(path)
    separators = None if indent else (',', ':')
    encoded = json.dumps(value, indent=indent, separators=separators, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def should_promote(metrics):
    """Ties are failures; all three comparisons must be finite held-out metrics."""
    try:
        values = [metrics[name]['macroF1'] for name in ('rules', 'ml', 'blend')]
        return (all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and 0 <= value <= 1 for value in values)
                and values[2] > values[0] and values[2] > values[1])
    except (KeyError, TypeError):
        return False


def _config(rules):
    if rules is None:
        return json.loads(DEFAULT_RULES.read_text(encoding='utf-8'))
    if isinstance(rules, (str, Path)):
        return json.loads(Path(rules).read_text(encoding='utf-8'))
    return rules


def _rows(sessions):
    try:
        x, y, groups, policy = load_sessions(sessions, return_policy=True)
    except OverflowError as exc:
        raise ValueError('Training measurements must contain finite representable numbers.') from exc
    classes = [level for level in CLASSES if level in set(y)]
    if len(set(groups)) < 3:
        raise ValueError('At least three participants are required for leave-one-person-out evaluation.')
    counts = Counter(y.tolist())
    if len(x) < 10 or len(classes) < 2 or min(counts.values()) < 3:
        raise ValueError('Need at least 10 ratings, two classes and three ratings per included class.')
    for train, _ in LeaveOneGroupOut().split(x, y, groups):
        if set(y[train]) != set(classes):
            raise ValueError('Every training fold must include every included class; collect broader participant label coverage.')
    return x, y, groups, classes, policy


def compare_sessions(sessions, rules=None):
    """Fit each scaler/logistic model only on the training people in that fold."""
    from pipeline.rule_engine import RuleEngine

    rules = _config(rules)
    x, y, groups, classes, policy = _rows(sessions)
    engine = RuleEngine(rules)
    from pipeline.config import validate_feature_policy
    if policy != validate_feature_policy(rules['featurePolicy']):
        raise ValueError('Training snapshot feature policy must match the selected rules featurePolicy.')
    bands = sorted(rules['probabilityBands'], key=lambda band: band['min'])
    class_scores = {band['level']: band['score'] for band in bands}

    def level(score):
        return next(band['level'] for band in reversed(bands) if score >= band['min'])

    rule_results = [engine.evaluate(dict(zip(FEATURES, row)), state='fatigue') for row in x]
    if any(not result['available'] for result in rule_results):
        raise ValueError('Configured fatigue rules cannot evaluate every eligible training row.')
    scores = np.asarray([result['score'] for result in rule_results])
    predictions = {'rules': np.asarray([result['level'] for result in rule_results]),
                   'ml': np.empty(len(y), dtype=object), 'blend': np.empty(len(y), dtype=object)}
    folds = []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        model = new_pipeline().fit(x[train], y[train])
        probabilities = model.predict_proba(x[test])
        ml_scores = probabilities @ np.asarray([class_scores[label] for label in model.classes_])
        predictions['ml'][test] = [level(score) for score in ml_scores]
        predictions['blend'][test] = [level(score) for score in
                                      ((1 - BLEND_WEIGHT) * scores[test] + BLEND_WEIGHT * ml_scores)]
        # The deployed JSON model keeps rules outside its observed feature range.
        # Evaluate that same fallback, with ranges fitted on training people only.
        out_of_range = np.any((x[test] < x[train].min(axis=0) - 1e-9)
                              | (x[test] > x[train].max(axis=0) + 1e-9), axis=1)
        predictions['blend'][test[out_of_range]] = predictions['rules'][test[out_of_range]]
        folds.append(dict(heldOutParticipant=str(groups[test][0]),
                          outOfRangeFallbackSamples=int(out_of_range.sum()),
                          trainingParticipants=sorted(set(groups[train].tolist())),
                          trainingSamples=len(train), testSamples=len(test)))
    metrics = {name: dict(macroF1=float(f1_score(y, values, labels=classes, average='macro', zero_division=0)),
                          accuracy=float(accuracy_score(y, values)),
                          confusionMatrix=confusion_matrix(y, values, labels=CLASSES).tolist())
               for name, values in predictions.items()}
    approved = should_promote(metrics)
    return dict(schemaVersion=1, approved=approved,
                reason='Held-out-person blend macro-F1 strictly exceeds both baselines.' if approved else
                       'Blend did not strictly exceed both baselines; existing model preserved.',
                source='participant-self-report', method='leave-one-participant-out',
                ruleConfigVersion=rules['version'], featurePolicy=policy, blendWeight=BLEND_WEIGHT,
                metrics=metrics, folds=folds, classOrder=classes, confusionMatrixClassOrder=CLASSES,
                scoring='Probability-weighted class rule score, fixed 0.5 blend, configured bands.',
                coverage=dict(samples=len(y), participants=len(set(groups)),
                              classCounts=dict(Counter(y.tolist())),
                              unseenLevels=[level for level in CLASSES if level not in classes]),
                limitations=['Internal held-out-person comparison, not external worker validation.',
                             'Unseen fatigue levels have no labeled evaluation coverage.',
                             'Rule scores and their blend are heuristic severity scores, not calibrated physiological probabilities.'])


def train_sessions(sessions, output, report, rules=None):
    """Always report a valid comparison; preserve the candidate file unless approved."""
    output, report = Path(output), Path(report)
    if output.resolve() == report.resolve():
        raise ValueError('Report must not overwrite the model output.')
    result = compare_sessions(sessions, rules)
    if result['approved']:
        x, y, groups = load_sessions(sessions)
        artifact, _ = fit_portable(x, y, groups, source='participant-self-report',
                                   feature_policy=result['featurePolicy'])
        # Retain the portable/browser evaluation schema; the reported deployment
        # metrics describe the same blend that is approved, with its comparison.
        artifact['evaluation'].update(status='available',
                                     accuracy=result['metrics']['blend']['accuracy'],
                                     macroF1=result['metrics']['blend']['macroF1'],
                                     confusionMatrix=result['metrics']['blend']['confusionMatrix'],
                                     classOrder=CLASSES, comparison=result,
                                     decisionMethod='fixed-0.5-blend-with-rule-fallback')
        artifact['deployment'] = dict(approved=True, ruleConfigVersion=result['ruleConfigVersion'],
                                      blendWeight=BLEND_WEIGHT, metrics=result['metrics'])
        artifact['modelId'] = 'polar-' + hashlib.sha256(
            json.dumps({key: value for key, value in artifact.items() if key != 'modelId'},
                       sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
        # Commit the evaluation first: an I/O failure cannot replace the model without a report.
        atomic_json(report, result)
        atomic_json(output, artifact)
    else:
        atomic_json(report, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', nargs='+', type=Path, required=True,
                        help='Schema v2 live JSON exports with independent participant self-reports.')
    parser.add_argument('--rules', type=Path, default=DEFAULT_RULES)
    parser.add_argument('--output', type=Path, required=True, help='Winning candidate JSON; never auto-deployed.')
    parser.add_argument('--report', type=Path, required=True, help='Held-out-person comparison report JSON.')
    args = parser.parse_args(argv)
    try:
        paths = [*args.sessions, args.rules]
        if any(destination.resolve() == source.resolve() for destination in (args.output, args.report) for source in paths):
            raise ValueError('Outputs must not overwrite an input session or rules config.')
        result = train_sessions([json.loads(path.read_text(encoding='utf-8')) for path in args.sessions],
                                args.output, args.report, args.rules)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    if result['approved']:
        print(f'Wrote approved candidate {args.output}; deployment requires explicitly configuring the server.')
    else:
        print('No model written; existing candidate preserved.')


if __name__ == '__main__':
    main()
