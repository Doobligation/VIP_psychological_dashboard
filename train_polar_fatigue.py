#!/usr/bin/env python3
"""Train an experimental Polar classifier; synthetic bootstrap is never validation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['hrChangePct', 'rmssdChangePct', 'sdnnChangePct']
CLASSES = ['none', 'low', 'medium', 'high', 'very-high']
SOURCES = {'synthetic-bootstrap', 'participant-self-report'}


def finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def feature_policy_from_snapshot(snapshot):
    """Validate recorded protocol; missing metadata means the original legacy policy."""
    if not isinstance(snapshot, dict):
        raise ValueError('A measurement snapshot is required.')
    # Lazy import keeps the portable fitting helpers independent at import time.
    from pipeline.config import LEGACY_FEATURE_POLICY, validate_feature_policy
    policy = snapshot['featurePolicy'] if 'featurePolicy' in snapshot else LEGACY_FEATURE_POLICY
    return validate_feature_policy(policy)


def features_from_snapshot(snapshot):
    """Match sensor/fatigue-model.js; never use the threshold result as a label."""
    if not isinstance(snapshot, dict):
        raise ValueError('A measurement snapshot is required.')
    policy = feature_policy_from_snapshot(snapshot)
    if (snapshot.get('fresh') is not True or snapshot.get('baselineState') != 'ready'
            or not finite(snapshot.get('requiredWindowSeconds'))
            or snapshot['requiredWindowSeconds'] != policy['windowSeconds']
            or not finite(snapshot.get('windowSeconds')) or snapshot['windowSeconds'] < policy['windowSeconds'] - policy['freshSeconds']
            or not finite(snapshot.get('rrCount')) or snapshot['rrCount'] < policy['minRr']):
        raise ValueError(f"A fresh, complete {policy['windowSeconds']:g}-second window and resting baseline are required.")
    baseline = snapshot.get('baseline')
    if not isinstance(baseline, dict):
        raise ValueError('Missing resting baseline.')
    values = []
    for key in ['hr', 'rmssd', 'sdnn']:
        current, reference = snapshot.get(key), baseline.get(key)
        if (not finite(current) or not finite(reference) or reference <= 0
                or current < 0 or (key == 'hr' and current == 0)):
            raise ValueError(f'Invalid {key} or baseline; valid finite values are required.')
        change = (current - reference) / reference * 100
        if not math.isfinite(change):
            raise ValueError('Feature overflow.')
        values.append(change)
    return values


def load_sessions(sessions, *, return_policy=False):
    """Reject a malformed export rather than silently cherry-picking its records."""
    features, labels, participants = [], [], []
    seen = set()
    dataset_policy = None
    for session in sessions:
        if not isinstance(session, dict) or session.get('mode') != 'live' or session.get('schemaVersion') != 2:
            raise ValueError('Only schema v2 live session exports can train a participant model; demos are excluded.')
        participant, session_id = session.get('participant'), session.get('sessionId')
        if not isinstance(participant, str) or not participant.strip() or not isinstance(session_id, str) or not session_id:
            raise ValueError('Each session needs a participant ID and session ID.')
        if session_id in seen:
            raise ValueError('Duplicate session; export each session only once.')
        seen.add(session_id)
        ratings = session.get('fatigueRatings')
        if not isinstance(ratings, list) or not ratings:
            raise ValueError('No independent fatigue ratings in session.')
        previous = None
        for rating in ratings:
            if not isinstance(rating, dict) or rating.get('source') != 'participant-self-report':
                raise ValueError('Labels must be independent participant self-reports.')
            if rating.get('level') not in CLASSES:
                raise ValueError('Unknown fatigue label.')
            policy = feature_policy_from_snapshot(rating.get('snapshot'))
            if dataset_policy is None:
                dataset_policy = policy
            elif policy != dataset_policy:
                raise ValueError('All training snapshots must use the same feature policy; mixed feature policies are not supported.')
            try:
                when = datetime.fromisoformat(rating['time'].replace('Z', '+00:00'))
                if when.tzinfo is None:
                    raise ValueError('Timestamp needs a timezone.')
                epoch = when.timestamp()
            except (KeyError, TypeError, AttributeError, ValueError) as exc:
                raise ValueError('Rating timestamp must be an ISO time with timezone.') from exc
            if previous is not None and epoch - previous < policy['windowSeconds']:
                raise ValueError(f"Ratings must be ordered and at least {policy['windowSeconds']:g} seconds apart.")
            previous = epoch
            features.append(features_from_snapshot(rating.get('snapshot')))
            labels.append(rating['level'])
            participants.append(participant.strip())
    if not features:
        raise ValueError('No training observations.')
    arrays = np.asarray(features, dtype=float), np.asarray(labels), np.asarray(participants)
    return (*arrays, dataset_policy) if return_policy else arrays


def bootstrap_samples(seed=1031, per_class=600):
    """Balanced synthetic threshold examples, NOT measurements of workers."""
    rng = np.random.default_rng(seed)
    collected = {level: [] for level in CLASSES}
    thresholds = [('very-high', 25, -40, -35), ('high', 15, -35, -30),
                  ('medium', 10, -25, -20), ('low', 5, -15, -15)]
    while any(len(rows) < per_class for rows in collected.values()):
        for row in rng.uniform([-20, -95, -95], [120, 100, 100], size=(10000, 3)):
            label = next((level for level, hr, rmssd, sdnn in thresholds
                          if row[0] > hr and row[1] < rmssd and row[2] < sdnn), 'none')
            if len(collected[label]) < per_class:
                collected[label].append(row)
    x = np.asarray([row for level in CLASSES for row in collected[level]])
    y = np.asarray([level for level in CLASSES for _ in collected[level]])
    return x, y, np.full(len(y), 'synthetic')


def new_pipeline():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))


def evaluate(x, y, groups, classes, source):
    if source == 'synthetic-bootstrap':
        return dict(status='not-applicable', reason='Synthetic rule examples cannot establish worker fatigue accuracy.')
    if len(set(groups)) < 2:
        return dict(status='unavailable', reason='Only one participant; collect other participants for independent evaluation.')
    predictions = np.empty(len(y), dtype=object)
    folds = []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        if set(y[train]) != set(classes):
            return dict(status='unavailable', reason='A held-out participant leaves a training fold without every class. Collect broader label coverage.')
        # Fit preprocessing only on this fold's training participants.
        model = new_pipeline().fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
        folds.append(dict(fold=len(folds) + 1, trainingParticipants=len(set(groups[train])),
                          trainingSamples=len(train), testSamples=len(test)))
    return dict(status='available', method='leave-one-participant-out', folds=folds,
                accuracy=float(accuracy_score(y, predictions)),
                macroF1=float(f1_score(y, predictions, labels=classes, average='macro', zero_division=0)),
                confusionMatrix=confusion_matrix(y, predictions, labels=classes).tolist(),
                classOrder=classes,
                limitation='Small-study internal evaluation; not external construction-worker validation.')


def train_model(x, y, groups, *, source, feature_policy=None):
    x, y, groups = np.asarray(x, dtype=float), np.asarray(y), np.asarray(groups)
    if source not in SOURCES:
        raise ValueError('Unknown training provenance.')
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or len(x) != len(y) or len(y) != len(groups):
        raise ValueError('Training needs finite three-feature rows, labels and participant groups.')
    counts = Counter(y.tolist())
    if not set(counts) <= set(CLASSES) or len(counts) < 2 or len(y) < 10 or min(counts.values()) < 3:
        raise ValueError('Need at least 10 ratings, at least two fatigue levels, and at least three ratings per included level.')
    classes = [level for level in CLASSES if level in counts]
    evaluation = evaluate(x, y, groups, classes, source)
    pipeline = new_pipeline().fit(x, y)
    scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    coef, intercept = classifier.coef_, classifier.intercept_
    if len(classifier.classes_) == 2:
        # Binary sklearn stores log odds for class 1. Softmax([0, log_odds])
        # yields the same probabilities; symmetric +/- logits would double it.
        coef = np.vstack([np.zeros(3), coef[0]])
        intercept = np.asarray([0.0, intercept[0]])
    order = [list(classifier.classes_).index(level) for level in classes]
    policy = feature_policy_from_snapshot({} if feature_policy is None else {'featurePolicy': feature_policy})
    artifact = dict(schemaVersion=1, modelType='multinomial-logistic-regression', featurePolicy=policy,
                    features=FEATURES, classes=classes, mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                    coefficients=coef[order].tolist(), intercepts=intercept[order].tolist(),
                    featureMin=x.min(axis=0).tolist(), featureMax=x.max(axis=0).tolist(),
                    training=dict(source=source, samples=len(y), participants=0 if source == 'synthetic-bootstrap' else len(set(groups)),
                                  classCounts=dict(counts), regularizationC=1.0, sklearnVersion=sklearn.__version__,
                                  labelDefinition='Assignment threshold labels' if source == 'synthetic-bootstrap' else 'Current fatigue level selected by participant',
                                  scope='synthetic-starter' if source == 'synthetic-bootstrap' else 'personal' if len(set(groups)) == 1 else 'study'),
                    evaluation=evaluation)
    artifact['modelId'] = 'polar-' + hashlib.sha256(json.dumps(artifact, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
    return artifact, pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bootstrap', action='store_true', help='Fit synthetic threshold examples, not worker data.')
    mode.add_argument('--sessions', nargs='+', type=Path, help='Live session JSON files with independent participant ratings.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    temporary = None
    try:
        if args.sessions and any(args.output.resolve() == path.resolve() for path in args.sessions):
            raise ValueError('Output must not overwrite an input session.')
        feature_policy = None
        if args.bootstrap:
            x, y, groups = bootstrap_samples()
            source = 'synthetic-bootstrap'
        else:
            x, y, groups, feature_policy = load_sessions(
                [json.loads(path.read_text(encoding='utf-8')) for path in args.sessions], return_policy=True)
            source = 'participant-self-report'
        artifact, _ = train_model(x, y, groups, source=source, feature_policy=feature_policy)
        encoded = json.dumps(artifact, indent=2, allow_nan=False) + '\n'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', dir=args.output.parent, suffix='.tmp', delete=False, encoding='utf-8') as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        temporary.replace(args.output)
        print(f"Saved {artifact['modelId']} to {args.output}; source={source}, samples={len(y)}, participants={artifact['training']['participants']}.")
        print(json.dumps(artifact['evaluation'], indent=2))
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
