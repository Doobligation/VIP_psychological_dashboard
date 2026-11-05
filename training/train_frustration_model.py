#!/usr/bin/env python3
"""Train, evaluate and export the Team 3 frustration model from labelled sessions.

Input: a folder of 1 Hz session CSVs named like `P7_Active_High.csv` (participant,
condition and Low/Medium/High label in any order) with EDA and skin-temperature columns.

Two profiles:

--profile unseen (deployed): models are compared with nested leave-one-participant-out (LOPO)
  cross-validation; for each held-out person the normalization, family and settings are chosen
  by an inner grouped CV over the remaining people only. Every reported accuracy is on people
  the model never trained on, and the report adds two controls: how well the same setup "learns"
  fake per-participant labels, and a permutation test of the LOPO score against chance.
--profile seen (previous model, kept to document its scores): a deep, unregularized XGBoost on
  heavily overlapping windows, scored with a random window split, so test windows come from
  participants (and sessions) seen in training. Much of that score is recognizing the person:
  its report records the accuracy on unseen people and on fake labels.

Features come from pipeline.frustration_model.window_features, the same code the live
server runs. Participant data stays outside the repository (default: .local/data).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.frustration_model import (CLASSES, FEATURES, MIN_WINDOW_SECONDS, NORMALIZATIONS, WINDOW_SECONDS,
                                        FrustrationModel, signal_scale, window_features)
from training.train_model import atomic_json

HOP_SECONDS = 30
SEEN_HOP_SECONDS = 10
SEEN_MODEL = ('xgboost', dict(max_depth=6, n_estimators=500, learning_rate=.1, subsample=1.0,
                              colsample_bytree=1.0, reg_lambda=1.0))
# Two sessions run 30-80 minutes while most last 2-7; cap each so no person dominates.
SESSION_CAP_SECONDS = 600
SEED = 0
# Selection scored baseline-z and rolling-z alike (inner macro-F1 0.622 and 0.612), so this was decided
# on a test the labels cannot run: on long real recordings, predictions from a first-minute reference
# change with the session's start time (see docs). Its family and settings still come from inner CV.
DEPLOY_NORMALIZATION = 'rolling-z'
FAKE_LABEL_ROUNDS = 10
PERMUTATION_ROUNDS = 200


# ---------------------------------------------------------------- data

def _column(df, kind):
    """Pick the real signal column, skipping exported index/timestamp ramps."""
    candidates = []
    for name in df.columns:
        if kind not in name.lower() or name == 'timestamp_local':
            continue
        values = pd.to_numeric(df[name], errors='coerce').to_numpy(float)
        if np.isfinite(values).sum() > 2 and abs(np.corrcoef(np.arange(len(values)), np.nan_to_num(values))[0, 1]) > .9999:
            continue
        candidates.append((name, values))
    named = [c for c in candidates if not c[0].lower().startswith('unnamed: 0')]
    if not (named or candidates):
        raise ValueError(f'no {kind} column')
    return (named or candidates)[-1][1]


def load_sessions(folder):
    sessions = []
    for path in sorted(Path(folder).glob('*.csv')):
        parts = path.stem.lower().split('_')
        participant = next((p.upper() for p in parts if re.fullmatch(r'p\d+', p)), None)
        label = next((p for p in parts if p in CLASSES), None)
        condition = next((p for p in parts if p in ('active', 'passive')), None)
        if not (participant and label and condition):
            raise ValueError(f'{path.name}: expected participant, condition and Low/Medium/High in the name')
        df = pd.read_csv(path)
        sessions.append(dict(name=path.stem, participant=participant, label=label, condition=condition,
                             eda=_column(df, 'eda'), temp=_column(df, 'temp')))
    if not sessions:
        raise ValueError(f'No session CSVs in {folder}')
    return sessions


def build_windows(sessions, window=WINDOW_SECONDS, hop=HOP_SECONDS, cap=SESSION_CAP_SECONDS,
                  normalization=NORMALIZATIONS[0]):
    """Windows scaled per wearer from data up to the window end, exactly as the live server does."""
    rows = []
    for s in sessions:
        n = min(len(s['eda']), len(s['temp']), cap)
        for start in range(0, n - window + 1, hop):
            end = start + window
            segment = {}
            for key in ('eda', 'temp'):
                values = s[key][start:end]
                if np.isnan(values).mean() > 1 - MIN_WINDOW_SECONDS / window:
                    break
                history = s[key][:end]
                history = history[~np.isnan(history)].tolist()
                filled = np.where(np.isnan(values), np.nanmean(values), values)
                segment[key] = (filled.tolist(), signal_scale(history, history[:window], min(history), max(history),
                                                              normalization, window))
            else:
                features = window_features(segment['eda'][0], segment['temp'][0], segment['eda'][1], segment['temp'][1])
                rows.append(dict(features, label=s['label'], participant=s['participant'],
                                 session=s['name'], condition=s['condition']))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- models

def candidates(normalizations=NORMALIZATIONS):
    """Small, deliberately regularized grids: few settings means little room to overfit selection."""
    return [(family, dict(params, normalization=normalization)) for normalization in normalizations
            for family, params in _family_grid()]


def _family_grid():
    out = [('logistic', dict(C=c)) for c in (0.1, 0.3, 1.0)]
    out += [('random-forest', dict(max_depth=d)) for d in (3, 5)]
    out += [('extra-trees', dict(max_depth=d, min_samples_leaf=leaf)) for d in (3, 5) for leaf in (5, 10)]
    try:
        import xgboost  # noqa: F401
        out += [('xgboost', dict(max_depth=d)) for d in (2, 3)]
    except ImportError:
        warnings.warn('xgboost is not installed; skipping that family.')
    return out


def make_model(family, params):
    params = {k: v for k, v in params.items() if k != 'normalization'}
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if family == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(C=params['C'], max_iter=5000))
    if family == 'random-forest':
        return RandomForestClassifier(200, max_depth=params['max_depth'], min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    if family == 'extra-trees':
        return ExtraTreesClassifier(200, max_depth=params['max_depth'], min_samples_leaf=params['min_samples_leaf'],
                                    random_state=SEED, n_jobs=-1)
    from xgboost import XGBClassifier
    defaults = dict(n_estimators=200, learning_rate=.05, subsample=.8, colsample_bytree=.8, reg_lambda=2.0)
    return XGBClassifier(**{**defaults, **params}, random_state=SEED, n_jobs=4, verbosity=0)


def fit(family, params, X, y, weights):
    """Every session counts equally and classes are balanced, whatever the window counts.

    weights=None fits unweighted, as the seen-participant profile does."""
    model = make_model(family, params)
    if weights is None:
        return model.fit(X, y)
    w = weights.copy()
    for c in np.unique(y):
        w[y == c] /= w[y == c].sum()
    w *= len(w) / w.sum()
    if family == 'logistic':
        model.fit(X, y, logisticregression__sample_weight=w)
    else:
        model.fit(X, y, sample_weight=w)
    return model


def macro_f1(y, pred):
    from sklearn.metrics import f1_score
    return f1_score(y, pred, average='macro', labels=range(len(CLASSES)), zero_division=0)


def select(configs, tables, y, weights, groups):
    """Inner grouped CV over the given people only; returns the best config and all scores.

    tables maps each normalization to its feature matrix; every matrix has the same rows."""
    from sklearn.model_selection import GroupKFold
    folds = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    scores = {}
    for family, params in configs:
        X = tables[params['normalization']]
        pred = np.zeros(len(y), int)
        for train, test in folds.split(X, y, groups):
            pred[test] = fit(family, params, X[train], y[train], weights[train]).predict(X[test])
        scores[(family, json.dumps(params, sort_keys=True))] = macro_f1(y, pred)
    best = max(scores, key=scores.get)
    return (best[0], json.loads(best[1])), scores


# ---------------------------------------------------------------- evaluation

def summarize(frame, y, pred):
    from sklearn.metrics import accuracy_score, confusion_matrix
    weights = 1 / frame.groupby('session')['session'].transform('count').to_numpy()
    by_session = pd.DataFrame(dict(session=frame['session'], true=y, pred=pred)).groupby('session')
    votes = by_session.agg(true=('true', 'first'), pred=('pred', lambda p: np.bincount(p, minlength=3).argmax()))
    matrix = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    return dict(
        windowAccuracy=round(float(accuracy_score(y, pred)), 4),
        sessionWeightedAccuracy=round(float(accuracy_score(y, pred, sample_weight=weights)), 4),
        macroF1=round(float(macro_f1(y, pred)), 4),
        sessionVoteAccuracy=round(float((votes['true'] == votes['pred']).mean()), 4),
        recall={c: round(float(matrix[i, i] / matrix[i].sum()), 4) if matrix[i].sum() else None for i, c in enumerate(CLASSES)},
        confusion=dict(rows='true', columns='predicted', labels=CLASSES, matrix=matrix.tolist()))


def bootstrap_ci(frame, y, pred, rounds=2000):
    """95% interval for window accuracy, resampling participants (the unit that generalizes)."""
    rng = np.random.default_rng(SEED)
    people = frame['participant'].unique()
    correct = pd.Series(y == pred).groupby(frame['participant'].to_numpy())
    hits, counts = correct.sum(), correct.count()
    stats = []
    for _ in range(rounds):
        sample = rng.choice(people, len(people), replace=True)
        stats.append(hits[sample].sum() / counts[sample].sum())
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def nested_lopo(tables, y, weights, groups, configs):
    """Outer LOPO; inner selection per family, per normalization and overall, from the same inner scores."""
    families = list(dict.fromkeys(f for f, _ in configs))
    normalizations = list(dict.fromkeys(p['normalization'] for _, p in configs))
    preds = {name: np.zeros(len(y), int) for name in families + normalizations + ['selected']}
    chosen = []
    for person in np.unique(groups):
        train, test = groups != person, groups == person
        _, scores = select(configs, {name: table[train] for name, table in tables.items()},
                           y[train], weights[train], groups[train])
        best_overall = max(scores, key=scores.get)
        fitted = {}

        def predict(key):
            if key not in fitted:
                params = json.loads(key[1])
                X = tables[params['normalization']]
                fitted[key] = fit(key[0], params, X[train], y[train], weights[train]).predict(X[test])
            return fitted[key]

        for family in families:
            preds[family][test] = predict(max((k for k in scores if k[0] == family), key=scores.get))
        for normalization in normalizations:
            matching = [k for k in scores if json.loads(k[1])['normalization'] == normalization]
            preds[normalization][test] = predict(max(matching, key=scores.get))
        preds['selected'][test] = predict(best_overall)
        chosen.append(dict(heldOut=str(person), family=best_overall[0], params=json.loads(best_overall[1])))
        print(f'  held out {person}: chose {best_overall[0]} {best_overall[1]}', flush=True)
    return preds, chosen


def baselines(frame, X, y, weights, groups, final):
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
    majority = np.zeros(len(y), int)
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        majority[test] = DummyClassifier(strategy='most_frequent').fit(X[train], y[train]).predict(X[test])
    leaky = np.zeros(len(y), int)
    for train, test in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        leaky[test] = fit(*final, X[train], y[train], weights[train]).predict(X[test])
    return summarize(frame, y, majority), summarize(frame, y, leaky)


def fake_labels(frame, rng):
    """Give each participant a random label drawn from the real per-participant labels."""
    people = frame['participant'].unique()
    real = frame.groupby('participant')['label'].agg(lambda v: v.mode()[0])
    mapping = dict(zip(people, rng.permutation(real[people].to_numpy())))
    return frame['participant'].map(mapping).map(CLASSES.index).to_numpy()


def controls(frame, X, y, weights, groups, final, final_lopo_accuracy):
    """Does the deployed setup score on labels that carry no information about frustration?

    Fake labels + random split: a high score means it recognizes the person.
    Fake labels + LOPO (a permutation test): how often chance alone matches the real LOPO score."""
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(SEED)
    scores = []
    for _ in range(FAKE_LABEL_ROUNDS):
        labels = fake_labels(frame, rng)
        pred = np.zeros(len(y), int)
        for train, test in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, labels):
            pred[test] = fit(*final, X[train], labels[train], weights[train]).predict(X[test])
        scores.append(float((pred == labels).mean()))
    chance = []
    people = np.unique(groups)
    for _ in range(PERMUTATION_ROUNDS):
        labels = fake_labels(frame, rng)
        pred = np.zeros(len(y), int)
        for person in people:
            test = groups == person
            pred[test] = fit(*final, X[~test], labels[~test], weights[~test]).predict(X[test])
        chance.append(float((pred == labels).mean()))
    chance = np.array(chance)
    return dict(
        fakeLabelRandomSplitAccuracy=round(float(np.mean(scores)), 4),
        fakeLabelRandomSplitRange=[round(min(scores), 4), round(max(scores), 4)],
        permutation=dict(rounds=PERMUTATION_ROUNDS, realLopoAccuracy=round(final_lopo_accuracy, 4),
                         chanceMean=round(float(chance.mean()), 4),
                         chance95th=round(float(np.percentile(chance, 95)), 4),
                         pValue=round(float((1 + (chance >= final_lopo_accuracy).sum()) / (1 + len(chance))), 4)))


def previous_model_scores():
    """The scores of the model this one replaced, shown beside the new ones so nothing is hidden."""
    try:
        report = json.loads((ROOT / 'models/team3_frustration_seen_report.json').read_text())
    except OSError:
        return None
    by_name = {e['name']: e['accuracy'] for e in report['evaluations']}
    return dict(family=report['selectedModel']['family'], params=report['selectedModel']['params'],
                randomSplitAccuracy=by_name['Random window split (5-fold)'],
                newPeopleAccuracy=by_name['New people (leave one participant out)'],
                fakeLabelRandomSplitAccuracy=by_name['Fake labels, random split (control)'])


# ---------------------------------------------------------------- export

def export(family, model):
    if family == 'logistic':
        scaler, clf = model.steps[0][1], model.steps[-1][1]
        return dict(modelType='logistic', mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                    coefficients=clf.coef_.tolist(), intercepts=clf.intercept_.tolist())
    if family in ('random-forest', 'extra-trees'):
        trees = []
        for estimator in model.estimators_:
            t = estimator.tree_
            trees.append(dict(left=t.children_left.tolist(), right=t.children_right.tolist(),
                              feature=[max(0, int(f)) for f in t.feature], threshold=t.threshold.tolist(),
                              value=t.value[:, 0, :].tolist()))
        return dict(modelType='tree-ensemble', trees=trees)
    trees = []
    for dump in model.get_booster().get_dump(dump_format='json'):
        nodes, order = {}, []
        def visit(node):
            nodes[node['nodeid']] = node
            order.append(node['nodeid'])
            for child in node.get('children', []):
                visit(child)
        visit(json.loads(dump))
        index = {nid: i for i, nid in enumerate(order)}
        tree = dict(strict=True, left=[], right=[], feature=[], threshold=[], value=[])
        for nid in order:
            node = nodes[nid]
            leaf = 'leaf' in node
            tree['left'].append(-1 if leaf else index[node['yes']])
            tree['right'].append(-1 if leaf else index[node['no']])
            tree['feature'].append(0 if leaf else int(node['split'].lstrip('f')))
            tree['threshold'].append(0.0 if leaf else float(node['split_condition']))
            tree['value'].append([float(node['leaf'])] if leaf else [0.0])
        trees.append(tree)
    return dict(modelType='xgboost', trees=trees, baseMargins=[0.0] * len(CLASSES))


def calibrate_xgboost(exported, model, X):
    """Recover XGBoost's per-class base margin so exported margins match exactly."""
    import xgboost
    margins = model.get_booster().predict(xgboost.DMatrix(X), output_margin=True)
    walked = np.zeros_like(margins)
    for index, tree in enumerate(exported['trees']):
        for row, x in enumerate(X):
            walked[row, index % len(CLASSES)] += FrustrationModel._walk(tree, list(x))[0]
    exported['baseMargins'] = (margins - walked).mean(axis=0).tolist()


def export_checked(family, model, X, meta):
    """Export, then refuse to save unless the portable model reproduces the library exactly."""
    exported = export(family, model)
    if exported['modelType'] == 'xgboost':
        calibrate_xgboost(exported, model, X)
    exported.update(meta)
    portable = FrustrationModel.__new__(FrustrationModel)
    portable.data, portable.kind = exported, exported['modelType']
    mine = np.array([portable.predict_proba(dict(zip(FEATURES, row))) for row in X])
    gap = float(np.abs(model.predict_proba(X) - mine).max())
    if gap > 1e-5:
        raise SystemExit(f'Exported model disagrees with the trained model (max probability gap {gap:.2e}); not saved.')
    return exported, gap


def model_meta(frame, sessions, family, params, hop, evaluation):
    normalization = params.get('normalization', NORMALIZATIONS[0])
    return dict(schemaVersion=1, features=FEATURES, classes=CLASSES, family=family, params=params,
                normalization=normalization,
                window=dict(seconds=WINDOW_SECONDS, minSeconds=MIN_WINDOW_SECONDS, hopSeconds=hop,
                            sessionCapSeconds=SESSION_CAP_SECONDS, scaling=normalization),
                inputs=dict(eda='any unit (scale-free)', temperature='any unit (scale-free)'),
                training=dict(source='labelled-sessions', participants=int(frame.participant.nunique()),
                              sessions=len(sessions), windows=len(frame),
                              classWindows={c: int((frame.label == c).sum()) for c in CLASSES},
                              trainedAt=datetime.now(timezone.utc).isoformat(timespec='seconds')),
                evaluation=evaluation)


def train_seen(args, sessions):
    """The team's chosen high-capacity model, with every way of scoring it recorded side by side."""
    from sklearn.model_selection import StratifiedKFold
    family, params = SEEN_MODEL
    frame = build_windows(sessions, hop=SEEN_HOP_SECONDS).reset_index(drop=True)
    X, y = frame[FEATURES].to_numpy(float), frame['label'].map(CLASSES.index).to_numpy()
    groups = frame['participant'].to_numpy()
    print(f'{len(sessions)} sessions, {frame.participant.nunique()} participants, {len(frame)} windows '
          f'(60 s every {SEEN_HOP_SECONDS} s); {family} {params}', flush=True)

    def random_split(labels):
        pred = np.zeros(len(labels), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, labels):
            pred[te] = fit(family, params, X[tr], labels[tr], None).predict(X[te])
        return pred

    def held_out(test_masks):
        pred, mask = np.zeros(len(y), int), np.zeros(len(y), bool)
        for te in test_masks:
            pred[te] = fit(family, params, X[~te], y[~te], None).predict(X[te])
            mask |= te
        return float((pred[mask] == y[mask]).mean())

    split_pred = random_split(y)
    headline = summarize(frame, y, split_pred)
    train_acc = float((fit(family, params, X, y, None).predict(X) == y).mean())
    late = np.zeros(len(y), bool)
    early = np.zeros(len(y), bool)
    for _, rows in frame.groupby('session').groups.items():
        rows = list(rows)
        cut = int(len(rows) * .7)
        early[rows[:cut]] = True
        late[rows[cut + 5:]] = True  # skip windows overlapping the training part
    temporal = float((fit(family, params, X[early], y[early], None).predict(X[late]) == y[late]).mean())
    two = frame.groupby('participant')['session'].nunique()
    other_session = held_out([(frame.session == s).to_numpy() for s in frame.session.unique()
                              if two[frame.participant[frame.session == s].iloc[0]] > 1])
    unseen = held_out([groups == person for person in np.unique(groups)])
    rng = np.random.default_rng(SEED)
    people = frame.participant.unique()
    real = frame.groupby('participant')['label'].agg(lambda v: v.mode()[0])
    fake_scores = []
    for _ in range(10):
        fake = dict(zip(people, rng.permutation(real[people].to_numpy())))
        labels = frame.participant.map(fake).map(CLASSES.index).to_numpy()
        fake_scores.append(float((random_split(labels) == labels).mean()))
    evaluations = [
        dict(name='Random window split (5-fold)', accuracy=headline['windowAccuracy'], scope='seen participants',
             meaning='Headline. Test windows come from people and sessions in the training data.'),
        dict(name='Training data itself', accuracy=round(train_acc, 4), scope='seen participants',
             meaning='Memorization ceiling.'),
        dict(name='Later part of each session', accuracy=round(temporal, 4), scope='seen participants',
             meaning='Train on the first 70% of every session, test on the end without overlap.'),
        dict(name="Same person's other session", accuracy=round(other_session, 4), scope='seen participants',
             meaning='A new session from someone in the training data.'),
        dict(name='New people (leave one participant out)', accuracy=round(unseen, 4), scope='unseen participants',
             meaning='Expected accuracy for workers not in the training data.'),
        dict(name='Fake labels, random split (control)', accuracy=round(float(np.mean(fake_scores)), 4),
             scope='seen participants', range=[round(min(fake_scores), 4), round(max(fake_scores), 4)],
             meaning='Each participant given a random label; a high score here is person recognition, not frustration.'),
    ]
    model = fit(family, params, X, y, None)
    meta = model_meta(frame, sessions, family, params, SEEN_HOP_SECONDS, dict(
        method='random window split (5-fold)', scope='seen participants', metrics=headline,
        unseenParticipantAccuracy=round(unseen, 4), fakeLabelAccuracy=round(float(np.mean(fake_scores)), 4)))
    exported, gap = export_checked(family, model, X, meta)
    report = dict(profile='seen-participants', generatedAt=meta['training']['trainedAt'],
                  data=dict(meta['training'], folder=args.data.name),
                  selectedModel=dict(family=family, params=params, exportCheckMaxGap=gap),
                  headline=dict(accuracy=headline['windowAccuracy'], scope='seen participants',
                                method='random window split (5-fold)'),
                  evaluations=evaluations, randomSplitDetail=headline)
    return exported, report, evaluations, gap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', choices=('seen', 'unseen'), default='unseen')
    parser.add_argument('--normalizations', nargs='+', choices=NORMALIZATIONS, default=list(NORMALIZATIONS),
                        help='Per-wearer scalings the nested selection may choose from')
    parser.add_argument('--deploy-normalization', choices=NORMALIZATIONS, default=DEPLOY_NORMALIZATION,
                        help='Scaling of the exported model; the reported accuracy still lets nested selection choose')
    parser.add_argument('--data', type=Path, default=ROOT / '.local/data/Frustration')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    warnings.filterwarnings('ignore', category=UserWarning)
    stem = 'team3_frustration_seen' if args.profile == 'seen' else 'team3_frustration'
    args.output = args.output or ROOT / f'models/{stem}_model.json'
    args.report = args.report or ROOT / f'models/{stem}_report.json'

    sessions = load_sessions(args.data)
    if args.profile == 'seen':
        exported, report, evaluations, gap = train_seen(args, sessions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, exported, indent=None)
        atomic_json(args.report, report)
        print()
        for e in evaluations:
            print(f"  {e['name']:42s} {e['accuracy']:.3f}  ({e['scope']})")
        print(f'Saved {args.output} and {args.report}; export check max gap {gap:.1e}')
        return
    frames = {name: build_windows(sessions, normalization=name) for name in args.normalizations}
    tables = {name: f[FEATURES].to_numpy(float) for name, f in frames.items()}
    frame = frames[args.normalizations[0]]
    y = frame['label'].map(CLASSES.index).to_numpy()
    groups = frame['participant'].to_numpy()
    weights = 1 / frame.groupby('session')['session'].transform('count').to_numpy(float)
    configs = candidates(args.normalizations)
    print(f'{len(sessions)} sessions, {frame.participant.nunique()} participants, {len(frame)} windows '
          f'({dict(frame.label.value_counts())}); {len(configs)} candidate settings', flush=True)

    print('Nested leave-one-participant-out evaluation:', flush=True)
    preds, chosen = nested_lopo(tables, y, weights, groups, configs)
    results = {name: summarize(frame, y, p) for name, p in preds.items()}
    results['selected']['windowAccuracy95CI'] = bootstrap_ci(frame, y, preds['selected'])
    per_person = pd.DataFrame(dict(participant=groups, label=frame['label'], correct=preds['selected'] == y))
    results['selected']['perParticipant'] = [
        dict(participant=str(pid), labels=sorted(set(rows['label'])), windows=int(len(rows)), accuracy=round(float(rows['correct'].mean()), 4))
        for pid, rows in per_person.groupby('participant', sort=False)]

    print('Final selection on all participants:', flush=True)
    deployable = [c for c in configs if c[1]['normalization'] == args.deploy_normalization] or configs
    final, final_scores = select(deployable, tables, y, weights, groups)
    _, all_scores = select(configs, tables, y, weights, groups)
    X = tables[final[1]['normalization']]
    majority, leaky = baselines(frame, X, y, weights, groups, final)
    # The final settings alone (no outer selection), which is what the permutation test repeats.
    fixed = np.zeros(len(y), int)
    for person in np.unique(groups):
        test = groups == person
        fixed[test] = fit(*final, X[~test], y[~test], weights[~test]).predict(X[test])
    print(f'Controls ({FAKE_LABEL_ROUNDS} fake-label rounds, {PERMUTATION_ROUNDS} permutations):', flush=True)
    control = controls(frame, X, y, weights, groups, final, float((fixed == y).mean()))
    previous = previous_model_scores()
    model = fit(*final, X, y, weights)
    meta = model_meta(frame, sessions, final[0], final[1], HOP_SECONDS, dict(
        method='nested leave-one-participant-out', scope='new people', metrics=results['selected'],
        deployedNormalizationAccuracy=results.get(final[1]['normalization'], {}).get('windowAccuracy'),
        majorityBaseline=majority['windowAccuracy'], randomSplitAccuracy=leaky['windowAccuracy']))
    exported, gap = export_checked(final[0], model, X, meta)

    report = dict(profile='unseen-participants', generatedAt=meta['training']['trainedAt'], data=dict(meta['training'], folder=args.data.name),
                  selectedModel=dict(family=final[0], params=final[1], exportCheckMaxGap=gap),
                  nestedLopo=results, outerFoldChoices=chosen,
                  baselines=dict(majorityClassLopo=majority, randomWindowSplitLeaky=leaky),
                  controls=control, previousModel=previous,
                  finalSelectionInnerMacroF1={f'{k[0]} {k[1]}': round(float(v), 4) for k, v in all_scores.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, exported, indent=None)
    atomic_json(args.report, report)

    print('\nResults on unseen participants (nested LOPO):')
    print(f"  {'model':16s} accuracy  macro-F1  session-vote  recall low/med/high")
    for name in [*dict.fromkeys(f for f, _ in configs), *args.normalizations, 'selected']:
        r = results[name]
        recall = '/'.join(f"{r['recall'][c]:.2f}" for c in CLASSES)
        print(f"  {name:16s}  {r['windowAccuracy']:.3f}    {r['macroF1']:.3f}      {r['sessionVoteAccuracy']:.3f}      {recall}")
    ci = results['selected']['windowAccuracy95CI']
    print(f"  majority class    {majority['windowAccuracy']:.3f}")
    print(f"\nDeployed: {final[0]} {final[1]}; nested LOPO accuracy with the scaling also selected "
          f"{results['selected']['windowAccuracy']:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f})")
    print(f"Random window split (leaky, for comparison only): {leaky['windowAccuracy']:.3f}")
    print(f"Fake labels, random split (control): {control['fakeLabelRandomSplitAccuracy']:.3f}")
    print(f"Permutation test: chance mean {control['permutation']['chanceMean']:.3f}, 95th percentile "
          f"{control['permutation']['chance95th']:.3f}, p = {control['permutation']['pValue']}")
    print(f'Saved {args.output} and {args.report}; export check max gap {gap:.1e}')


if __name__ == '__main__':
    main()
