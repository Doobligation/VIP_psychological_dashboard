#!/usr/bin/env python3
"""Replay a labelled session's EDA and skin temperature through the frustration pipeline.

The replay goes through the same FeedbackPipeline as live data, one sample per second, and
prints each new ML prediction with the 3-consecutive-High feedback decision. By default the
model is retrained without the replayed participant (same family and settings as the
deployed model), so the output shows how it does on a person it has never seen. Use
--deployed-model to replay with the shipped model instead (optimistic for its own training
participants). Participant data stays in .local/data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def holdout_model(sessions, participant, deployed, directory):
    """Fit the deployed family/settings on everyone except `participant` and export it."""
    from pipeline.frustration_model import CLASSES, FEATURES
    from training.train_frustration_model import build_windows, calibrate_xgboost, export, fit
    # Same recipe as the deployed model: its window step, and unweighted for the seen profile.
    normalization = deployed.get('normalization', 'running-minmax')
    frame = build_windows(sessions, hop=deployed['window']['hopSeconds'], normalization=normalization)
    train = frame[frame.participant != participant]
    X, y = train[FEATURES].to_numpy(float), train['label'].map(CLASSES.index).to_numpy()
    weights = (None if deployed['evaluation'].get('scope') == 'seen participants'
               else 1 / train.groupby('session')['session'].transform('count').to_numpy(float))
    model = fit(deployed['family'], deployed['params'], X, y, weights)
    exported = dict(export(deployed['family'], model), schemaVersion=1, features=FEATURES, classes=CLASSES,
                    family=deployed['family'], params=deployed['params'], normalization=normalization,
                    heldOutParticipant=participant)
    if exported['modelType'] == 'xgboost':
        calibrate_xgboost(exported, model, X)
    path = Path(directory) / f'holdout_{participant}.json'
    path.write_text(json.dumps(exported))
    return path


def main(argv=None):
    from pipeline import FeedbackPipeline
    from pipeline.config import load_config
    from training.train_frustration_model import load_sessions
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session', help='Session name, e.g. P15_Passive_High')
    parser.add_argument('--data', type=Path, default=ROOT / '.local/data/Frustration')
    parser.add_argument('--deployed-model', action='store_true', help='Use the shipped model (not held out).')
    args = parser.parse_args(argv)
    sessions = load_sessions(args.data)
    session = next((s for s in sessions if s['name'].lower() == args.session.lower()), None)
    if session is None:
        parser.error(f"unknown session; choose from: {', '.join(sorted(s['name'] for s in sessions))}")
    rules = load_config(None, 'rules.json')
    deployed = json.loads((ROOT / rules['states']['frustration']['modelFile']).read_text())
    with tempfile.TemporaryDirectory() as directory:
        if not args.deployed_model:
            rules['states']['frustration']['modelFile'] = str(holdout_model(sessions, session['participant'], deployed, directory))
        pipe = FeedbackPipeline(rules=rules)
        print(f"{session['name']}: true label {session['label'].upper()} · "
              f"{'held-out model (never saw ' + session['participant'] + ')' if not args.deployed_model else 'deployed model'}")
        counts, last = {}, None
        print('  time   level   probabilities                     (printed when the level changes or feedback fires)')
        for i, (eda, temp) in enumerate(zip(session['eda'], session['temp'])):
            raw = {k: float(v) for k, v in (('eda_us', eda), ('temp_c', temp)) if np.isfinite(v)}
            out = pipe.process(dict(session_id='replay', participant=session['participant'], mode='demo',
                                    sample_id=str(i), timestamp_ms=i * 1000, raw=raw))
            state = out['states']['frustration']
            if state['model']['status'] != 'ml':
                continue
            counts[state['level']] = counts.get(state['level'], 0) + 1
            p = state['model']['probabilities']
            line = (f"  {i // 60:2d}:{i % 60:02d}  {state['level']:<6}  low {p['low']:.2f} med {p['medium']:.2f} high {p['high']:.2f}"
                    f"  high streak {state['streak']['count']}")
            if state['alert']:
                line += f"  -> FEEDBACK: {state['alert']['message']}"
            if state['level'] != last or state['alert']:
                print(line)
            last = state['level']
        total = sum(counts.values())
        if total:
            share = ', '.join(f'{k} {v / total:.0%}' for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
            print(f"Per-second predictions: {share} (true: {session['label']}); "
                  f"agreement {counts.get(session['label'], 0) / total:.0%}")


if __name__ == '__main__':
    main()
