#!/usr/bin/env python3
"""Retrain the frustration model with workers' check-in answers; switch only if it is not worse.

Steps 9 and 10 of the Closed-Loop Feedback guide (Nov 11, 2026):

  python3 training/retrain_from_checkins.py           # report: agreement, current vs candidate, verdict
  python3 training/retrain_from_checkins.py --apply   # switch the live model if the verdict allows it

The candidate keeps the deployed model's family, settings and per-wearer scaling. It is trained on the
study windows (.local/data/Frustration) plus every answered check-in that has ML features, each answer
weighted as --weight study windows. An answer of "Yes" is High frustration and "No" is Low; --little
decides "A little". Fatigue answers are scored for agreement only, because fatigue is rules-only.

The candidate is tested two ways against the current model:
  answers  leave one worker out: train without that worker's answers and predict them (balanced
           accuracy, High versus not High). The current model never trained on answers, so it
           predicts them directly.
  study    leave one study participant out, trained with and without the answers (balanced
           accuracy over Low, Medium and High).

--apply switches only when the candidate is at least as good on the answers and loses no more than
--max-study-loss on the study participants (--force overrides that verdict). The old model is copied
to .local/model_backups/ first; restart python3 serve_dashboard.py to load the new one. Reports are
saved under .local/retrain/ because they name workers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline import checkins as checkin_rules
from pipeline.frustration_model import CLASSES, FEATURES, FrustrationModel
from training import train_frustration_model as study
from training.train_model import atomic_json

DEFAULT_DB = ROOT / '.local' / 'closed_loop.sqlite3'
DEFAULT_MODEL = ROOT / 'models' / 'team3_frustration_model.json'
DEFAULT_DATA = ROOT / '.local' / 'data' / 'Frustration'
BACKUP_DIR = ROOT / '.local' / 'model_backups'
REPORT_DIR = ROOT / '.local' / 'retrain'
HIGH, LOW = CLASSES.index('high'), CLASSES.index('low')


def read_checkins(db):
    """Every check-in and a connection for looking up the readings behind them (read-only)."""
    if not Path(db).exists():
        raise SystemExit(f'No database at {db}. Start the dashboard server and collect check-ins first.')
    connection = sqlite3.connect(f'file:{Path(db).resolve()}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute('SELECT data_json FROM checkins ORDER BY asked_ms, sequence').fetchall()
    except sqlite3.OperationalError:
        rows = []  # a database from before check-ins existed
    return [json.loads(row['data_json']) for row in rows], connection


def moment(connection, record):
    """(features, normalization) of the frustration model at the check-in's moment, or (None, reason).

    A question links the reading it asked about. A self-report sent before a delayed upload arrived
    has no reading yet, so the session's ML reading nearest its time is used once it exists."""
    query = ("SELECT result_json FROM readings WHERE {} AND "
             "json_extract(result_json,'$.states.frustration.model.status')='ml' {}")
    row = None
    if record.get('reading_id'):
        row = connection.execute(query.format('id=?', ''), (record['reading_id'],)).fetchone()
    if row is None:
        window = checkin_rules.SELF_REPORT_MATCH_MS
        row = connection.execute(query.format('session_id=? AND timestamp_ms BETWEEN ? AND ?', 'ORDER BY ABS(timestamp_ms-?) LIMIT 1'),
                                 (record['session_id'], record['moment_ms'] - window, record['moment_ms'] + window,
                                  record['moment_ms'])).fetchone()
    if row is None:
        return None, 'no ML features at that moment (no EDA and skin temperature, or the data has not arrived)'
    model = json.loads(row['result_json'])['states']['frustration']['model']
    return model.get('features'), model.get('normalization')


def answer_label(answer, little):
    return HIGH if checkin_rules.counts_as_felt(answer, little) else LOW


def collect_answers(records, connection, little, include_demo, normalization):
    """Answered check-ins usable for training, and why the others were left out."""
    usable, skipped = [], {}
    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1
    for record in records:
        answer = (record.get('answers') or {}).get('frustration')
        if record['status'] != 'answered' or not answer:
            continue
        if record.get('mode') == 'demo' and not include_demo:
            skip('demo session (use --include-demo to practise)')
            continue
        features, found = moment(connection, record)
        if features is None:
            skip(found)
            continue
        if found != normalization:
            skip(f'features scaled with {found}, the model uses {normalization}')
            continue
        usable.append(dict(id=record['id'], worker=record['participant'], session=record['session_id'],
                           kind=record['kind'], answer=answer, label=answer_label(answer, little),
                           x=[float(features[name]) for name in FEATURES]))
    return usable, skipped


def balanced(true, pred):
    from sklearn.metrics import balanced_accuracy_score
    return round(float(balanced_accuracy_score(true, pred)), 4)


def evaluate(current, family, params, study_frame, answers, weight):
    """Current model versus the candidate, on the answers (leave one worker out) and the study (LOPO)."""
    X_s = study_frame[FEATURES].to_numpy(float)
    y_s = study_frame['label'].map(CLASSES.index).to_numpy()
    groups = study_frame['participant'].to_numpy()
    w_s = 1 / study_frame.groupby('session')['session'].transform('count').to_numpy(float)
    X_a = np.array([a['x'] for a in answers], float)
    y_a = np.array([a['label'] for a in answers])
    workers = np.array([a['worker'] for a in answers])
    # One answer weighs as much as `weight` typical study windows.
    w_a = np.full(len(answers), weight * w_s.mean())

    current_answers = np.array([CLASSES.index(current.predict(dict(zip(FEATURES, x)))[0]) for x in X_a])
    candidate_answers = np.zeros(len(answers), int)
    for worker in np.unique(workers):
        test = workers == worker
        model = study.fit(family, params, np.vstack([X_s, X_a[~test]]), np.concatenate([y_s, y_a[~test]]),
                          np.concatenate([w_s, w_a[~test]]))
        candidate_answers[test] = model.predict(X_a[test])

    current_study, candidate_study = np.zeros(len(y_s), int), np.zeros(len(y_s), int)
    for person in np.unique(groups):
        test = groups == person
        current_study[test] = study.fit(family, params, X_s[~test], y_s[~test], w_s[~test]).predict(X_s[test])
        candidate_study[test] = study.fit(family, params, np.vstack([X_s[~test], X_a]), np.concatenate([y_s[~test], y_a]),
                                          np.concatenate([w_s[~test], w_a])).predict(X_s[test])
    high = y_a == HIGH
    return dict(
        answers=dict(method='leave one worker out; balanced accuracy, High versus not High', answers=len(answers),
                     workers=int(len(np.unique(workers))),
                     current=balanced(high, current_answers == HIGH), candidate=balanced(high, candidate_answers == HIGH)),
        study=dict(method='leave one study participant out; balanced accuracy over Low, Medium and High',
                   participants=int(len(np.unique(groups))), windows=int(len(y_s)),
                   current=balanced(y_s, current_study), candidate=balanced(y_s, candidate_study),
                   candidateMetrics=study.summarize(study_frame, y_s, candidate_study)),
    ), (X_s, y_s, w_s, X_a, y_a, w_a)


def verdict(evaluation, max_study_loss):
    """Switch only if as good on the answers and at most max_study_loss worse on the study."""
    a, s = evaluation['answers'], evaluation['study']
    reasons, switch = [], True
    if a['candidate'] >= a['current']:
        reasons.append(f"Answers: candidate {a['candidate']:.3f} is at least the current {a['current']:.3f}.")
    else:
        switch = False
        reasons.append(f"Answers: candidate {a['candidate']:.3f} is below the current {a['current']:.3f}.")
    loss = round(s['current'] - s['candidate'], 4)
    if loss <= 0:
        reasons.append(f"Study: candidate gains {-loss:.3f} balanced accuracy ({s['current']:.3f} to {s['candidate']:.3f}).")
    elif loss <= max_study_loss + 1e-9:
        reasons.append(f"Study: candidate loses {loss:.3f} balanced accuracy (allowed {max_study_loss:.3f}).")
    else:
        switch = False
        reasons.append(f"Study: candidate loses {loss:.3f} balanced accuracy, more than the allowed {max_study_loss:.3f}.")
    if a['workers'] < 2:
        reasons.append('Only one worker answered, so leaving that worker out trains on no answers: '
                       'the answer test cannot show an improvement yet.')
    return dict(switch=switch, reasons=reasons)


def print_agreement(summary):
    counts = summary['counts']
    print(f"Check-ins: {counts['asked']} asked, {counts['answered']} answered, {counts['noAnswer']} no answer, "
          f"{counts['dismissed']} not now, {counts['open']} open; {counts['selfReports']} self-reports.")
    little = 'counts as felt' if summary['littleCountsAs'] == 'yes' else 'counts as not felt'
    if not summary['agreement']:
        print('Agreement: no answered check-ins with a prediction yet.')
        return
    print(f'Agreement per source ("A little" {little}):')
    for row in summary['agreement']:
        high = (f"High predicted {row['predictedHigh']}: {row['yes']} yes, {row['aLittle']} a little, {row['no']} no"
                + (f" ({row['confirmedRate']:.0%} confirmed)" if row['confirmedRate'] is not None else ''))
        print(f"  {row['state'].capitalize():12s} {row['source']}: {row['answers']} answers; {high}; "
              f"all answers agree {row['agreement']:.0%}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='Deployed frustration model to compare and replace')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA, help='Labelled study sessions (outside the repository)')
    parser.add_argument('--little', choices=('yes', 'no'), default='yes', help='Does "A little" count as felt frustrated?')
    parser.add_argument('--weight', type=float, default=3.0, help='Study windows one answer counts as')
    parser.add_argument('--min', type=int, default=20, help='Answers needed, including both felt and "No"')
    parser.add_argument('--max-study-loss', type=float, default=0.02, help='Balanced accuracy the study may lose')
    parser.add_argument('--include-demo', action='store_true', help='Practice only: also train on demo-session answers')
    parser.add_argument('--apply', action='store_true', help='Switch the live model if the verdict allows it')
    parser.add_argument('--force', action='store_true', help='With --apply: switch whatever the verdict')
    parser.add_argument('--report', type=Path, help='Where to save the JSON report (default .local/retrain/)')
    parser.add_argument('--backup-dir', type=Path, default=BACKUP_DIR)
    args = parser.parse_args(argv)
    if args.weight <= 0 or args.min < 2 or args.max_study_loss < 0:
        parser.error('--weight must be positive, --min at least 2 and --max-study-loss not negative.')
    if args.force and not args.apply:
        parser.error('--force only changes what --apply does.')
    if args.apply and args.include_demo:
        parser.error('--apply never trains the live model on demo answers; drop --include-demo.')
    warnings.filterwarnings('ignore', category=UserWarning)

    current = FrustrationModel(args.model)
    family, params, normalization = current.data['family'], current.data['params'], current.normalization
    records, connection = read_checkins(args.db)
    stamp = datetime.now(timezone.utc)
    report = dict(generatedAt=stamp.isoformat(timespec='seconds'), database=str(args.db),
                  model=dict(path=str(args.model), family=family, params=params, normalization=normalization,
                             trainedAt=current.data.get('training', {}).get('trainedAt')),
                  settings=dict(little=args.little, weight=args.weight, min=args.min, maxStudyLoss=args.max_study_loss,
                                includeDemo=args.include_demo),
                  agreement=checkin_rules.summarize(records, args.little), evaluation=None, verdict=None, applied=False)
    print_agreement(report['agreement'])
    try:
        answers, skipped = collect_answers(records, connection, args.little, args.include_demo, normalization)
    finally:
        connection.close()
    labels = [a['label'] for a in answers]
    report['answers'] = dict(usable=len(answers), high=labels.count(HIGH), low=labels.count(LOW),
                             workers=len({a['worker'] for a in answers}), skipped=skipped)
    print(f"\nAnswers usable for training: {len(answers)} (felt frustrated {labels.count(HIGH)}, not {labels.count(LOW)}) "
          f"from {report['answers']['workers']} worker(s)" + (f"; left out: {', '.join(f'{n} {r}' for r, n in skipped.items())}" if skipped else '') + '.')

    enough = len(answers) >= args.min and HIGH in labels and LOW in labels
    if not enough:
        report['verdict'] = dict(switch=False, reasons=[
            f'Not enough answers yet: {len(answers)} usable; need at least {args.min}, including both "felt frustrated" and "No".'])
    else:
        if not args.data.is_dir():
            raise SystemExit(f'Study data not found at {args.data} (labelled sessions stay out of the repository).')
        sessions = study.load_sessions(args.data)
        frame = study.build_windows(sessions, normalization=normalization).reset_index(drop=True)
        print(f'Study: {len(sessions)} sessions, {frame.participant.nunique()} participants, {len(frame)} windows. '
              f'Model: {family} {params}.', flush=True)
        evaluation, data = evaluate(current, family, params, frame, answers, args.weight)
        report['evaluation'] = evaluation
        report['verdict'] = verdict(evaluation, args.max_study_loss)
        a, s = evaluation['answers'], evaluation['study']
        print('\n                          current  candidate')
        print(f"  answers (balanced acc.)  {a['current']:.3f}    {a['candidate']:.3f}    leave one worker out, {a['workers']} worker(s)")
        print(f"  study   (balanced acc.)  {s['current']:.3f}    {s['candidate']:.3f}    leave one participant out, {s['participants']} participants")
    print(f"\nVerdict: {'switch' if report['verdict']['switch'] else 'keep the current model'}.")
    for reason in report['verdict']['reasons']:
        print(f'  {reason}')

    if args.apply and report['evaluation'] and (report['verdict']['switch'] or args.force):
        X_s, y_s, w_s, X_a, y_a, w_a = data
        X, y, w = np.vstack([X_s, X_a]), np.concatenate([y_s, y_a]), np.concatenate([w_s, w_a])
        model = study.fit(family, params, X, y, w)
        meta = study.model_meta(frame, sessions, family, params, study.HOP_SECONDS, dict(
            method='leave one study participant out (fixed settings), check-in answers in training',
            scope='new people', metrics=evaluation['study']['candidateMetrics'],
            balancedAccuracy=evaluation['study']['candidate'],
            checkinAnswers={k: evaluation['answers'][k] for k in ('method', 'answers', 'workers', 'current', 'candidate')},
            previous=dict(trainedAt=current.data.get('training', {}).get('trainedAt'),
                          method=current.data.get('evaluation', {}).get('method'),
                          windowAccuracy=current.data.get('evaluation', {}).get('metrics', {}).get('windowAccuracy'))))
        meta['training'].update(source='labelled-sessions + check-in answers', checkins=dict(
            answers=len(answers), feltFrustrated=labels.count(HIGH), notFrustrated=labels.count(LOW),
            workers=report['answers']['workers'], weight=args.weight, littleCountsAs=args.little,
            forced=bool(args.force and not report['verdict']['switch'])))
        exported, _ = study.export_checked(family, model, X, meta)
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / f"{args.model.stem}-{stamp.strftime('%Y%m%dT%H%M%SZ')}.json"
        shutil.copy2(args.model, backup)
        atomic_json(args.model, exported, indent=None)
        report.update(applied=True, backup=str(backup))
        forced = ' (forced with --force; the verdict was to keep the current model)' if not report['verdict']['switch'] else ''
        print(f'\nSwitched{forced}: {args.model} now holds the candidate. The old model is kept at {backup}.')
        print('Restart python3 serve_dashboard.py to load it. To undo, copy the backup over the model and restart.')
    elif args.apply:
        print('\nNot applied: the current model stays live.')

    target = args.report or REPORT_DIR / f"retrain-{stamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    atomic_json(target, report)
    print(f'Report: {target}')
    return report


if __name__ == '__main__':
    main()
