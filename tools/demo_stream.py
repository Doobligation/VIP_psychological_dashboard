#!/usr/bin/env python3
"""Scripted Team 3 frustration/fatigue demo replay; no hardware or labels.

Each step of config/demo_scenario.json is one prediction: baseline % changes for EDA,
SCR, HR, RMSSD and SDNN. By default the steps run through FeedbackPipeline locally.
Use --url http://127.0.0.1:8000 to POST each demo sample to /samples instead. The
output is synthetic software behavior, never evidence of physiological performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCENARIO = ROOT / 'config' / 'demo_scenario.json'


def demo_steps(scenario=None):
    """Yield (phase, changes) per prediction, in scenario order."""
    scenario = scenario or json.loads(SCENARIO.read_text(encoding='utf-8'))
    for phase in scenario['phases']:
        for name in phase['steps']:
            yield phase, scenario['patterns'][name]


def demo_samples(session_id='synthetic-demo', *, start_ms=1800000000000, feature_policy=None, scenario=None):
    """Fresh synthetic summary windows, one second apart, at the active feature protocol."""
    from pipeline.config import load_config, validate_feature_policy
    policy = validate_feature_policy(feature_policy if feature_policy is not None else
                                     load_config(None, 'rules.json')['featurePolicy'])
    scenario = scenario or json.loads(SCENARIO.read_text(encoding='utf-8'))
    baseline = scenario['baseline']
    for index, (phase, changes) in enumerate(demo_steps(scenario)):
        current = {key: baseline[key] * (1 + changes[key] / 100) for key in baseline}
        snapshot = dict(current, edaFresh=True, baseline=dict(baseline), baselineState='ready', fresh=True,
                        rrCount=max(60, policy['minRr']), windowSeconds=policy['windowSeconds'],
                        requiredWindowSeconds=policy['windowSeconds'], featurePolicy=dict(policy),
                        postureCalibrated=True, posture=dict(level='good', pitch=0, roll=0))
        yield dict(session_id=session_id, participant='synthetic-demo-participant', mode='demo',
                   sample_id=f'demo-{index:04d}', timestamp_ms=start_ms + index * 1000,
                   snapshot=snapshot, raw=dict(source='synthetic-script', phase=phase['title']))


def run_demo(*, url=None, session_id='synthetic-demo', start_ms=1800000000000, interval=0):
    """Run through the same pipeline as ingestion, or send demo-only API readings."""
    if url:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('Demo API URL must use loopback HTTP.')
        endpoint = url.rstrip('/') + '/samples'
        pipeline = None
    else:
        from pipeline import FeedbackPipeline
        pipeline = FeedbackPipeline()
    for sample in demo_samples(session_id, start_ms=start_ms):
        if pipeline is not None:
            result = pipeline.process(sample)
        else:
            request = urllib.request.Request(endpoint, data=json.dumps(sample, allow_nan=False).encode(),
                                             headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.load(response)
        yield result
        if interval:
            time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', help='Optional loopback server origin.')
    parser.add_argument('--session-id', default=None)
    parser.add_argument('--interval', type=float, default=0, help='Wall-clock seconds between samples; default fast replay.')
    parser.add_argument('--json', action='store_true', help='Print full JSON results instead of a summary line per prediction.')
    args = parser.parse_args(argv)
    if not 0 <= args.interval <= 60:
        parser.error('--interval must be between 0 and 60 seconds.')
    try:
        for result in run_demo(url=args.url, session_id=args.session_id or 'synthetic-demo-' + uuid.uuid4().hex,
                               start_ms=int(time.time() * 1000), interval=args.interval):
            if args.json:
                print(json.dumps(result, allow_nan=False))
                continue
            cells = []
            for name in ('frustration', 'fatigue'):
                state = result['states'][name]
                streak = state['streak']
                cells.append(f"{name}={state['level']:<8} high {streak['count']}/{streak['required']}")
            alerts = ' | '.join(f"FEEDBACK {a['state']}: {a['message']}" for a in result['alerts'])
            print('  '.join(cells), alerts)
    except (ValueError, OSError, urllib.error.URLError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
