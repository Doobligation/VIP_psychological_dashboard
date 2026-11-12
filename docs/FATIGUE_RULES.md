# Current fatigue rules

> **Update (2026-10-29):** Feedback timing is superseded by the Team 3 assignment rules in
> [the Team 3 guide](TEAM3_FRUSTRATION_FATIGUE.md): risk levels are Low/Medium/High (no
> separate "no rule matched" level), Medium never gives feedback, and High gives feedback
> after 3 consecutive High predictions instead of a 30/60-second dwell. The thresholds
> below are unchanged.

The active fatigue workflow is **rule-based**. Its authority is the section after
“USE THIS THIS IS THE NEWEST SET OF INSTRUCTIONS” in `fatigue`,
specifically section 6, “Fatigue-Related Stress Thresholds.” This replaces the older
Polar fatigue thresholds and model-preview workflow. Group 1 posture rules are unchanged.

**Temporary troubleshooting setting:** the resting baseline and rolling feature
window are currently **5 seconds**, with at least **3 valid RR intervals** and
**3 HR observations**, and a 2-second freshness limit. This replaces the previous
60-second/30-observation settings at the user's request. Feedback persistence
remains separately configured. Restore the longer window after troubleshooting.

## Quick test score

At the user's request, **Fatigue score · quick test** appears immediately from a
nonzero reported HR, even if contact is flagged bad and RR intervals are absent.
It uses the arbitrary display scale `round(clamp((HR - 60) / 80 × 100, 0, 100))`.
For example, 111 bpm produces 64/100. The 60–140 bpm endpoints are test settings,
not evidence-based fatigue thresholds, and this is not a measured fatigue percentage.
Contact warnings remain visible. No resting baseline, HRV, EDA, or ML is used.

An invalid/zero HR packet cannot create a score. The last nonzero HR may be held
for at most 10 seconds with its age displayed; the score then clears. Reconnecting
resets it. Exports include `snapshot.quickFatigue` (or `observations[].quickFatigue`)
with `source: hr-only-debug`, `provisional: true`, the measurement time, source HR,
scale, and contact warning. The latest API also clears it when the saved reading
is stale. Historical exports preserve what was shown at observation time.

This score does not supply missing features, trigger the full-rule feedback loop,
or make a measurement eligible for participant ratings. The full fatigue rules
continue to report their own availability and classification below it.

## Full fatigue rules

Each signal is compared with its own personal resting baseline:

```text
change_percent = 100 × (current − baseline) / baseline
```

| Result | All required conditions |
| --- | --- |
| Low | RMSSD change < −15%; SDNN change < −15% |
| Medium | RMSSD change < −25%; SDNN change < −20% |
| High | RMSSD change < −35%; SDNN change < −30%; HR change < +10%; EDA change < +15% |

The inequalities are strict: exactly −25% RMSSD does not satisfy the medium row,
and exactly +10% HR or +15% EDA does not satisfy high. The highest matching row wins.
There is no very-high rule in this source. HR and EDA are **upper bounds** in the
high rule, not the increasing-HR thresholds used by the previous implementation.

Low and medium need both HRV changes. Missing HRV or an invalid/zero HRV baseline
makes the result unavailable. High additionally needs measured HR and EDA changes.
When medium matches but high lacks an otherwise-needed signal, the result reports
medium **and explicitly lists the unassessed high rule**. Missing EDA is never zero-filled.
“No rule matched” describes available evidence; it does not establish absence of fatigue.

## What to do in the dashboard

Run `python3 serve_dashboard.py`, open the dashboard, and use **Group 1 · Fatigue &
posture**. No model file is needed. The model upload/estimate panel has been removed
from this workflow; the backend also bypasses model loading and blending for the
active configuration, even if a model path is supplied.

For live Polar use, enter a participant ID, connect, capture the resting baseline
and calibrate upright posture. H10 provides HR/RR and acceleration, but no EDA.
Its stream can assess low and medium; it cannot establish the complete high rule.
The EDA card and explanation make this limitation visible.

**Start demo** provides clearly labeled synthetic HR/RR and EDA, including a
baseline, medium/high patterns and recovery. This demonstrates the rule logic and
feedback loop, not actual Empatica or Polar measurements. Independent self-ratings
remain available for live sessions to evaluate the rules; demo ratings remain disabled.

### If fatigue stays unavailable

The saved-reading counter includes received packets that fail the signal checks.
Use the **Last HR packet** line and its explanation to distinguish a zero heart
rate, missing RR intervals, reported contact loss, and a stale stream. The EXG
waveform uses a separate stream: a moving waveform alone does not establish that
the native HR/RR input is usable. Once HR/RR is current, click **Capture resting
baseline (5s)** and rest through calibration; the temporary setting requires at
least 3 valid RR intervals and 3 HR observations over that period.

Browser session JSON observations include `rawHeartRate` (the latest received
HR/RR packet, including rejected values) and `heartSignal` (its current status).
The processed `hr`, `rmssd`, and `sdnn` fields can remain null when the packet is
unsuitable. Raw packet timestamps identify repeated snapshots of the same packet.
**Download saved session** includes every saved HR packet under `sample.raw`.
Neither export is a full raw ECG/ACC recording like the separate Polar visualizer.

## EDA and external inputs

EDA uses microSiemens. An optional measured EDA stream is accepted alongside HR/RR
in `POST /samples` raw readings:

```json
{
  "session_id": "session-unique-id",
  "participant": "P01",
  "mode": "live",
  "sample_id": "sample-unique-id",
  "timestamp_ms": 1792684800000,
  "raw": {"hr": 80, "rr_intervals_ms": [740, 760], "eda_us": 2.1}
}
```

Use actual current timestamps for live input. Raw ingestion automatically captures
the initial resting baseline; EDA needs its own complete fresh window during that
baseline. Gaps or invalid EDA invalidate its current window without discarding
otherwise-usable HRV. A zero EDA baseline cannot support a percentage comparison.

Already-computed snapshot inputs may supply `eda`, `baseline.eda`, and explicit
`edaFresh: true`, alongside the existing baseline/window/quality metadata. Only set
that flag after checking EDA freshness and window coverage. The server recomputes
baseline changes rather than trusting a supplied classification.

The existing Empatica cloud reader is **not newly connected to this fatigue flow**:
it currently reads EDA, temperature and pulse-rate summaries without the beat data
needed for RMSSD/SDNN. A measured Empatica session must supply appropriate beat
intervals and aligned EDA to the input contract above. No data or baseline is invented.

## Feedback and files

- `config/rules.json`: the three source-backed fatigue rules and their provenance.
- `config/feedback_table.json`: retained medium/high messages and 30s/60s persistence.
- `pipeline/`: raw features, rule evaluation, observation-driven feedback decisions.
- `sensor/`: browser processing, rule display, EDA availability and ratings.
- `storage/` and `serving/`: persistent readings, alerts, responses and outcomes.

The new file supplies fatigue thresholds, but its feedback table describes
**frustration**. Those messages are not reassigned to fatigue. Existing fatigue
messages, timing, rapid-rise policy, cooldowns and warning caps are retained as
documented prototype choices. Rule scores are heuristic severity values, not stress
probabilities. The source's other stress-cause tables and ML probability formula are
outside this Group 1 fatigue change; the user's explicit rule-only instruction governs.

Restart the server and refresh the page after changing configuration. Older ML
experiments remain in the repository for reference and are not loaded by this flow.
Tests for those experiments use the old rules in an explicit test fixture.

## Verification

```bash
npm test
npm run build
venv/bin/python -m unittest discover -s tests -p 'test_*.py'
npm run test:browser
python3 tools/demo_stream.py
```

Set `POLAR_TEST_PORT=8001` to test while the regular server runs on port 8000.
The rule tests cover every strict threshold, high-rule missing inputs, fresh/stale
EDA, nonzero baselines and model bypass in live and demo modes. Hardware measurements
and physiological validity still require an actual study session.
