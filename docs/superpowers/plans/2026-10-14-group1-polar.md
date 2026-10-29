# Group 1 Polar implementation plan

> Use test-driven implementation and independent final review. Work in the existing feature checkout; preserve the user's untracked assignment sources.

**Goal:** Explain Group 1's assignment and deliver a usable experimental fatigue/posture feedback workflow on the existing dashboard.

**Sources:** `prompt`; `Polar H10 Handoff.pdf`; `POLAR H10 PREDICTION - Decision Based Rule (AutoRecovered).pdf`; `ML-Feedback-Decision Based rule.pdf`, especially page 51 fatigue and pages 65–66 posture. The referenced Experimental setup.xlsx is absent.

**Architecture:** Keep the upstream EXG RMS signal, and add independent HR/RR and accelerometer inputs to a browser-local session. A pure rule engine supplies the panel, demo, and downloadable experiment log. A separate Python converter handles recorded CSV/XLSX timestamps and accelerometer angles. Existing wristband ML stays independent.

## Decisions and constraints

- No trained fatigue model or labeled training set exists. Results are experimental rule classifications, never an ML probability or verified medical diagnosis.
- Fatigue requires all three strict thresholds relative to baseline: low +5/-15/-15; medium +10/-25/-20; high +15/-35/-30; very high +25/-40/-35 (HR/RMSSD/SDNN percentages). Highest matched tier wins; no match means no rule matched, not healthy.
- Proposed defaults where the assignment is silent: 60-second baseline and rolling window, at least 30 RR intervals, five-second freshness, RR plausibility 300–2000 ms, sample SDNN. Invalid intervals and gaps must not be bridged.
- Feedback dwell selects the lower end of the provided ranges: medium 30 s, high 60 s, very high 120 s. No fabricated rapid-deterioration rule. Text always available, opt-in browser speech once per eligible event; no feedback on unavailable data.
- HRV alone cannot identify heat, sleep loss, or work duration. Explain the measured cardiovascular pattern; optional operator-reported context must be clearly separate from detection.
- Posture uses the supplied atan2 pitch/roll formulas and absolute upright-relative angles; poor >20/10 degrees, critical >45/20. No yaw, overhead reach, or validated ergonomic risk claims from a chest accelerometer.
- Preserve original source documents, startup without npm, both HTML hooks and existing EXG visualization. Rebuild the tracked bundle.

## Tasks

- [x] Engine: `sensor/fatigue.js` plus Node tests for HRV formulas, thresholds/boundaries, baseline, invalid/stale inputs, dwell, posture reference, session resets.
- [x] Transport and UI: HR/RR decoding, guarded controller wiring, optional ACC failure handling; baseline/calibration/feedback/participant/log/demo controls within existing panel. Browser tests exercise the shipped bundle and sensor packets.
- [x] Recordings: `polar_recordings.py`, CSV examples, optional XLSX dependency, Python tests. Explicit Unix units and column mappings; preserve ECG data and compute ACC angles without inventing HR.
- [x] Handoff: `docs/GROUP1_HANDOFF.md` explains assignment, exact threshold/cause/output table, assumptions, demo/live/conversion instructions, what to submit and remaining hardware/data validation. Update README to reflect actual feature scope.
- [x] Verification: Node, Python, build, browser suite, screenshot review, independent code review; fix material findings and leave changes reviewable in this checkout.

## Review focus

Missing RR despite fresh HR; gaps during baseline and sustained feedback; baseline accidentally reused after reconnect; degree/sign/unit mistakes; demo data leaking into live outputs; malformed recordings and timestamp precision; participant changes during a session.

## Completion evidence

Build succeeded; 42 Node tests, 14 Python tests with XLSX enabled, and five Chromium browser scenarios passed. Desktop and mobile screenshots inspected. Independent review findings about retry exhaustion, stale exported baseline metadata and stale calibration text were reproduced and fixed. Source assignment files remain untouched; changes are left in the existing feature checkout.
