# Polar H10 fatigue model

**Current update (2026-10-22):** The user supplied a new `fatigue` file and requested rules only. The active three-level rules, EDA requirement, and removed model-upload workflow are documented in [Current fatigue rules](FATIGUE_RULES.md). Older threshold/model instructions below are historical and do not enable ML in the active fatigue pipeline.

The model family is **multinomial logistic regression** on HR, RMSSD and SDNN
percentage changes from a personal baseline. It estimates current category, not future
onset. The professor's pathway now requires **rules first**: live decisions come from
the local server, and synthetic models are limited to demo previews. Sensor analysis
readings are sent only to the local server for persistent logging.

Use [the current workflow guide](PROFESSOR_WORKFLOW.md) for real-label collection and
`training/train_model.py`: leave-one-person-out comparison of rules, ML and blend,
with a candidate saved only when the blend wins. The older trainer below remains an
experiment/bootstrap utility; its output alone is not eligible for live deployment.

## What the shipped model learned

`models/polar_fatigue_model.json` is a **synthetic starter**, fitted to 3,000 generated
feature combinations labeled using the assignment original assignment thresholds.
There are 600 examples for each of none, low, medium, high and very-high. Generated
changes span approximately −20% to +120% HR and −95% to +100% RMSSD/SDNN. The seed is
1031. These ranges and balanced class counts are engineering choices, not measured
worker distributions. Logistic regression only approximates the conjunctions in the
rules, so the model estimate and threshold result can disagree.

**No construction workers were in this training set. There is no worker accuracy
estimate or calibrated fatigue probability.** A normal-looking estimate does not
establish absence of fatigue. The UI identifies synthetic training explicitly. The
live rule pipeline controls sustained feedback and speech until an approved real-label blend is configured. Demo model estimates do not trigger live actions. Missing or out-of-range measurements produce
an unavailable model estimate.

Construction research supports studying HRV alongside independently reported fatigue,
but does not supply this model's weights or validate it. For example, [Anwer et al.
(2023)](https://doi.org/10.1061/JCEMD4.COENG-13100) studied 15 construction workers
performing bar bending/fixing and collected Borg-20 ratings. This project's five-choice
self-rating is **not** that instrument and is **not** a Borg-20 or CR10 score.
The [author's research record](https://researchportal.northumbria.ac.uk/en/publications/identification-and-classification-of-physical-fatigue-in-construc/)
describes that study. A usable labeled worker dataset was not retrieved during this
implementation. Published accuracy figures must not be attributed to this prototype.

## Wear the monitor and collect a session

1. Run `python3 serve_dashboard.py` and open the dashboard in Chrome/Edge on localhost.
2. Enter a stable participant ID before connecting the H10. Use the same ID for all
   sessions belonging to that person; do not use one ID for different people.
3. Capture a resting baseline for 60 seconds. Rule classification becomes available when
   there is a complete fresh window with at least 30 valid RR intervals.
4. During representative work, select **How fatigued do you feel?**, optionally enter
   the current task, and click **Record fatigue rating**. Rate your own experience
   independently of the displayed estimates. Recording is enabled only in live mode
   with valid measurements. Ratings must be at least 60 seconds apart.
5. Click **Export session JSON** before starting another session or leaving the page.
   The export includes the ratings and their exact measurement snapshots, model
   artifacts, model IDs on predictions, baseline, observations and events. It does
   not include raw ECG or beat-by-beat RR. The browser copy is held in page memory; Download saved session exports the local database log. Check its separate save status.

Sensor signals alone are not independent labels. They are paired with the manual
rating at the time it is recorded. Recording a rating does not change the running
model; retraining is an explicit offline step. Demo ratings are disabled. Disconnect
retains the exportable log but clears live estimates; reconnect requires a new baseline.

## Older personal-model experiment (demo preview only)

Use Python 3.10+ in an environment with the ML requirements:

```bash
python3 -m pip install -r requirements-polar-ml.txt
python3 train_polar_fatigue.py --sessions session-P01.json session-P02.json --output models/polar_fatigue_personal.json
```

This reads schema v2 **live** exports only. It rejects malformed/stale snapshots,
non-independent label sources, repeated session IDs, and ratings less than 60 seconds
apart within a session. Features are recomputed from measurements, not trusted from
cached rule results. Keep only the latest export of each session.

The engineering minimum is 10 ratings across at least two levels, with at least three
ratings per included level. This permits testing the pipeline; **it is not a sufficient
sample-size recommendation or validation standard**. Do not intentionally induce
extreme fatigue to fill categories. Levels not represented in training cannot be
predicted by that artifact. A model trained on one person is marked personal, with
evaluation unavailable. Train a study model using multiple people with representative
tasks and fatigue levels.

When every held-out-participant fold leaves all included classes in training, the
trainer reports leave-one-participant-out accuracy, macro-F1 and a confusion matrix.
Each fold fits its own scaler and classifier on its training participants. If fold
coverage is inadequate it reports evaluation unavailable rather than a misleading
score. Fixed `C=1.0` and balanced class weights are used; there is no hyperparameter
search or random split of overlapping sensor windows. These internal results still
need independent construction-task evaluation before deployment decisions.

Use **Load a retrained model → Load trained model JSON** to preview the output during
a demo. Imports cannot enable live blending. To train and enable an eligible live
candidate, follow the new comparison trainer and `serving.server --model` commands in
[the workflow guide](PROFESSOR_WORKFLOW.md). Export before refreshing to retain browser-only
ratings and model provenance.

## Reproduce the starter and verify the software

```bash
python3 train_polar_fatigue.py --bootstrap --output models/polar_fatigue_model.json
npm run build
npm test
python3 -m unittest discover -s tests -p 'test_*.py'
npm run test:browser
```

The bootstrap command replaces the chosen model artifact; keep your real model under
a distinct filename if you need both. The browser suite can use an installed browser
with `PLAYWRIGHT_CHROMIUM_EXECUTABLE=/path/to/chromium`. Python/browser parity tests,
simulated Bluetooth checks, and unit tests verify software behavior, not real H10 radio
performance or physiological accuracy. Physical-device troubleshooting remains to be
done while wearing the monitor.
