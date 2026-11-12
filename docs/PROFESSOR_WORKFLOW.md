# Group 1: professor's closed-loop pathway

**Current update (2026-10-22):** The user supplied a new `fatigue` file and requested rules only. The active three-level rules, EDA requirement, and removed model-upload workflow are documented in [Current fatigue rules](FATIGUE_RULES.md). Older threshold/model instructions below are historical and do not enable ML in the active fatigue pipeline.

The instructions saved in `work` fit this project. They extend the Polar
H10 fatigue/posture prototype into a recorded feedback loop. Group 1 scope was
confirmed: stress, frustration, EDA and SCR belong to separate work and are not
inferred from H10 data here.

The main change is **rules first**. The shipped synthetic logistic regression is
demo-only. Live decisions use the configured rules until an independently labeled
model blend wins a held-out-person comparison. No construction-worker accuracy has
been established.

## Assignment mapping

| Professor's requirement | Implementation |
| --- | --- |
| States, causes, thresholds and minimum evidence | `config/rules.json`; fatigue HR/RMSSD/SDNN baseline changes, posture calibrated absolute pitch/roll |
| Message, task action, timing, goal; low means no feedback | `config/feedback_table.json`, validated CSV/XLSX importer and template |
| Ingest, clean, compute features and personal baseline | `pipeline/ingest.py`, `features.py`; browser Bluetooth HR/RR and calibrated acceleration snapshot bridge |
| Highest matching rule; missing signals never normal | `pipeline/rule_engine.py`; configurable browser threshold display |
| Model fallback, blend and score | `pipeline/model.py`; portable JSON logistic coefficients, explicit live promotion gate |
| Persistence, rapid rise, cooldown, warning cap | `pipeline/decision_engine.py`; observation-driven timing, no timer advancement from polling |
| Connect the stages | `pipeline/pipeline.py` |
| Display and audio | Existing Polar dashboard plus `serving/dashboard.html` entry point; issued server alerts control live speech |
| Readings, alerts, responses, outcomes, export | `storage/feedback_store.py`, SQLite, dashboard response/outcome buttons and saved export |
| API | `serving/server.py`, also integrated into `serve_dashboard.py` |
| Unseen-person comparison, save only if blend wins | `training/train_model.py` |
| Scripted rest → strain/posture → recovery | `tools/demo_stream.py` and clearly marked browser demo |
| Tests | Pipeline, storage/API, training, browser transport and real built dashboard checks |

Rules and messages are read from JSON at server startup and browser refresh. After
editing either file, restart the server and refresh the dashboard. Change the rule
version when changing thresholds or score bands; an artifact approved against a
different version falls back to rules.

## Run and collect

Python 3.10+ runs the rule pipeline and SQLite server without extra packages:

```bash
python3 serve_dashboard.py
```

Open `http://127.0.0.1:8000/psychosocial_dashboard.html` in Chrome/Edge. Enter a stable
pseudonymous participant ID, connect the H10, capture a resting baseline and
calibrate upright posture. A blank ID gets a session-only anonymous storage identity;
independent fatigue ratings require an explicit participant ID. Keep the same ID
for each person's repeated sessions.

The **Local feedback loop** reports saved, pending and dropped readings. It shows
the current decision and suppression reason, then a message/action/goal when an
alert is issued. Enable spoken feedback to hear live alerts. Use **Acknowledge
feedback**, **I took a break**, or **Feedback wasn't helpful**, followed by a
**Better / The same / Worse** outcome. These are linked to the displayed last alert;
they do not automatically become fatigue training labels.

Use **Record fatigue rating** for an independent label. The local database links it
to the latest eligible saved reading; the browser export retains the snapshot at
the button click. Database-save failure is shown separately and the browser copy
remains exportable. Do not merge two exports of the same session into training.

**Download saved session** exports the durable readings, decisions, alerts, ratings,
responses and outcomes. Its link survives a page reload. **Export session JSON**
also remains available for the browser's latest 10,000 summaries, local ratings and
model provenance. Pending network deliveries and browser-only ratings are not durable;
wait for pending deliveries to drain and download the browser export before leaving.

SQLite data lives in `.local/closed_loop.sqlite3`, excluded from Git and static
HTTP access. Keep downloaded participant exports/reports under `.local/` too.
The server listens on loopback and accepts same-origin browser writes. The analysis
log records each submitted HR/RR notification and its posture snapshot; it is not
an archive of every raw 130 Hz ECG or 25 Hz accelerometer sample. Browser transport
is capped at 256 pending readings, with losses shown if that bound is exceeded.
Switching sessions retains queued deliveries; their results cannot change the new session’s decisions.
Delayed live readings are logged with unavailable evidence and cannot issue late alerts.

## Demo without a sensor

Click **Start demo** for synthetic 20× playback, or run:

```bash
python3 tools/demo_stream.py
python3 tools/demo_stream.py --url http://127.0.0.1:8000
```

The command-line demo processes 221 synthetic readings and includes recovery.
Demo mode permits simulated timestamps; its labels are excluded from real training.
The browser's legacy threshold cues demonstrate basic persistence; the Local feedback
loop demonstrates the full cooldown/rapid-rise/cap policy and records its issued alerts.

## Table import

The original `ML-Feedback-Decision Based rule.xlsx` is absent. Current threshold
values come from the supplied PDF, pages 51 and 65–66, as documented in the
[Group 1 handoff](GROUP1_HANDOFF.md). Do not claim the missing workbook was imported.

Normalize its feedback rows to the supplied `config/feedback_template.csv` columns:
`state,level,cause,message,action,goal,seconds,enabled`. Then:

```bash
python3 training/build_feedback_table.py config/feedback_template.csv --output .local/feedback-candidate.json
python3 -m pip install -r requirements.txt
python3 training/build_feedback_table.py feedback-normalized.xlsx --output .local/feedback-candidate.json
```

CSV requires only the standard library; XLSX requires `openpyxl`. Review the candidate
against the professor's sheet, then replace `config/feedback_table.json` and restart.
The importer validates rows and preserves decision settings; it does not guess
unknown workbook headings or physiological thresholds.

## Train and evaluate after collecting labels

The model is standardized multinomial logistic regression with fixed `C=1.0` and
balanced class weights. Inputs are HR, RMSSD and SDNN percentage changes from baseline;
the target is the manually selected fatigue category. It estimates current state,
not future fatigue onset. Linear regression is not used for these categorical labels.

```bash
python3 -m pip install -r requirements.txt
python3 training/train_model.py --sessions .local/P01.json .local/P02.json .local/P03.json --output .local/fatigue-blend.json --report .local/comparison.json
```

At least three distinct participants and adequate class coverage in every training
fold are required. These are software minimums, not a study sample-size recommendation.
Each fold fits its scaler and classifier on the other people. The report compares
rules, ML and the fixed 0.5 blend using macro-F1 and accuracy. The deployment range
fallback is included in blend evaluation. A candidate is written **only if blend
macro-F1 strictly exceeds both rules and ML**; otherwise only the report is written
and any existing model file is preserved. No synthetic or alert-generated labels qualify.

After reviewing a winning report, explicitly load the candidate when starting the server:

```bash
python3 -m serving.server --model .local/fatigue-blend.json
```

The loader checks provenance, participant count, rule version and winning metrics.
Missing, malformed, unapproved or out-of-range models fall back to rules. Browser
model imports are demo previews and cannot bypass this live gate. The older
`train_polar_fatigue.py` remains a bootstrap/personal-model experiment; its artifacts
alone cannot activate live blending.

## Declared choices and remaining evidence

The existing strict fatigue and posture thresholds are preserved. Baseline/window
length (60 seconds), minimum baseline observations (30), RR filter (300–2000 ms),
freshness (5 seconds), cooldown (120 seconds per state), cap (3 total warnings in
600 seconds), and rapid rise (2 severity levels within 30 seconds, minimum 15-second
persistence) are configurable prototype choices requiring study review. Normal/low
levels issue no feedback. Rapid rise cannot bypass cooldown or the warning cap.

Severity scores and band boundaries are explicit engineering mappings, not calibrated
fatigue probabilities. A cardiovascular strain pattern does not establish heat exposure,
sleep loss or a specific task cause. The PDF's optional weighted score is not invented
because its component definitions are missing.

Remaining work needs evidence that is not in this repository: check the original
workbook, test while wearing the H10 during the study protocol, collect independent
labels across people, and assess whether any blend actually wins. Software tests
and synthetic demo runs do not establish physiological accuracy.

## Verify changes

```bash
npm test
npm run build
python3 -m unittest discover -s tests -p 'test_*.py'
npm run test:browser
```

ML tests need the requirements installed. XLSX round-trip tests skip when optional
`openpyxl` is absent. Browser tests use an isolated temporary database and simulated
Bluetooth; port 8000 must be free. Use `PLAYWRIGHT_CHROMIUM_EXECUTABLE` for an existing
Chromium executable. These checks cover software behavior, not physical radio operation.
