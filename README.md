# Integrated Psychosocial Dashboard + Live Wristband ML

This package combines the historical psychosocial dashboard with the live wristband stream and the machine-learning classifiers built from your uploaded data.

## Team 3: Frustration and fatigue prediction and feedback

Predicts **Low / Medium / High** frustration and fatigue risk from % change against a
resting baseline (EDA, SCR, HR, RMSSD, SDNN) using the DBR masterfile rules. **Only High
risk gives feedback, and only after 3 consecutive High predictions**; a Low or Medium
prediction resets the count. Thresholds and messages live in `config/`, not in code.

When EDA and skin temperature are streamed, an Extra-Trees model sets the frustration level and
the DBR rules name the cause where they can. **Expect about 66% accuracy on a new worker** (nested
leave-one-participant-out, 95% interval 43–83%, 15 participants). The previous model's 87.5% was a
random window split; it scored 48.9% on new people. Retrain with
`python3 training/train_frustration_model.py` (labelled sessions in `.local/data/Frustration`, never
committed). Evaluation reports are in `models/` and on the dashboard. See
[the write-up](docs/TEAM3_FRUSTRATION_FATIGUE.md) and [the response to the November review](docs/REVIEW_RESPONSE_2026-11.md).

Empatica EmbracePlus data arrives through AWS S3 in 30-minute files, so it is delayed. Run
`python3 tools/empatica_bridge.py --url http://127.0.0.1:8000 --participant <enrollment> --since <UTC start> --session-id <name> --poll 60`
to feed it in; its predictions are logged, and cues that would have fired are recorded but not sent. Then pick
the session in the dashboard's Session menu. Details are in the Team 3 guide.

```bash
python3 serve_dashboard.py
# open http://127.0.0.1:8000/serving/dashboard.html and click "Run scripted demo"
```

Course handouts (the DBR masterfile PDFs and instructor notes) are not included in this
public repository; docs cite them by name. The design, derived thresholds, tests and limitations are in
[the Team 3 guide](docs/TEAM3_FRUSTRATION_FATIGUE.md). The Polar H10 panel below uses
the same server-side decision engine.

## Group 1: Polar H10 fatigue and posture

Start with [the Group 1 handoff](docs/GROUP1_HANDOFF.md). It explains what the
assignment asks for, how it extends the earlier Polar handoff, and the exact
threshold/cause/feedback table to discuss with the team.

The active fatigue workflow follows the newest section of `fatigue` and
uses **rules only**. See [the current fatigue guide](docs/FATIGUE_RULES.md) for the
exact low/medium/high conditions and data requirements. No model JSON is needed.
The prior [professor workflow](docs/PROFESSOR_WORKFLOW.md) documents the underlying
storage and feedback architecture; its model-enablement instructions are superseded.

Live HR/RR provides RMSSD/SDNN and a personal resting baseline. Low and medium fatigue
use HRV changes; high additionally requires HR change below +10% and EDA change below
+15%. H10 has no EDA sensor, so missing EDA explicitly limits high-rule assessment.
The dashboard retains calibrated posture, feedback timing/cooldowns, local SQLite
readings/alerts/responses/outcomes, independent ratings and exports. **Start demo**
runs synthetic HR/RR plus EDA at 20× to demonstrate all three fatigue levels.

The Polar integration is on `feature/polar-h10-dashboard`, based on
`lexiliiii/VIP_psychological_dashboard`'s `polar-h10-integration` branch
(`c729e4d`) with this repository's voice-coach work merged in.

The sensor panel also retains the existing visualizer's **EXG 200 ms RMS** signal.
It reuses `polar-h10` for Bluetooth and the filter/RMS implementation extracted from
[polar-h10-visualizer](https://github.com/yhzhao343/polar-h10-visualizer):
`src/consts.ts` and `src/PolarH10VisualizerRow.ts`, commit
`771559bc6ba6e2f13b43f02e1e7b026b1ec7473e` (Yuhui Zhao, ISC).
The upstream defaults are Fili Butterworth highpass (`order: 4`, cutoff 25 Hz),
130 Hz EXG sampling, and a 200 ms / 26-sample RMS window over the filtered signal.
For Polar-only use, you need Python 3.10+ and desktop Chrome or Edge on a
platform that supports Web Bluetooth. The built panel is included in Git:
**no npm install, historical-data rebuild, wristband poller, or AWS setup is
needed to run it.** Keep the `assets/` folder alongside the HTML files.

```bash
python3 serve_dashboard.py
```

On Windows, use `py -3 serve_dashboard.py`. Keep the terminal running.
Open `http://127.0.0.1:8000/psychosocial_dashboard.html` in Chrome or Edge and click
**Connect Polar H10**. Use localhost or HTTPS and allow Bluetooth access. For
firmware 4+, pair the H10 in your OS first. The panel shows one RMS value and one
RMS chart. Disconnect clears the session; missing samples become unavailable after
five seconds, and unexpected disconnections trigger bounded reconnection attempts.
Enter a participant ID before connecting, capture a resting baseline, and calibrate
upright posture. Every reconnection needs fresh calibration. The panel's rule
details explain thresholds and measurement availability.

The integration files are `sensor/driver.js` (existing device library),
`sensor/controller.js` (stream connection), `sensor/data.js` (upstream filter/RMS),
`sensor/fatigue.js` (HRV, calibration, rules and persistence), `sensor/panel.js`
(RMS display), and `sensor/study-panel.js` (Group 1 workflow, demo, speech and export).
The build emits `assets/polar-panel.js`. The accelerometer runs at 25 Hz with a
4 g range. Only pitch/roll tilt is evaluated; yaw and arm elevation are not inferred.
Existing wristband data, psychological models, and historical dashboard views
remain independent. The worker/scenario/playback controls do not assign the
Polar signal to a worker. The explicit participant field applies to the Polar
session only. **Export session JSON** downloads up to the latest 10,000 summary
observations, baseline values, feedback events, independent fatigue ratings. This browser export is not a raw ECG recording. **Download saved session**
exports the durable database log, including submitted HR/RR packets and decisions.
The **Local feedback loop** shows pending or dropped deliveries; wait for pending
readings and export browser-only ratings before leaving. Data is stored locally under
`.local/`, excluded from Git and static serving. Polar speech is delivered by the
browser from issued local-server alerts; the separate wristband voice coach remains
independent.

For recorded CSV/XLSX data, `polar_recordings.py` converts explicit Unix timestamp
units to UTC and computes accelerometer bending angles. Examples, optional Excel
dependencies and column-mapping commands are in the handoff. CSV needs no extra
Python packages; XLSX needs `python3 -m pip install -r requirements-polar.txt`.

The dashboard's **Live stream unavailable** message and the server's port 7000
message refer to the separate wristband feed. Check the status inside the
**EXG · 200 ms RMS** panel for Polar status. Use the local URL above, not a
`file://` page. Reloading or leaving the page ends the sensor session.

### Sensor check

1. Enable Bluetooth and prepare/wear the H10 with moistened strap contacts.
2. Click **Connect Polar H10**, choose your sensor, and grant Bluetooth access.
3. Confirm **Receiving EXG**, a numeric µV RMS value, and a moving trace.
4. Confirm HR/RR arrive, capture a baseline while resting, and calibrate while
   upright. Verify posture changes with controlled movements and export a session.
5. Click **Disconnect** and confirm the value and chart clear. Connect again
   to start a new session. If pairing fails, check OS pairing, battery/range,
   permissions, and whether another app is using the H10.

### Developing and verifying the panel

Node.js 22+ is needed only to edit/build/test the JavaScript:

```bash
npm ci
npm test
npm run build
npx playwright install chromium
npm run test:browser
```

Commit `assets/polar-panel.js` whenever `sensor/` changes. The lockfile pins
`polar-h10`, Fili, and build/test tooling. Third-party attribution is in
`assets/polar-licenses.txt`. If Chromium is already installed elsewhere, set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` to its executable path. Browser tests start the
Python server with a temporary study database automatically (port 8000 must be free); set
`PYTHON` if your Python command has a different name.

Tests cover the filter/RMS pipeline, HR/RR decoding, baseline/HRV calculations,
fatigue and posture thresholds, feedback persistence, stale data, connection
failures, bounded retries, exports, CSV/XLSX conversion, and the shipped bundle
with simulated Bluetooth packets. They do **not**
validate real radio connectivity or sensor accuracy; complete the sensor check
above with an H10 before a live session. Historical regeneration preservation
can also be checked with `python3 -m unittest discover -s tests -p 'test_*.py'`
in an environment with pandas installed. Regeneration updates the historical
JSON without overwriting the maintained HTML, server, or README.

## What changed
- The dashboard now pulls live biomarker predictions from `avro_stream_poller_with_ml.py`.
- A **Live worker** is injected into the worker queue for both dashboard lanes:
  - **Frustration**
  - **Mental Demand**
- The dashboard auto-refreshes every 5 seconds.
- The detail panel now shows:
  - live Frustration label and confidence
  - live Mental Demand label and confidence
  - pulse rate from the stream
- Zone, summary, alerts, and worker queue all update to include the live worker.

## Run order
This older worker-queue view reads per-minute predictions from the poller below. The Team 3
frustration and fatigue dashboard does not: it uses `tools/empatica_bridge.py`, described above.

### 1) Start the live wristband poller
```bash
PORT=7000 python avro_stream_poller_with_ml.py
```
That serves the live endpoint at:
```text
http://127.0.0.1:7000/latest
```

### 2) Start the integrated dashboard server
In a second terminal:
```bash
python serve_dashboard.py
```
Then open:
```text
http://127.0.0.1:8000/psychosocial_dashboard.html
```

## Notes
- The dashboard still includes the original historical sessions.
- The live worker appears with `mode = Live` and updates as new minute-level data arrives.
- Frustration may show **Pending** until the second minute sample arrives, because that model needs two minute-level points.
- The current trained live models use **EDA + temperature + minute index**. Pulse rate is displayed in the dashboard and used for dashboard proxy scoring, but it is not part of the current trained classifier inputs.

## Voice safety coach (construction-worker feedback)

`worksite_voice_coach.py` turns the live predictions into spoken safety guidance
instead of spoken labels. Run it in a third terminal:

```bash
pip install pyttsx3 requests      # Linux also needs: sudo apt-get install espeak
python worksite_voice_coach.py
```

It tracks **how long** strain has been sustained and escalates:

| Sustained at high strain | What it says |
| --- | --- |
| immediately | "Are you feeling tired? Please have some water before your next lift." |
| 10 minutes | step into the shade, loosen your vest, five slow breaths |
| 25 minutes | hand the task to your partner, reset your grip |
| 45 minutes | "Please take a break. You have been stressed for forty five minutes." |
| 60 minutes | safety stop, flag the supervisor for a rest rotation |

It also raises separate advisories from the raw sensors (skin temperature, pulse
rate, EDA spikes), holds the sustained-strain clock through a short dip rather
than resetting on one good minute, confirms recovery, and writes every utterance
to `voice_coach_log.csv`.

For a presentation, `--demo` replays a scripted 90-minute shift with no network
or AWS needed, and `--list-voices` / `--voice-name` pick the voice:

```bash
python worksite_voice_coach.py --demo --announce-start --verbose
```

## Files
- `worksite_voice_coach.py` — spoken construction-site safety coach (see above)
- `speak_biomarker_labels.py` — minimal text-to-speech label reader
- `serve_dashboard.py` — runs the integrated dashboard server
- `live_dashboard_backend.py` — polls the live stream and injects the live worker into dashboard data
- `psychosocial_dashboard.html` — updated dashboard front end
- `dashboard_data.json` — historical baseline dataset
- `avro_stream_poller_with_ml.py` — legacy wristband polling + older ML prediction API for this view and the voice coach
- `biomarker_runtime.py` — runtime predictor helper
- `models/` — trained model files and metadata
