# Integrated Psychosocial Dashboard + Live Wristband ML

This package combines the historical psychosocial dashboard with the live wristband stream and the machine-learning classifiers built from your uploaded data.

## Polar H10 EXG · 200 ms RMS

The Polar integration is on `feature/polar-h10-dashboard`, based on
`lexiliiii/VIP_psychological_dashboard`'s `polar-h10-integration` branch
(`c729e4d`) with this repository's voice-coach work merged in.

The sensor panel displays only the existing visualizer's **EXG 200 ms RMS** signal.
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

The integration files are `sensor/driver.js` (existing device library),
`sensor/controller.js` (EXG connection), `sensor/data.js` (upstream filter/RMS), and
`sensor/panel.js` (RMS display). The build emits `assets/polar-panel.js`.
No Polar heart-rate, RR/HRV, acceleration, or raw waveform views are included.
Existing wristband data, psychological models, and historical dashboard views
remain independent. The worker/scenario/playback controls do not assign the
Polar signal to a worker. RMS readings stay in browser memory and are not
saved, exported, or sent to the psychological models or voice coach.

The dashboard's **Live stream unavailable** message and the server's port 7000
message refer to the separate wristband feed. Check the status inside the
**EXG · 200 ms RMS** panel for Polar status. Use the local URL above, not a
`file://` page. Reloading or leaving the page ends the sensor session.

### Sensor check

1. Enable Bluetooth and prepare/wear the H10 with moistened strap contacts.
2. Click **Connect Polar H10**, choose your sensor, and grant Bluetooth access.
3. Confirm **Receiving EXG**, a numeric µV RMS value, and a moving trace.
4. Click **Disconnect** and confirm the value and chart clear. Connect again
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
Python server automatically (or reuse a running server on port 8000); set
`PYTHON` if your Python command has a different name.

Tests cover the filter/RMS pipeline, freshness, connection failures, retries,
and the shipped bundle with simulated Bluetooth packets. They do **not**
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
### 1) Start the live wristband poller
```bash
python avro_stream_poller_with_ml.py
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
- `avro_stream_poller_with_ml.py` — live wristband polling + ML prediction API
- `biomarker_runtime.py` — runtime predictor helper
- `models/` — trained model files and metadata
