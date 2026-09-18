# Integrated Psychosocial Dashboard + Live Wristband ML

This package combines the historical psychosocial dashboard with the live wristband stream and the machine-learning classifiers built from your uploaded data.

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
