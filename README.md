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

## Files
- `serve_dashboard.py` — runs the integrated dashboard server
- `live_dashboard_backend.py` — polls the live stream and injects the live worker into dashboard data
- `psychosocial_dashboard.html` — updated dashboard front end
- `dashboard_data.json` — historical baseline dataset
- `avro_stream_poller_with_ml.py` — live wristband polling + ML prediction API
- `biomarker_runtime.py` — runtime predictor helper
- `models/` — trained model files and metadata
