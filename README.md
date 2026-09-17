# Integrated Psychosocial Dashboard + Live Wristband ML

This package combines the historical psychosocial dashboard with the live wristband stream and the machine-learning classifiers built from your uploaded data.

## Polar H10 EXG · 200 ms RMS

The sensor panel displays only the existing visualizer's **EXG 200 ms RMS** signal.
It reuses `polar-h10` for Bluetooth and the filter/RMS implementation extracted from
[polar-h10-visualizer](https://github.com/yhzhao343/polar-h10-visualizer):
`src/consts.ts` and `src/PolarH10VisualizerRow.ts`, commit
`771559bc6ba6e2f13b43f02e1e7b026b1ec7473e` (Yuhui Zhao, ISC).
The upstream defaults are Fili Butterworth highpass (`order: 4`, cutoff 25 Hz),
130 Hz EXG sampling, and a 200 ms / 26-sample RMS window over the filtered signal.


```bash
python3 serve_dashboard.py
```

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
remain independent and unchanged. No test suite is included.

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
