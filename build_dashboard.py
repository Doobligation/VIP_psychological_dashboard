from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = Path('/mnt/data/OneDrive_1_4-8-2026(1).zip')
OUTPUT_HTML = BASE_DIR / 'psychosocial_dashboard.html'
OUTPUT_JSON = BASE_DIR / 'dashboard_data.json'
README = BASE_DIR / 'README.md'
SERVE = BASE_DIR / 'serve_dashboard.py'

ZONE_NAMES = [
    'Material Handling Zone',
    'Concrete Zone',
    'Steel Zone',
    'Scaffolding Zone',
    'Lift / Overhead Zone',
    'Break & Recovery Area',
]


@dataclass
class SessionInfo:
    category: str
    file_name: str
    participant_id: str
    mode: str
    label: str
    records: list[dict]
    metrics: dict
    alerts: list[dict]
    summary: dict


def safe_slug(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')


def parse_filename(category: str, name: str) -> tuple[str, str, str]:
    stem = Path(name).stem
    parts = stem.replace('__', '_').split('_')

    participant = parts[0].upper() if parts else 'UNKNOWN'

    mode = 'Unknown'
    label = 'Medium'

    for part in parts[1:]:
        p = part.strip().lower()
        if p in {'active', 'passive'}:
            mode = p.title()
        elif p in {'low', 'medium', 'high'}:
            label = p.title()

    if category.lower() == 'mental demand' and mode == 'Unknown':
        mode = 'Active'

    return participant, mode, label


def normalize_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').ffill().bfill().fillna(0.0)
    lo = float(s.min())
    hi = float(s.max())
    if math.isclose(lo, hi):
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).std().fillna(0.0)


def label_boost(label: str) -> float:
    return {'Low': 0.10, 'Medium': 0.22, 'High': 0.34}.get(label, 0.20)


def mode_modifier(mode: str) -> float:
    return {'Active': 0.08, 'Passive': 0.02}.get(mode, 0.04)


def zone_for_session(participant: str, category: str, mode: str) -> str:
    num_match = re.search(r'(\d+)', participant)
    n = int(num_match.group(1)) if num_match else 0
    idx = (n + len(category) + len(mode)) % len(ZONE_NAMES)
    return ZONE_NAMES[idx]


def build_records(df: pd.DataFrame, label: str, mode: str, category: str) -> tuple[list[dict], dict, list[dict], dict]:
    work = df.copy()
    work['timestamp_local'] = pd.to_datetime(work['timestamp_local'], errors='coerce')
    work = work.dropna(subset=['timestamp_local']).sort_values('timestamp_local').reset_index(drop=True)

    eda = normalize_series(work['eda_eda'])
    bvp = normalize_series(work['bvp_bvp'])
    temp = normalize_series(work['temperature_temp'])

    eda_smooth = rolling_mean(eda, 12)
    bvp_smooth = rolling_mean(bvp, 12)
    temp_smooth = rolling_mean(temp, 20)
    eda_vol = rolling_std(eda, 10)
    bvp_vol = rolling_std(bvp, 10)

    fatigue_trend = pd.Series(range(len(work)), dtype=float)
    fatigue_trend = fatigue_trend / max(len(work) - 1, 1)

    stress = (0.48 * eda_smooth) + (0.22 * bvp_smooth) + (0.12 * temp_smooth) + (0.18 * eda_vol.clip(0, 1))
    fatigue = (0.40 * temp_smooth) + (0.20 * eda_smooth) + (0.20 * fatigue_trend) + (0.20 * bvp_vol.clip(0, 1))
    cognitive = (0.45 * stress) + (0.20 * eda_vol.clip(0, 1)) + (0.15 * (1 - rolling_mean(temp, 30))) + (0.20 * bvp_vol.clip(0, 1))

    risk = (
        0.42 * stress
        + 0.28 * fatigue
        + 0.30 * cognitive
        + label_boost(label)
        + mode_modifier(mode)
        + (0.06 if category.lower() == 'mental demand' else 0.04)
    ).clip(0, 1)

    data_confidence = (1 - (0.4 * work[['eda_eda', 'bvp_bvp', 'temperature_temp']].isna().mean(axis=1))).clip(0.6, 1.0)
    if mode == 'Passive':
        data_confidence = (data_confidence - 0.05).clip(0.55, 0.98)

    records: list[dict] = []
    alerts: list[dict] = []

    last_alert_kind = None
    last_alert_index = -999

    for i, row in work.iterrows():
        stress_score = float((stress.iloc[i] * 100))
        fatigue_score = float((fatigue.iloc[i] * 100))
        cognitive_score = float((cognitive.iloc[i] * 100))
        overall = float((risk.iloc[i] * 100))
        confidence = float((data_confidence.iloc[i] * 100))

        if overall >= 78:
            risk_band = 'High'
        elif overall >= 60:
            risk_band = 'Caution'
        else:
            risk_band = 'Normal'

        dominant = max(
            [('Stress', stress_score), ('Fatigue', fatigue_score), ('Cognitive Load', cognitive_score)],
            key=lambda x: x[1],
        )[0]

        recommendation = build_recommendation(risk_band, dominant, confidence)

        records.append(
            {
                'index': i,
                'timestamp': row['timestamp_local'].strftime('%Y-%m-%d %H:%M:%S'),
                'eda': round(float(row['eda_eda']), 4),
                'bvp': round(float(row['bvp_bvp']), 4),
                'temperature': round(float(row['temperature_temp']), 4),
                'stress': round(stress_score, 1),
                'fatigue': round(fatigue_score, 1),
                'cognitiveLoad': round(cognitive_score, 1),
                'risk': round(overall, 1),
                'dataConfidence': round(confidence, 1),
                'riskBand': risk_band,
                'dominantRisk': dominant,
                'recommendation': recommendation,
            }
        )

        should_alert = risk_band in {'High', 'Caution'} and (i - last_alert_index >= 20 or dominant != last_alert_kind)
        if should_alert:
            alerts.append(
                {
                    'time': row['timestamp_local'].strftime('%H:%M:%S'),
                    'level': risk_band,
                    'type': dominant,
                    'message': build_alert_message(risk_band, dominant),
                    'recommendation': recommendation,
                    'score': round(overall, 1),
                }
            )
            last_alert_index = i
            last_alert_kind = dominant

    metric_summary = {
        'avgStress': round(mean(r['stress'] for r in records), 1),
        'avgFatigue': round(mean(r['fatigue'] for r in records), 1),
        'avgCognitiveLoad': round(mean(r['cognitiveLoad'] for r in records), 1),
        'peakRisk': round(max(r['risk'] for r in records), 1),
        'avgConfidence': round(mean(r['dataConfidence'] for r in records), 1),
    }

    summary = {
        'start': records[0]['timestamp'] if records else None,
        'end': records[-1]['timestamp'] if records else None,
        'durationMinutes': round(len(records) / 60, 1),
        'finalRisk': records[-1]['risk'] if records else 0,
        'finalBand': records[-1]['riskBand'] if records else 'Normal',
    }

    return records, metric_summary, alerts, summary


def build_alert_message(level: str, dominant: str) -> str:
    prefix = 'Immediate attention recommended' if level == 'High' else 'Monitor closely'
    mapping = {
        'Stress': f'{prefix}: sustained physiological arousal is trending upward.',
        'Fatigue': f'{prefix}: fatigue markers are accumulating across the session.',
        'Cognitive Load': f'{prefix}: cognitive load is elevated relative to the current task state.',
    }
    return mapping[dominant]


def build_recommendation(level: str, dominant: str, confidence: float) -> str:
    confidence_note = 'Data confidence is strong.' if confidence >= 85 else 'Verify sensor placement while responding.'
    if level == 'High':
        if dominant == 'Stress':
            return f'Pause the task, perform a supervisor check-in, and provide a short recovery break. {confidence_note}'
        if dominant == 'Fatigue':
            return f'Rotate the worker to a lighter task, provide hydration/rest, and review upcoming workload. {confidence_note}'
        return f'Reduce task complexity, confirm instructions, and schedule a short reset before resuming work. {confidence_note}'
    if dominant == 'Stress':
        return f'Watch for escalation over the next 10 minutes and prepare a short check-in. {confidence_note}'
    if dominant == 'Fatigue':
        return f'Plan an earlier break or minor task rotation if the trend continues. {confidence_note}'
    return f'Keep the worker under observation and simplify task sequencing if risk increases. {confidence_note}'


def build_dataset() -> dict:
    sessions: dict[str, list[dict]] = {'Frustration': [], 'Mental Demand': []}

    with zipfile.ZipFile(ZIP_PATH) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith('.csv') and not n.endswith('/')]
        for name in csv_names:
            category = Path(name).parts[0]
            if category not in sessions:
                continue
            participant, mode, label = parse_filename(category, name)
            raw = io.BytesIO(zf.read(name))
            df = pd.read_csv(raw)
            needed = {'timestamp_local', 'eda_eda', 'bvp_bvp', 'temperature_temp'}
            if not needed.issubset(df.columns):
                continue
            records, metrics, alerts, summary = build_records(df, label, mode, category)
            sessions[category].append(
                {
                    'sessionId': safe_slug(f'{category}-{Path(name).stem}'),
                    'fileName': Path(name).name,
                    'participantId': participant,
                    'workerName': f'Worker {participant}',
                    'mode': mode,
                    'taskType': category,
                    'groundTruthLabel': label,
                    'zone': zone_for_session(participant, category, mode),
                    'records': records,
                    'metrics': metrics,
                    'alerts': alerts,
                    'summary': summary,
                }
            )

    for key in sessions:
        sessions[key].sort(key=lambda x: (x['participantId'], x['mode'], x['fileName']))

    return {
        'title': 'Psychosocial Risk Monitoring Dashboard',
        'subtitle': 'Construction manager prototype using uploaded wearable CSV sessions',
        'generatedFrom': ZIP_PATH.name,
        'categories': sessions,
    }


def write_files(dataset: dict) -> None:
    OUTPUT_JSON.write_text(json.dumps(dataset, indent=2), encoding='utf-8')
    OUTPUT_HTML.write_text(build_html(dataset), encoding='utf-8')
    README.write_text(build_readme(), encoding='utf-8')
    SERVE.write_text(build_server_script(), encoding='utf-8')


def build_readme() -> str:
    return """# Psychosocial Risk Monitoring Dashboard

This package contains a ready-to-open web prototype and the Python builder used to generate it from the uploaded ZIP of CSV files.

## Files
- `psychosocial_dashboard.html` — main dashboard (open in a browser)
- `dashboard_data.json` — processed session data used by the dashboard
- `build_dashboard.py` — rebuilds the dashboard from the ZIP file
- `serve_dashboard.py` — optional tiny local web server

## What the dashboard does
- Reads uploaded wearable CSV sessions
- Simulates a construction manager dashboard
- Computes simple proxy scores for stress, fatigue, cognitive load, and overall psychosocial risk
- Surfaces alerts and recommendations

## Run options
### Option 1: open the HTML directly
Double-click `psychosocial_dashboard.html`.

### Option 2: serve locally with Python
```bash
python serve_dashboard.py
```
Then open the local address shown in the terminal.

## Notes
This is an interface prototype for coursework. The risk scores are proxy scores derived from uploaded EDA, BVP, and temperature time series plus file labels. They are not clinical predictions.
"""


def build_server_script() -> str:
    return """from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import webbrowser

ROOT = Path(__file__).resolve().parent
PORT = 8000

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

if __name__ == '__main__':
    url = f'http://127.0.0.1:{PORT}/psychosocial_dashboard.html'
    print(f'Serving dashboard at {url}')
    try:
        webbrowser.open(url)
    except Exception:
        pass
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
"""


def build_html(dataset: dict) -> str:
    data_json = json.dumps(dataset)
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Psychosocial Risk Monitoring Dashboard</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #38bdf8;
      --good: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
      --line: #334155;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
      color: var(--text);
    }}
    .app {{ padding: 20px; max-width: 1500px; margin: 0 auto; }}
    .header {{
      display: flex; justify-content: space-between; align-items: flex-start; gap: 16px;
      margin-bottom: 18px;
    }}
    .title h1 {{ margin: 0 0 6px 0; font-size: 28px; }}
    .title p {{ margin: 0; color: var(--muted); }}
    .pill-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .pill {{ background: rgba(56, 189, 248, 0.12); color: #bae6fd; border: 1px solid rgba(56, 189, 248, 0.3); padding: 8px 12px; border-radius: 999px; font-size: 12px; }}
    .controls {{
      display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px;
    }}
    .panel {{
      background: rgba(17, 24, 39, 0.96);
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 18px;
      box-shadow: 0 18px 38px rgba(0, 0, 0, 0.25);
    }}
    .control-card {{ padding: 14px; }}
    label {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
    select, input[type="range"] {{ width: 100%; }}
    select {{
      background: var(--panel-2); color: var(--text); border: 1px solid var(--line); border-radius: 10px; padding: 10px;
    }}
    .metrics {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .metric {{ padding: 14px; }}
    .metric .label {{ font-size: 12px; color: var(--muted); }}
    .metric .value {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    .metric .sub {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
    .layout {{ display: grid; grid-template-columns: 360px 1fr; gap: 16px; align-items: start; }}
    .sidebar {{ padding: 14px; max-height: calc(100vh - 230px); overflow: auto; }}
    .main {{ display: grid; gap: 16px; }}
    .worker-card {{
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid transparent;
      border-radius: 14px;
      padding: 12px;
      margin-bottom: 10px;
      cursor: pointer;
      transition: 140ms ease;
    }}
    .worker-card:hover {{ border-color: rgba(56, 189, 248, 0.35); transform: translateY(-1px); }}
    .worker-card.active {{ border-color: rgba(56, 189, 248, 0.7); background: rgba(56, 189, 248, 0.10); }}
    .worker-top {{ display: flex; justify-content: space-between; gap: 8px; align-items: center; }}
    .worker-name {{ font-weight: 700; font-size: 15px; }}
    .badge {{ padding: 5px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
    .badge.normal {{ background: rgba(34, 197, 94, 0.18); color: #86efac; }}
    .badge.caution {{ background: rgba(245, 158, 11, 0.18); color: #fcd34d; }}
    .badge.high {{ background: rgba(239, 68, 68, 0.18); color: #fca5a5; }}
    .muted {{ color: var(--muted); }}
    .worker-meta {{ display: flex; flex-wrap: wrap; gap: 8px; font-size: 12px; margin-top: 8px; color: var(--muted); }}
    .progress {{ height: 8px; border-radius: 999px; background: rgba(148,163,184,0.14); overflow: hidden; margin-top: 12px; }}
    .progress > span {{ display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #38bdf8, #ef4444); }}
    .detail-grid {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 16px; }}
    .detail-card {{ padding: 16px; }}
    .detail-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; margin-bottom: 8px; }}
    .detail-head h2, .detail-card h3 {{ margin: 0; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }}
    .stat-box {{ background: rgba(255,255,255,0.03); border-radius: 14px; padding: 12px; border: 1px solid rgba(148,163,184,0.12); }}
    .stat-box .k {{ font-size: 12px; color: var(--muted); }}
    .stat-box .v {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    canvas {{ width: 100%; height: 280px; background: rgba(255,255,255,0.02); border-radius: 14px; border: 1px solid rgba(148,163,184,0.12); }}
    .alerts, .zones {{ display: grid; gap: 10px; }}
    .alert-item, .zone-item {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(148,163,184,0.12); border-radius: 14px; padding: 12px; }}
    .alert-top, .zone-top {{ display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-bottom: 6px; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .recommendation {{ font-size: 14px; line-height: 1.5; color: #dbeafe; background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.18); border-radius: 14px; padding: 12px; }}
    .small {{ font-size: 12px; color: var(--muted); }}
    .footer-note {{ margin-top: 14px; font-size: 12px; color: var(--muted); }}
    .timeline-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: center; margin-bottom: 10px; }}
    @media (max-width: 1200px) {{
      .metrics {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .layout {{ grid-template-columns: 1fr; }}
      .detail-grid, .grid-2, .controls {{ grid-template-columns: 1fr; }}
      .sidebar {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div class="title">
        <h1>Psychosocial Risk Monitoring Dashboard</h1>
        <p>Manager-facing prototype for exoskeleton-assisted construction teams. Built from your uploaded CSV sessions.</p>
      </div>
      <div class="pill-row">
        <div class="pill" id="sourcePill"></div>
        <div class="pill">Real-time proxy scoring</div>
        <div class="pill">Actionable alerts</div>
      </div>
    </div>

    <div class="controls">
      <div class="panel control-card">
        <label for="taskSelect">Monitoring scenario</label>
        <select id="taskSelect"></select>
      </div>
      <div class="panel control-card">
        <label for="riskFilter">Risk filter</label>
        <select id="riskFilter">
          <option value="All">All workers</option>
          <option value="High">High only</option>
          <option value="Caution">Caution or High</option>
          <option value="Normal">Normal only</option>
        </select>
      </div>
      <div class="panel control-card">
        <label for="timeSlider">Playback time</label>
        <input id="timeSlider" type="range" min="0" max="100" value="100" />
        <div class="small" id="timeLabel"></div>
      </div>
      <div class="panel control-card">
        <label for="modeFilter">Mode filter</label>
        <select id="modeFilter">
          <option value="All">All modes</option>
          <option value="Active">Active</option>
          <option value="Passive">Passive</option>
        </select>
      </div>
    </div>

    <div class="metrics" id="summaryMetrics"></div>

    <div class="layout">
      <div class="panel sidebar">
        <div class="detail-head">
          <h3>Worker queue</h3>
          <div class="small" id="workerCountLabel"></div>
        </div>
        <div id="workerList"></div>
      </div>

      <div class="main">
        <div class="detail-grid">
          <div class="panel detail-card">
            <div class="detail-head">
              <div>
                <h2 id="workerTitle">Worker</h2>
                <div class="muted" id="workerSubtitle"></div>
              </div>
              <div id="workerBadge" class="badge normal">Normal</div>
            </div>
            <div class="timeline-head">
              <div class="small">Stress, fatigue, cognitive load, and overall risk over time</div>
              <div class="small" id="detailTimestamp"></div>
            </div>
            <canvas id="trendChart" width="900" height="280"></canvas>
            <div class="stats-grid" id="detailStats"></div>
          </div>

          <div class="panel detail-card">
            <h3>Current interpretation</h3>
            <div class="recommendation" id="recommendationBox"></div>
            <div class="stats-grid" style="margin-top:14px;" id="signalStats"></div>
            <div class="footer-note">Proxy scores are derived from uploaded EDA, BVP, temperature, task label, and session context.</div>
          </div>
        </div>

        <div class="grid-2">
          <div class="panel detail-card">
            <div class="detail-head">
              <h3>Live alert queue</h3>
              <div class="small">Most recent actionable events</div>
            </div>
            <div class="alerts" id="alertList"></div>
          </div>
          <div class="panel detail-card">
            <div class="detail-head">
              <h3>Site zone snapshot</h3>
              <div class="small">Workers grouped by area</div>
            </div>
            <div class="zones" id="zoneList"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const DATA = {data_json};

    const state = {{
      category: Object.keys(DATA.categories)[0],
      riskFilter: 'All',
      modeFilter: 'All',
      workerId: null,
      playbackPercent: 100,
    }};

    const taskSelect = document.getElementById('taskSelect');
    const riskFilter = document.getElementById('riskFilter');
    const modeFilter = document.getElementById('modeFilter');
    const timeSlider = document.getElementById('timeSlider');

    document.getElementById('sourcePill').textContent = `Source: ${'{'}DATA.generatedFrom{'}'}`;

    function init() {{
      Object.keys(DATA.categories).forEach(category => {{
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        taskSelect.appendChild(option);
      }});
      taskSelect.value = state.category;

      taskSelect.addEventListener('change', () => {{
        state.category = taskSelect.value;
        state.workerId = null;
        render();
      }});
      riskFilter.addEventListener('change', () => {{
        state.riskFilter = riskFilter.value;
        render();
      }});
      modeFilter.addEventListener('change', () => {{
        state.modeFilter = modeFilter.value;
        render();
      }});
      timeSlider.addEventListener('input', () => {{
        state.playbackPercent = Number(timeSlider.value);
        renderSummaryAndDetails();
      }});
      render();
    }}

    function currentSessions() {{
      return DATA.categories[state.category] || [];
    }}

    function playbackIndex(records) {{
      if (!records || !records.length) return 0;
      const idx = Math.floor((records.length - 1) * (state.playbackPercent / 100));
      return Math.max(0, Math.min(records.length - 1, idx));
    }}

    function getCurrentRecord(session) {{
      return session.records[playbackIndex(session.records)];
    }}

    function passesFilters(session) {{
      const rec = getCurrentRecord(session);
      if (!rec) return false;
      if (state.modeFilter !== 'All' && session.mode !== state.modeFilter) return false;
      if (state.riskFilter === 'High') return rec.riskBand === 'High';
      if (state.riskFilter === 'Caution') return ['High', 'Caution'].includes(rec.riskBand);
      if (state.riskFilter === 'Normal') return rec.riskBand === 'Normal';
      return true;
    }}

    function filteredSessions() {{
      return currentSessions().filter(passesFilters).sort((a, b) => getCurrentRecord(b).risk - getCurrentRecord(a).risk);
    }}

    function ensureSelectedWorker(list) {{
      if (!list.length) {{
        state.workerId = null;
        return null;
      }}
      const exists = list.find(item => item.sessionId === state.workerId);
      if (!exists) state.workerId = list[0].sessionId;
      return list.find(item => item.sessionId === state.workerId) || list[0];
    }}

    function riskClass(band) {{
      return band.toLowerCase();
    }}

    function render() {{
      const list = filteredSessions();
      const selected = ensureSelectedWorker(list);
      renderWorkerList(list);
      renderSummaryAndDetails(selected, list);
    }}

    function renderWorkerList(list) {{
      const workerList = document.getElementById('workerList');
      const workerCountLabel = document.getElementById('workerCountLabel');
      workerCountLabel.textContent = `${'{'}list.length{'}'} shown`;
      workerList.innerHTML = '';

      if (!list.length) {{
        workerList.innerHTML = '<div class="small">No workers match the current filters.</div>';
        return;
      }}

      list.forEach(session => {{
        const rec = getCurrentRecord(session);
        const div = document.createElement('div');
        div.className = `worker-card ${'{'}session.sessionId === state.workerId ? 'active' : ''{'}'}`;
        div.innerHTML = `
          <div class="worker-top">
            <div>
              <div class="worker-name">${'{'}session.workerName{'}'}</div>
              <div class="small">${'{'}session.taskType{'}'} • ${'{'}session.mode{'}'} • ${'{'}session.zone{'}'}</div>
            </div>
            <div class="badge ${'{'}riskClass(rec.riskBand){'}'}">${'{'}rec.riskBand{'}'}</div>
          </div>
          <div class="worker-meta">
            <span>Risk ${'{'}rec.risk.toFixed(1){'}'}</span>
            <span>Stress ${'{'}rec.stress.toFixed(1){'}'}</span>
            <span>Confidence ${'{'}rec.dataConfidence.toFixed(0){'}'}%</span>
          </div>
          <div class="progress"><span style="width:${'{'}Math.max(6, rec.risk){'}'}%"></span></div>
        `;
        div.addEventListener('click', () => {{
          state.workerId = session.sessionId;
          render();
        }});
        workerList.appendChild(div);
      }});
    }}

    function renderSummaryAndDetails(selected, listOverride) {{
      const list = listOverride || filteredSessions();
      renderSummaryMetrics(list);
      renderZoneSnapshot(list);
      renderTimeLabel(selected);
      if (!selected) {{
        clearDetailState();
        return;
      }}
      const rec = getCurrentRecord(selected);
      renderDetail(selected, rec);
      renderAlerts(selected, rec);
    }}

    function renderTimeLabel(selected) {{
      const label = document.getElementById('timeLabel');
      if (!selected) {{
        label.textContent = 'No session selected';
        return;
      }}
      const rec = getCurrentRecord(selected);
      label.textContent = `Viewing: ${'{'}rec.timestamp{'}'}`;
    }}

    function renderSummaryMetrics(list) {{
      const metrics = document.getElementById('summaryMetrics');
      metrics.innerHTML = '';
      const records = list.map(getCurrentRecord);
      const count = records.length;
      const high = records.filter(r => r.riskBand === 'High').length;
      const caution = records.filter(r => r.riskBand === 'Caution').length;
      const avgRisk = count ? records.reduce((sum, r) => sum + r.risk, 0) / count : 0;
      const avgConfidence = count ? records.reduce((sum, r) => sum + r.dataConfidence, 0) / count : 0;
      const dominant = (() => {{
        const tally = {{'Stress':0,'Fatigue':0,'Cognitive Load':0}};
        records.forEach(r => tally[r.dominantRisk] = (tally[r.dominantRisk] || 0) + 1);
        return Object.entries(tally).sort((a,b)=>b[1]-a[1])[0]?.[0] || 'N/A';
      }})();
      const cards = [
        ['Workers shown', String(count), 'Filtered live sessions'],
        ['High risk', String(high), 'Immediate attention candidates'],
        ['Caution', String(caution), 'Monitor and prepare response'],
        ['Average risk', avgRisk.toFixed(1), 'Across current queue'],
        ['Avg confidence', `${'{'}avgConfidence.toFixed(0){'}'}%`, 'Sensor/data confidence'],
        ['Dominant pattern', dominant, 'Most common risk driver'],
      ];
      cards.forEach(([label, value, sub]) => {{
        const div = document.createElement('div');
        div.className = 'panel metric';
        div.innerHTML = `<div class="label">${'{'}label{'}'}</div><div class="value">${'{'}value{'}'}</div><div class="sub">${'{'}sub{'}'}</div>`;
        metrics.appendChild(div);
      }});
    }}

    function renderDetail(session, rec) {{
      document.getElementById('workerTitle').textContent = session.workerName;
      document.getElementById('workerSubtitle').textContent = `${'{'}session.participantId{'}'} • ${'{'}session.taskType{'}'} • ${'{'}session.mode{'}'} • Ground truth ${'{'}session.groundTruthLabel{'}'} • ${'{'}session.zone{'}'}`;
      const badge = document.getElementById('workerBadge');
      badge.className = `badge ${'{'}riskClass(rec.riskBand){'}'}`;
      badge.textContent = rec.riskBand;
      document.getElementById('detailTimestamp').textContent = rec.timestamp;
      document.getElementById('recommendationBox').textContent = rec.recommendation;

      const detailStats = document.getElementById('detailStats');
      detailStats.innerHTML = '';
      const stats = [
        ['Current overall risk', rec.risk.toFixed(1)],
        ['Dominant risk', rec.dominantRisk],
        ['Peak session risk', session.metrics.peakRisk.toFixed(1)],
        ['Average confidence', `${'{'}session.metrics.avgConfidence.toFixed(0){'}'}%`],
      ];
      stats.forEach(([k,v]) => {{
        const box = document.createElement('div');
        box.className = 'stat-box';
        box.innerHTML = `<div class="k">${'{'}k{'}'}</div><div class="v">${'{'}v{'}'}</div>`;
        detailStats.appendChild(box);
      }});

      const signalStats = document.getElementById('signalStats');
      signalStats.innerHTML = '';
      const signalCards = [
        ['EDA', rec.eda.toFixed(3)],
        ['BVP', rec.bvp.toFixed(3)],
        ['Temp', rec.temperature.toFixed(3)],
        ['Data confidence', `${'{'}rec.dataConfidence.toFixed(0){'}'}%`],
      ];
      signalCards.forEach(([k,v]) => {{
        const box = document.createElement('div');
        box.className = 'stat-box';
        box.innerHTML = `<div class="k">${'{'}k{'}'}</div><div class="v">${'{'}v{'}'}</div>`;
        signalStats.appendChild(box);
      }});

      drawTrendChart(session.records, playbackIndex(session.records));
    }}

    function clearDetailState() {{
      document.getElementById('workerTitle').textContent = 'No worker selected';
      document.getElementById('workerSubtitle').textContent = '';
      document.getElementById('detailTimestamp').textContent = '';
      document.getElementById('recommendationBox').textContent = 'Change the filters or choose a worker to view recommendations.';
      document.getElementById('detailStats').innerHTML = '';
      document.getElementById('signalStats').innerHTML = '';
      document.getElementById('alertList').innerHTML = '<div class="small">No active alerts.</div>';
      const ctx = document.getElementById('trendChart').getContext('2d');
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }}

    function renderAlerts(session, rec) {{
      const alertList = document.getElementById('alertList');
      alertList.innerHTML = '';
      const upto = playbackIndex(session.records);
      const activeAlerts = session.alerts.filter(alert => {{
        const match = session.records.find(r => r.timestamp.endsWith(alert.time));
        if (!match) return true;
        return match.index <= upto;
      }}).slice(-6).reverse();

      if (!activeAlerts.length) {{
        const div = document.createElement('div');
        div.className = 'small';
        div.textContent = 'No caution/high alerts have triggered yet at the current playback time.';
        alertList.appendChild(div);
        return;
      }}

      activeAlerts.forEach(alert => {{
        const div = document.createElement('div');
        div.className = 'alert-item';
        div.innerHTML = `
          <div class="alert-top">
            <strong>${'{'}alert.type{'}'}</strong>
            <div class="badge ${'{'}riskClass(alert.level){'}'}">${'{'}alert.level{'}'}</div>
          </div>
          <div class="small">${'{'}alert.time{'}'} • Score ${'{'}alert.score.toFixed(1){'}'}</div>
          <div style="margin-top:6px;">${'{'}alert.message{'}'}</div>
          <div class="small" style="margin-top:8px;">${'{'}alert.recommendation{'}'}</div>
        `;
        alertList.appendChild(div);
      }});
    }}

    function renderZoneSnapshot(list) {{
      const zoneList = document.getElementById('zoneList');
      zoneList.innerHTML = '';
      const grouped = new Map();
      list.forEach(session => {{
        const rec = getCurrentRecord(session);
        const current = grouped.get(session.zone) || {{ count: 0, high: 0, caution: 0, avgRisk: 0 }};
        current.count += 1;
        current.avgRisk += rec.risk;
        if (rec.riskBand === 'High') current.high += 1;
        if (rec.riskBand === 'Caution') current.caution += 1;
        grouped.set(session.zone, current);
      }});
      const zones = Array.from(grouped.entries()).map(([zone, stats]) => ({{
        zone,
        count: stats.count,
        high: stats.high,
        caution: stats.caution,
        avgRisk: stats.avgRisk / Math.max(1, stats.count),
      }})).sort((a,b)=>b.avgRisk-a.avgRisk);

      if (!zones.length) {{
        zoneList.innerHTML = '<div class="small">No zone data for the current filters.</div>';
        return;
      }}

      zones.forEach(z => {{
        const div = document.createElement('div');
        div.className = 'zone-item';
        div.innerHTML = `
          <div class="zone-top">
            <strong>${'{'}z.zone{'}'}</strong>
            <span class="small">Avg risk ${'{'}z.avgRisk.toFixed(1){'}'}</span>
          </div>
          <div class="small">Workers: ${'{'}z.count{'}'} • High: ${'{'}z.high{'}'} • Caution: ${'{'}z.caution{'}'}</div>
        `;
        zoneList.appendChild(div);
      }});
    }}

    function drawTrendChart(records, upto) {{
      const canvas = document.getElementById('trendChart');
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = 'rgba(255,255,255,0.02)';
      ctx.fillRect(0, 0, w, h);

      const pad = {{ left: 48, right: 18, top: 20, bottom: 32 }};
      const chartW = w - pad.left - pad.right;
      const chartH = h - pad.top - pad.bottom;
      const lines = [
        {{ key: 'stress', color: '#38bdf8', label: 'Stress' }},
        {{ key: 'fatigue', color: '#f59e0b', label: 'Fatigue' }},
        {{ key: 'cognitiveLoad', color: '#c084fc', label: 'Cognitive' }},
        {{ key: 'risk', color: '#ef4444', label: 'Overall' }},
      ];

      ctx.strokeStyle = 'rgba(148,163,184,0.20)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {{
        const y = pad.top + (chartH / 5) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();
        const value = 100 - i * 20;
        ctx.fillStyle = 'rgba(148,163,184,0.9)';
        ctx.font = '12px sans-serif';
        ctx.fillText(String(value), 8, y + 4);
      }}

      const visible = records.slice(0, upto + 1);
      const maxPoints = visible.length;
      if (!maxPoints) return;

      lines.forEach(line => {{
        ctx.strokeStyle = line.color;
        ctx.lineWidth = line.key === 'risk' ? 3 : 2;
        ctx.beginPath();
        visible.forEach((r, idx) => {{
          const x = pad.left + (idx / Math.max(1, maxPoints - 1)) * chartW;
          const y = pad.top + chartH - (r[line.key] / 100) * chartH;
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.stroke();
      }});

      const current = visible[visible.length - 1];
      const markerX = pad.left + ((visible.length - 1) / Math.max(1, maxPoints - 1)) * chartW;
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(markerX, pad.top);
      ctx.lineTo(markerX, h - pad.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      const legendX = pad.left;
      lines.forEach((line, i) => {{
        const x = legendX + i * 140;
        const y = h - 10;
        ctx.fillStyle = line.color;
        ctx.fillRect(x, y - 10, 18, 4);
        ctx.fillStyle = 'rgba(229,231,235,0.92)';
        ctx.font = '12px sans-serif';
        ctx.fillText(line.label, x + 24, y - 6);
      }});
    }}

    init();
  </script>
</body>
</html>'''


def main() -> None:
    dataset = build_dataset()
    write_files(dataset)
    print(f'Wrote {OUTPUT_HTML}')
    print(f'Wrote {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
