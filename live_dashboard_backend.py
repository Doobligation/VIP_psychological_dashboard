from __future__ import annotations

import copy
import json
import math
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


class LiveDashboardBridge:
    def __init__(self, base_json_path: Path | str, live_url: str = 'http://127.0.0.1:7000/latest', poll_seconds: int = 5):
        self.base_json_path = Path(base_json_path)
        self.live_url = live_url
        self.poll_seconds = poll_seconds
        self.lock = threading.Lock()

        self.base_data = json.loads(self.base_json_path.read_text(encoding='utf-8'))
        self.raw_history = deque(maxlen=180)
        self.session_records = {'Frustration': [], 'Mental Demand': []}
        self.session_alerts = {'Frustration': [], 'Mental Demand': []}
        self.latest_sample: dict[str, Any] | None = None
        self.last_signature = None
        self.status = {
            'connected': False,
            'message': 'Waiting for live stream.',
            'last_poll_at': None,
            'last_update_at': None,
            'live_url': self.live_url,
        }

    def start(self):
        thread = threading.Thread(target=self._poll_loop, daemon=True)
        thread.start()

    def _poll_loop(self):
        while True:
            try:
                self.poll_once()
            except Exception as exc:
                with self.lock:
                    self.status.update({
                        'connected': False,
                        'message': f'Live poll failed: {exc}',
                        'last_poll_at': datetime.now().isoformat(timespec='seconds'),
                    })
            time.sleep(self.poll_seconds)

    def _fetch_latest(self) -> dict[str, Any]:
        req = Request(self.live_url, headers={'Cache-Control': 'no-cache'})
        with urlopen(req, timeout=4) as resp:
            return json.loads(resp.read().decode('utf-8'))

    def poll_once(self):
        now_iso = datetime.now().isoformat(timespec='seconds')
        try:
            payload = self._fetch_latest()
        except URLError as exc:
            with self.lock:
                self.status.update({
                    'connected': False,
                    'message': f'Live stream unavailable: {exc.reason}',
                    'last_poll_at': now_iso,
                })
            return
        except Exception as exc:
            with self.lock:
                self.status.update({
                    'connected': False,
                    'message': f'Live stream unavailable: {exc}',
                    'last_poll_at': now_iso,
                })
            return

        signature = (
            payload.get('participant_id'),
            payload.get('updated_at'),
            payload.get('eda_time'),
            payload.get('temp_time'),
            payload.get('pulse_time'),
            payload.get('eda'),
            payload.get('temp'),
            payload.get('pulse_rate'),
        )

        with self.lock:
            self.status.update({
                'connected': True,
                'message': payload.get('note') or 'Connected to live stream.',
                'last_poll_at': now_iso,
                'last_update_at': payload.get('updated_at'),
            })
            self.latest_sample = payload

            if not payload.get('updated_at'):
                return
            if signature == self.last_signature:
                return

            participant_id = payload.get('participant_id') or 'LIVE'
            if self.last_signature and participant_id != self.last_signature[0]:
                self.raw_history.clear()
                self.session_records = {'Frustration': [], 'Mental Demand': []}
                self.session_alerts = {'Frustration': [], 'Mental Demand': []}

            self.last_signature = signature
            self._append_live_sample(payload)

    def _append_live_sample(self, payload: dict[str, Any]):
        sample_time = payload.get('eda_time') or payload.get('temp_time') or payload.get('pulse_time') or payload.get('updated_at')
        timestamp = self._format_timestamp(sample_time)
        eda = self._safe_float(payload.get('eda'))
        temp = self._safe_float(payload.get('temp'))
        pulse = self._safe_float(payload.get('pulse_rate'))

        predictions = payload.get('predictions') or {}
        frustration_info = predictions.get('frustration') or {}
        mental_info = predictions.get('mental_demand') or {}

        point = {
            'timestamp': timestamp,
            'eda': eda,
            'temp': temp,
            'pulse': pulse,
            'frustration_label': self._normalize_label(frustration_info.get('label')),
            'frustration_confidence': self._safe_float(frustration_info.get('confidence')),
            'mental_label': self._normalize_label(mental_info.get('label')),
            'mental_confidence': self._safe_float(mental_info.get('confidence')),
        }
        self.raw_history.append(point)

        self.session_records['Frustration'].append(self._build_record(payload, point, task_type='Frustration'))
        self.session_records['Mental Demand'].append(self._build_record(payload, point, task_type='Mental Demand'))

        for task_type in ('Frustration', 'Mental Demand'):
            self._maybe_add_alert(task_type)

    def _build_record(self, payload: dict[str, Any], point: dict[str, Any], task_type: str) -> dict[str, Any]:
        idx = len(self.session_records[task_type])
        minute_index = int((payload.get('predictions') or {}).get('session_minute_index') or idx)

        eda_norm = self._normalize_recent('eda', point['eda'])
        temp_norm = self._normalize_recent('temp', point['temp'])
        pulse_norm = self._normalize_recent('pulse', point['pulse'])
        eda_vol = self._recent_volatility('eda')
        pulse_vol = self._recent_volatility('pulse')
        minute_progress = min(1.0, minute_index / 10.0)

        frustration_score = self._label_score(point['frustration_label'])
        mental_score = self._label_score(point['mental_label'])

        stress = 100 * (0.34 * frustration_score + 0.28 * eda_norm + 0.18 * pulse_norm + 0.20 * eda_vol)
        fatigue = 100 * (0.34 * temp_norm + 0.22 * pulse_norm + 0.24 * minute_progress + 0.20 * pulse_vol)
        cognitive = 100 * (0.44 * mental_score + 0.16 * eda_vol + 0.20 * minute_progress + 0.20 * pulse_vol)

        if task_type == 'Frustration':
            overall = 0.50 * stress + 0.18 * fatigue + 0.32 * cognitive
        else:
            overall = 0.22 * stress + 0.18 * fatigue + 0.60 * cognitive
        overall = self._clamp(overall, 0, 100)

        if overall >= 78:
            risk_band = 'High'
        elif overall >= 60:
            risk_band = 'Caution'
        else:
            risk_band = 'Normal'

        dominant = max(
            [('Stress', stress), ('Fatigue', fatigue), ('Cognitive Load', cognitive)],
            key=lambda item: item[1],
        )[0]

        data_confidence = self._compute_data_confidence(point)
        recommendation = self._build_recommendation(
            task_type=task_type,
            risk_band=risk_band,
            dominant=dominant,
            frustration_label=point['frustration_label'],
            frustration_conf=point['frustration_confidence'],
            mental_label=point['mental_label'],
            mental_conf=point['mental_confidence'],
        )

        return {
            'index': idx,
            'timestamp': point['timestamp'],
            'eda': round(point['eda'], 4) if not math.isnan(point['eda']) else 0.0,
            'bvp': round(point['pulse'], 4) if not math.isnan(point['pulse']) else 0.0,
            'pulseRate': round(point['pulse'], 2) if not math.isnan(point['pulse']) else None,
            'temperature': round(point['temp'], 4) if not math.isnan(point['temp']) else 0.0,
            'stress': round(stress, 1),
            'fatigue': round(fatigue, 1),
            'cognitiveLoad': round(cognitive, 1),
            'risk': round(overall, 1),
            'dataConfidence': round(data_confidence, 1),
            'riskBand': risk_band,
            'dominantRisk': dominant,
            'recommendation': recommendation,
            'frustrationLabel': point['frustration_label'].title() if point['frustration_label'] != 'pending' else 'Pending',
            'frustrationConfidence': self._pct(point['frustration_confidence']),
            'mentalDemandLabel': point['mental_label'].title() if point['mental_label'] != 'pending' else 'Pending',
            'mentalDemandConfidence': self._pct(point['mental_confidence']),
            'liveSampleNote': payload.get('note'),
            'sessionMinuteIndex': minute_index,
        }

    def _maybe_add_alert(self, task_type: str):
        records = self.session_records[task_type]
        if not records:
            return
        rec = records[-1]
        if rec['riskBand'] == 'Normal':
            return

        alerts = self.session_alerts[task_type]
        should_add = True
        if alerts:
            prev = alerts[-1]
            should_add = (
                prev['level'] != rec['riskBand']
                or prev['type'] != rec['dominantRisk']
                or rec['index'] - prev['recordIndex'] >= 2
            )
        if not should_add:
            return

        alerts.append({
            'time': self._time_only(rec['timestamp']),
            'level': rec['riskBand'],
            'type': rec['dominantRisk'],
            'message': self._build_alert_message(rec['riskBand'], rec['dominantRisk']),
            'recommendation': rec['recommendation'],
            'score': rec['risk'],
            'recordIndex': rec['index'],
        })

    def get_dashboard_data(self) -> dict[str, Any]:
        with self.lock:
            data = copy.deepcopy(self.base_data)
            data['generatedFrom'] = f"{self.base_data.get('generatedFrom', 'historical sessions')} + live wristband stream"
            data['liveStatus'] = copy.deepcopy(self.status)
            data['liveLatestSample'] = copy.deepcopy(self.latest_sample)

            participant_id = (self.latest_sample or {}).get('participant_id') or 'LIVE'
            for task_type in ('Frustration', 'Mental Demand'):
                records = copy.deepcopy(self.session_records[task_type])
                if not records:
                    continue
                alerts = copy.deepcopy(self.session_alerts[task_type])
                session = {
                    'sessionId': f"live-{task_type.lower().replace(' ', '-')}-{participant_id.lower()}",
                    'fileName': 'live wristband stream',
                    'participantId': participant_id,
                    'workerName': f"Live Worker {participant_id}",
                    'mode': 'Live',
                    'taskType': task_type,
                    'groundTruthLabel': 'Live model inference',
                    'zone': 'Live Sensor Stream',
                    'records': records,
                    'metrics': self._session_metrics(records),
                    'alerts': alerts,
                    'summary': self._session_summary(records),
                }
                data['categories'].setdefault(task_type, [])
                data['categories'][task_type] = data['categories'][task_type] + [session]
            return data

    def _session_metrics(self, records: list[dict[str, Any]]) -> dict[str, float]:
        if not records:
            return {'avgStress': 0.0, 'avgFatigue': 0.0, 'avgCognitiveLoad': 0.0, 'peakRisk': 0.0, 'avgConfidence': 0.0}
        return {
            'avgStress': round(sum(r['stress'] for r in records) / len(records), 1),
            'avgFatigue': round(sum(r['fatigue'] for r in records) / len(records), 1),
            'avgCognitiveLoad': round(sum(r['cognitiveLoad'] for r in records) / len(records), 1),
            'peakRisk': round(max(r['risk'] for r in records), 1),
            'avgConfidence': round(sum(r['dataConfidence'] for r in records) / len(records), 1),
        }

    def _session_summary(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {'start': None, 'end': None, 'durationMinutes': 0.0, 'finalRisk': 0.0, 'finalBand': 'Normal'}
        return {
            'start': records[0]['timestamp'],
            'end': records[-1]['timestamp'],
            'durationMinutes': round(max(1, len(records)) / 1.0, 1),
            'finalRisk': records[-1]['risk'],
            'finalBand': records[-1]['riskBand'],
        }

    def _compute_data_confidence(self, point: dict[str, Any]) -> float:
        fields = [point['eda'], point['temp'], point['pulse']]
        available = sum(0 if math.isnan(v) else 1 for v in fields)
        sensor_pct = 100 * (available / 3)
        model_conf = [self._pct(point['frustration_confidence']), self._pct(point['mental_confidence'])]
        model_conf = [m for m in model_conf if m is not None]
        if model_conf:
            return self._clamp(0.55 * sensor_pct + 0.45 * (sum(model_conf) / len(model_conf)), 55, 100)
        return self._clamp(sensor_pct, 55, 100)

    def _label_score(self, label: str) -> float:
        return {
            'low': 0.25,
            'medium': 0.58,
            'high': 0.90,
            'pending': 0.50,
        }.get(label, 0.50)

    def _normalize_recent(self, key: str, current_value: float) -> float:
        values = [item[key] for item in self.raw_history if not math.isnan(item[key])]
        if math.isnan(current_value):
            return 0.5
        if not values:
            return 0.5
        lo = min(values)
        hi = max(values)
        if math.isclose(lo, hi):
            return 0.5
        return self._clamp((current_value - lo) / (hi - lo), 0.0, 1.0)

    def _recent_volatility(self, key: str, window: int = 5) -> float:
        values = [item[key] for item in list(self.raw_history)[-window:] if not math.isnan(item[key])]
        if len(values) <= 1:
            return 0.1
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        span = max(values) - min(values)
        if math.isclose(span, 0.0):
            return 0.1
        return self._clamp(std / span, 0.0, 1.0)

    def _build_alert_message(self, level: str, dominant: str) -> str:
        prefix = 'Immediate attention recommended' if level == 'High' else 'Monitor closely'
        mapping = {
            'Stress': f'{prefix}: live frustration-linked arousal is trending upward.',
            'Fatigue': f'{prefix}: live fatigue indicators are accumulating across the session.',
            'Cognitive Load': f'{prefix}: mental-demand pressure is elevated in the current task window.',
        }
        return mapping.get(dominant, f'{prefix}: psychosocial risk is elevated.')

    def _build_recommendation(
        self,
        task_type: str,
        risk_band: str,
        dominant: str,
        frustration_label: str,
        frustration_conf: float | None,
        mental_label: str,
        mental_conf: float | None,
    ) -> str:
        fr_text = f"Frustration {frustration_label.title()}"
        fr_pct = self._pct(frustration_conf)
        if fr_pct is not None:
            fr_text += f" ({fr_pct:.0f}% conf)"
        md_text = f"Mental Demand {mental_label.title()}"
        md_pct = self._pct(mental_conf)
        if md_pct is not None:
            md_text += f" ({md_pct:.0f}% conf)"

        if risk_band == 'High':
            if dominant == 'Stress':
                action = 'Pause the task, check in with the worker, and consider a short recovery break.'
            elif dominant == 'Fatigue':
                action = 'Plan a task rotation, hydration break, and workload adjustment.'
            else:
                action = 'Reduce task complexity, repeat critical instructions, and slow the task pace.'
        elif risk_band == 'Caution':
            if dominant == 'Stress':
                action = 'Watch the next few minutes closely and prepare a supervisor check-in.'
            elif dominant == 'Fatigue':
                action = 'Prepare an earlier rest or rotation if the trend continues.'
            else:
                action = 'Monitor comprehension and simplify the next task step if needed.'
        else:
            action = 'Continue monitoring; no immediate intervention is needed.'

        return f"{action} Live model output: {fr_text}; {md_text}. Dashboard lane: {task_type}."

    def _format_timestamp(self, value: Any) -> str:
        if not value:
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            return datetime.fromisoformat(str(value).replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            try:
                return str(value).replace('T', ' ')[:19]
            except Exception:
                return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _time_only(self, value: str) -> str:
        return value.split(' ')[-1] if value else ''

    def _normalize_label(self, value: Any) -> str:
        if value is None:
            return 'pending'
        text = str(value).strip().lower()
        if text in {'low', 'medium', 'high'}:
            return text
        return 'pending'

    def _safe_float(self, value: Any) -> float:
        try:
            if value is None:
                return float('nan')
            return float(value)
        except Exception:
            return float('nan')

    def _pct(self, value: float | None) -> float | None:
        if value is None or math.isnan(value):
            return None
        return self._clamp(value * 100.0, 0.0, 100.0)

    def _clamp(self, value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))
