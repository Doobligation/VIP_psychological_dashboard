import json
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class LiveBiomarkerPredictor:
    def __init__(self, model_dir, max_gap_minutes=5):
        model_dir = Path(model_dir)
        self.frustration_model = joblib.load(model_dir / "frustration_model.joblib")
        self.mental_demand_model = joblib.load(model_dir / "mental_demand_model.joblib")
        metadata_path = model_dir / "model_metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        self.max_gap_minutes = max_gap_minutes
        self.reset()

    def reset(self):
        self.history = deque(maxlen=10)
        self.last_participant_id = None
        self.last_timestamp = None
        self.session_minute_index = -1

    def _parse_timestamp(self, sample):
        for key in ("eda_time", "temp_time", "pulse_time", "updated_at"):
            value = sample.get(key)
            if not value:
                continue
            try:
                return pd.to_datetime(value)
            except Exception:
                continue
        return pd.Timestamp(datetime.now())

    def _maybe_reset(self, participant_id, current_ts):
        if self.last_participant_id is None:
            return

        if participant_id != self.last_participant_id:
            self.reset()
            return

        if self.last_timestamp is not None:
            gap_minutes = abs((current_ts - self.last_timestamp).total_seconds()) / 60.0
            if gap_minutes > self.max_gap_minutes:
                self.reset()

    def _predict_with_confidence(self, model, features_df):
        label = model.predict(features_df)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(features_df)[0]
                confidence = float(np.max(probs))
            except Exception:
                confidence = None
        return label, confidence

    def update(self, sample):
        participant_id = sample.get("participant_id")
        current_ts = self._parse_timestamp(sample)

        self._maybe_reset(participant_id, current_ts)
        self.last_participant_id = participant_id
        self.last_timestamp = current_ts
        self.session_minute_index += 1

        current_point = {
            "eda": float(sample["eda"]) if sample.get("eda") is not None else np.nan,
            "temp": float(sample["temp"]) if sample.get("temp") is not None else np.nan,
            "pulse_rate": float(sample["pulse_rate"]) if sample.get("pulse_rate") is not None else np.nan,
            "minute_index": self.session_minute_index,
            "timestamp": str(current_ts),
        }
        self.history.append(current_point)

        response = {
            "session_minute_index": self.session_minute_index,
            "history_size": len(self.history),
            "model_notes": [
                "Predictions use EDA + temperature + session minute index.",
                "Pulse rate is stored in the live buffer but is not used by the current trained models.",
            ],
            "frustration": {
                "label": None,
                "confidence": None,
                "ready": False,
                "reason": "Need at least 2 minute-level samples for the frustration model.",
            },
            "mental_demand": {
                "label": None,
                "confidence": None,
                "ready": False,
                "reason": "Need at least 1 minute-level sample for the mental-demand model.",
            },
        }

        md_features = pd.DataFrame(
            [
                {
                    "eda": current_point["eda"],
                    "temp": current_point["temp"],
                    "minute_index": current_point["minute_index"],
                }
            ]
        )
        md_label, md_conf = self._predict_with_confidence(self.mental_demand_model, md_features)
        response["mental_demand"] = {
            "label": md_label,
            "confidence": md_conf,
            "ready": True,
            "reason": None,
        }

        if len(self.history) >= 2:
            prev = self.history[-2]
            cur = self.history[-1]
            fr_features = pd.DataFrame(
                [
                    {
                        "eda_t_minus_1": prev["eda"],
                        "temp_t_minus_1": prev["temp"],
                        "eda_t_0": cur["eda"],
                        "temp_t_0": cur["temp"],
                        "eda_delta_t_0": cur["eda"] - prev["eda"],
                        "temp_delta_t_0": cur["temp"] - prev["temp"],
                        "minute_index": cur["minute_index"],
                    }
                ]
            )
            fr_label, fr_conf = self._predict_with_confidence(self.frustration_model, fr_features)
            response["frustration"] = {
                "label": fr_label,
                "confidence": fr_conf,
                "ready": True,
                "reason": None,
            }

        return response
