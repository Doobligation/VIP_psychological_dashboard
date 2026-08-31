import os
import time
from datetime import datetime, timezone, timedelta

import boto3
import pandas as pd
from flask import Flask, jsonify

from biomarker_runtime import LiveBiomarkerPredictor

AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION = "us-east-1"

S3_BUCKET = "empatica-us-east-1-prod-data"
S3_BASE_PREFIX = "v2/597/1/1/participant_data/"
POLL_SECONDS = 30

FORCE_DOWNLOAD_EVERY_POLL = False

LOCAL_BIOMARKER_DIR = r""
os.makedirs(LOCAL_BIOMARKER_DIR, exist_ok=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
predictor = LiveBiomarkerPredictor(MODEL_DIR)

app = Flask(__name__)
latest_sample = {
    "eda": None,
    "temp": None,
    "pulse_rate": None,
    "eda_time": None,
    "temp_time": None,
    "pulse_time": None,
    "participant_id": None,
    "source_prefix": None,
    "updated_at": None,
    "note": "waiting for first sample",
    "predictions": None,
}


@app.route("/latest", methods=["GET"])
def latest():
    return jsonify(latest_sample)


def make_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def list_objects_all(s3, prefix):
    out = []
    token = None
    while True:
        args = {"Bucket": S3_BUCKET, "Prefix": prefix}
        if token:
            args["ContinuationToken"] = token
        resp = s3.list_objects_v2(**args)
        out.extend(resp.get("Contents", []))
        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            break
    return out


def download_key(s3, key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(S3_BUCKET, key, local_path)


def looks_like_epoch_ms(x):
    try:
        return float(x) > 1e11
    except Exception:
        return False


def epoch_ms_to_iso(v):
    try:
        dt = datetime.fromtimestamp(float(v) / 1000.0, tz=timezone.utc).astimezone()
        return dt.isoformat(timespec="seconds")
    except Exception:
        return None


def find_timestamp_col(df):
    for c in df.columns:
        cl = c.lower()
        if "timestamp_iso" == cl:
            return c
    for c in df.columns:
        cl = c.lower()
        if "timestamp_unix" == cl:
            return c
    for c in df.columns:
        cl = c.lower()
        if "time" in cl or "stamp" in cl or "unix" in cl or "epoch" in cl:
            return c
    return None


def choose_value_col(df, kind):
    preferred = {
        "eda": ["eda_scl_usiemens", "eda", "EDA", "value"],
        "temp": ["temperature_celsius", "temp_celsius", "temperature", "temp", "value"],
        "pulse": ["pulse_rate", "pulse-rate", "pulse", "hr", "heart_rate", "bpm", "value"],
    }[kind]

    for c in preferred:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c

    ts_col = find_timestamp_col(df)
    for c in df.columns:
        if c != ts_col and pd.api.types.is_numeric_dtype(df[c]):
            try:
                if not looks_like_epoch_ms(df.iloc[-1][c]):
                    return c
            except Exception:
                return c
    return None


def read_last_value(csv_path, kind):
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, None

    ts_col = find_timestamp_col(df)
    val_col = choose_value_col(df, kind)
    if not val_col:
        print(f"[DEBUG] No value column for {kind}: {list(df.columns)}")
        return None, None

    for i in range(len(df) - 1, -1, -1):
        row = df.iloc[i]
        val = row.get(val_col, None)
        if pd.isna(val) or looks_like_epoch_ms(val):
            continue

        ts_val = None
        if ts_col and ts_col in df.columns:
            ts_raw = row.get(ts_col, None)
            if pd.notna(ts_raw):
                if looks_like_epoch_ms(ts_raw):
                    ts_val = epoch_ms_to_iso(ts_raw)
                else:
                    ts_val = str(ts_raw)

        return ts_val, float(val)

    return None, None


def utc_dates_to_check(days_back=1):
    now_utc = datetime.now(timezone.utc)
    return [(now_utc - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(0, days_back + 1)]


def pick_most_recent_participant_and_keys(s3, date_prefixes, date_strings):
    candidates = []

    for prefix, day in zip(date_prefixes, date_strings):
        objs = list_objects_all(s3, prefix)
        for o in objs:
            k = o["Key"]
            if "/digital_biomarkers/aggregated_per_minute/" not in k:
                continue
            if not k.lower().endswith(".csv"):
                continue
            if f"_{day}_" not in k:
                continue
            candidates.append(o)

    if not candidates:
        return None, {}, {}, None

    groups = {}
    for o in candidates:
        parts = o["Key"].split("/")
        pid = None
        for day in date_strings:
            if day in parts:
                try:
                    pid = parts[parts.index(day) + 1]
                except Exception:
                    pid = None
                if pid:
                    break
        if not pid:
            continue
        groups.setdefault(pid, []).append(o)

    if not groups:
        return None, {}, {}, None

    best_pid = max(groups, key=lambda p: max(x["LastModified"] for x in groups[p]))
    keys = {}
    mods = {}
    used_prefix = None
    best_objs = groups[best_pid]

    def newest_ending(suffix):
        matches = [o for o in best_objs if o["Key"].endswith(suffix)]
        if not matches:
            return None
        return max(matches, key=lambda x: x["LastModified"])

    eda_o = newest_ending("_eda.csv")
    temp_o = newest_ending("_temperature.csv")
    pulse_o = newest_ending("_pulse-rate.csv")

    if eda_o:
        keys["eda"] = eda_o["Key"]
        mods["eda"] = eda_o["LastModified"]
    if temp_o:
        keys["temp"] = temp_o["Key"]
        mods["temp"] = temp_o["LastModified"]
    if pulse_o:
        keys["pulse"] = pulse_o["Key"]
        mods["pulse"] = pulse_o["LastModified"]

    newest_obj = max(best_objs, key=lambda x: x["LastModified"])
    newest_key_parts = newest_obj["Key"].split("/")
    for day in date_strings:
        if day in newest_key_parts:
            used_prefix = f"{S3_BASE_PREFIX}{day}/"
            break

    return best_pid, keys, mods, used_prefix


def poll_loop():
    s3 = make_s3_client()
    last_sig = None
    last_mods = {}

    print("Polling biomarker CSV updates...")

    while True:
        try:
            days = utc_dates_to_check(days_back=1)
            prefixes = [f"{S3_BASE_PREFIX}{d}/" for d in days]
            pid, keys, mods, used_prefix = pick_most_recent_participant_and_keys(s3, prefixes, days)

            if not pid or not all(k in keys for k in ("eda", "temp", "pulse")):
                latest_sample["note"] = "waiting for biomarker files"
                time.sleep(POLL_SECONDS)
                continue

            changed = FORCE_DOWNLOAD_EVERY_POLL or any(mods[k] != last_mods.get(k) for k in mods)

            print(f"[DEBUG] Participant: {pid}")
            print(f"[DEBUG] S3 LastModified eda={mods.get('eda')} temp={mods.get('temp')} pulse={mods.get('pulse')}")
            print(f"[DEBUG] changed={changed} (force={FORCE_DOWNLOAD_EVERY_POLL})")

            if not changed:
                time.sleep(POLL_SECONDS)
                continue

            local_day = used_prefix.split("/")[-2] if used_prefix else days[0]
            local_dir = os.path.join(LOCAL_BIOMARKER_DIR, local_day, pid)
            paths = {
                "eda": os.path.join(local_dir, "eda.csv"),
                "temp": os.path.join(local_dir, "temperature.csv"),
                "pulse": os.path.join(local_dir, "pulse-rate.csv"),
            }

            for k in paths:
                download_key(s3, keys[k], paths[k])
                last_mods[k] = mods[k]

            eda_ts, eda_val = read_last_value(paths["eda"], "eda")
            temp_ts, temp_val = read_last_value(paths["temp"], "temp")
            pulse_ts, pulse_val = read_last_value(paths["pulse"], "pulse")

            sig = (eda_ts, temp_ts, pulse_ts, eda_val, temp_val, pulse_val)

            if sig == last_sig:
                latest_sample["note"] = "S3 updated but last valid value unchanged (no new minute row yet)"
                print("[DEBUG] last valid value unchanged:", sig)
            else:
                latest_sample.update(
                    {
                        "eda": eda_val,
                        "temp": temp_val,
                        "pulse_rate": pulse_val,
                        "eda_time": eda_ts,
                        "temp_time": temp_ts,
                        "pulse_time": pulse_ts,
                        "participant_id": pid,
                        "source_prefix": used_prefix if used_prefix else prefixes[0],
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "note": "new minute detected",
                    }
                )
                latest_sample["predictions"] = predictor.update(latest_sample)
                last_sig = sig
                print("Updated /latest:", latest_sample)

        except Exception as e:
            print("Poll error:", e)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    import threading

    threading.Thread(target=poll_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=7000)
