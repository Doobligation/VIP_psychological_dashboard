"""
speak_biomarker_labels.py

Polls the live biomarker endpoint served by `avro_stream_poller_with_ml.py`
(from the VIP_psychological_dashboard project) and reads new Frustration /
Mental Demand label updates aloud using offline text-to-speech (pyttsx3).

WHY pyttsx3:
- Runs fully offline, no API key or network call needed per utterance
- Works on Windows (SAPI5), macOS (NSSpeechSynthesizer), and Linux (espeak)
- Well suited to a continuously-running local monitoring loop like this one,
  where you don't want cloud latency, cost, or a dependency on internet
  access just to announce a label change

SETUP
-----
1. Make sure the live poller from the project is already running in another
   terminal, as described in the project's README:

       python avro_stream_poller_with_ml.py

   This exposes live predictions at http://127.0.0.1:7000/latest

2. Install dependencies for this script:

       pip install pyttsx3 requests

   On Linux, pyttsx3 also needs espeak installed at the OS level, e.g.:

       sudo apt-get install espeak

3. Run this script in a separate terminal:

       python speak_biomarker_labels.py

USAGE NOTES
-----------
- This script only SPEAKS. It does not modify the dashboard or the poller.
- The /latest JSON structure is confirmed from the actual project source
  (avro_stream_poller_with_ml.py + biomarker_runtime.py) -- see
  extract_labels() below for the exact fields used.
- The script only announces a label out loud when it CHANGES (or the first
  time it's seen), so it doesn't repeat "Frustration: High" every 5 seconds
  if nothing has changed.
- Frustration needs 2 minute-level samples and Mental Demand needs 1 before
  either model is "ready". Until then this script announces "Pending" once,
  then announces the real label once it resolves.

SECURITY NOTE
-------------
avro_stream_poller_with_ml.py (as provided) contains hardcoded AWS
credentials. Those should be rotated immediately and moved to environment
variables -- they should never live in source code, especially in a public
repo. This script does not need or use any AWS credentials; it only talks
to your local http://127.0.0.1:7000/latest endpoint.
"""

import argparse
import sys
import time
from typing import Optional

import requests

try:
    import pyttsx3
except ImportError:
    print(
        "pyttsx3 is not installed. Install it with:\n"
        "    pip install pyttsx3\n"
        "(On Linux you may also need: sudo apt-get install espeak)",
        file=sys.stderr,
    )
    raise

DEFAULT_ENDPOINT = "http://127.0.0.1:7000/latest"
POLL_INTERVAL_SECONDS = 5  # matches the dashboard's own auto-refresh cadence


def build_tts_engine(rate: int = 175, volume: float = 1.0) -> "pyttsx3.Engine":
    """Create and configure a pyttsx3 TTS engine instance."""
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    return engine


def speak(engine: "pyttsx3.Engine", text: str) -> None:
    """Speak text aloud and wait until finished before returning."""
    print(f"[speaking] {text}")
    engine.say(text)
    engine.runAndWait()


def fetch_latest(endpoint: str) -> Optional[dict]:
    """Fetch the latest JSON payload from the live poller endpoint."""
    try:
        resp = requests.get(endpoint, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"[warning] could not reach {endpoint}: {exc}", file=sys.stderr)
        return None
    except ValueError:
        print("[warning] response was not valid JSON", file=sys.stderr)
        return None


def extract_labels(payload: dict) -> dict:
    """
    Pull out frustration / mental demand label + confidence from the raw
    /latest JSON payload produced by avro_stream_poller_with_ml.py.

    Confirmed exact shape (from biomarker_runtime.py's
    LiveBiomarkerPredictor.update()):

        payload["predictions"] = {
            "session_minute_index": int,
            "history_size": int,
            "model_notes": [...],
            "frustration": {
                "label": str or None,
                "confidence": float or None,
                "ready": bool,
                "reason": str or None,
            },
            "mental_demand": {
                "label": str or None,
                "confidence": float or None,
                "ready": bool,
                "reason": str or None,
            },
        }

    "ready" is False when there isn't yet enough history for that model
    (frustration needs 2 minute-level samples, mental_demand needs 1). When
    not ready, we surface the label as "Pending" (matching the dashboard's
    own terminology from the README) along with the reason.
    """
    predictions = payload.get("predictions")
    if not isinstance(predictions, dict):
        # e.g. still "waiting for first sample" / "waiting for biomarker files"
        return {
            "frustration_label": None,
            "frustration_confidence": None,
            "mental_demand_label": None,
            "mental_demand_confidence": None,
        }

    def _read(block_name):
        block = predictions.get(block_name) or {}
        if block.get("ready"):
            return block.get("label"), block.get("confidence")
        # Not ready yet -- surface as "Pending" so the change gets announced,
        # then a later real label will differ from "Pending" and get spoken.
        return "Pending", None

    frustration_label, frustration_conf = _read("frustration")
    mental_demand_label, mental_demand_conf = _read("mental_demand")

    return {
        "frustration_label": frustration_label,
        "frustration_confidence": frustration_conf,
        "mental_demand_label": mental_demand_label,
        "mental_demand_confidence": mental_demand_conf,
    }


def format_utterance(kind: str, label, confidence) -> str:
    if label is None:
        return ""
    if confidence is not None:
        try:
            pct = round(float(confidence) * 100)
            return f"{kind} is now {label}, {pct} percent confidence."
        except (TypeError, ValueError):
            pass
    return f"{kind} is now {label}."


def run_loop(endpoint: str, interval: int, debug: bool) -> None:
    engine = build_tts_engine()
    last_frustration = None
    last_mental_demand = None

    print(f"Polling {endpoint} every {interval}s. Press Ctrl+C to stop.")

    while True:
        payload = fetch_latest(endpoint)
        if payload is not None:
            if debug:
                print(f"[debug] raw payload: {payload}")

            labels = extract_labels(payload)

            f_label = labels["frustration_label"]
            f_conf = labels["frustration_confidence"]
            m_label = labels["mental_demand_label"]
            m_conf = labels["mental_demand_confidence"]

            if f_label is not None and f_label != last_frustration:
                utterance = format_utterance("Frustration", f_label, f_conf)
                if utterance:
                    speak(engine, utterance)
                last_frustration = f_label

            if m_label is not None and m_label != last_mental_demand:
                utterance = format_utterance("Mental demand", m_label, m_conf)
                if utterance:
                    speak(engine, utterance)
                last_mental_demand = m_label

            if f_label is None and m_label is None and payload.get("note") not in (
                "waiting for first sample",
                "waiting for biomarker files",
            ):
                print(
                    f"[info] no predictions yet -- poller note: {payload.get('note')!r}",
                    file=sys.stderr,
                )

        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Poll the live biomarker endpoint and speak Frustration / "
            "Mental Demand label updates aloud."
        )
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"URL of the live poller's /latest endpoint (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=POLL_INTERVAL_SECONDS,
        help=f"Seconds between polls (default: {POLL_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the raw JSON payload on every poll for troubleshooting",
    )
    args = parser.parse_args()

    try:
        run_loop(args.endpoint, args.interval, args.debug)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
