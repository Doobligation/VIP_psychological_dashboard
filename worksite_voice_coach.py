"""
worksite_voice_coach.py

A specialized text-to-speech safety coach for CONSTRUCTION WORKERS, built on top
of the live biomarker stream from `avro_stream_poller_with_ml.py`.

WHAT MAKES THIS DIFFERENT FROM speak_biomarker_labels.py
--------------------------------------------------------
`speak_biomarker_labels.py` is a label reader: it says "Frustration is now High"
every time the model output flips. On a real job site that is noise -- a worker
in a harness forty feet up does not need a running commentary of ML labels.

This script is a COACH. It answers three questions a foreman actually cares
about:

    1. Is this worker under strain RIGHT NOW?          (label + biomarkers)
    2. For HOW LONG have they been under strain?       (sustained-load timer)
    3. What should they DO about it?                   (escalating instructions)

Design decisions, and why:

  * ESCALATION LADDER, not repetition. The same "high" reading means something
    different at minute 1 than at minute 45. Round one is a nudge ("have some
    water"). Round five is a safety stop ("flagging your supervisor"). See TIERS.

  * THE TIMER SURVIVES A DIP. If a worker drops from "high" to "medium" the
    sustained-strain clock keeps running through a grace window instead of
    resetting. Real fatigue does not reset because one minute looked better.

  * SPEAKS ONLY WHEN IT EARNS IT. A global quiet gap plus a per-message cooldown
    means the coach cannot nag. Nagging PPE gets switched off, and a safety
    device that gets switched off protects nobody.

  * FEMALE VOICE BY DEFAULT. Chosen deliberately: on a loud site, a voice that
    contrasts with the predominantly male crew chatter and low-frequency
    equipment rumble is easier to pick out of the noise floor.

  * HANDS-FREE AND EYES-FREE. A worker on a ladder cannot look at a dashboard.
    Every piece of feedback is audible and every instruction is a physical
    action ("step into the shade", "hand off to your partner").

  * IT LOGS. Every utterance is written to a CSV so the crew can review what was
    said, when, and on what readings -- useful for the presentation and for
    after-action review.

RUN IT
------
    pip install pyttsx3 requests
    # Linux only: sudo apt-get install espeak

    # Terminal 1 -- live stream (already part of the project)
    python avro_stream_poller_with_ml.py

    # Terminal 2 -- this coach
    python worksite_voice_coach.py

PRESENTATION / DEMO MODE
------------------------
You cannot wait 45 real minutes in front of a class, and the AWS stream may not
be live in the room. Two flags solve that:

    # Fully self-contained scripted shift -- no poller, no AWS, no network.
    python worksite_voice_coach.py --demo

    # Against the real stream, but 60x clock so "45 minutes" lands in 45 seconds.
    python worksite_voice_coach.py --time-scale 60

Other useful flags:

    --list-voices          show installed voices, then exit (pick one to use)
    --voice-name Samantha  force a specific voice
    --endpoint URL         point at the poller (auto-detects :7000 / :7001)
    --log-file PATH        where to write the spoken-event CSV
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import requests

try:
    import pyttsx3
except ImportError:
    print(
        "pyttsx3 is not installed. Install it with:\n"
        "    pip install pyttsx3 requests\n"
        "(On Linux you may also need: sudo apt-get install espeak)",
        file=sys.stderr,
    )
    raise


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# The poller binds to $PORT (default 7001), while the project README documents
# 7000. Rather than make the user guess, we probe both.
CANDIDATE_ENDPOINTS = (
    "http://127.0.0.1:7000/latest",
    "http://127.0.0.1:7001/latest",
)

POLL_INTERVAL_SECONDS = 5

# Anti-nag controls (in SIMULATED seconds, so --time-scale scales them too).
MIN_GAP_BETWEEN_UTTERANCES = 20
ADVISORY_COOLDOWN = 8 * 60

# How long a worker may sit at "medium" before the sustained-strain clock gives
# up and resets. A dip is not a recovery.
MEDIUM_GRACE_SECONDS = 5 * 60

# Only celebrate a recovery if they were actually loaded for a while first --
# otherwise the coach congratulates people for nothing.
RECOVERY_MIN_SUSTAINED_SECONDS = 5 * 60

# Biomarker thresholds for the physiological advisories. These are deliberately
# conservative starting points for a wrist sensor, not clinical cutoffs; tune
# them per crew against your own baseline data before relying on them.
SKIN_TEMP_HOT_C = 35.5
PULSE_ELEVATED_BPM = 110.0
EDA_SPIKE_USIEMENS = 8.0

LABEL_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass
class Tier:
    """One rung of the escalation ladder."""

    after_seconds: int
    name: str
    message: str


# Round one is a nudge; the ladder ends in a hard safety stop. Wording is
# written for someone holding a tool, so every line ends in a physical action.
TIERS: List[Tier] = [
    Tier(
        after_seconds=0,
        name="round-1-hydrate",
        message=(
            "Heads up. Your stress readings just went high. "
            "Are you feeling tired? Please have some water before your next lift."
        ),
    ),
    Tier(
        after_seconds=10 * 60,
        name="round-2-cool-down",
        message=(
            "You have been running high for ten minutes. "
            "Step into the shade, loosen your vest, and take five slow breaths."
        ),
    ),
    Tier(
        after_seconds=25 * 60,
        name="round-3-swap-task",
        message=(
            "Twenty five minutes at high strain. "
            "If you have a partner, hand the task over now and reset your grip."
        ),
    ),
    Tier(
        after_seconds=45 * 60,
        name="round-4-mandatory-break",
        message=(
            "Please take a break. You have been stressed for forty five minutes. "
            "Stop the task, step away from the equipment, and sit down for five minutes."
        ),
    ),
    Tier(
        after_seconds=60 * 60,
        name="round-5-supervisor",
        message=(
            "This is a safety call. You have been at high strain for a full hour. "
            "I am flagging your supervisor for a rest rotation. Please stop work now."
        ),
    ),
]

RECOVERY_MESSAGE = (
    "Good. Your readings are back to normal. Ease back into the task, "
    "and keep that water close."
)

MEDIUM_ENTRY_MESSAGE = (
    "Your strain is climbing. Slow the pace a little and check your footing."
)

# Advisories keyed by name so each gets its own cooldown.
ADVISORY_HEAT = (
    "Your skin temperature is high. Get out of the sun, drink water now, "
    "and wet your neck if you can."
)
ADVISORY_PULSE = (
    "Your heart rate is elevated. Put the load down, stand up straight, "
    "and breathe until it settles."
)
ADVISORY_COGNITIVE = (
    "Mental demand is high while your body is calm. "
    "Slow down and re-read the work order. Do not skip the checklist."
)
ADVISORY_EDA = (
    "Sharp stress spike detected. Stop what you are doing for ten seconds "
    "and look around you before you continue."
)


# --------------------------------------------------------------------------
# Clock -- lets the presentation compress 45 minutes into 45 seconds
# --------------------------------------------------------------------------


class Clock:
    """
    A wall clock that can run fast.

    Every duration in this script (tiers, cooldowns, grace windows) is expressed
    in SIMULATED seconds and read through this clock. With --time-scale 60 one
    real second counts as sixty simulated seconds, so the full escalation ladder
    can be demonstrated in about a minute without changing any of the thresholds
    that would ship to a real site.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self._start = time.monotonic()

    def now(self) -> float:
        return (time.monotonic() - self._start) * self.scale

    def advance(self) -> None:
        """No-op: a wall clock advances on its own."""


class StepClock:
    """
    A clock that advances a fixed amount per poll instead of with wall time.

    Demo mode uses this so the scripted shift is reproducible. With a scaled
    wall clock, the seconds spent actually SPEAKING count against the escalation
    timers -- at 60x a five-second sentence burns five simulated minutes, which
    can vault past a whole rung of the ladder and skip it in front of the class.
    Stepping the clock once per scripted reading removes that coupling entirely.
    """

    def __init__(self, seconds_per_step: float) -> None:
        self.seconds_per_step = seconds_per_step
        self._t = 0.0

    def now(self) -> float:
        return self._t

    def advance(self) -> None:
        self._t += self.seconds_per_step


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


# --------------------------------------------------------------------------
# Voice
# --------------------------------------------------------------------------

# Common female voice names across macOS / Windows SAPI5 / Linux espeak.
FEMALE_VOICE_HINTS = (
    "samantha", "victoria", "karen", "moira", "tessa", "fiona", "ava",
    "allison", "susan", "zira", "hazel", "eva", "catherine", "linda",
    "female", "f2", "f3",
)


def is_english(voice) -> bool:
    """
    True if this voice speaks English.

    Worth being careful here: a naive "first female voice" search on macOS picks
    the Italian voice 'Alice', which then reads English safety instructions with
    Italian phonetics -- unusable on a site. `languages` is the reliable signal
    where it exists; the id/name check covers the platforms where it is empty.
    """
    langs = getattr(voice, "languages", None) or []
    for lang in langs:
        if isinstance(lang, bytes):
            lang = lang.decode("utf-8", "ignore")
        if str(lang).lower().lstrip("\x05").startswith("en"):
            return True
    if langs:
        return False

    haystack = f"{voice.id or ''} {voice.name or ''}".lower()
    return "en-" in haystack or "en_" in haystack or "english" in haystack


def is_female(voice) -> bool:
    gender = str(getattr(voice, "gender", "") or "").lower()
    if "female" in gender:
        return True
    if gender and ("male" in gender or "neuter" in gender):
        return False
    haystack = f"{voice.name or ''} {voice.id or ''}".lower()
    return any(hint in haystack for hint in FEMALE_VOICE_HINTS)


def pick_voice(engine, preferred_name: Optional[str], prefer_female: bool) -> Optional[str]:
    """
    Choose a voice id, best match first.

    A female voice is the default on purpose -- on a site full of male crew
    chatter and low-frequency engine noise, a contrasting voice is easier to
    pick out of the noise floor. Intelligibility still wins over timbre, so an
    English male voice beats a non-English female one, and if nothing matches we
    fall back to the system default: a coach that will not speak is worse than
    one with the wrong voice.
    """
    try:
        voices = engine.getProperty("voices")
    except Exception:
        return None

    if preferred_name:
        needle = preferred_name.lower()
        for v in voices:
            if needle in (v.name or "").lower() or needle in (v.id or "").lower():
                return v.id
        print(
            f"[warning] no voice matching {preferred_name!r}; falling back. "
            "Run with --list-voices to see what is installed.",
            file=sys.stderr,
        )

    english = [v for v in voices if is_english(v)]

    if prefer_female:
        # Named preferences first: these are the clearest full-quality English
        # female voices across macOS, Windows SAPI5 and espeak.
        for hint in ("samantha", "zira", "hazel", "ava", "allison", "victoria", "karen"):
            for v in english:
                if hint in (v.name or "").lower():
                    return v.id
        for v in english:
            if is_female(v):
                return v.id

    return english[0].id if english else None


def list_voices(show_all: bool = False) -> None:
    """Print the installed voices, English-only unless show_all, marking the default pick."""
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    chosen = pick_voice(engine, None, prefer_female=True)

    shown = voices if show_all else [v for v in voices if is_english(v)]
    scope = "installed" if show_all else "English"
    print(f"{len(shown)} {scope} voice(s) of {len(voices)} installed:\n")
    for v in shown:
        marker = "  <-- default pick" if v.id == chosen else ""
        sex = "female" if is_female(v) else "other"
        print(f"  {v.name!r}  ({sex}){marker}")
    print('\nUse one with:  --voice-name "Samantha"')
    if not show_all:
        print("Pass --list-voices --all to see every installed voice.")


class Voice:
    """
    Speech output, with rate and volume tuned for a noisy site.

    IMPORTANT -- why a new engine per utterance:

    Reusing one pyttsx3 engine across a long-running loop is the obvious
    implementation, and it is silently broken on macOS (confirmed here on
    pyttsx3 2.99). The FIRST runAndWait() speaks normally; every later one
    returns in about ten milliseconds without producing any sound, because the
    NSSpeechSynthesizer driver's run loop is stopped after the first call and
    never restarted. Nothing raises, and the script goes on printing every
    event, so it looks like it is working while the worker hears silence --
    the worst possible failure mode for a safety device.

    Building a fresh engine for each utterance sidesteps it and is verifiably
    audible on repeat calls. The cost is roughly a third of a second of setup
    per line, which is irrelevant for a coach that speaks a few times an hour.

    (The same bug affects any script built on the one-engine pattern, including
    the original speak_biomarker_labels.py.)
    """

    def __init__(self, rate: int, volume: float, voice_name: Optional[str], prefer_female: bool):
        # Slower than the default 200: instructions competing with machinery
        # need to survive being half-heard.
        self.rate = rate
        self.volume = volume

        probe = pyttsx3.init()
        self.voice_id = pick_voice(probe, voice_name, prefer_female)
        try:
            chosen = self.voice_id
            for v in probe.getProperty("voices"):
                if v.id == chosen:
                    print(f"[info] voice: {v.name}")
                    break
        except Exception:
            pass
        probe.stop()
        del probe

    def _build_engine(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        if self.voice_id:
            try:
                engine.setProperty("voice", self.voice_id)
            except Exception as exc:
                print(f"[warning] could not set voice: {exc}", file=sys.stderr)
        return engine

    def say(self, text: str) -> None:
        engine = self._build_engine()
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            # A coach that crashes mid-shift is worse than one that misses a
            # line, so a failed utterance is reported and the loop continues.
            print(f"[warning] speech failed: {exc}", file=sys.stderr)
        finally:
            try:
                engine.stop()
            except Exception:
                pass


# --------------------------------------------------------------------------
# Reading the live stream
# --------------------------------------------------------------------------


def discover_endpoint(explicit: Optional[str]) -> Optional[str]:
    """Return a reachable /latest URL, probing the usual ports if none given."""
    if explicit:
        return explicit
    for url in CANDIDATE_ENDPOINTS:
        try:
            requests.get(url, timeout=2).raise_for_status()
            print(f"[info] found the live poller at {url}")
            return url
        except requests.RequestException:
            continue
    print(
        "[warning] no poller found on "
        + " or ".join(CANDIDATE_ENDPOINTS)
        + "\n          Start it with:  python avro_stream_poller_with_ml.py"
        + "\n          Or rehearse offline with:  python worksite_voice_coach.py --demo",
        file=sys.stderr,
    )
    return CANDIDATE_ENDPOINTS[0]


_fetch_failures = {"count": 0}


def fetch_latest(endpoint: str) -> Optional[dict]:
    """
    Fetch /latest, reporting outages without flooding the console.

    A dropped connection is normal on a site (the poller restarts, wifi dips).
    We print the first failure and then stay quiet until it either recovers or
    has failed for a while, so a long outage does not bury the spoken-event log
    under thousands of identical lines.
    """
    try:
        resp = requests.get(endpoint, timeout=5)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        _fetch_failures["count"] += 1
        n = _fetch_failures["count"]
        if n == 1 or n % 12 == 0:
            print(f"[warning] could not reach {endpoint} ({n} in a row): {exc}", file=sys.stderr)
        return None
    except ValueError:
        print("[warning] response was not valid JSON", file=sys.stderr)
        return None

    if _fetch_failures["count"]:
        print(f"[info] reconnected to {endpoint}")
        _fetch_failures["count"] = 0
    return payload


@dataclass
class Reading:
    """One normalized snapshot of the worker, built from the /latest payload."""

    frustration: Optional[str] = None
    mental_demand: Optional[str] = None
    frustration_confidence: Optional[float] = None
    mental_demand_confidence: Optional[float] = None
    eda: Optional[float] = None
    temp: Optional[float] = None
    pulse_rate: Optional[float] = None

    @property
    def strain(self) -> Optional[str]:
        """
        Worst-of the two model labels.

        Deliberately pessimistic: on a site, a worker who is cognitively
        overloaded OR frustrated is a worker at elevated risk around moving
        equipment. We take the higher of the two rather than averaging, because
        averaging hides exactly the case we care about.
        """
        ranks = [
            LABEL_RANK[l.lower()]
            for l in (self.frustration, self.mental_demand)
            if isinstance(l, str) and l.lower() in LABEL_RANK
        ]
        if not ranks:
            return None
        return ("low", "medium", "high")[max(ranks)]


def parse_reading(payload: dict) -> Reading:
    """
    Normalize the /latest payload into a Reading.

    Shape confirmed against biomarker_runtime.LiveBiomarkerPredictor.update():
    payload["predictions"]["frustration"] / ["mental_demand"] each carry
    label / confidence / ready / reason. A block that is not `ready` yet has no
    usable label, so we leave it as None rather than inventing "Pending" --
    the coach has nothing to advise on until a real label exists.
    """
    reading = Reading(
        eda=payload.get("eda"),
        temp=payload.get("temp"),
        pulse_rate=payload.get("pulse_rate"),
    )

    predictions = payload.get("predictions")
    if not isinstance(predictions, dict):
        return reading

    fr = predictions.get("frustration") or {}
    if fr.get("ready"):
        reading.frustration = fr.get("label")
        reading.frustration_confidence = fr.get("confidence")

    md = predictions.get("mental_demand") or {}
    if md.get("ready"):
        reading.mental_demand = md.get("label")
        reading.mental_demand_confidence = md.get("confidence")

    return reading


# --------------------------------------------------------------------------
# The coach
# --------------------------------------------------------------------------


@dataclass
class CoachState:
    high_since: Optional[float] = None   # simulated seconds, start of strain run
    medium_since: Optional[float] = None  # when the run dropped to medium
    next_tier: int = 0
    last_spoke_at: float = -1e9
    last_said: Dict[str, float] = field(default_factory=dict)
    announced_medium: bool = False


class WorksiteVoiceCoach:
    def __init__(self, voice: Voice, clock: Union[Clock, StepClock], log_path: Optional[str]) -> None:
        self.voice = voice
        self.clock = clock
        self.state = CoachState()
        self.log_path = log_path
        self._log_writer = None
        self._log_file = None
        if log_path:
            self._open_log(log_path)

    # -- logging ----------------------------------------------------------

    def _open_log(self, path: str) -> None:
        is_new = not os.path.exists(path)
        self._log_file = open(path, "a", newline="", encoding="utf-8")
        self._log_writer = csv.writer(self._log_file)
        if is_new:
            self._log_writer.writerow(
                [
                    "wall_time", "sim_seconds", "event", "sustained",
                    "frustration", "mental_demand", "eda", "temp", "pulse_rate",
                    "message",
                ]
            )
            self._log_file.flush()

    def _log(self, event: str, reading: Reading, message: str, sustained: Optional[float]) -> None:
        if not self._log_writer:
            return
        self._log_writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                round(self.clock.now(), 1),
                event,
                format_duration(sustained) if sustained is not None else "",
                reading.frustration,
                reading.mental_demand,
                reading.eda,
                reading.temp,
                reading.pulse_rate,
                message,
            ]
        )
        self._log_file.flush()

    def close(self) -> None:
        if self._log_file:
            self._log_file.close()

    # -- speaking ---------------------------------------------------------

    def _speak(
        self,
        key: str,
        message: str,
        reading: Reading,
        sustained: Optional[float] = None,
        cooldown: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """
        Speak, unless doing so would make the coach a nag.

        Two gates: a global quiet gap so two triggers firing in the same poll do
        not talk over each other, and a per-message cooldown so the same advice
        is not repeated every five seconds. `force` bypasses the global gap for
        escalation rungs -- a safety stop waits for nothing.
        """
        now = self.clock.now()

        if not force and now - self.state.last_spoke_at < MIN_GAP_BETWEEN_UTTERANCES:
            return False

        if cooldown is not None:
            last = self.state.last_said.get(key)
            if last is not None and now - last < cooldown:
                return False

        stamp = time.strftime("%H:%M:%S")
        held = f" [sustained {format_duration(sustained)}]" if sustained is not None else ""
        print(f"[{stamp}] ({key}){held} {message}")

        self.voice.say(message)
        self.state.last_spoke_at = self.clock.now()
        self.state.last_said[key] = self.state.last_spoke_at
        self._log(key, reading, message, sustained)
        return True

    # -- the decision logic ----------------------------------------------

    def handle(self, reading: Reading) -> None:
        strain = reading.strain
        if strain is None:
            # No ready model output yet -- stay quiet rather than guess.
            return

        if strain == "high":
            self._handle_high(reading)
        elif strain == "medium":
            self._handle_medium(reading)
        else:
            self._handle_low(reading)

        self._physiological_advisories(reading)

    def _handle_high(self, reading: Reading) -> None:
        now = self.clock.now()
        st = self.state

        if st.high_since is None:
            # Start of a new strain run. Round one fires immediately.
            st.high_since = now
            st.next_tier = 0
            st.announced_medium = False

        st.medium_since = None  # back to high, so the grace window closes
        sustained = now - st.high_since

        # Walk the ladder. A worker whose stream stalls and resumes can cross
        # more than one rung at once; we only speak the highest reached so the
        # coach does not deliver a monologue.
        rung_to_speak: Optional[Tier] = None
        while st.next_tier < len(TIERS) and sustained >= TIERS[st.next_tier].after_seconds:
            rung_to_speak = TIERS[st.next_tier]
            st.next_tier += 1

        if rung_to_speak is not None:
            self._speak(
                rung_to_speak.name,
                rung_to_speak.message,
                reading,
                sustained=sustained,
                force=True,
            )

    def _handle_medium(self, reading: Reading) -> None:
        now = self.clock.now()
        st = self.state

        if st.high_since is not None:
            # Mid-run dip. Hold the sustained-strain clock through a grace
            # window -- fatigue does not reset because one minute looked better.
            if st.medium_since is None:
                st.medium_since = now
            elif now - st.medium_since >= MEDIUM_GRACE_SECONDS:
                self._end_run(reading, now, recovered=True)
            return

        if not st.announced_medium:
            st.announced_medium = True
            self._speak(
                "medium-entry",
                MEDIUM_ENTRY_MESSAGE,
                reading,
                cooldown=ADVISORY_COOLDOWN,
            )

    def _handle_low(self, reading: Reading) -> None:
        now = self.clock.now()
        st = self.state
        st.announced_medium = False
        if st.high_since is not None:
            self._end_run(reading, now, recovered=True)

    def _end_run(self, reading: Reading, now: float, recovered: bool) -> None:
        """Close out a strain run, and say so if it was long enough to matter."""
        st = self.state
        sustained = now - (st.high_since or now)
        st.high_since = None
        st.medium_since = None
        st.next_tier = 0

        if recovered and sustained >= RECOVERY_MIN_SUSTAINED_SECONDS:
            self._speak("recovered", RECOVERY_MESSAGE, reading, sustained=sustained)

    def _physiological_advisories(self, reading: Reading) -> None:
        """
        Advice driven by the raw sensors rather than the models.

        Worth keeping separate: the trained classifiers use EDA + temperature +
        minute index only, so pulse rate carries information the models never
        see. Heat and heart rate are also the two things that put a construction
        worker in an ambulance, and they deserve a voice even on a "low" label.
        """
        temp = reading.temp
        if isinstance(temp, (int, float)) and temp >= SKIN_TEMP_HOT_C:
            self._speak("advisory-heat", ADVISORY_HEAT, reading, cooldown=ADVISORY_COOLDOWN)

        pulse = reading.pulse_rate
        if isinstance(pulse, (int, float)) and pulse >= PULSE_ELEVATED_BPM:
            self._speak("advisory-pulse", ADVISORY_PULSE, reading, cooldown=ADVISORY_COOLDOWN)

        eda = reading.eda
        if isinstance(eda, (int, float)) and eda >= EDA_SPIKE_USIEMENS:
            self._speak("advisory-eda", ADVISORY_EDA, reading, cooldown=ADVISORY_COOLDOWN)

        # Head overloaded but body calm: a different failure mode from fatigue,
        # and it wants a different instruction (slow down, re-check) rather than
        # "drink water".
        md = (reading.mental_demand or "").lower()
        fr = (reading.frustration or "").lower()
        if md == "high" and fr == "low":
            self._speak(
                "advisory-cognitive",
                ADVISORY_COGNITIVE,
                reading,
                cooldown=ADVISORY_COOLDOWN,
            )


# --------------------------------------------------------------------------
# Demo mode -- a scripted shift, so the presentation never depends on AWS
# --------------------------------------------------------------------------


def demo_readings() -> Callable[[], Optional[dict]]:
    """
    A scripted ninety-minute stretch of a shift, replayed one reading per poll.

    Each step is five simulated minutes (--demo-step-minutes), so the whole
    escalation ladder plays out in about a minute and a half of real time.
    Sequence: an easy start, a climb into sustained high strain that walks all
    five rungs, a dip at 15 minutes that does NOT reset the sustained-strain
    clock, heat / pulse / EDA advisories along the way, and a real recovery once
    the break is taken.
    """
    script: List[Tuple[str, str, float, float, float]] = [
        # (frustration, mental_demand, eda, temp_c, pulse)   step -> sustained
        ("low", "low", 2.1, 33.0, 78),          # 00m  easy start
        ("low", "low", 2.4, 33.2, 81),          # 05m
        ("low", "medium", 3.6, 33.6, 88),       # 10m  -> "strain is climbing"
        ("medium", "high", 5.2, 34.1, 96),      # 15m  -> ROUND 1 (hydrate)
        ("high", "high", 6.0, 34.4, 101),       # 20m  held 05m
        ("high", "high", 5.8, 34.6, 104),       # 25m  held 10m -> ROUND 2
        ("medium", "medium", 8.6, 34.5, 99),    # 30m  dip; clock holds; EDA spike
        ("high", "high", 7.0, 34.8, 112),       # 35m  back to high; pulse advisory
        ("high", "high", 7.2, 35.2, 114),       # 40m  held 25m -> ROUND 3
        ("high", "high", 7.5, 35.7, 116),       # 45m  heat advisory
        ("high", "high", 7.3, 35.8, 118),       # 50m
        ("high", "high", 7.0, 35.3, 113),       # 55m
        ("high", "high", 6.9, 35.0, 110),       # 60m  held 45m -> ROUND 4 (break)
        ("high", "high", 6.6, 34.8, 108),       # 65m  break not taken
        ("high", "high", 6.3, 34.6, 106),       # 70m
        ("high", "high", 6.0, 34.4, 104),       # 75m  held 60m -> ROUND 5 (supervisor)
        ("medium", "medium", 4.4, 34.0, 97),    # 80m  work stopped, coming down
        ("low", "low", 2.8, 33.4, 85),          # 85m  -> recovery
        ("low", "low", 2.3, 33.1, 80),          # 90m  back to work
    ]

    step = {"i": 0}

    def next_payload() -> Optional[dict]:
        i = step["i"]
        if i >= len(script):
            return None
        fr, md, eda, temp, pulse = script[i]
        step["i"] = i + 1
        return {
            "eda": eda,
            "temp": temp,
            "pulse_rate": pulse,
            "note": "demo",
            "predictions": {
                "frustration": {"label": fr, "confidence": 0.8, "ready": True, "reason": None},
                "mental_demand": {"label": md, "confidence": 0.7, "ready": True, "reason": None},
            },
        }

    return next_payload


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    clock = StepClock(args.demo_step_minutes * 60) if args.demo else Clock(scale=args.time_scale)
    voice = Voice(
        rate=args.rate,
        volume=args.volume,
        voice_name=args.voice_name,
        prefer_female=not args.any_voice,
    )
    coach = WorksiteVoiceCoach(voice, clock, args.log_file)

    if args.demo:
        source = demo_readings()
        print(
            f"Worksite Voice Coach -- DEMO shift: one scripted reading every "
            f"{args.interval}s, each worth {args.demo_step_minutes:g} minutes on the "
            f"job. Ctrl+C to stop.\n"
        )
    else:
        endpoint = discover_endpoint(args.endpoint)
        source = lambda: fetch_latest(endpoint)  # noqa: E731
        print(
            f"Worksite Voice Coach -- polling {endpoint} every {args.interval}s, "
            f"clock running at {args.time_scale:g}x. Ctrl+C to stop.\n"
        )

    if args.announce_start:
        voice.say("Worksite voice coach is online. I will check on you as you work.")

    try:
        while True:
            payload = source()
            if payload is None:
                if args.demo:
                    print("\n[demo] scripted shift complete.")
                    break
            else:
                if args.debug:
                    print(f"[debug] {payload}")
                reading = parse_reading(payload)
                if args.verbose and reading.strain:
                    held = ""
                    if coach.state.high_since is not None:
                        held = f"  held {format_duration(clock.now() - coach.state.high_since)}"
                    print(
                        f"    strain={reading.strain:<6} "
                        f"frustration={reading.frustration} "
                        f"mental_demand={reading.mental_demand} "
                        f"pulse={reading.pulse_rate} temp={reading.temp}{held}"
                    )
                coach.handle(reading)
                clock.advance()  # demo only; the wall clock ignores this

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        coach.close()
        if args.log_file:
            print(f"Spoken-event log written to {args.log_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construction-worker voice safety coach driven by live biomarker "
            "predictions: escalating spoken guidance based on how long strain "
            "has been sustained, not just the current label."
        )
    )
    parser.add_argument("--endpoint", default=None,
                        help="Poller /latest URL (default: auto-detect port 7000 then 7001)")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL_SECONDS,
                        help=f"Real seconds between polls (default: {POLL_INTERVAL_SECONDS})")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Speed up the escalation clock against the LIVE stream, e.g. 60 "
                             "makes the 45-minute rung fire after 45 seconds (default: 1). "
                             "Ignored in --demo mode, which steps its own clock")
    parser.add_argument("--demo", action="store_true",
                        help="Replay a scripted shift instead of polling; needs no network")
    parser.add_argument("--demo-step-minutes", type=float, default=5.0,
                        help="Job-site minutes each demo reading represents (default: 5)")
    parser.add_argument("--voice-name", default=None,
                        help='Force a voice, e.g. --voice-name "Samantha"')
    parser.add_argument("--any-voice", action="store_true",
                        help="Do not prefer a female voice; use the system default")
    parser.add_argument("--list-voices", action="store_true",
                        help="List the English voices installed, and exit")
    parser.add_argument("--all", action="store_true",
                        help="With --list-voices, show every voice, not just English ones")
    parser.add_argument("--rate", type=int, default=165,
                        help="Speech rate, words per minute (default: 165, slower than "
                             "default so instructions survive site noise)")
    parser.add_argument("--volume", type=float, default=1.0, help="Volume 0.0-1.0 (default: 1.0)")
    parser.add_argument("--log-file", default="voice_coach_log.csv",
                        help="CSV log of everything spoken (default: voice_coach_log.csv; "
                             'pass "" to disable)')
    parser.add_argument("--announce-start", action="store_true",
                        help="Say a line on startup to confirm audio works before the demo")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each reading even when the coach stays silent")
    parser.add_argument("--debug", action="store_true", help="Print the raw payload each poll")
    args = parser.parse_args()

    if args.list_voices:
        list_voices(show_all=args.all)
        return

    run(args)


if __name__ == "__main__":
    main()
