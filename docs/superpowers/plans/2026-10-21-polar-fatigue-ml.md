# Polar Fatigue ML Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Ship an explicitly synthetic starter fatigue classifier with live inference and independent label collection for retraining.

**Architecture:** Python fits standardized multinomial logistic regression and exports JSON. A separate browser module validates and evaluates artifacts against eligible FatigueSession snapshots. Study controls record independent ratings in existing session exports.

**Tech Stack:** Existing Python/scikit-learn, browser JavaScript, Node tests and Playwright.

**Spec:** ../specs/2026-10-21-polar-fatigue-ml.md

## Global Constraints

- Work in place; preserve pre-existing uncommitted Polar implementation.
- Starter training data is synthetic; never claim worker validation or calibrated confidence.
- Features are baseline-relative HR, RMSSD and SDNN percentages, in that order.
- Window is 60 seconds with at least 55 seconds coverage, 30 RR intervals, and fresh HR/RR.
- No model-based speech or interventions. Keep existing threshold feedback.
- Real labels must be manually supplied during live sessions with participant ID.

## Review Focus

- Malformed or stale snapshots must not become numeric estimates or training records.
- Failed model import must not replace a working model.
- Demo exports and rule-derived labels must never enter real training.
- Participants and overlapping samples must not leak between training and evaluation.
- Python feature normalization and browser inference must agree for the exported artifact.

### Task 1: Train and serialize

Files: `train_polar_fatigue.py`, `tests/test_polar_fatigue_model.py`, `models/polar_fatigue_model.json`, `requirements-polar-ml.txt`.
Produces: JSON schema v1; fixed three features and five ordered classes; training/evaluation provenance.

- [x] Write tests for snapshot validation, demo rejection, duplicate sessions, sparse labels, participant-isolated evaluation, and serialized prediction parity.
- [x] Run tests to establish failures, implement trainer, rerun Python tests.
- [x] Generate reproducible starter artifact with synthetic provenance and no physiological accuracy claim.

### Task 2: Browser inference and collection

Files: `sensor/fatigue-model.js`, `sensor/study-panel.js`, `tests/fatigue-model.test.js`, `tests/browser/polar.spec.js`, built bundle.
Consumes: Task 1 JSON artifact and FatigueSession snapshots.
Produces: unavailable/estimated ML result, independently recorded rating snapshots, schema v2 session exports.

- [x] Write failing Node tests for shape validation, hand-computed logits, stale/invalid windows, and rating admission.
- [x] Implement model inference, model load/import and independent live ratings; preserve rule feedback.
- [x] Add browser tests exercising real bundle, live rating/export, demo rejection and model import errors.
- [x] Build and run Node, Python and browser suites.

### Task 3: Handoff and review

Files: `docs/POLAR_FATIGUE_ML.md`, `README.md`.

- [x] Document model provenance, source limitations, data collection, retraining and evaluation commands.
- [x] Review final changes independently, resolve material issues and verify affected checks.
- [x] Report shipped behavior and unperformed hardware/worker validation.

## Verification and review record

- `npm run build`: passed; committed bundle target rebuilt in the working tree.
- `npm test`: 47 passed.
- `venv/bin/python -m unittest discover -s tests -p 'test_*.py'`: 24 passed,
  one existing optional XLSX test skipped because openpyxl is not installed.
- Playwright with installed Chromium 1248: all 9 tests passed. The default
  Playwright browser build was absent; no browser download was necessary.
- Python CLI successfully trained/exported a model from temporary test-session
  fixtures. This was a software smoke test, not worker validation.
- Independent read-only review identified three issues. Regression tests reproduced
  each before fixes: browser-history restoration permanently disabling model imports;
  invalid local import canceling a pending default load; alphabetical sklearn tie
  breaking disagreeing with browser severity order. All are fixed and covered.
- Desktop and narrow-screen screenshots inspected. `git diff --check` passed.
- No physical H10 testing or worker accuracy validation was performed. Published
  research informs the intended study, not the shipped starter's fitted coefficients.
- Existing changes remain in place; no commits, branch changes or deployment.
