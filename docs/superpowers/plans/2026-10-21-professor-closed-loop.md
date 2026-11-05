# Professor Closed-Loop Implementation Plan

> Use superpowers:subagent-driven-development for independent component tasks and review. The user authorized implementation directly; continue without repeated approval prompts.

**Goal:** Implement the applicable pathway in `work` for the Polar fatigue/posture system.
**Architecture:** Config-driven Python processing and decisions, SQLite/API, browser Bluetooth bridge, held-out-person model promotion.
**Tech Stack:** Python stdlib/SQLite, existing sklearn, JavaScript/esbuild/Playwright.
**Spec:** ../specs/2026-10-21-professor-closed-loop.md

## Global Constraints

- Preserve existing dirty working tree; no staging, commits, worktree moves or publication.
- Group 1 active; never invent stress/frustration thresholds or missing spreadsheet data.
- Missing measurements never mean normal. All thresholds/messages live in config JSON.
- Rules only live until real-label held-out-person blend beats both rules and ML.
- Local participant data is gitignored and excluded from static serving.
- Interfaces and schemas are defined in the spec; coordinate any changes before edits.

## Review Focus

Missing evidence, timestamp/duplicate/retry behavior, session isolation, score semantics,
training leakage/promotion gate, persistent/private storage, browser offline and reconnect.

### Task 1: Config and pipeline

Owner: pipeline agent. Files: config/rules.json, config/feedback_table.json, pipeline/*.py,
tests/test_pipeline.py. Do not edit other agents' files.
Produces RuleEngine, DecisionEngine, FeedbackPipeline interfaces from spec.
- [x] Write failing tests for thresholds, missing evidence, baseline, timing and session isolation.
- [x] Implement configs, ingest/features/rule/model/decision/pipeline; run tests.
- [x] Review component behavior and report results.

### Task 2: Storage and API

Owner: service agent. Files: storage/*.py, serving/server.py, serving/dashboard.html,
serve_dashboard.py, .gitignore, tests/test_feedback_service.py.
Consumes FeedbackPipeline.process(sample) and returns reading_id with its result.
- [x] Write failing database and HTTP tests with temporary stores.
- [x] Implement durable logs, idempotence, responses/outcomes/context/exports, API and integration.
- [x] Ensure runtime files cannot be served; test restart durability and session references.

### Task 3: Training, importer and demo

Owner: training agent. Files: training/*.py, tools/demo_stream.py,
config/feedback_template.csv, tests/test_closed_loop_training.py.
Consumes config/RuleEngine and prior train_polar_fatigue helpers; produces deployment metadata.
- [x] Write failing tests for held-out-person comparison, promotion/no-overwrite and import validation.
- [x] Implement training gate, normalized spreadsheet importer, deterministic scripted demo.
- [x] Verify commands and report synthetic versus empirical evidence explicitly.

### Task 4: Browser bridge and documentation

Owner: root. Files: sensor/*.js integration, tests/browser/polar.spec.js and Node tests,
README.md, requirements.txt, docs/PROFESSOR_WORKFLOW.md and relevant existing guides.
- [x] Load shared configs and gate synthetic model to demo.
- [x] Add bounded sample transport, local logging status, decisions and feedback/outcome controls.
- [x] Preserve live baseline/reconnect/demo; prevent duplicate audio; test real bundle.
- [x] Write alignment table and commands, with missing workbook/hardware/labels listed honestly.

### Task 5: Review and verify

- [x] Independent component/final review, resolve material findings.
- [x] Run full Python/Node/browser tests and demo; inspect UI; rebuild bundle.
- [x] Report completed requirements and evidence-dependent remaining work.

## Verification result (2026-10-22)

- Node: 54 passed. Python: 87 run, 85 passed and 2 optional openpyxl tests skipped.
- Playwright: all 11 passed, including actual API readings/alert/response/outcome/export
  and visible offline storage errors. Browser databases are temporary and isolated.
- CLI demo: 221 synthetic readings, 3 alerts, ending fatigue none / posture good.
- Inspected desktop feedback and narrow-screen screenshots; rebuilt shipped bundle.
- Independent final review's 3 material findings fixed and reviewed again: retained
  pending deliveries, consistent recorded feature policy, latency-safe rating spacing.
- No original workbook, physical-device test, or real-label performance result available.
  These evidence-dependent steps are documented, not represented as completed.
