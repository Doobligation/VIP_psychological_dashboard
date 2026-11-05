# Professor's wearable feedback pathway

Authority: the user supplied `work` (called work.txt in chat) and explicitly asked
to perform it if aligned with the project. It aligns with the existing Polar work.
User-confirmed scope is Group 1 fatigue and posture; stress/frustration remain explicitly
unconfigured until their team's thresholds and sensor source are supplied. Shared infrastructure remains reusable.

The professor's instruction supersedes the prior live synthetic starter: live operation
uses rules until independent labels support a model/blend that wins on held-out people.
Synthetic models may remain available for demos only. Rule scores are heuristic severity
scores, never calibrated physiological probabilities. Cause is the observed cardiovascular
strain or chest tilt pattern, not inferred heat, sleep loss, or medical diagnosis.

## Interfaces and data

Root directories follow work: config, pipeline, storage, serving, training, tools.
Keep the existing dashboard, server and browser Bluetooth driver. Preserve the dirty
working tree; no commits or branch changes. Runtime data goes in gitignored `.local/`
and must not be served by the static HTTP handler. Bind services to loopback.

`config/rules.json`: schemaVersion=1, version, featurePolicy (windowSeconds=60,
freshSeconds=5, minRr=30, rrMinMs=300, rrMaxMs=2000, baselineSamples=30),
probabilityBands [{level,severity,min,score}], states {fatigue,posture,stress,frustration}.
Bands: none(0,0,0), low(1,.1,.2), medium(2,.35,.45), high(3,.6,.7), very-high(4,.85,.9).
Each state has enabled, defaultLevel, requiredSignals, minEvidence, rules. Each rule
has cause, level, severity, score, match ('all'/'any'), minEvidence, conditions
[{signal,operator,value}], explanation. Signals for fatigue: hrChangePct, rmssdChangePct,
sdnnChangePct; posture: absPitch, absRoll. Fatigue thresholds retain existing strict
5/-15/-15, 10/-25/-20, 15/-35/-30, 25/-40/-35 percentages. Posture OR thresholds retain
20/10 poor, 45/20 critical. Missing signals are skipped as evidence, never zero-filled;
minimum evidence prevents partial data from becoming a normal result.

`config/feedback_table.json`: schemaVersion=1, version, decision settings, states.
states[state][level] has cause, enabled, message, action, goal, seconds. Level 1/low
has feedback disabled. Fatigue persistence 30/60/120; posture 30/60. Provisional
decision defaults: freshSeconds=5, cooldownSeconds=120, warningWindowSeconds=600,
maxWarnings=3; rapidRise {levels:2, withinSeconds:30, minPersistenceSeconds:15}.
Rapid rise may shorten persistence only with fresh evidence; never bypass cooldown
or warning cap. Timers advance on observations, not GET requests; no alerts on
missing data, duplicates or reversed timestamps. Recovery resets streaks; episode
cooldown/history persists. Version and score semantics are logged. All defaults
not supplied by the PDFs are marked implementation choices.

Samples: {session_id,participant,mode:'live'|'demo',sample_id,timestamp_ms,snapshot?,raw?}.
Browser sends each HR/RR notification it uses, with raw packet and current snapshot.
Demo sends each simulated summary window. Raw file ingestion may instead supply
hr/rr_intervals_ms/contact fields; FeaturePipeline computes a baseline and HRV. Native
browser snapshot fields match FatigueSession, including freshness and baseline status.
Derived features are absent on stale/ineligible snapshots. Optional EDA/SCR can be
computed when measured, but no stress threshold is invented. All sample IDs and
session IDs are bounded nonempty strings; malformed numeric values are rejected.

`RuleEngine(rules).evaluate(features,state='fatigue')` ->
{state,available,level,severity,cause,score,score_kind:'rule-score',evidence,reason}.
`DecisionEngine(feedback).update(result,timestamp_ms)` returns the result plus
alert (id/state/cause/level/message/action/goal/time) or null, suppression and timing.
`FeedbackPipeline(rules=None,feedback=None,model_path=None).process(sample)` ->
{session_id,sample_id,timestamp_ms,mode,features,states:{fatigue:decision,posture:decision},alerts:[]}.
State is isolated per session. Expose reset(session_id). Imported model errors fall
back to rules with a visible reason; only approved real-data blends can affect live
decisions. Use no pickle. Model data follows prior portable logistic JSON with a
deployment object {approved,ruleConfigVersion,blendWeight,metrics}. BlendWeight=.5
is fixed before evaluation. Score is the class-probability-weighted rule score, then
blended with rule score, then binned using probabilityBands.

## Persistent closed loop and endpoints

SQLite stores readings (raw/features/decisions), alerts, responses, outcomes and session
context. Persist all ingested analysis readings, including insufficient-data readings.
Deduplicate retries by session_id+sample_id before advancing the pipeline. Store alert
IDs and reference them from responses/outcomes; never accept cross-session references.
Each rating links an eligible fresh reading and the manually selected level. Demo
labels cannot train a live model. Labels are not automatically derived from alerts.

`ClosedLoopService(db_path,model_path=None)` owns pipeline/store. API mixin
`FeedbackAPIMixin.handle_feedback_api()` returns True if it handled the request;
it can be used by the existing server without changing historical API routes.
Expose POST /samples; GET /latest and /decision; GET/POST /context;
POST /feedback; GET /summary; GET /export. Session-scoped GET uses ?session_id=.
POST /samples accepts one sample and returns pipeline result plus reading_id.
GET latest/decision returns that same result with stale=true and unavailable states
if no new sample was received for >5s; never advance timers on polling.
POST /feedback: {session_id,reading_id,kind:'rating',level,task?,observed_at_ms?} OR
{session_id,alert_id,kind:'response',response:'acknowledged'|'break-taken'|'not-helpful'|'dismissed'}
OR {session_id,alert_id,kind:'outcome',outcome:'better'|'same'|'worse'}.
POST /context: {session_id,participant,mode,context:{...}}.
GET /summary returns counts for readings, alerts, responses, outcomes, ratings.
GET /export returns schemaVersion=2,sessionId,participant,mode,fatigueRatings
[{time,source:'participant-self-report',level,task,snapshot}], readings,alerts,responses,outcomes.
Runtime snapshots are never default shared artifacts. Reject malformed/oversize
JSON and cross-origin mutations; return concise JSON errors. Reading/alert IDs are
opaque strings. Origin policy permits absent Origin for local CLI and same-host HTTP.

## Training and configuration import

training/train_model.py consumes schema2 exports from browser or database. Reuse the
prior feature validation and standardized logistic fitting. For every held-out person,
fit preprocessing/model only on training people. Compare rule/ML/blend macro-F1 and
accuracy with fixed blendWeight=.5. Require at least three participants and all
included classes in each training fold. Save a deployable artifact only when blend
macro-F1 is strictly greater than BOTH rules and ML. Otherwise write a report only
and preserve any existing deployed model. Do not auto-deploy, fabricate labels or
report synthetic-data physiological accuracy. Keep the old bootstrap CLI explicitly
demo-only. Report unseen-level and dataset coverage limits.

training/build_feedback_table.py imports a documented normalized CSV/XLSX sheet
with state,level,cause,message,action,goal,seconds,enabled columns. Validate values and
write atomically; use the existing feedback config's decision defaults. Supply a CSV
template. The original DBR workbook is absent, so verify its columns/values when it
becomes available rather than guessing them. tools/demo_stream.py scripts rest ->
fatigue/posture -> recovery with synthetic provenance, optionally posts to the API.

## Browser integration

Fetch rules and feedback JSON on page load; edits take effect on refresh/server restart
without source changes. Browser fallback calculations use the same configuration.
Live model card must clearly say rule-only unless a validated deployment exists;
synthetic starter predictions are allowed only in demo. Add local-storage status,
server decision card, response/outcome buttons and persistent export link. Local server
feedback is the sole live audio source when connected, preventing duplicate speech
from browser timers; demo stays clearly labeled. Network failures are visible; bound
the retry queue and count/report dropped readings, never claim unsaved data is stored.
Existing manual ratings also post to /feedback using the latest eligible reading ID.

## Verification

Test threshold boundaries/missing evidence, config-driven changes, baseline/HRV,
timers/cooldowns/rapid-rise/cap, duplicate/reversed timestamps, session isolation,
SQLite durability and foreign references, HTTP routes/privacy/errors, rules fallback,
training gate and participant isolation, synthetic demo, real-bundle browser wiring.
Hardware use and actual unseen-worker superiority remain pending real observations.

Implementation notes (2026-10-22): Native snapshots record featurePolicy. Legacy
exports without metadata use the original 60s/5s/30-RR policy. Training rejects
mixed protocols and runtime model loading checks policy compatibility. Rating times
use validated observed_at_ms (default reading time), preserving spacing despite
network delay. The browser keeps pending deliveries across session changes and
filters old decisions out of the selected session.
