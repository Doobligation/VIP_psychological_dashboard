# Polar fatigue model

The user wants a working fatigue model now, connected to live Polar H10 readings,
with troubleshooting and data collection when they wear the monitor. They have
authorized implementation in this conversation. Existing Polar work is uncommitted;
extend it in place without staging, reverting, or moving those changes.

Use standardized multinomial logistic regression on HR, RMSSD, and SDNN percentage
changes from the participant's resting baseline. Class order: none, low, medium,
high, very-high. An explicitly synthetic starter is fitted to the assignment's
threshold labels; it approximates those rules and is not worker-trained or validated.
Public construction-worker studies motivate collecting HRV and independent ratings,
but their published results are not this model's performance. No usable public
labeled training file was retrieved in this session.

Keep the threshold result and its sustained feedback. Add a separate ML estimate
card with provenance and data-quality status. No automatic interventions or confidence
percentages from the starter model. Missing model, missing baseline, stale or incomplete
windows produce unavailable estimates. Disconnection and baseline recapture clear them.

Add a manual current-fatigue selector with no default answer, a task description,
and a record button. Require participant ID, live connection, eligible measurements,
and at least 60 seconds between accepted ratings. Store the exact snapshot, timestamp,
and independently chosen rating in session JSON. Never treat demos or rule predictions
as participant labels. Keep existing session fields compatible.

A Python CLI creates the starter artifact or trains from one or more live session
exports. Recompute features from validated snapshots; reject synthetic sessions,
invalid labels, duplicate sessions, and insufficient data. Use fixed regularization
and a scaler fitted only to training folds. Evaluate by held-out participant where
possible; report why evaluation is unavailable otherwise. Single-participant training
is explicitly personal and unvalidated. No inference artifacts use pickle.

Export a versioned JSON artifact containing feature order, scaler, coefficients,
classes, provenance, training support, and evaluation. Browser can fetch the default
artifact or load a local artifact. Validate schema and numeric dimensions before use;
failed import retains the previous usable model. Record artifact identity with each
ML prediction for later troubleshooting.

Verify feature parity, numerical inference, stale/invalid inputs, independent labels,
group isolation, real CLI-to-JS artifact compatibility, and browser collection/export.
