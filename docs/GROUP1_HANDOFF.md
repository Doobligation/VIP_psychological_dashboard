# Group 1: Polar H10, fatigue, and posture handoff

**Current update (2026-10-22):** The user supplied a new `fatigue` file and requested rules only. The active three-level rules, EDA requirement, and removed model-upload workflow are documented in [Current fatigue rules](FATIGUE_RULES.md). Older threshold/model instructions below are historical and do not enable ML in the active fatigue pipeline.

**Update (2026-10-22):** The professor's closed-loop pathway now adds shared JSON
configuration, local persistent readings/alerts/responses/outcomes, cooldowns and
warning limits. Live decisions start with rules; synthetic ML is demo-only. See
[the current workflow](PROFESSOR_WORKFLOW.md) and [ML guide](POLAR_FATIGUE_ML.md).
The source thresholds below are retained; current runtime policy is in `config/`.

## What the assignment asks for

Group 1 (Palash, Dean, Kimi, Andrew) owns **Polar H10 and fatigue prediction** for the upcoming drywall experiment. For each feedback rule, the team must explain:

1. **Threshold:** which measurements or combination of measurements trigger it.
2. **Cause:** what observed condition led to it, and what additional context is needed to identify a cause.
3. **Output:** what the worker sees/hears and what action is recommended.

Group 2 owns platform refinement and the stress/frustration feedback system. Those existing wristband models are separate from this Polar workflow. The assignment prompt refers to the second worksheet, “Feedback System and Decision-Based Rule,” in `Experimental setup.xlsx`. That workbook was not supplied. The PDFs provide usable rules, but the original worksheet and any later revisions still need to be checked. The prompt gives relative deadlines (“tonight” and “Monday”); it does not establish calendar dates.

The Polar prediction document, pp. 1–2 also asks for Unix timestamp conversion from ECG/accelerometer spreadsheets and bending-angle calculation. These are included as a separate recordings converter.

## What existed before, and what this implementation adds

The original Polar handoff, pp. 1–3 describes an EXG RMS display: connect/disconnect, a current value in microvolts, a roughly ten-second trace, and connection recovery. Its 200 ms window is **26 samples at 130 Hz**, not a 200 Hz sample rate. It uses the existing 25 Hz highpass filter. It did not provide HR, HRV, accelerometer posture, fatigue rules, participant association, or session export. EXG RMS is not used as an HRV or fatigue measurement.

The extended Polar panel adds independent HR/RR measurements, RMSSD and SDNN, a resting-baseline workflow, explicit fatigue thresholds, upright-relative pitch/roll, sustained-condition feedback, optional spoken feedback, a hardware-free demo, and a downloadable session JSON. Participant ID belongs to this Polar session; the historical dashboard's worker/scenario/playback controls do not assign the sensor to a worker. The original EXG trace remains available.

This is an **experimental decision-rule implementation**. No fatigue ML model has been trained on real worker labels, and the displayed classification is not a probability, measured prediction accuracy, or medically validated diagnosis. The provided PDF proposes an ML architecture, but supplies neither fatigue training data nor a trained model. The existing stress/frustration models are not repurposed as a fatigue model.

## Fatigue rules to submit

Source: ML feedback document, p. 51. HR is measured in beats/minute; RR, RMSSD, and SDNN are in milliseconds. Each current feature is compared with its own resting baseline:

```text
change_percent = 100 × (current_value − baseline_value) / baseline_value
```

All three conditions in a row must hold. Evaluate the highest matching level first. The strict `>` and `<` operators come from the source; a value exactly on a boundary does not satisfy that condition.

| Result | HR change | RMSSD change | SDNN change | Observed condition / cause of the rule | Feedback output |
| --- | --- | --- | --- | --- | --- |
| No rule matched | No complete row below matches | — | — | Available measurements do not satisfy the combined fatigue rule; this does not establish absence of fatigue | No fatigue intervention |
| Low fatigue | > +5% | < −15% | < −15% | HR elevation together with reduced RR variability relative to the participant baseline | No intervention, as specified in the source |
| Medium fatigue | > +10% | < −25% | < −20% | The same measured cardiovascular pattern crosses the medium thresholds | After 30 seconds sustained: “Fatigue pattern sustained. Take several slow breaths and monitor your effort.” |
| High fatigue | > +15% | < −35% | < −30% | The combined changes cross the high thresholds | After 60 seconds sustained: “High fatigue pattern sustained. Slow your pace and take a short recovery break if safe.” |
| Very high fatigue | > +25% | < −40% | < −35% | The combined changes cross the very-high thresholds | After 120 seconds sustained: “Very high fatigue pattern sustained. Pause the task and take a recovery break when safe.” |
| Unavailable | Missing, invalid, insufficient, or stale input | — | — | There is insufficient usable evidence to classify | Explain the missing requirement; do not issue a fatigue intervention |

The source gives persistence **ranges** of 30–60 seconds, 60–120 seconds, and 2–3 minutes. The implementation selects their lower bounds: 30, 60, and 120 seconds. These choices are implementation defaults, not additional source thresholds. The source also mentions “rapid deterioration” but gives no definition; the closed-loop implementation now provides an explicitly provisional rapid-rise setting in configuration (see the workflow guide). The displayed generic messages adapt the feedback on pp. 37–38 to describe a measured pattern without declaring a proven cause. Persistence/action guidance comes from pp. 40–41. The accompanying action is to reduce pace briefly for medium, reduce workload and take a recovery break for high, and pause the task and recover for very high, with safe task interruption considered.

The app reports the measured HR/HRV pattern. It does **not** determine heat exposure, insufficient sleep, prolonged exertion, or long work duration from HRV alone. The source's cause signatures are qualitative arrows, have no numeric cutoffs, and overlap. In particular, cardiovascular fatigue and long work duration share the same pattern. Specific causal claims require independently recorded task/environment/sleep context and validation. For comparison, the source's cardiovascular messages are “Take several slow breaths and monitor your effort,” “High fatigue detected. Slow your pace and recover,” and “Critical fatigue detected. Stop and recover immediately.” The generic implementation wording above is a deliberate adaptation of those messages.

The PDF's optional weighted score (`0.4 × HRscore + 0.3 × RMSSDscore + 0.3 × SDNNscore`) is not implemented because the component score definitions are absent.

## HRV and data-quality assumptions

The assignment does not specify a baseline protocol, HRV window, artifact policy, or sample requirements. The implementation uses the following declared defaults:

| Item | Implementation default |
| --- | --- |
| Resting baseline | 60 seconds of usable HR/RR data; participant remains resting |
| Current HRV window | Rolling 60 seconds |
| Required window coverage | At least 55 seconds of retained HR and RR data, allowing for the five-second freshness tolerance |
| Minimum RR count | 30 valid intervals |
| RR plausibility check | 300–2000 ms, inclusive |
| Freshness | Five seconds; HR and RR availability are checked independently |
| RMSSD | Square root of the mean squared differences between consecutive valid RR intervals |
| SDNN | Sample standard deviation of RR intervals, using denominator `n − 1` |
| Missing/invalid RR | Do not join adjacent-difference calculations across an invalid interval or a data gap |
| Invalid baseline | Zero/nonfinite baseline features cannot be used for percentage changes |
| Sustained feedback | Requires usable, current sensor observations at the same tier; missing input, gaps, and tier changes restart the dwell period |

These are software safeguards and provisional study settings, not an ECG/HRV artifact correction method validated for drywall work. A short-window estimate can change with motion, breathing, posture, and measurement quality. Review the baseline and window protocol with the study team before interpreting collected results.

Polar's [official H10 SDK documentation](https://github.com/polarofficial/polar-ble-sdk/blob/master/documentation/products/PolarH10.md) states HR in bpm, RR intervals in ms, ECG in microvolts at 130 Hz, and accelerometer axes in milli-g. Live pitch/roll use ratios, so a uniform milli-g scale cancels from the angle formulas. The recordings converter explicitly normalizes acceleration to g; recorded files require a unit selection.

## Posture rules and limits

Source: ML feedback document, pp. 65–66. The Polar-labeled formulas are:

```text
pitch_degrees = atan2(Ax, sqrt(Ay² + Az²)) × 180 / π
roll_degrees  = atan2(Ay, sqrt(Ax² + Az²)) × 180 / π
```

Calibrating while upright records the reference orientation. Classification uses absolute differences from that reference; this calibration and use of absolute angles are explicit implementation choices to accommodate mounting orientation.

| Result | Trigger | Interpretation / output |
| --- | --- | --- |
| Within angle limits | Absolute pitch ≤20° and absolute roll ≤10° | No posture intervention; describes the measured angles only |
| Poor posture | Absolute pitch >20° OR absolute roll >10° | After 30 seconds sustained: “Adjust your posture and maintain a neutral working position.” |
| Critical posture | Absolute pitch >45° OR absolute roll >20° | After 60 seconds sustained: “Correct posture immediately if safe.” |
| Unavailable / uncalibrated | No usable current acceleration or no upright reference | Request data/calibration; do not label posture as good |

The source's table includes the 20°/10° boundaries in the good range, while its preceding prose uses strict `<` for good. This implementation follows the table by triggering poor/critical only above their stated limits. Critical takes precedence over poor. Posture dwell defaults select the lower bounds of the source's 30–60 and 60–120 second ranges on pp. 57–58. The shorter Polar document gives a different pitch formula in an **EmotiBit** subsection; this implementation consistently uses the explicitly Polar-labeled formulas above.

Only chest-sensor pitch and roll are assessed. The source also lists yaw/twisting, overhead work, asymmetric lifting, and static-posture duration, but it does not supply sufficient algorithms to infer those reliably from this implementation. Those classifications are not implemented. The panel is not a validated ergonomic assessment, and “within angle limits” does not establish safe lifting or a safe work task.

## Run a demonstration without a sensor

Run from the repository root with Python 3.10 or newer:

```bash
python3 serve_dashboard.py
```

Open `http://127.0.0.1:8000/psychosocial_dashboard.html` in a desktop browser. The prebuilt `assets/polar-panel.js` is included; npm is not needed just to run the dashboard. Do not open the HTML using a `file://` URL. The separate wristband stream may be unavailable while the Polar panel is working.

1. In the Polar panel, enter a **Participant ID** for the demonstration.
2. Click **Start demo**. The panel labels the session **Demo**, uses a generated 60-second baseline, and runs an accelerated synthetic sequence. No Bluetooth device is used.
3. Observe the changing HR/HRV values, matched fatigue rules, and posture values. Synthetic time is accelerated; this does not demonstrate real-world detection speed or accuracy.
4. Optionally select **Enable spoken feedback**. Speech uses the browser's available speech synthesis; text remains visible when speech is disabled or unavailable.
5. Click **Export session JSON** to retain the demonstration record. Keep its Demo provenance when sharing it.
6. Click **Stop demo** before starting a live session.

## Run a live session

Use desktop Chrome or Edge with Web Bluetooth support, OS/browser Bluetooth permission, and a nearby Polar H10. Prepare and wear the strap according to the sensor setup.

1. Start the server as above and enter the participant's study identifier in the Polar panel.
2. Click **Connect Polar H10** and choose the device. Confirm incoming HR/RR data rather than relying only on the EXG trace.
3. While the participant rests, click **Capture resting baseline (60s)**. Wait for successful completion. Missing RR data or a flat/invalid HRV baseline must be resolved before fatigue classification.
4. With the sensor in its intended position and the participant upright, click **Calibrate upright posture**. Recalibrate if sensor placement changes.
5. Begin the task and monitor measurement availability, percentage changes, classifications, and feedback. Optionally enable spoken feedback.
6. Click **Export session JSON** while the session is available, then disconnect. Browser memory is not durable storage; retain the downloaded file before closing/reloading the page or starting a replacement session.

An accelerometer stream failure should leave the HR/RR workflow usable while posture is unavailable. Fresh HR without usable RR is insufficient for fatigue classification. Reconnecting or switching between Demo and Live requires a new live baseline/reference; synthetic measurements must not become live measurements. Hardware behavior, connection recovery, and representative motion artifacts still require a real-device experiment.

## Convert recorded ECG or accelerometer files

The separate [polar_recordings.py](../polar_recordings.py) tool reads CSV with the Python standard library. XLSX input additionally requires:

```bash
python3 -m pip install -r requirements-polar.txt
```

For the supplied synthetic accelerometer CSV:

```bash
python3 polar_recordings.py examples/polar_acc_demo.csv --kind acc --timestamp-column timestamp --timestamp-unit ms --acc-unit g --output /tmp/polar-angles.csv
```

The input headers are `timestamp,ax,ay,az,note`. The output preserves those columns and appends `timestamp_utc,ax_g,ay_g,az_g,acceleration_magnitude_g,pitch_deg,roll_deg,posture_level`. The synthetic sequence produces Good, Poor, Critical, Poor, Critical labels. The converter's label `Good` means within the supplied tilt limits, not an ergonomic safety finding.

For the supplied synthetic ECG CSV:

```bash
python3 polar_recordings.py examples/polar_ecg_demo.csv --kind ecg --timestamp-column timestamp --timestamp-unit ns --output /tmp/polar-ecg-utc.csv
```

Its headers are `timestamp,ecg_uv,note`; output appends `timestamp_utc` while preserving the ECG values. The example retains nanosecond timestamp precision.

Specify the actual file's Unix timestamp unit explicitly: `s`, `ms`, `us`, or `ns`. Select acceleration units using `--acc-unit g`, `mg`, or `m/s2`; this converter uses 1 g = 9.80665 m/s². Map nonstandard axis columns with `--ax-column`, `--ay-column`, and `--az-column`. For XLSX, `--sheet NAME` selects a worksheet; otherwise the first worksheet is used. Store long Unix timestamps as text in Excel to prevent Excel's numeric rounding before conversion.

Converted timestamps are UTC, not an inferred local timezone. The ACC path calculates the supplied pitch/roll formulas. If a separately measured neutral reference is available, subtract it with `--reference-pitch-deg` and `--reference-roll-deg`; both default to zero, so uncalibrated file angles are not automatically equivalent to live upright-relative angles. ECG conversion preserves the signal and converts time without inventing HR or detecting RR peaks. This converter does not import recordings into the live fatigue panel.

Run `python3 polar_recordings.py --help` for file, worksheet, column-mapping, and unit options. The examples are synthetic and are not experiment results. The missing original spreadsheets mean their exact column names, timestamp scale, worksheet selection, and sensor mounting convention still need confirmation.

## What to hand in and what remains to validate

Submit this explanation and the thresholds/cause/output table with the implementation, a clearly labeled Demo export or screenshot, and the commands needed to reproduce it. For real experiment evidence, record the participant baseline, sensor placement, independent task context, data-quality issues, and the actual exported session. Compare those records with worker-reported outcomes before making claims about accuracy.

Automated checks establish rule calculations and software behavior, not physiological validity. The remaining study work is to recover the original `Experimental setup.xlsx`, agree on baseline/window/calibration procedures, test the H10 during representative tasks, and collect labeled observations to evaluate the rules or train a future fatigue model. Source quotations such as “critical fatigue detected” are feedback text from the assignment framework, not evidence of clinical validation.

## Verification performed

The implementation passed 42 Node tests, 14 Python tests (including XLSX input), and five Chromium browser tests. Browser checks exercised the built bundle with simulated Bluetooth HR/RR, ECG and acceleration packets, baseline capture, missing RR, reconnect calibration resets, demo speech and export, both dashboard pages, and a narrow-screen layout. The build and whitespace checks passed. Real Bluetooth radio performance and physiological accuracy were not tested without an H10 and participant data.

The JSON download contains summary observations and feedback events, not raw ECG or beat-by-beat RR recordings. Each observation includes its own baseline/calibration status; the top-level baseline reflects the current live session and clears when recapturing or reconnecting. A stopped demo retains its labeled demonstration log for download.
