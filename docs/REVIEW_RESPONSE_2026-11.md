# Response to the November review

This covers each item in the review of the frustration and fatigue dashboard. I re-ran the code from before and after the changes on recordings from the study bucket, so the numbers are measured, not estimated. The evidence is in [the write-up](TEAM3_FRUSTRATION_FATIGUE.md).

**The main finding: the Oct 30 test data came from a device that was not being worn.** Empatica's wear detection reads 0% for 917 of its 918 minutes (the other has no value), its own pulse rate, EDA and PRV are blank all day, and median "skin" temperature is 21.9 °C. The bridge ignored this and produced 271 cues. That explains the near-zero EDA and much of the implausible HRV on that day. Worn recordings had the same defects for other reasons, covered below.

## Status

| # | Item | Status |
| --- | --- | --- |
| 1 | Accuracy overstated | Fixed; caveats below |
| 2 | Cause never identified | **Partly fixed** |
| 3 | Unrealistic HRV | Fixed; reads low |
| 4 | EDA near zero | Fixed; low signal remains coarse |
| 5 | No Avro for Nov 1 | The files exist; the bridge now explains itself |
| 6 | Too many alerts | Fixed |
| 7 | "Follow live" Polar-only | Fixed |
| 8 | Two Empatica pipelines | Chose the bridge |
| 9 | AWS keys | No keys found in git history |
| 10 | requirements.txt | Done |

## Item by item

**1. Accuracy.** I reproduced your numbers exactly: 87.5% random split, 48.9% new people, 86.0% fake labels. The dashboard now leads with accuracy on new people, and shows the previous model's numbers beside the new ones.
- The deployed model scores **65.6% on new people** (nested leave-one-participant-out, 95% interval 43–83%). A permutation test puts chance at 33% on average and 55% at the 95th percentile (p = 0.015).
- Fake-label accuracy fell from 86% to 59%, so some person recognition remains.
- High recall is 47%, and four of 15 people score 0%.
- **Per-person normalization helped less than expected.** It is in, but on its own it did not remove person recognition: the old model still scored 83–92% on fake labels under every scaling I tried. Most of the improvement comes from a smaller, regularized model and less window overlap.

**2. Cause.** The pipeline did compute SCR. It was lost because 18–65% of windows have no skin response, and a zero baseline cannot anchor a percentage. SCR now has a floor, and all five signals are present in 51–74% of readings instead of mostly 0%. But that did not solve it: all five arrows hold together in only about 1% of those predictions. I added a labelled "closest pattern" match at 4 of 5 signals. **Only 21 of 549 High predictions (3.8%) now name a cause**, and most cues still use the general message. See the first decision below.

**3. HRV.** I reproduced the 512 ms baseline. There were two causes: missed beats that pass the 300–2000 ms range check, and a baseline taken from the first 5 seconds. Beats are now cleaned, and the baseline is the wearer's recent median. Median RMSSD went from 180–266 ms to 60–87 ms on worn recordings. Windows under 60% clean beats report no HRV, so about two-thirds of worn minutes have it. Against Empatica's own RMSSD, rank correlation rose from 0.74 to 0.83, but cleaned values read roughly 20–30 ms low. Fatigue High went from 35% of readings on Oct 30 to none, and from 30% to 1% on Nov 3.

**4. EDA.** Near-zero EDA on Oct 30 was the unworn device. The recent worn recordings really are low, with medians around 0.05–0.25 µS. Means under 0.05 µS are now reported as not measured. Most of the instability came from drift against a first-minute baseline: on Nov 1, the median EDA change went from +143% to +2%.

**5. Avro.** Nov 1 does have 18 Avro files, starting at 17:09 UTC. I cannot tell what you hit, but there are two likely causes: the enrollment name changed by a letter between Oct 30 and Nov 1, and uploads can lag by hours. The bridge now has `--list`, and when nothing is processed it says who is there or that Avro has not arrived. Per-minute CSVs cannot be a fallback, because one value per minute cannot feed 60 s features.

**6. Alerts.** On Oct 30 I got 40 frustration and 231 fatigue cues. "3 consecutive" meant 3 seconds, because predictions ran every second on windows that overlap by 98%. There is now one prediction every 30 s, and an identical message does not repeat within 15 minutes. Frustration cues by day: 40 → 0, 111 → 14, 44 → 6, but **12 → 15 on Nov 3**, where the new model calls High more often. The most in any 23 minutes of the Nov 4 retest session is 2.

**7. Follow.** The dashboard has a Session menu listing every saved session, and the address bar then holds a shareable link.

**8. Pipelines.** The dashboard uses `tools/empatica_bridge.py`. I did not delete `avro_stream_poller_with_ml.py`, because the older worker-queue view and the voice coach read its `/latest` endpoint; it is now labelled as legacy. The README said port 7000 while the code defaults to 7001, and that is corrected.

**9. Keys.** `.env` was already git-ignored. I searched all history on every fetched branch, including for the actual key values, and found none; the old file held empty strings. So rotation is not needed because of this repository. If the copy with keys filled in was shared or pushed anywhere else, rotate them. I added `.env.example`.

**10. Requirements.** Added as asked, plus `flask`, which the legacy poller imports. scikit-learn 1.8.0 needs Python 3.11 or newer.

## Decisions I would like your view on

1. **How should causes be named?** Physiology alone rarely matches a full pattern. Options: keep the strict 4-of-5 rule and accept that most cues are general (current); relax to 3 of 5, which reaches about 29% of predictions with all five signals, but is usually a 4- or 5-way tie; or add task context, as the DBR architecture intends. I would add task context.
2. **Is a rolling baseline acceptable?** The DBR says "resting baseline", but sessions have no rest period, and a first-minute baseline drifts by hundreds of percent. I used the wearer's last 20 minutes, so elevation lasting beyond about 10 minutes starts to read as normal.

## Known gaps

- The study's S3 recordings have no frustration labels, so predictions on them cannot be scored. The model calls 19–41% of them High, and I cannot say whether that is right.
- One download of a 69 MB session export was cut short (1 of 12 attempts). I could not reproduce it or find the cause, and it is not fixed. Retrying worked.
