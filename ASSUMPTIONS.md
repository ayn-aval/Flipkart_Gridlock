# Assumptions & Limitations

Every assumption, heuristic and data limitation in the prototype. This document
describes **what the code currently does**. Where an earlier version of the system
differed, that is called out explicitly, because the gap between this file and the
code was itself one of the project's problems.

---

## 0. The target-leakage correction

The single most important thing on this page.

### What was wrong

`severity_tier` was assigned as `df["severity_tier"] = df["priority"]`, and the
severity classifier reported **91.4% accuracy** against it.

`priority` is not a measure of impact. It is an administrative flag:

| | High | Low |
| :--- | ---: | ---: |
| On one of the 21 named corridors | 4,967 | 6 |
| Non-corridor | 7 | 3,077 |

"On a named corridor → High" is correct for 8,044 of 8,057 rows — **99.84%**. The
model was learning to answer *"is this on a main road?"*, which the dispatcher knows
before opening the app.

The code did try to prevent this, dropping `corridor` from the feature set with the
comment *"to prevent data leakage, as it administratively dictates Priority"*. The
reasoning was right and the fix did not work, because `latitude`, `longitude`,
`junction` and `police_station` remained. With 7,162 distinct coordinate pairs across
8,057 rows, the trees simply partition space and reconstruct the corridor boundaries.

| Feature set | Accuracy |
| :--- | ---: |
| Majority-class baseline | 0.6174 |
| As shipped (the reported 91%) | 0.9299 |
| Without text | 0.9398 |
| Description text only | 0.6414 |
| **Without lat/lon/junction/police_station** | **0.6458** |
| **Cause + time + closure only** | **0.6402** |

A second consequence: `ForecastRequest` had no latitude or longitude field, so every
live call fell back to the city centre. The feature carrying nearly all the model's
signal was frozen at a constant, and a sweep of all 2,112 form combinations returned
**High for every single one**.

### What it is now

```
High    — required a road closure, OR took more than 90 minutes to clear
Medium  — cleared in 30-90 minutes, no closure
Low     — cleared in under 30 minutes, no closure
```

Defined in `data_cleaning.engineer_severity_tier`. Thresholds are
`SEVERITY_MEDIUM_MIN = 30` and `SEVERITY_HIGH_MIN = 90`.

Consequences, accepted deliberately:

1. **Only outcome-bearing rows can be labelled.** 2,523 of 8,057 (31%). Both models
   train on that subset. The remaining 5,534 appear in the dashboard as "no measured
   outcome" and are excluded from training. Training a classifier on 8,057 rows of a
   label that was not a label is not a better trade.
2. **`requires_road_closure` cannot be a severity feature**, because it is part of
   the label. `forecasting.SEVERITY_NUMERICAL_FEATURES` excludes it. The duration
   model still uses it — there it is a legitimate predictor.
3. **Geography is now an honest feature.** Corridor median durations range from 32 to
   78 minutes, which is real signal about clearance time rather than a restatement of
   the label. `evaluate_metrics.py` re-runs the ablation on every invocation and fails
   loudly if removing geography ever costs more than 0.20 accuracy again.

Current performance: **51.5%** accuracy against a **40.0%** baseline. Modest, real,
and reported next to the baseline everywhere it appears.

---

## 1. Data audit

### Verified limitations

1. **No `end_datetime` for most rows.** `(closed_datetime − start_datetime)` is used
   as the duration proxy. This conflates event duration with administrative closure
   delay and is the main reason duration is hard to predict — see §3.

   **Do not try to recover the missing labels from `modified_datetime`.** It looks
   like an easy win: only 31% of rows carry a duration, `modified_datetime` is 100%
   populated, and on the rows where both exist it equals `closed_datetime` to within
   a minute 98.2% of the time. Using it would roughly triple the training set.

   It is a trap. Of the 3,956 closed events with no `closed_datetime`, **91.6% have a
   `modified_datetime` falling exactly at `:35` past the hour**, against 1.9% of the
   genuinely closed ones. Those events were closed by a recurring sweep, not by anyone
   observing the road clear. Their implied durations cluster tightly around 147
   minutes (p25 129, p75 167) where real closures are heavy-tailed (median 52, mean
   523). Feeding them in would add ~3,800 synthetic labels stacked near the sweep
   interval, pull every estimate toward it, and — because the test split would inherit
   the same artefact — report the result as an accuracy improvement. That is the same
   failure as the original `priority` leak wearing different clothes.

   The 69% of events without a real close time are not recoverable from this export.
   Fixing it requires the upstream system to record an actual resolution timestamp.
2. **No ground-truth resource data.** No field records officers deployed, barricades
   placed, diversions used, or whether the response was adequate. The recommendation
   engine is therefore an explicit rule layer, and every API response says so.
3. **Diurnal reporting bias.** Most events are logged between 19:00 and 08:00 IST.
   This reflects patrol and reporting shifts, not incident rates. We do not claim
   traffic is safest at noon. `hour_of_day`, `is_peak_hour`, `is_night` and `time_bin`
   are still model inputs and therefore partly encode shift patterns; the
   `/events/distributions` response carries a note stating this.
4. **Event-driven causes are thin.** The causes central to the brief are 66
   processions, 45 public events, 14 protests and 4 VIP movements. Forecasts for
   these are flagged `is_sparse_cause` and the UI leads with analogues.
5. **`zone` is sparse** (~3,400 of 8,057) and **`junction` sparser** (~2,500). Both
   are filled with `"unknown"` and used as ordinary categorical levels.
6. **Anonymisation.** IDs, user IDs, vehicle numbers and some address details are
   anonymised. No effect on modelling; it does prevent cross-referencing.
7. **Descriptions are multilingual.** Predominantly Kannada script mixed with
   transliterated English ("tyear blost", "woter logging"); 17% are empty.

### Assumptions

1. **Timestamps are UTC**; Bengaluru is UTC+5:30. All hour and day features are
   derived after converting to IST.
2. **`requires_road_closure` is boolean.** Unmappable values become `False`.
3. **Corridor centroids** are the mean of member-event coordinates. Corridors are
   linear, not circular, so this is an approximation.

---

## 2. Cleaning & feature engineering

1. **Cause normalisation.** `Debris` merges into `debris`. Causes with fewer than 5
   occurrences merge into `others` — currently `test_demo` (3) and
   `fog_low_visibility` (2). *This was documented before but not implemented; it is
   now done in `clean_event_cause`.* It also stops the dashboard offering "test_demo"
   as a forecastable option.
2. **Duration outliers** are capped at 24 hours (1,440 min). Values outside
   `0 < duration ≤ 1440` become `NaN`. *An earlier version of this document said 7
   days; the code has used 1,440 for some time.*
3. **Coordinate cleaning.** Out-of-Bengaluru coordinates and `0` sentinels become
   `NaN`.
4. **`event_span_km`** is the haversine distance between start and end coordinates,
   **capped at 10 km**. *It was previously set to `0.0` when it exceeded 10 km, which
   made a 12 km closure indistinguishable from a point event — the opposite of the
   truth.*
5. **Corridor adjacency** is the top 5 nearest corridors by haversine distance
   between centroids, written by `compute_corridor_adjacency`. *This step was
   documented from the beginning but never implemented; `corridor_adjacency.csv` was
   read by three modules and written by none, surviving only because it was committed.
   Deleting `data/processed/` and re-running the documented pipeline used to leave the
   API unable to boot.* The caveat stands: centroid proximity is not road connectivity.

---

## 3. Forecasting

### Why the primary estimator is not a learned model

The duration target is dominated by administrative closure delay. On a held-out split:

| Estimator | MAE | Median AE |
| :--- | ---: | ---: |
| Naive: global median | 87.6 | 29.2 |
| Naive: per-cause median | 84.8 | 28.9 |
| **Hierarchical empirical (primary)** | **84.9** | **29.0** |
| XGBoost regressor (cross-check) | ~85 | ~30 |
| k-NN analogue median | ~85 | ~31 |

Every method lands within noise of a one-line `groupby`. Given no accuracy advantage,
the estimator that reports a calibrated interval and can state how many past events
each estimate rests on is more useful than one that emits a point estimate. The
XGBoost models are still trained and served as cross-checks with published metrics,
so the comparison stays visible.

**The empirical estimator also produces severity**, from the observed class mix in the
same stratum (Laplace-smoothed, `α = 1`). This is both more accurate than the
classifier (51.5% vs 46.3%) and better calibrated — mean top probability 0.505 against
observed accuracy 0.515 — and it makes contradiction impossible. The classifier could
return "High at 99% confidence" alongside a 19-minute duration estimate, which is Low
by the definition severity is derived from.

Backoff order, minimum 20 observations per stratum:

```
(cause, corridor)  →  (cause, road_closure)  →  (cause)  →  global
```

A requested road closure returns High with probability 1 and stratum `"definition"` —
it is High by definition, so there is nothing to infer.

### Honest metric interpretation

1. **Severity: 51.5% against a 40.0% baseline.** A ~11-point lift on a 3-class
   problem. All three classes are predicted with comparable precision (0.48–0.60);
   `Low` recall is weakest at 0.26.
2. **Duration: MAE 84.9 min, median AE 29.0 min, P10–P90 covering 76.8%.** The median
   AE is the operationally useful figure: for a typical event the estimate is about
   half an hour off. MAE is inflated by the right tail. Where a stratum's
   interquartile spread exceeds 4×, the response carries a `duration_warning` and the
   UI tells the dispatcher to plan against the range — pot_holes, for example, is
   bimodal (15 events under 30 min, 11 over 90 min, almost nothing between).
3. **k-NN analogues** return a real median and P10–P90 across 25 neighbours.
   *Previously `method: "knn_analog_fallback"` was returned as a label while the
   duration came from the regressor and no median was ever computed; two consumers
   checked for the missing field and silently did nothing.*

### Features

Categorical: `event_cause`, `event_type`, `corridor`, `zone`, `police_station`,
`direction`, `veh_type`, `junction`, `time_bin`.
Numerical: `hour_of_day`, `day_of_week`, `is_weekend`, `requires_road_closure_int`
(duration model only), `is_peak_hour`, `is_night`, `has_vehicle`, `event_span_km`,
`latitude`, `longitude`.
Text: `description`, vectorised with **character n-grams (3–5)**. *English word
tokens with an English stop-word list were previously used on a mostly-Kannada
corpus; in ablation that text block cost about a point of accuracy.*

Coordinates now come from the request, defaulting to the **corridor centroid** rather
than the city centre.

### What the model does not know

Traffic volume at the time, road geometry, weather, expected crowd size, VIP security
level, or any historical deployment outcome.

---

## 4. Resource recommendations

**Entirely rule-based.** The dataset has zero ground truth for officers deployed,
barricades placed, diversions used, or adequacy of response. This is stated in every
API response and on the dashboard. The layer demonstrates how a forecast becomes an
operational decision; it is not a claim of learned resource optimisation.

Counts are anchored to:

1. **Closure frequency by cause** — VIP movement 80%, public event 46%, protest 40%,
   tree fall 39%, construction 26%, procession 26%, vehicle breakdown 4.3%.
2. **Duration patterns** — construction runs hours, accidents clear in ~40 minutes.
3. **Severity tier** from the forecast.
4. **Standard practice** — a typical junction has 2–4 officers; a major closure needs
   6–10; large events historically get 15–30+.

### Rule coverage

All 55 rules exist for both closure states at each severity. *Previously most
High-severity rules existed only with `requires_road_closure=True`, so a High forecast
with closure set to "No" fell through to the closure variant and printed a plan whose
first instruction was to close the road, while the response reported
`road_closure_used: False`. The fallback chain also tried High before Medium and Low,
so a Medium event could receive the full High closure plan.* The chain now never
escalates: exact match, then the other closure state at the same severity, then
strictly downward.

### Diversions

Centroid haversine distance from the adjacency table, hour-aware (compares historical
event frequency on both corridors at that hour), warns when the alternate is busier,
and states plainly when the event is not on a monitored corridor. Does **not** account
for real-time traffic, road connectivity or driving distance.

---

## 5. Computer vision

A per-frame vehicle counter, not an incident detector.

- YOLOv8n, filtered to car / motorcycle / bus / truck.
- Thresholds are **vehicles per megapixel** plus **fraction of frame covered by
  boxes**. *Raw box counts were previously used, which made the threshold a function
  of camera zoom and resolution: a wide shot of free-flowing traffic tripped it while
  a tight shot of a real jam did not.*
- Heavy-vehicle presence in dense traffic raises a `needs_human_review` flag.
- A single frame cannot distinguish stopped from moving traffic. Stationarity across
  consecutive frames, or bounding-box overlap, is the upgrade path. Every response
  carries a `method_note` saying so.
- Inference runs in a **separate process** (`backend/cv_worker.py`). xgboost and torch
  each load their own OpenMP runtime; on macOS the combination aborts the process when
  inference is called from a request thread. Isolation fixes that and keeps torch out
  of the web process entirely. The lazy loader is now lock-guarded — the startup
  warm-up and an early request could previously construct the model concurrently.

---

## 6. The learning loop

Officers submit actual outcomes; `POST /models/retrain` folds them into training,
recomputing severity from the corrected duration so the label stays consistent with
the cleaning pipeline, then hot-swaps the models.

*This was previously claimed in four documents and implemented in none: the log was
written and displayed, never read back into training.*

Two related corrections:

1. **The committed log was poisoned.** All 13,097 rows had `predicted_duration_min`
   exactly equal to ground truth, an artefact of a bug fixed in `bc3437a` whose data
   was never purged — and 468 rows predicted durations beyond the 1,440-minute
   training cap, which the model cannot produce. The dashboard's "13.3 min MAE / 95.1%
   accuracy" came entirely from that. The log is now reset and gitignored.
2. **`/feedback` no longer invents predictions.** For an unrecognised ID it used to
   fabricate one by perturbing the answer the user had just supplied, guaranteeing a
   flattering figure; it also crashed with `TypeError` when the duration was omitted.
   It now returns 404 for IDs with no prediction on record.

Live feedback metrics are computed only over rows that can actually be scored, and
are displayed separately from the offline benchmark, each labelled as what it is.
