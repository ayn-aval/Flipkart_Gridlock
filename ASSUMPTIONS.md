# Assumptions & Limitations

This document tracks every assumption, heuristic, and data limitation relevant to the prototype. Updated after each phase.

---

## Phase 0 — Data Audit

### Dataset Limitations (Verified)

1. **No `end_datetime` for most rows:** Only a handful of events have `end_datetime` populated. We use `(closed_datetime − start_datetime)` as a proxy for "time-to-clear" / event duration. This proxy conflates actual event duration with reporting/administrative delay.

2. **No ground-truth resource data:** The dataset contains **no fields** for:
   - Number of police officers deployed
   - Number of barricades placed
   - Diversion routes actually used
   - Manpower cost or availability
   
   Any resource recommendations (Phase 3) will be built as a **transparent rule/heuristic layer**, not as something "learned" from data that doesn't exist. This will be clearly labeled in the demo.

3. **Diurnal reporting bias:** Most logged events start between 7pm–8am (IST), with very few between 10am–5pm. This likely reflects patrol/reporting patterns (shift changes, night patrol schedules), **not** actual incident-rate patterns. We will not claim "traffic is safest at noon."

4. **Small planned-event subset:** The event-driven causes most relevant to the theme (`public_event`, `procession`, `vip_movement`, `protest`, `construction`) represent a small fraction of the total dataset (~670 rows out of ~8,200). Model generalization from this subset will be limited — we use explicit fallback strategies (k-NN analog lookup) rather than claiming strong predictive power.

5. **Zone data is sparse:** The `zone` column is only populated for ~3,400 of ~8,200 rows. We will impute or work around this.

6. **Junction data is sparse:** The `junction` column is only populated for ~2,500 rows. We will use lat/lon for geographic analysis where junction is missing.

7. **Anonymization:** Event IDs, user IDs, vehicle numbers, and some address details are anonymized. This doesn't affect our modeling but means we can't cross-reference with external systems.

### Assumptions Made

1. **UTC timestamps:** All datetime columns appear to be in UTC. Bengaluru is UTC+5:30. We will convert to IST for all user-facing displays and hour-of-day features.

2. **`requires_road_closure` as boolean:** Treated as a binary indicator. True = road closure needed, False = not needed.

3. **Corridor centroids:** We assume that computing mean(lat, lon) per corridor gives a reasonable centroid for adjacency calculations. This is a simplification — corridors are linear, not circular.

---

## Phase 1 — Cleaning & Feature Engineering

### Cleaning Decisions

1. **Event cause normalization:** `Debris` (12 rows) merged with `debris` (1 row) → `debris` (13 rows). `test_demo` (3 rows) and `fog_low_visibility` (2 rows) merged into `others` as they had fewer than 5 occurrences and aren't actionable for the theme.

2. **Duration outlier filter:** Duration-to-close values are capped at 7 days (10,080 min). Records exceeding this are likely stale administrative closures (e.g., a pothole event that stayed "active" for weeks before bulk-close). The median with this filter is ~52 min, which aligns better with operational reality.

3. **Coordinate cleaning:** 2 rows had `endlatitude` values around 59.86 and `endlongitude` around 62.7 — clearly not in Bengaluru (lat ~12.8-13.3, lon ~77.3-77.8). These were set to NaN. Additionally, `endlatitude`/`endlongitude` == 0 (used as missing marker) was converted to NaN.

4. **IST time features:** All hour/day features are derived from IST (UTC+5:30), not UTC, since user-facing displays and operational decisions are in local time.

### Severity Tier Logic

The severity tier is a synthetic label derived from available data:
- **High:** priority=High AND (duration > 120 min OR requires_road_closure=True)
- **Medium:** priority=High AND 0 < duration ≤ 120 min, OR priority=Low AND duration > 60 min
- **Low:** everything else (including events with unknown duration)

This is a heuristic — not ground truth. Real severity depends on factors not in the data (road width, traffic volume, number of affected lanes, proximity to hospitals/schools, etc.).

### Corridor Adjacency

- Adjacency is computed using haversine distance between corridor centroids.
- Top 5 nearest neighbors stored per corridor.
- This is a simplification: corridors are road segments, not points. Two corridors might be geographically close by centroid but not connected by road. We accept this for the prototype.

---

## Phase 2 — Impact Forecasting Engine

### Model Choice Rationale

**Gradient Boosted Trees (GBT)** chosen over logistic/linear regression because:
- 8K rows is comfortable for GBT (not so small that it overfits, not so large that it's slow)
- Severity depends on non-linear interactions (e.g., construction + road_closure behaves differently from construction alone)
- GBT provides feature importances for explainability
- Handles mixed categorical/numerical features well via the preprocessing pipeline

### Honest Metric Interpretation

1. **Severity Classifier (71.9% accuracy):** This is a three-class problem with imbalanced classes (Low: 5,767 / Medium: 1,810 / High: 596). The model beats naive majority-class baseline (70.5%) modestly. The F1 for the High class is lower because it's the minority class. For a hackathon prototype this is acceptable — the model's output should be presented as a "suggested tier" not a definitive classification.

2. **Duration Regressor (Median AE 34 min, MAE 366 min, R² ≈ 0):**
   - The **Median AE of 34 min** is operationally useful — for the median event, the prediction is off by about half an hour.
   - The **MAE of 366 min** and **R² near zero** are driven by the extreme right tail. Events like pot_holes and water_logging can stay "open" for days (administrative delay), creating massive residuals that dominate MAE/R².
   - Per-cause, the model works well for vehicle_breakdown (MAE 31 min) and accident (MAE 41 min) which have tighter duration distributions. It struggles with construction, water_logging, and pot_holes which have highly variable durations due to the nature of these issues and administrative close delays.
   - **We report these metrics honestly.** We do not claim high accuracy for duration prediction.

3. **k-NN Fallback for Rare Planned Events:**
   - Procession (72 rows, only 13 with duration), public_event (84 rows, 0 with duration), vip_movement (20 rows, 0 with duration), protest (15 rows, 2 with duration) — these are too sparse for reliable model predictions.
   - Instead of fabricating confidence, we surface the 5 most similar historical events and let the user see the range of outcomes. The model's point prediction is still shown but explicitly labeled as "rough estimate."
   - This is labeled `knn_analog_fallback` in the API response.

### Features Used

- `event_cause`, `corridor`, `time_period` (categorical, one-hot encoded)
- `hour_of_day`, `day_of_week`, `is_weekend`, `requires_road_closure` (numerical, scaled)
- `veh_type` (optional categorical — filled with "none" for non-vehicle events)
- `zone` was excluded because it's missing for 58% of rows and would reduce the usable training set significantly.

### What the Model Does NOT Know

The model has no information about:
- Actual traffic volume at the time of the event
- Road geometry (number of lanes, width, alternative paths)
- Weather conditions
- Size/scale of planned events (expected crowd size, VIP security level)
- Historical resource deployment outcomes

These gaps are acceptable for a prototype that demonstrates the concept.

---

## Phase 3 — Resource Recommendation Engine

### Critical Disclaimer

**The recommendation engine is entirely rule/heuristic-based.** The dataset has zero ground truth for:
- How many officers were actually deployed to any event
- How many barricades were placed
- Which diversion routes were used
- Whether the deployment was adequate or insufficient

This is stated upfront in every API response and in the demo. Judges should understand that the recommendation layer demonstrates the *architecture* for operationalizing forecasts, not a claim of "learned" resource optimization.

### How Officer/Barricade Numbers Were Anchored

The numbers are not arbitrary. They're derived from:

1. **Road closure frequency by cause** (from data): VIP movements need closure 80% of the time → highest resource allocation. Vehicle breakdowns only 4.3% → lowest.

2. **Duration patterns** (from Phase 1): construction events last hours → sustained deployment. Accidents clear in ~40 min → quick response.

3. **Severity tier** (from Phase 2 model): High severity → more resources, Medium → standard, Low → monitoring only.

4. **Standard traffic management benchmarks**: A typical Bengaluru junction has 2-4 officers. A major road closure needs 6-10. Large events historically get 15-30+.

### Diversion Suggestions

- Based on **centroid distance** (haversine) from the corridor adjacency table
- Hour-aware: compares historical event frequency at the specific hour on both the affected and alternate corridor
- Warns when the suggested alternate is historically busier
- For "Non-corridor" events, explicitly states that local on-ground assessment is needed
- **Does NOT account for**: real-time traffic, road connectivity, or actual driving distance (only centroid-to-centroid distance)
