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
