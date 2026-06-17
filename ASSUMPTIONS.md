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
