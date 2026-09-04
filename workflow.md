# Namma Route: Operational Workflow

Step-by-step flow from the perspective of a Bengaluru Traffic Police dispatcher.

Throughout: the system produces **estimates with stated uncertainty**, not
predictions. Every forecast reports the range, the number of past events behind it,
and how it compares to a naive baseline. That framing is the product.

---

## Workflow A — Unplanned events (camera-assisted)

### 1. Density monitoring
The YOLOv8 pipeline reads junction camera stills and reports vehicle density,
normalised per megapixel so the threshold means the same thing on a wide shot and a
tight one.

### 2. Congestion flag
Density crosses the threshold on the Outer Ring Road. The pipeline raises an alert
with the annotated frame and a `needs_human_review` flag when heavy vehicles are
present in dense traffic.

**What this is:** a congestion signal worth a human look. **What it is not:** accident
detection. A single frame cannot tell a stopped vehicle from a moving one — the
response says so in its `method_note`.

### 3. Dashboard alert
The dashboard shows the alert with the annotated frame. The frame is fetched once per
alert rather than on every poll.

### 4. Forecast
The alert routes into the estimator, which returns:
- an **impact tier** (Low / Medium / High) with calibrated probabilities
- an **expected clearance time** as a median plus a typical range
- the **evidence**: "median of 244 past vehicle breakdown events on Mysore Road"
- cross-checks from the k-NN analogues and the regression model

If the interquartile spread is wide, the response carries a warning telling the
dispatcher to plan against the range rather than the midpoint.

### 5. Dispatch
The Response Planner returns officer and barricade counts, an action checklist and
the best adjacent corridors. The checklist matches the closure decision that was
actually entered — asking for a High-severity plan without a closure returns
lane-control actions, not instructions to close the road.

---

## Workflow B — Planned events (proactive)

### 1. Registration
A procession is scheduled for Friday 17:00 near MG Road.

### 2. Parameters
The dispatcher enters cause `Procession`, corridor `CBD 1`, hour `17:00`, day
`Friday`, road closure `Yes`.

### 3. Output
- **High impact tier.** With a closure requested this is High *by definition*, and the
  response says so rather than implying a model inferred it.
- **Expected clearance ~46 min, typical range 22–85 min**, based on past processions.
- A **thin-history flag**: processions are 66 events in the dataset. The UI leads with
  the analogue list and the range rather than the midpoint.

### 4. Deployment
The planner returns 8–15 officers, 10–20 barricades, a route-closure checklist and
diversion corridors with hour-aware rationale.

---

## Workflow C — The learning loop

### 1. Log the outcome
After the event clears, the dispatcher opens **Post-Event Learning** and enters the
actual severity and clearance time against the event ID — either a historical `FKID…`
or a `SIM-…` id returned by the planner. IDs with no prediction on record are
rejected rather than scored against an invented one.

### 2. Live scoring
The panel shows accuracy computed **only from logged feedback**, over rows that can
actually be scored, and displays the offline benchmark separately and labelled as
such. Both start empty and move as feedback arrives.

### 3. Retrain
**Retrain with feedback** folds the logged outcomes into the training set, recomputes
severity from the corrected durations, retrains and hot-swaps the models without a
restart. The button reports when it finishes.

This is a real retraining path, not a description of one. It can also be run from the
command line with `python3 backend/forecasting.py --feedback`.
