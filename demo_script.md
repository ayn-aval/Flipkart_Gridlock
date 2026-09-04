# Namma Route — 3-Minute Demo Script

**[Bracketed text]** is what to click. The rest is roughly what to say.

The spine of this demo is that the system is **calibrated**: it tells you how
confident it is, what evidence it is standing on, and where it is weak. That is
harder to build than a confident-looking number and it is the thing worth showing.

---

## 1. The hook — landing page

**[Start on the landing page.]**

> "Namma Route is a decision-support tool for the Bengaluru Traffic Police. When an
> event happens today, the call on how many officers to send is made from experience
> and instinct. We turn 8,000 historical events into an estimate you can act on — and,
> just as importantly, an honest statement of how much to trust it."

**[Click Launch Command Center.]**

---

## 2. The data — dashboard and map

> "8,057 real Astram events. Note the severity breakdown: about a third have a
> measured outcome. The rest we show as 'no measured outcome' rather than inventing a
> label for them."

**[Click Live Map.]**

> "Historical events across the city. But knowing where traffic *was* isn't the job —
> the job is knowing what happens next."

---

## 3. The forecast — Response Planner

**[Response Planner. Select: Cause → Procession, Corridor → CBD 1, Hour → 5 PM,
Day → Friday, Road Closure → Yes. Click Run Forecast.]**

> "High impact tier — and look at the reasoning line: with a road closure requested,
> that's High *by definition*. The system isn't pretending to infer something it was
> told.
>
> Clearance: about 46 minutes, typical range 22 to 85, based on past processions on
> this corridor. We show the range because the point estimate on its own would be
> false precision.
>
> And it flags that processions are thin in our data — 66 events — so it tells you to
> lean on the analogue list below rather than the midpoint."

**[Scroll to the deployment plan.]**

> "8 to 15 officers, 10 to 20 barricades, a route-closure checklist and diversion
> corridors. This layer is a transparent rule table, not a model — the dataset has no
> record of what was actually deployed, so claiming we learned it would be a lie. The
> disclaimer ships in the API response itself."

**[Optional, if asked about accuracy:]**

> "51.5% on a three-class problem against a 40% baseline. That's modest and it's real.
> An earlier version of this reported 91% — that number came from a leak, the label
> was really just 'is this on a main road', and we took it out. `evaluate_metrics.py`
> re-runs the leak test on every invocation so it can't come back."

---

## 4. The camera layer — CCTV

**[CCTV tab, click a camera.]**

> "YOLOv8 counts vehicles and reports density per megapixel, so the threshold means
> the same thing whether the camera is zoomed wide or tight.
>
> We're careful about what we claim here: this flags congestion, not accidents. One
> frame can't tell a stopped car from a moving one. Detecting an actual incident needs
> stationarity across frames — that's the next build, not something we'd claim today."

---

## 5. The loop — Post-Event Learning

**[Post-Event Learning tab.]**

> "When an event clears, the officer logs what actually happened. These two counters
> are computed purely from that logged feedback — they start at zero and move as real
> outcomes come in. The offline benchmark is shown separately, labelled as a benchmark,
> so nobody confuses the two."

**[Enter a real event ID and an actual duration. Submit. Then click Retrain with
feedback.]**

> "And this genuinely retrains. It folds the logged outcomes into the training set,
> recomputes the labels and hot-swaps the models without a restart."

---

## 6. Close

> "Namma Route turns a traffic event into a resourcing decision, and it tells you how
> much to trust that decision. We think the honesty is the feature — a dispatcher who
> can see the range and the sample size can use the number. One who's given a
> confident-looking point estimate can't."

**[Questions.]**

---

## If a judge pushes on the numbers

- **"51% isn't very good."** Correct — against a 40% baseline on three classes. The
  data supports about eleven points of lift and we report it next to the baseline
  everywhere. The alternative was a 91% figure that measured nothing.
- **"Why not deep learning?"** We tried gradient boosting; it ties a one-line groupby
  (MAE 84.9 vs 84.8). The target measures administrative ticket closure, not
  clearance, so there is little learnable structure. We ship the estimator that
  reports a calibrated interval and its sample size, and publish the comparison.
- **"How do you know there's no leakage?"** `evaluate_metrics.py` runs a feature
  ablation every time and fails loudly if removing geography costs more than 0.20
  accuracy — the signature of the original bug.
