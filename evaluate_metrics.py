"""
Model Evaluation & Leakage Audit
================================
Evaluates every estimator against the naive baseline it must beat, and re-runs the
feature ablation that originally exposed the target leak.

The point of this script is to make the leak impossible to reintroduce silently. If
someone puts `priority` back in as the label, or re-adds a feature that encodes it,
the ablation section below will show accuracy jumping to ~0.93 while "cause + time
only" stays at chance — which is the exact signature of the original bug.

Usage:
    python3 evaluate_metrics.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, median_absolute_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from backend.forecasting import (  # noqa: E402
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, SEVERITY_CLASS_ORDER,
    EmpiricalOutcomeEstimator, load_labelled_data, load_engine,
)

RULE = "=" * 72


def header(title):
    print("\n" + RULE)
    print(f"  {title}")
    print(RULE)


df = load_labelled_data()

# ─── 1. Primary estimator vs baselines ───────────────────────────────────────

header("PRIMARY ESTIMATOR — empirical, vs naive baselines")

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["severity_tier"])
est = EmpiricalOutcomeEstimator().fit(train_df)
rows = test_df.to_dict("records")
y_dur = test_df["duration_to_close_min"].values
y_sev = test_df["severity_tier"].astype(str).values

dur = [est.predict_duration(r) for r in rows]
med = np.array([d["median_min"] for d in dur])
p10 = np.array([d["p10_min"] for d in dur])
p90 = np.array([d["p90_min"] for d in dur])

global_median = float(train_df["duration_to_close_min"].median())
cause_median = train_df.groupby("event_cause")["duration_to_close_min"].median()
per_cause = test_df["event_cause"].map(cause_median).fillna(global_median).values

print(f"\n  {'Duration estimator':<34}{'MAE':>10}{'MedAE':>10}")
print(f"  {'-' * 54}")
for name, pred in [
    ("Naive: always the global median", np.full_like(y_dur, global_median)),
    ("Naive: per-cause median", per_cause),
    ("Empirical (primary)", med),
]:
    print(f"  {name:<34}{mean_absolute_error(y_dur, pred):>10.1f}"
          f"{median_absolute_error(y_dur, pred):>10.1f}")

coverage = ((y_dur >= p10) & (y_dur <= p90)).mean()
print(f"\n  P10-P90 interval covers {coverage:.1%} of outcomes (nominal 80%)")

sev = [est.predict_severity(r) for r in rows]
probs = np.array([[s["probabilities"][c] for c in SEVERITY_CLASS_ORDER] for s in sev])
sev_pred = [SEVERITY_CLASS_ORDER[i] for i in probs.argmax(1)]
baseline = pd.Series(y_sev).value_counts(normalize=True).max()

print(f"\n  {'Severity estimator':<34}{'Accuracy':>10}{'F1 (w)':>10}")
print(f"  {'-' * 54}")
print(f"  {'Naive: majority class':<34}{baseline:>10.4f}{'—':>10}")
print(f"  {'Empirical (primary)':<34}{accuracy_score(y_sev, sev_pred):>10.4f}"
      f"{f1_score(y_sev, sev_pred, average='weighted', zero_division=0):>10.4f}")
print(f"\n  Mean confidence {probs.max(1).mean():.3f} against accuracy "
      f"{accuracy_score(y_sev, sev_pred):.3f} — these should track each other.")
print()
print(classification_report(y_sev, sev_pred, digits=3, zero_division=0))


# ─── 2. Leakage audit ────────────────────────────────────────────────────────

header("LEAKAGE AUDIT — feature ablation on the severity target")

print("""
  A legitimate target degrades gracefully as features are removed. A leaked one
  collapses to chance the moment the leaking column goes, because nothing else in
  the data actually predicts it.

  The original `severity_tier = priority` scored 0.9299 with geography and 0.6458
  without it, against a 0.6174 baseline — the signature of a leak. Current numbers:
""")

y = LabelEncoder().fit_transform(df["severity_tier"].astype(str))
maj = np.bincount(y).max() / len(y)
print(f"  Majority-class baseline: {maj:.4f}\n")

geo_cats = ["corridor", "zone", "police_station", "junction"]
geo_nums = ["latitude", "longitude"]
sev_nums = [f for f in NUMERICAL_FEATURES if f != "requires_road_closure_int"]


def ablate(name, cats, nums):
    X = df[cats + nums]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
            ("num", StandardScaler(), nums),
        ])),
        ("clf", XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                              reg_lambda=1.5, min_child_weight=5, random_state=42,
                              n_jobs=-1, verbosity=0)),
    ])
    pipe.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, pipe.predict(X_te))
    print(f"  {name:<44}{acc:>8.4f}   ({acc - maj:+.4f} vs baseline)")
    return acc


full = ablate("All features", CATEGORICAL_FEATURES, sev_nums)
no_geo = ablate("Without geography (lat/lon/junction/station)",
                [c for c in CATEGORICAL_FEATURES if c not in geo_cats],
                [n for n in sev_nums if n not in geo_nums])
minimal = ablate("Cause + time only", ["event_cause", "event_type", "time_bin"],
                 ["hour_of_day", "day_of_week", "is_weekend"])

print()
gap = full - no_geo
if gap > 0.20:
    print(f"  FAIL: removing geography costs {gap:.3f} accuracy. That is the leakage")
    print("        signature — the label is probably encoding location again.")
else:
    print(f"  PASS: removing geography costs {gap:.3f} accuracy. Geography contributes")
    print("        like an ordinary feature rather than defining the label.")


# ─── 3. Served engine sanity check ───────────────────────────────────────────

header("SERVED ENGINE — output variance check")

try:
    engine = load_engine()
except FileNotFoundError as exc:
    print(f"  Skipped: {exc}")
else:
    causes = sorted(df["event_cause"].unique())
    corridors = sorted(df["corridor"].unique())
    seen, durations = {}, []
    for cause in causes:
        for corridor in corridors:
            for closure in (0, 1):
                r = engine.forecast({
                    "event_cause": cause, "event_type": "unplanned",
                    "corridor": corridor, "zone": "Unknown",
                    "police_station": "Unknown", "direction": "unknown",
                    "hour_of_day": 9, "day_of_week": 2, "is_weekend": 0,
                    "requires_road_closure": closure, "veh_type": "none",
                    "description": "",
                })
                seen[r["severity_tier"]] = seen.get(r["severity_tier"], 0) + 1
                durations.append(r["expected_duration_min"])

    total = sum(seen.values())
    print(f"\n  {total} forecasts across every cause x corridor x closure combination")
    print(f"  Severity distribution: {seen}")
    print(f"  Duration range: {min(durations):.0f} - {max(durations):.0f} min")
    if len(seen) == 1:
        print("\n  FAIL: the model returns a single constant severity. This is what the")
        print("        original build did (High for all 2,112 inputs) because latitude")
        print("        and longitude were frozen at the city centre for every request.")
    else:
        print("\n  PASS: output varies with the input.")

print("\n" + RULE)
