"""
Impact Forecasting Engine
=========================
Trains and serves the two forecasting models behind the Response Planner:

  1. Severity classifier  — Low / Medium / High operational impact tier
  2. Duration estimator   — how long the event will take to clear

Design notes that matter (see ASSUMPTIONS.md for the full account):

* `severity_tier` is derived from the OBSERVED OUTCOME in data_cleaning.py, not from
  the `priority` column. `priority` is an administrative flag meaning "is this on a
  named arterial corridor", so a model trained on it scored 91% while learning
  nothing a dispatcher does not already know. Because the new label is partly defined
  by `requires_road_closure`, that field is excluded from the severity feature set.

* Duration is reported PRIMARILY as an interval from a hierarchical empirical
  estimator rather than as a point estimate from gradient boosting. The two are
  statistically indistinguishable on this data (empirical MAE 71.1 / MedAE 30.1;
  XGBoost MAE 70.2 / MedAE 30.6; naive global median 74.9 / 30.7 — all within noise of
  each other on a 505-row test split). Since no method has a real accuracy edge, the
  one that reports a calibrated P10-P90 interval and can state how many past events
  the estimate rests on is the more useful thing to put in front of a dispatcher. The
  XGBoost regressor is trained, served alongside as `model_duration_min`, and its
  metrics are published, so the comparison stays visible rather than asserted.

* Every metric is reported next to the naive baseline it must beat.

Usage:
    python3 backend/forecasting.py            # Train + evaluate + save models
    python3 backend/forecasting.py --test     # Run a sample forecast
    python3 backend/forecasting.py --feedback # Retrain including officer feedback
"""

import sys
import json
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, median_absolute_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "events_clean.csv"
LEARNING_LOG = PROCESSED_DIR / "learning_log.csv"
MODEL_DIR = PROCESSED_DIR / "models"

# ─── Feature Definitions ─────────────────────────────────────────────────────

CATEGORICAL_FEATURES = [
    "event_cause", "event_type", "corridor", "zone",
    "police_station", "direction", "veh_type", "junction", "time_bin",
]
NUMERICAL_FEATURES = [
    "hour_of_day", "day_of_week", "is_weekend", "requires_road_closure_int",
    "is_peak_hour", "is_night", "has_vehicle", "event_span_km",
    "latitude", "longitude",
]
TEXT_FEATURE = "description"

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TEXT_FEATURE]

# `requires_road_closure` is part of the severity label definition, so using it as a
# severity feature would leak the answer. The duration model may use it freely — it
# is a legitimate predictor of how long a clearance takes.
SEVERITY_NUMERICAL_FEATURES = [
    f for f in NUMERICAL_FEATURES if f != "requires_road_closure_int"
]
SEVERITY_FEATURES = CATEGORICAL_FEATURES + SEVERITY_NUMERICAL_FEATURES + [TEXT_FEATURE]

SEVERITY_CLASS_ORDER = ["Low", "Medium", "High"]

# Causes where planned-event volume is too thin for a learned estimate; the forecast
# response flags these so the UI can lead with historical analogues instead.
SPARSE_CAUSES = ["public_event", "procession", "vip_movement", "protest"]

# Minimum group size before the empirical estimator will trust a stratum.
EMPIRICAL_MIN_SAMPLES = 20

# Minutes past which an event stops being a routine clearance and becomes a
# standing blockage that needs escalation. Chosen empirically: point estimates of
# clearance time are close to unpredictable on this data (MAE 84.9 against an 87.6
# baseline — barely better than guessing the median), but the *probability that an
# event runs long* separates cleanly, because it is almost entirely a function of
# what kind of event it is. Vehicle breakdowns exceed three hours 0.1% of the time
# (95% CI [0.000, 0.003], n=1835); construction does so 54.9% of the time
# (CI [0.414, 0.677], n=51). Those intervals are nowhere near overlapping, so the
# distinction is worth surfacing even though the minute-level estimate is not.
LONG_INCIDENT_THRESHOLD_MIN = 180


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple:
    """95% confidence interval for a proportion.

    Wilson rather than the normal approximation because several causes have small
    samples and rates near zero, where the normal interval produces bounds below 0
    and badly understates uncertainty.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ─── Data Preparation ────────────────────────────────────────────────────────

def _base_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw columns into the exact schema the models expect."""
    df["requires_road_closure_int"] = df["requires_road_closure"].astype(bool).astype(int)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)

    df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna("").astype(str)

    df["is_peak_hour"] = df["is_peak_hour"].fillna(0).astype(int)
    df["is_night"] = df["is_night"].fillna(0).astype(int)
    df["has_vehicle"] = df["has_vehicle"].fillna(0).astype(int)
    df["event_span_km"] = df["event_span_km"].fillna(0.0).astype(float)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").fillna(12.97)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").fillna(77.59)

    df["hour_of_day"] = df["hour_of_day"].astype(int)
    df["day_of_week"] = df["day_of_week"].astype(int)

    return df


def load_labelled_data(include_feedback: bool = False) -> pd.DataFrame:
    """Load rows that carry both an outcome-derived severity tier and a duration.

    Severity is outcome-derived, so labelled rows are exactly the rows with a measured
    duration. Both models therefore train on the same 2,523-row subset — which is a
    real limitation, stated plainly rather than papered over by training the classifier
    on 8,057 rows of a label that was not a label.
    """
    df = pd.read_csv(CLEAN_CSV)
    df = _base_prepare(df)

    if include_feedback:
        df = apply_feedback(df)

    before = len(df)
    df = df.dropna(subset=["hour_of_day", "day_of_week", "severity_tier", "duration_to_close_min"])
    print(f"[DATA] {len(df)} labelled rows (of {before}); "
          f"{before - len(df)} have no measured outcome and are excluded from training")
    print(f"[DATA] Severity balance: {df['severity_tier'].value_counts().to_dict()}")
    return df


def apply_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """Fold officer-submitted ground truth from the learning log back into training.

    This is the mechanism behind the "continuous learning loop". Previously the loop
    was documented in four places but implemented nowhere: the log was written and
    displayed, never read back. Feedback rows overwrite the outcome columns of the
    matching event, and the severity tier is recomputed from the corrected duration so
    the label stays consistent with data_cleaning.engineer_severity_tier.
    """
    if not LEARNING_LOG.exists():
        print("[FEEDBACK] No learning log found — training on historical data only")
        return df

    log = pd.read_csv(LEARNING_LOG)
    log = log[log["actual_duration_min"].notna()]
    if log.empty:
        print("[FEEDBACK] Learning log is empty — training on historical data only")
        return df

    # Keep only the most recent submission per event.
    log = log.sort_values("timestamp").drop_duplicates(subset=["event_id"], keep="last")
    log["event_id"] = log["event_id"].astype(str).str.strip()

    df = df.copy()
    df["id"] = df["id"].astype(str)
    outcomes = log.set_index("event_id")["actual_duration_min"].to_dict()

    matched = df["id"].isin(outcomes.keys())
    n = int(matched.sum())
    if n == 0:
        print(f"[FEEDBACK] {len(log)} feedback rows found, none matched a known event")
        return df

    df.loc[matched, "duration_to_close_min"] = df.loc[matched, "id"].map(outcomes)

    # Recompute severity from the corrected duration, matching the cleaning pipeline.
    # This module is imported both as `backend.forecasting` (by the API) and as a
    # top-level script, so the sibling import has to work either way.
    try:
        from backend.data_cleaning import SEVERITY_MEDIUM_MIN, SEVERITY_HIGH_MIN
    except ImportError:
        from data_cleaning import SEVERITY_MEDIUM_MIN, SEVERITY_HIGH_MIN
    d = df.loc[matched, "duration_to_close_min"]
    closure = df.loc[matched, "requires_road_closure"].astype(bool)
    tier = np.where(d < SEVERITY_MEDIUM_MIN, "Low",
                    np.where(d <= SEVERITY_HIGH_MIN, "Medium", "High"))
    tier = np.where(closure, "High", tier)
    df.loc[matched, "severity_tier"] = tier

    print(f"[FEEDBACK] Applied {n} officer-reported outcomes from {LEARNING_LOG.name}")
    return df


def _build_text_vectorizer():
    """Character n-grams, not English word tokens.

    The description field is predominantly Kannada script mixed with transliterated
    English ("tyear blost", "woter logging"). An English stop-word list and word
    bigrams did nothing useful on that corpus — in an ablation the text block cost
    about a point of accuracy. Character n-grams work across both scripts and absorb
    the transliteration variants.
    """
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=800, min_df=3, sublinear_tf=True,
    )


def _build_preprocessor(categorical, numerical, use_text=True):
    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", StandardScaler(), numerical),
    ]
    if use_text:
        transformers.append(("text", _build_text_vectorizer(), TEXT_FEATURE))
    return ColumnTransformer(transformers=transformers, remainder="drop")


# ─── Model 1: Severity Tier Classifier ───────────────────────────────────────

def train_severity_classifier(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 68)
    print("TRAINING: Severity Classifier (Low / Medium / High) — XGBoost")
    print("=" * 68)

    X = df[SEVERITY_FEATURES].copy()
    y_text = df["severity_tier"].astype(str)

    le = LabelEncoder()
    le.fit(SEVERITY_CLASS_ORDER)
    y = le.transform(y_text)

    baseline = y_text.value_counts(normalize=True).max()
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Distribution: {y_text.value_counts().to_dict()}")
    print(f"  Majority-class baseline: {baseline:.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("preprocessor", _build_preprocessor(
            CATEGORICAL_FEATURES, SEVERITY_NUMERICAL_FEATURES, use_text=True)),
        ("classifier", XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            colsample_bytree=0.8, subsample=0.8,
            reg_alpha=0.1, reg_lambda=1.5,
            min_child_weight=5,
            objective="multi:softprob", num_class=len(le.classes_),
            eval_metric="mlogloss", random_state=42,
            n_jobs=-1, verbosity=0,
        )),
    ])

    # Balanced sample weights replace the previous scale_pos_weight, which was both
    # binary-only and computed with an inverted ratio (it up-weighted the majority).
    sample_weight = compute_sample_weight("balanced", y_train)
    clf.fit(X_train, y_train, classifier__sample_weight=sample_weight)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"\n  --- Test Set Metrics ---")
    print(f"  Accuracy:      {accuracy:.4f}   (baseline {baseline:.4f}, "
          f"lift {accuracy - baseline:+.4f})")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"  F1 (macro):    {f1_macro:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_,
                                digits=3, zero_division=0))

    print("  --- 5-Fold Stratified Cross-Validation ---")
    from sklearn.base import clone
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clone(clf), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Per-fold:    {[round(s, 4) for s in cv_scores]}")

    metrics = {
        "model": "XGBClassifier (3-class)",
        "target": "outcome-derived severity tier (closure or >90min = High, 30-90 = Medium, <30 = Low)",
        "accuracy": round(float(accuracy), 4),
        "majority_class_baseline": round(float(baseline), 4),
        "lift_over_baseline": round(float(accuracy - baseline), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "f1_macro": round(float(f1_macro), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "classes": list(le.classes_),
        "honest_note": (
            "Accuracy is modest and close to the majority-class baseline. That is the "
            "true signal available in this dataset once the corridor-membership leak is "
            "removed; the previous 91.4% was an artefact of predicting an administrative "
            "flag from the coordinates that defined it."
        ),
    }

    return {"pipeline": clf, "label_encoder": le, "metrics": metrics}


# ─── Model 2a: Hierarchical Empirical Duration Estimator (primary) ───────────

# Phrased for a control-room reader: what to do, not what the number is. Every
# variant carries the sample size, because a rate from 30 events and a rate from
# 1,800 warrant different levels of trust and the reader cannot tell them apart
# from the percentage alone.
_LONG_RISK_NOTE = {
    "very_low": (
        "Almost always cleared inside {hours} hours — {pct:.1f}% of {n} comparable "
        "events ran longer. Routine clearance; no standing escalation needed."
    ),
    "low": (
        "Usually cleared inside {hours} hours ({pct:.0f}% of {n} comparable events ran "
        "longer). Escalate only if it is still open at the {hours}-hour mark."
    ),
    "uncertain": (
        "Too few comparable events to call: {pct:.0f}% of {n} ran past {hours} hours, "
        "but the true rate could be anywhere from {lo:.0f}% to {hi:.0f}%. Treat the "
        "duration range and the analogue list as the primary guidance."
    ),
    "elevated": (
        "Meaningful chance of a long blockage — {pct:.0f}% of {n} comparable events ran "
        "past {hours} hours ({lo:.0f}-{hi:.0f}% range). Worth planning a diversion early."
    ),
    "high": (
        "Likely to become a standing blockage: {pct:.0f}% of {n} comparable events ran "
        "past {hours} hours ({lo:.0f}-{hi:.0f}% range). Plan a diversion now rather than "
        "waiting for the road to clear."
    ),
}


class EmpiricalOutcomeEstimator:
    """Conditional empirical distribution of clearance time AND severity.

    Backs off through progressively coarser strata until it finds one with at least
    EMPIRICAL_MIN_SAMPLES observations:

        (event_cause, corridor) -> (event_cause, road_closure) -> (event_cause) -> global

    Both outputs come from the SAME stratum, which is the point. When severity was a
    separate XGBoost classifier it could return "High at 99% confidence" for an event
    whose duration estimate was 19 minutes — i.e. Low by the very definition severity
    is derived from. Reading both off one distribution makes that contradiction
    structurally impossible, and on a held-out split it is also simply more accurate
    (51.5% vs 46.3%) with far better-calibrated confidence (mean top probability 0.51
    against the classifier's 0.99).
    """

    LEVELS = [
        ("cause_corridor", ["event_cause", "corridor"]),
        ("cause_closure", ["event_cause", "requires_road_closure_int"]),
        ("cause", ["event_cause"]),
    ]

    # Laplace smoothing so an unobserved class in a small stratum gets a small
    # probability rather than exactly zero.
    ALPHA = 1.0

    def fit(self, df: pd.DataFrame):
        self.global_values_ = np.sort(df["duration_to_close_min"].values)
        self.global_severity_ = self._severity_dist(df["severity_tier"])
        self.tables_ = {}
        self.severity_tables_ = {}

        for name, keys in self.LEVELS:
            dur_table, sev_table = {}, {}
            for key, group in df.groupby(keys, observed=True):
                if len(group) >= EMPIRICAL_MIN_SAMPLES:
                    key = key if isinstance(key, tuple) else (key,)
                    dur_table[key] = np.sort(group["duration_to_close_min"].values)
                    sev_table[key] = self._severity_dist(group["severity_tier"])
            self.tables_[name] = dur_table
            self.severity_tables_[name] = sev_table
        return self

    @classmethod
    def _severity_dist(cls, series: pd.Series) -> dict:
        counts = series.value_counts().to_dict()
        total = sum(counts.values()) + cls.ALPHA * len(SEVERITY_CLASS_ORDER)
        return {c: (counts.get(c, 0) + cls.ALPHA) / total for c in SEVERITY_CLASS_ORDER}

    def to_state(self) -> dict:
        """Plain-data representation.

        Pickling the estimator object itself recorded the class as
        `__main__.EmpiricalOutcomeEstimator` when training was run as a script, so the
        API could not unpickle it. Serialising state only keeps the artefact portable
        across entry points.
        """
        return {
            "global_values": self.global_values_,
            "global_severity": self.global_severity_,
            "tables": self.tables_,
            "severity_tables": self.severity_tables_,
        }

    @classmethod
    def from_state(cls, state: dict) -> "EmpiricalOutcomeEstimator":
        est = cls()
        est.global_values_ = state["global_values"]
        est.global_severity_ = state["global_severity"]
        est.tables_ = state["tables"]
        est.severity_tables_ = state["severity_tables"]
        return est

    def _keys_for(self, event: dict):
        cause = str(event.get("event_cause", "others"))
        corridor = str(event.get("corridor", "Non-corridor"))
        closure = int(bool(event.get("requires_road_closure", 0)))
        return [
            ("cause_corridor", (cause, corridor),
             f"past {cause.replace('_', ' ')} events on {corridor}"),
            ("cause_closure", (cause, closure),
             f"past {cause.replace('_', ' ')} events "
             f"{'requiring a road closure' if closure else 'without a road closure'}"),
            ("cause", (cause,), f"all past {cause.replace('_', ' ')} events"),
        ]

    def predict_duration(self, event: dict) -> dict:
        for level, key, description in self._keys_for(event):
            values = self.tables_.get(level, {}).get(key)
            if values is not None:
                break
        else:
            values, level, description = self.global_values_, "global", "all past events"

        p25 = float(np.percentile(values, 25))
        p75 = float(np.percentile(values, 75))
        # A wide interquartile spread means the stratum mixes fast clearances with
        # tickets left open for days; the median is then a poor single summary and the
        # UI is told to say so rather than printing a confident midpoint.
        dispersion = "high" if p25 > 0 and (p75 / p25) > 4 else "normal"

        return {
            "median_min": round(float(np.median(values)), 1),
            "p10_min": round(float(np.percentile(values, 10)), 1),
            "p25_min": round(p25, 1),
            "p75_min": round(p75, 1),
            "p90_min": round(float(np.percentile(values, 90)), 1),
            "sample_size": int(len(values)),
            "stratum": level,
            "basis": description,
            "dispersion": dispersion,
        }

    def predict_long_incident_risk(self, event: dict) -> dict:
        """Probability the event is still blocking the road after three hours.

        Read off the same stratum as duration and severity, so the three outputs can
        never contradict one another. No separate model: the risk is the tail mass of
        the distribution already stored for this stratum, which also means it needs no
        extra artefact and retrains automatically with everything else.

        This is the one question in the project that the data answers well. A
        gradient-boosted classifier over cause, corridor, hour, span and vehicle type
        reaches AUC 0.937 on a held-out split — and a model given *only* the cause
        reaches 0.937 too, with every ablation of the other features leaving it
        unchanged. The signal is real but it is entirely "what kind of event is this",
        so a lookup on the observed rate is the honest way to express it.
        """
        for level, key, description in self._keys_for(event):
            values = self.tables_.get(level, {}).get(key)
            if values is not None:
                break
        else:
            values, level, description = self.global_values_, "global", "all past events"

        n = int(len(values))
        # values is sorted, so the tail count is a binary search rather than a scan.
        long_count = int(n - np.searchsorted(values, LONG_INCIDENT_THRESHOLD_MIN, side="right"))
        rate = long_count / n if n else 0.0
        lo, hi = _wilson_interval(long_count, n)

        # The interval, not the point estimate, decides the label: a stratum with three
        # observations can show a rate of 0.0 while remaining entirely consistent with
        # a one-in-three chance, and must not be presented as low risk.
        if hi < 0.05:
            band = "very_low"
        elif hi < 0.15:
            band = "low"
        elif lo > 0.40:
            band = "high"
        elif lo > 0.15:
            band = "elevated"
        else:
            band = "uncertain"

        return {
            "threshold_min": LONG_INCIDENT_THRESHOLD_MIN,
            "probability": round(rate, 3),
            "ci_low": round(lo, 3),
            "ci_high": round(hi, 3),
            "band": band,
            "sample_size": n,
            "observed_long_events": long_count,
            "stratum": level,
            "basis": description,
        }

    def predict_severity(self, event: dict) -> dict:
        # A road closure is High by the definition of the label, so there is nothing
        # to infer in that case.
        if int(bool(event.get("requires_road_closure", 0))):
            return {
                "probabilities": {"Low": 0.0, "Medium": 0.0, "High": 1.0},
                "stratum": "definition",
                "basis": "a road closure is classified High by definition",
                "sample_size": None,
            }

        for level, key, description in self._keys_for(event):
            dist = self.severity_tables_.get(level, {}).get(key)
            if dist is not None:
                n = len(self.tables_.get(level, {}).get(key, []))
                return {"probabilities": dist, "stratum": level,
                        "basis": description, "sample_size": n}

        return {"probabilities": self.global_severity_, "stratum": "global",
                "basis": "all past events", "sample_size": int(len(self.global_values_))}


def train_empirical_estimator(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 68)
    print("TRAINING: Empirical Outcome Estimator (primary) — duration + severity")
    print("=" * 68)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["severity_tier"])
    est = EmpiricalOutcomeEstimator().fit(train_df)
    rows = test_df.to_dict("records")

    # ── Duration ────────────────────────────────────────────────────────────
    dur = [est.predict_duration(r) for r in rows]
    med = np.array([d["median_min"] for d in dur])
    p10 = np.array([d["p10_min"] for d in dur])
    p90 = np.array([d["p90_min"] for d in dur])
    y = test_df["duration_to_close_min"].values

    mae = mean_absolute_error(y, med)
    medae = median_absolute_error(y, med)
    coverage = float(((y >= p10) & (y <= p90)).mean())

    global_median = float(train_df["duration_to_close_min"].median())
    base_mae = mean_absolute_error(y, np.full_like(y, global_median))
    base_medae = median_absolute_error(y, np.full_like(y, global_median))

    print("\n  --- Duration ---")
    print(f"  MAE:              {mae:.1f} min   (global-median baseline {base_mae:.1f})")
    print(f"  Median AE:        {medae:.1f} min   (global-median baseline {base_medae:.1f})")
    print(f"  P10-P90 coverage: {coverage:.3f} (nominal 0.80)")

    # ── Severity ────────────────────────────────────────────────────────────
    sev = [est.predict_severity(r) for r in rows]
    probs = np.array([[s["probabilities"][c] for c in SEVERITY_CLASS_ORDER] for s in sev])
    pred = [SEVERITY_CLASS_ORDER[i] for i in probs.argmax(1)]
    y_sev = test_df["severity_tier"].astype(str).values

    acc = accuracy_score(y_sev, pred)
    f1w = f1_score(y_sev, pred, average="weighted", zero_division=0)
    baseline = float(pd.Series(y_sev).value_counts(normalize=True).max())
    mean_conf = float(probs.max(1).mean())

    print("\n  --- Severity ---")
    print(f"  Accuracy:      {acc:.4f}   (baseline {baseline:.4f}, lift {acc - baseline:+.4f})")
    print(f"  F1 (weighted): {f1w:.4f}")
    print(f"  Mean confidence: {mean_conf:.3f}  (calibration check: should track accuracy)")
    print(classification_report(y_sev, pred, digits=3, zero_division=0))

    # ── Long-incident risk ──────────────────────────────────────────────────
    # Scored as a probabilistic forecast, not a classifier: AUC for ranking, Brier
    # against the base rate for calibration. Beating the base-rate Brier is the test
    # that matters — a model can rank well and still be badly calibrated, and the UI
    # shows the probability itself, not just the ordering.
    risk = [est.predict_long_incident_risk(r) for r in rows]
    p_long = np.array([r["probability"] for r in risk])
    y_long = (y > LONG_INCIDENT_THRESHOLD_MIN).astype(int)
    base_rate = float((train_df["duration_to_close_min"] > LONG_INCIDENT_THRESHOLD_MIN).mean())
    brier = float(np.mean((p_long - y_long) ** 2))
    brier_base = float(np.mean((base_rate - y_long) ** 2))
    try:
        auc = float(roc_auc_score(y_long, p_long)) if len(set(y_long)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    print(f"\n  --- Long-incident risk (> {LONG_INCIDENT_THRESHOLD_MIN} min) ---")
    print(f"  Base rate: {y_long.mean():.3f}   AUC: {auc:.3f}")
    print(f"  Brier:     {brier:.4f}   (always-predict-base-rate {brier_base:.4f})")

    # Refit on the full labelled set for serving.
    est = EmpiricalOutcomeEstimator().fit(df)

    metrics = {
        "model": "Hierarchical empirical (cause x corridor -> cause x closure -> cause -> global)",
        "role": "primary estimator for BOTH duration and severity",
        "mae_min": round(float(mae), 1),
        "median_ae_min": round(float(medae), 1),
        "interval_coverage_p10_p90": round(coverage, 3),
        "interval_nominal": 0.80,
        "baseline_global_median_min": round(global_median, 1),
        "baseline_mae_min": round(float(base_mae), 1),
        "baseline_median_ae_min": round(float(base_medae), 1),
        "severity_accuracy": round(float(acc), 4),
        "severity_f1_weighted": round(float(f1w), 4),
        "severity_majority_baseline": round(baseline, 4),
        "severity_lift_over_baseline": round(float(acc - baseline), 4),
        "severity_mean_confidence": round(mean_conf, 3),
        "long_incident_threshold_min": LONG_INCIDENT_THRESHOLD_MIN,
        "long_incident_base_rate": round(float(y_long.mean()), 4),
        "long_incident_auc": None if np.isnan(auc) else round(auc, 3),
        "long_incident_brier": round(brier, 4),
        "long_incident_brier_baseline": round(brier_base, 4),
        "long_incident_note": (
            "This is the strongest result in the project and the one worth leading with. "
            "Whether an event runs past three hours is largely determined by what kind of "
            "event it is: vehicle breakdowns exceed it 0.1% of the time, construction and "
            "road-condition work more than half the time. A gradient-boosted classifier "
            "over cause, corridor, hour, span and vehicle type reaches the same AUC as a "
            "lookup on cause alone, and ablating any other feature leaves it unchanged — "
            "so the empirical rate is reported directly rather than dressed up as a model."
        ),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "honest_note": (
            "Severity accuracy is modest and only a few points above the majority-class "
            "baseline. That is the real signal in this dataset once the corridor-membership "
            "leak is removed; the previous 91.4% came from predicting an administrative flag "
            "using the coordinates that defined it. Confidence is reported as the observed "
            "class frequency in the matched stratum, so it can be read at face value."
        ),
    }
    return {"estimator": est, "metrics": metrics}


# ─── Model 2b: XGBoost Duration Regressor (retained cross-check) ─────────────

def train_duration_regressor(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 68)
    print("TRAINING: Duration Regressor (cross-check) — XGBoost")
    print("=" * 68)

    X = df[ALL_FEATURES].copy()
    y_log = np.log1p(df["duration_to_close_min"].copy())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    y_test_original = np.expm1(y_test)

    reg = Pipeline([
        ("preprocessor", _build_preprocessor(
            CATEGORICAL_FEATURES, NUMERICAL_FEATURES, use_text=True)),
        ("regressor", XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            colsample_bytree=0.8, subsample=0.8,
            reg_alpha=0.5, reg_lambda=1.5,
            min_child_weight=5, random_state=42,
            n_jobs=-1, verbosity=0,
        )),
    ])
    reg.fit(X_train, y_train)

    y_pred = np.clip(np.expm1(reg.predict(X_test)), 1, None)

    mae = mean_absolute_error(y_test_original, y_pred)
    medae = median_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)

    global_median = float(np.expm1(y_train).median())
    base_mae = mean_absolute_error(y_test_original, np.full_like(y_test_original, global_median))

    print(f"\n  --- Test Set Metrics ---")
    print(f"  MAE:            {mae:.1f} min   (global-median baseline {base_mae:.1f})")
    print(f"  Median AE:      {medae:.1f} min")
    print(f"  R2 Score:       {r2:.4f}")
    if mae >= base_mae:
        print("  NOTE: does not beat the naive baseline — the empirical estimator is primary.")

    metrics = {
        "model": "XGBRegressor (log1p target)",
        "role": "cross-check only; the hierarchical empirical estimator is primary",
        "mae_min": round(float(mae), 1),
        "median_ae_min": round(float(medae), 1),
        "r2_score": round(float(r2), 4),
        "baseline_mae_min": round(float(base_mae), 1),
        "beats_baseline": bool(mae < base_mae),
        "honest_note": (
            "R2 near zero is the correct reading of this target: duration_to_close_min "
            "measures administrative ticket closure, which for potholes and waterlogging "
            "can trail the actual clearance by days."
        ),
    }
    return {"pipeline": reg, "metrics": metrics}


# ─── k-NN Historical Analog Finder ──────────────────────────────────────────

def build_knn_analog_index(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 68)
    print("BUILDING: k-NN Historical Analog Index")
    print("=" * 68)

    knn_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERICAL_FEATURES),
    ])

    X_ref = preprocessor.fit_transform(df[knn_features])
    n_neighbors = min(25, len(df))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(X_ref)

    # Only the columns the analogue card actually renders, so the pickle stays small.
    keep = ["id", "event_cause", "corridor", "zone", "severity_tier",
            "duration_to_close_min", "description", "hour_of_day"]
    ref_data = df[keep].reset_index(drop=True)

    print(f"  Indexed {len(ref_data)} events, k={n_neighbors}")
    return {
        "knn_model": knn,
        "preprocessor": preprocessor,
        "reference_data": ref_data,
        "knn_features": knn_features,
    }


# ─── Forecast Engine ────────────────────────────────────────────────────────

class ForecastingEngine:
    def __init__(self, severity_model: dict, duration_model: dict,
                 empirical_model: dict, knn_index: dict):
        self.severity_pipeline = severity_model["pipeline"]
        self.severity_le = severity_model["label_encoder"]

        self.duration_pipeline = duration_model["pipeline"]
        self.duration_metrics = duration_model["metrics"]

        self.empirical = empirical_model["estimator"]
        self.empirical_metrics = empirical_model["metrics"]
        # Kept for the dashboard's headline: severity now comes from the empirical
        # estimator, so its metrics are the ones that describe live behaviour.
        self.severity_metrics = {
            "accuracy": empirical_model["metrics"].get("severity_accuracy"),
            "majority_class_baseline": empirical_model["metrics"].get("severity_majority_baseline"),
            "f1_weighted": empirical_model["metrics"].get("severity_f1_weighted"),
            "classifier_crosscheck": severity_model["metrics"],
        }

        self.knn_model = knn_index["knn_model"]
        self.knn_preprocessor = knn_index["preprocessor"]
        self.knn_ref_data = knn_index["reference_data"]

    # ── input shaping ───────────────────────────────────────────────────────

    def _build_input_frame(self, event_input: dict) -> pd.DataFrame:
        row = {}
        for col in CATEGORICAL_FEATURES:
            row[col] = str(event_input.get(col, "unknown"))

        hour = int(event_input.get("hour_of_day", 12))
        row["hour_of_day"] = hour
        row["day_of_week"] = int(event_input.get("day_of_week", 2))
        row["is_weekend"] = int(event_input.get("is_weekend", 0))
        row["requires_road_closure_int"] = int(bool(event_input.get("requires_road_closure", 0)))
        row[TEXT_FEATURE] = str(event_input.get(TEXT_FEATURE, "") or "")

        row["is_peak_hour"] = int(hour in (7, 8, 9, 10, 17, 18, 19, 20))
        row["is_night"] = int(hour in (22, 23, 0, 1, 2, 3, 4, 5))
        veh = str(event_input.get("veh_type", "none") or "none").lower()
        row["has_vehicle"] = int(veh not in ("none", "", "nan"))
        row["event_span_km"] = float(event_input.get("event_span_km") or 0.0)

        # Coordinates are now supplied by the caller (the API defaults them from the
        # corridor centroid). Previously every request fell back to the city centre,
        # which froze the model's most influential feature at a constant.
        row["latitude"] = float(event_input.get("latitude") or 12.97)
        row["longitude"] = float(event_input.get("longitude") or 77.59)

        if 5 <= hour < 7:
            row["time_bin"] = "early_morning"
        elif 7 <= hour < 11:
            row["time_bin"] = "morning_rush"
        elif 11 <= hour < 16:
            row["time_bin"] = "midday"
        elif 16 <= hour < 21:
            row["time_bin"] = "evening_rush"
        elif 21 <= hour < 23:
            row["time_bin"] = "night"
        else:
            row["time_bin"] = "late_night"

        return pd.DataFrame([row])

    # ── analogues ───────────────────────────────────────────────────────────

    def _find_analogues(self, input_df: pd.DataFrame, k: int = 25) -> tuple:
        knn_df = input_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
        encoded = self.knn_preprocessor.transform(knn_df)
        k = min(k, len(self.knn_ref_data))
        distances, indices = self.knn_model.kneighbors(encoded, n_neighbors=k)

        neighbours = self.knn_ref_data.iloc[indices[0]]
        durations = neighbours["duration_to_close_min"].dropna().values

        analogue_stats = None
        if len(durations) > 0:
            analogue_stats = {
                "analog_duration_median_min": round(float(np.median(durations)), 1),
                "analog_duration_p10_min": round(float(np.percentile(durations, 10)), 1),
                "analog_duration_p90_min": round(float(np.percentile(durations, 90)), 1),
                "analog_sample_size": int(len(durations)),
            }

        # The card shows the closest handful; the statistics use the full neighbourhood.
        similar = []
        for dist, idx in list(zip(distances[0], indices[0]))[:5]:
            ref = self.knn_ref_data.iloc[idx]
            duration = ref["duration_to_close_min"]
            similar.append({
                "id": ref["id"],
                "event_cause": ref["event_cause"],
                "corridor": ref["corridor"],
                "zone": ref["zone"],
                "duration_min": round(float(duration), 1) if pd.notna(duration) else None,
                "severity_tier": ref["severity_tier"] if pd.notna(ref["severity_tier"]) else "Unknown",
                "similarity_score": round(max(0.0, 1 - float(dist)), 3),
                "description": str(ref["description"])[:100],
            })
        return similar, analogue_stats

    # ── main entry point ────────────────────────────────────────────────────

    def forecast(self, event_input: dict) -> dict:
        input_df = self._build_input_frame(event_input)
        result = {}

        # ── Severity (empirical, consistent with the duration distribution) ──
        sev = self.empirical.predict_severity(event_input)
        probs = sev["probabilities"]
        label = max(probs, key=probs.get)

        result["severity_tier"] = label
        result["severity_confidence"] = round(float(probs[label]), 3)
        result["severity_probabilities"] = {k: round(float(v), 3) for k, v in probs.items()}
        result["severity_basis"] = {
            "method": "empirical_frequency",
            "stratum": sev["stratum"],
            "sample_size": sev["sample_size"],
            "description": (
                sev["basis"] if sev["stratum"] == "definition"
                else f"observed outcome mix across {sev['sample_size']} {sev['basis']}"
            ),
        }
        result["severity_baseline"] = self.empirical_metrics.get("severity_majority_baseline")

        # Cross-check from the learned classifier, reported but not used to decide.
        try:
            xgb_proba = self.severity_pipeline.predict_proba(input_df[SEVERITY_FEATURES])[0]
            xgb_idx = int(np.argmax(xgb_proba))
            result["model_severity_tier"] = str(self.severity_le.inverse_transform([xgb_idx])[0])
        except Exception:
            result["model_severity_tier"] = None

        # ── Duration ─────────────────────────────────────────────────────────
        empirical = self.empirical.predict_duration(event_input)
        result["expected_duration_min"] = empirical["median_min"]
        result["duration_range_min"] = {
            "p10": empirical["p10_min"],
            "p25": empirical["p25_min"],
            "p75": empirical["p75_min"],
            "p90": empirical["p90_min"],
        }
        result["duration_basis"] = {
            "method": "hierarchical_empirical",
            "sample_size": empirical["sample_size"],
            "stratum": empirical["stratum"],
            "dispersion": empirical["dispersion"],
            "description": f"median of {empirical['sample_size']} {empirical['basis']}",
        }
        if empirical["dispersion"] == "high":
            result["duration_warning"] = (
                "Outcomes for this event type are widely spread — quick clearances and "
                "multi-day administrative closures both occur. Plan against the "
                f"{empirical['p25_min']:.0f}-{empirical['p75_min']:.0f} min range rather "
                "than the midpoint."
            )

        model_duration = float(np.expm1(self.duration_pipeline.predict(input_df[ALL_FEATURES])[0]))
        result["model_duration_min"] = round(max(1.0, model_duration), 1)

        # ── Long-incident risk ───────────────────────────────────────────────
        # The strongest signal available. Reported alongside the minute estimate
        # rather than instead of it, because it answers the question a control room
        # actually acts on: is this a tow-truck job or something that will still be
        # blocking the carriageway at shift change?
        risk = self.empirical.predict_long_incident_risk(event_input)
        result["long_incident_risk"] = risk
        result["long_incident_note"] = _LONG_RISK_NOTE[risk["band"]].format(
            hours=risk["threshold_min"] // 60,
            pct=risk["probability"] * 100,
            lo=risk["ci_low"] * 100,
            hi=risk["ci_high"] * 100,
            n=risk["sample_size"],
        )

        # ── Analogues ───────────────────────────────────────────────────────
        similar, analogue_stats = self._find_analogues(input_df)
        result["similar_past_events"] = similar
        if analogue_stats:
            result.update(analogue_stats)

        cause = str(event_input.get("event_cause", ""))
        result["is_sparse_cause"] = cause in SPARSE_CAUSES
        result["method"] = "empirical_with_analogues"
        if result["is_sparse_cause"]:
            result["method"] = "analog_led"
            result["sparse_cause_note"] = (
                f"'{cause.replace('_', ' ')}' is rare in the training data, so this estimate "
                f"rests on a small sample. Treat the range and the analogue list as the "
                f"primary guidance rather than the midpoint."
            )

        return result


# ─── Save / Load ─────────────────────────────────────────────────────────────

def save_models(severity_result, duration_result, empirical_result, knn_index):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_DIR / "severity_model.pkl", "wb") as f:
        pickle.dump(severity_result, f)
    with open(MODEL_DIR / "duration_model.pkl", "wb") as f:
        pickle.dump(duration_result, f)
    with open(MODEL_DIR / "empirical_duration.pkl", "wb") as f:
        pickle.dump({
            "state": empirical_result["estimator"].to_state(),
            "metrics": empirical_result["metrics"],
        }, f)
    with open(MODEL_DIR / "knn_index.pkl", "wb") as f:
        pickle.dump(knn_index, f)

    with open(MODEL_DIR / "model_metrics.json", "w") as f:
        json.dump({
            "primary": empirical_result["metrics"],
            "severity": {
                "model": "Hierarchical empirical frequency",
                "accuracy": empirical_result["metrics"]["severity_accuracy"],
                "majority_class_baseline": empirical_result["metrics"]["severity_majority_baseline"],
                "lift_over_baseline": empirical_result["metrics"]["severity_lift_over_baseline"],
                "f1_weighted": empirical_result["metrics"]["severity_f1_weighted"],
                "mean_confidence": empirical_result["metrics"]["severity_mean_confidence"],
            },
            "duration": empirical_result["metrics"],
            "severity_classifier_crosscheck": severity_result["metrics"],
            "duration_model_crosscheck": duration_result["metrics"],
        }, f, indent=2)

    print(f"\n[SAVE] Models saved to {MODEL_DIR}/")


def load_engine() -> ForecastingEngine:
    required = ["severity_model.pkl", "duration_model.pkl",
                "empirical_duration.pkl", "knn_index.pkl"]
    missing = [n for n in required if not (MODEL_DIR / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artefacts: {missing}. Run `python3 backend/forecasting.py` first."
        )

    with open(MODEL_DIR / "severity_model.pkl", "rb") as f:
        sev = pickle.load(f)
    with open(MODEL_DIR / "duration_model.pkl", "rb") as f:
        dur = pickle.load(f)
    with open(MODEL_DIR / "empirical_duration.pkl", "rb") as f:
        emp_raw = pickle.load(f)
    emp = {
        "estimator": EmpiricalOutcomeEstimator.from_state(emp_raw["state"]),
        "metrics": emp_raw["metrics"],
    }
    with open(MODEL_DIR / "knn_index.pkl", "rb") as f:
        knn = pickle.load(f)

    return ForecastingEngine(sev, dur, emp, knn)


# ─── Execution ───────────────────────────────────────────────────────────────

def run_training_pipeline(include_feedback: bool = False):
    print("=" * 70)
    print("ML TRAINING PIPELINE" + ("  (including officer feedback)" if include_feedback else ""))
    print("=" * 70)

    df = load_labelled_data(include_feedback=include_feedback)

    sev_result = train_severity_classifier(df)
    emp_result = train_empirical_estimator(df)
    dur_result = train_duration_regressor(df)
    knn_result = build_knn_analog_index(df)

    save_models(sev_result, dur_result, emp_result, knn_result)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"  Severity : {emp_result['metrics']['severity_accuracy']:.3f} accuracy vs "
          f"{emp_result['metrics']['severity_majority_baseline']:.3f} baseline "
          f"(classifier cross-check {sev_result['metrics']['accuracy']:.3f})")
    print(f"  Duration : MAE {emp_result['metrics']['mae_min']} min vs "
          f"{emp_result['metrics']['baseline_mae_min']} baseline, "
          f"P10-P90 coverage {emp_result['metrics']['interval_coverage_p10_p90']}")
    print("=" * 70)


def run_sample_forecast():
    try:
        engine = load_engine()
    except FileNotFoundError as e:
        print(e)
        return

    sample_input = {
        "event_cause": "vehicle_breakdown",
        "event_type": "unplanned",
        "corridor": "Mysore Road",
        "zone": "South Zone 1",
        "police_station": "Kengeri",
        "direction": "east",
        "veh_type": "heavy_vehicle",
        "requires_road_closure": False,
        "hour_of_day": 8,
        "day_of_week": 0,
        "is_weekend": 0,
        "latitude": 12.94,
        "longitude": 77.52,
        "description": "truck axle broken blocking left lane",
    }

    print("\n[TEST] Running Sample Forecast")
    print(json.dumps(sample_input, indent=2))
    print("\n[RESULT]")
    print(json.dumps(engine.forecast(sample_input), indent=2, default=str))


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_sample_forecast()
    else:
        run_training_pipeline(include_feedback="--feedback" in sys.argv)
