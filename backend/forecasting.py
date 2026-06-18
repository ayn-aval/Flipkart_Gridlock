"""
Impact Forecasting Engine
=========================
Phase 2: Train models for severity tier classification and duration-to-clear regression.
Includes a k-NN fallback for rare planned events not well represented in training data.

Usage:
    python3 backend/forecasting.py          # Train + evaluate + save models
    python3 backend/forecasting.py --test   # Run sample forecast
"""

import sys
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, median_absolute_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "events_clean.csv"
MODEL_DIR = PROCESSED_DIR / "models"

# ─── Feature Definitions ─────────────────────────────────────────────────────

# Features used for both models
CATEGORICAL_FEATURES = ["event_cause", "corridor", "time_period"]
NUMERICAL_FEATURES = ["hour_of_day", "day_of_week", "is_weekend", "requires_road_closure_int"]
# veh_type is optional (only present for vehicle events)
OPTIONAL_CAT_FEATURES = ["veh_type"]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + OPTIONAL_CAT_FEATURES

# Event-driven causes (for k-NN fallback logic)
EVENT_DRIVEN_CAUSES = ["public_event", "procession", "vip_movement", "protest", "construction"]


# ─── Data Preparation ────────────────────────────────────────────────────────

def load_and_prepare_data() -> pd.DataFrame:
    """Load cleaned data and prepare features for modeling."""
    df = pd.read_csv(CLEAN_CSV)
    
    # Convert boolean to int for modeling
    df["requires_road_closure_int"] = df["requires_road_closure"].astype(int)
    
    # Fill missing veh_type with 'none' (for non-vehicle events)
    df["veh_type"] = df["veh_type"].fillna("none")
    
    # Drop rows missing hour_of_day (these had unparseable start_datetime)
    before = len(df)
    df = df.dropna(subset=["hour_of_day", "day_of_week"])
    after = len(df)
    if before != after:
        print(f"[DATA] Dropped {before - after} rows with missing time features")
    
    # Ensure hour and day are int
    df["hour_of_day"] = df["hour_of_day"].astype(int)
    df["day_of_week"] = df["day_of_week"].astype(int)
    
    print(f"[DATA] Prepared {len(df)} rows with {len(ALL_FEATURES)} features")
    return df


def build_preprocessor():
    """Build a sklearn ColumnTransformer for feature encoding."""
    cat_features = CATEGORICAL_FEATURES + OPTIONAL_CAT_FEATURES
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("num", StandardScaler(), NUMERICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor


# ─── Model 1: Severity Tier Classifier ───────────────────────────────────────

def train_severity_classifier(df: pd.DataFrame) -> dict:
    """Train a Gradient Boosted Classifier for severity tier prediction.
    
    Why GBT over logistic regression:
    - The dataset has 8K rows which is comfortable for GBT
    - Severity depends on non-linear interactions (e.g., construction + road_closure
      is very different from construction alone)
    - GBT handles class imbalance better with its boosting mechanism
    - Feature importance is interpretable
    """
    print("\n" + "=" * 60)
    print("TRAINING: Severity Tier Classifier")
    print("=" * 60)
    
    # Use all rows (severity_tier is available for all)
    X = df[ALL_FEATURES].copy()
    y = df["severity_tier"].copy()
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Class distribution: {dict(zip(le.classes_, np.bincount(y_encoded)))}")
    
    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Build pipeline
    preprocessor = build_preprocessor()
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        ))
    ])
    
    # Train
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    print(f"\n  --- Test Set Metrics ---")
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  F1 (macro):   {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))
    
    # Feature importance (top 15)
    feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
    importances = clf.named_steps["classifier"].feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print(f"  Top 15 Feature Importances:")
    for i in top_idx:
        print(f"    {feature_names[i]:40s} {importances[i]:.4f}")
    
    metrics = {
        "model": "GradientBoostingClassifier",
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "classes": list(le.classes_),
    }
    
    return {
        "pipeline": clf,
        "label_encoder": le,
        "metrics": metrics,
    }


# ─── Model 2: Duration-to-Clear Regressor ────────────────────────────────────

def train_duration_regressor(df: pd.DataFrame) -> dict:
    """Train a Gradient Boosted Regressor for duration-to-clear prediction.
    
    Only uses rows where duration_to_close_min is available (2,711 rows).
    Predicts in minutes. Log-transforms the target to handle right-skew.
    """
    print("\n" + "=" * 60)
    print("TRAINING: Duration-to-Clear Regressor")
    print("=" * 60)
    
    # Filter to rows with valid duration
    df_dur = df[df["duration_to_close_min"].notna()].copy()
    print(f"  Rows with valid duration: {len(df_dur)}")
    
    X = df_dur[ALL_FEATURES].copy()
    y = df_dur["duration_to_close_min"].copy()
    
    # Log-transform target (add 1 to avoid log(0))
    y_log = np.log1p(y)
    
    print(f"  Target stats: median={y.median():.1f}, mean={y.mean():.1f}, "
          f"std={y.std():.1f}, min={y.min():.1f}, max={y.max():.1f}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    y_test_original = np.expm1(y_test)  # back to minutes for metrics
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Build pipeline
    preprocessor = build_preprocessor()
    reg = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        ))
    ])
    
    # Train
    reg.fit(X_train, y_train)
    
    # Evaluate (in original scale)
    y_pred_log = reg.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 1, None)  # Floor at 1 minute
    
    mae = mean_absolute_error(y_test_original, y_pred)
    medae = median_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    
    print(f"\n  --- Test Set Metrics ---")
    print(f"  MAE:            {mae:.1f} min")
    print(f"  Median AE:      {medae:.1f} min")
    print(f"  R² Score:       {r2:.4f}")
    
    # Error by event cause
    test_df = X_test.copy()
    test_df["actual"] = y_test_original.values
    test_df["predicted"] = y_pred
    test_df["abs_error"] = np.abs(test_df["actual"] - test_df["predicted"])
    
    cause_errors = test_df.groupby("event_cause").agg(
        mae=("abs_error", "mean"),
        median_ae=("abs_error", "median"),
        count=("abs_error", "count"),
    ).sort_values("count", ascending=False)
    
    print(f"\n  MAE by Event Cause:")
    for cause, row in cause_errors.iterrows():
        print(f"    {cause:25s} MAE={row['mae']:.1f}min  MedAE={row['median_ae']:.1f}min  (n={int(row['count'])})")
    
    # Feature importance (top 15)
    feature_names = reg.named_steps["preprocessor"].get_feature_names_out()
    importances = reg.named_steps["regressor"].feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print(f"\n  Top 15 Feature Importances:")
    for i in top_idx:
        print(f"    {feature_names[i]:40s} {importances[i]:.4f}")
    
    metrics = {
        "model": "GradientBoostingRegressor",
        "mae_min": round(mae, 1),
        "median_ae_min": round(medae, 1),
        "r2_score": round(r2, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "target_log_transformed": True,
    }
    
    return {
        "pipeline": reg,
        "metrics": metrics,
    }


# ─── k-NN Historical Analog Finder ──────────────────────────────────────────

def build_knn_analog_index(df: pd.DataFrame) -> dict:
    """Build a k-NN index for finding similar historical events.
    
    This is the fallback for planned events (public_event, procession,
    vip_movement, protest) which are poorly represented in the training data.
    Instead of relying on a thin training signal, we surface the most
    similar past events and let the user see what happened historically.
    
    Features for similarity: event_cause, corridor, hour_of_day, day_of_week,
    requires_road_closure, is_weekend.
    """
    print("\n" + "=" * 60)
    print("BUILDING: k-NN Historical Analog Index")
    print("=" * 60)
    
    # Prepare the reference dataset
    knn_features = ["event_cause", "corridor", "hour_of_day",
                     "day_of_week", "requires_road_closure_int", "is_weekend"]
    
    df_ref = df[df["duration_to_close_min"].notna()].copy()
    print(f"  Reference pool: {len(df_ref)} events with known duration")
    
    # Encode categoricals for distance computation
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["event_cause", "corridor"]),
            ("num", StandardScaler(),
             ["hour_of_day", "day_of_week", "requires_road_closure_int", "is_weekend"]),
        ]
    )
    
    X_ref = preprocessor.fit_transform(df_ref[knn_features])
    
    # Build k-NN index
    knn = NearestNeighbors(n_neighbors=min(10, len(df_ref)), metric="cosine")
    knn.fit(X_ref)
    
    # Store reference data for retrieval
    ref_data = df_ref[["id", "event_cause", "corridor", "hour_of_day",
                        "day_of_week", "is_weekend", "requires_road_closure",
                        "duration_to_close_min", "severity_tier", "priority",
                        "description"]].copy()
    ref_data = ref_data.reset_index(drop=True)
    
    print(f"  k-NN index built with {X_ref.shape[1]} features")
    
    return {
        "knn_model": knn,
        "preprocessor": preprocessor,
        "reference_data": ref_data,
        "knn_features": knn_features,
    }


# ─── Forecast Function ──────────────────────────────────────────────────────

class ForecastingEngine:
    """Main forecasting engine that combines:
    - GBT severity classifier
    - GBT duration regressor
    - k-NN historical analog finder (fallback for rare planned events)
    """
    
    def __init__(self, severity_model: dict, duration_model: dict, knn_index: dict):
        self.severity_pipeline = severity_model["pipeline"]
        self.severity_le = severity_model["label_encoder"]
        self.severity_metrics = severity_model["metrics"]
        
        self.duration_pipeline = duration_model["pipeline"]
        self.duration_metrics = duration_model["metrics"]
        
        self.knn_model = knn_index["knn_model"]
        self.knn_preprocessor = knn_index["preprocessor"]
        self.knn_ref_data = knn_index["reference_data"]
        self.knn_features = knn_index["knn_features"]
    
    def forecast(self, event_input: dict) -> dict:
        """
        Given a new/planned event description, return forecast + similar past events.
        
        Args:
            event_input: dict with keys matching ALL_FEATURES, e.g.:
                {
                    "event_cause": "procession",
                    "corridor": "Bellary Road 1",
                    "time_period": "morning_rush",
                    "hour_of_day": 8,
                    "day_of_week": 2,
                    "is_weekend": 0,
                    "requires_road_closure": 1,
                    "veh_type": "none",
                }
        
        Returns:
            {
                "severity_tier": "High",
                "severity_confidence": 0.78,
                "severity_probabilities": {"Low": 0.1, "Medium": 0.12, "High": 0.78},
                "expected_duration_min": 85.3,
                "method": "model" or "knn_analog_fallback",
                "similar_past_events": [...],
                "model_accuracy_note": "...",
            }
        """
        # Prepare input DataFrame
        input_df = pd.DataFrame([event_input])
        
        # Ensure correct types
        input_df["requires_road_closure_int"] = int(event_input.get("requires_road_closure", 0))
        input_df["veh_type"] = event_input.get("veh_type", "none")
        input_df["hour_of_day"] = int(event_input.get("hour_of_day", 12))
        input_df["day_of_week"] = int(event_input.get("day_of_week", 2))
        input_df["is_weekend"] = int(event_input.get("is_weekend", 0))
        
        result = {}
        
        # --- Severity prediction ---
        severity_pred = self.severity_pipeline.predict(input_df[ALL_FEATURES])[0]
        severity_proba = self.severity_pipeline.predict_proba(input_df[ALL_FEATURES])[0]
        severity_label = self.severity_le.inverse_transform([severity_pred])[0]
        
        proba_dict = {
            cls: round(float(p), 3)
            for cls, p in zip(self.severity_le.classes_, severity_proba)
        }
        
        result["severity_tier"] = severity_label
        result["severity_confidence"] = round(float(max(severity_proba)), 3)
        result["severity_probabilities"] = proba_dict
        
        # --- Duration prediction ---
        dur_pred_log = self.duration_pipeline.predict(input_df[ALL_FEATURES])[0]
        dur_pred = float(np.expm1(dur_pred_log))
        dur_pred = max(1.0, dur_pred)  # Floor at 1 minute
        result["expected_duration_min"] = round(dur_pred, 1)
        
        # --- Determine method ---
        cause = event_input.get("event_cause", "")
        is_rare_planned = cause in EVENT_DRIVEN_CAUSES and cause != "construction"
        # construction has enough data (480 rows); others are thin
        
        if is_rare_planned:
            result["method"] = "knn_analog_fallback"
            result["model_accuracy_note"] = (
                f"'{cause}' has very few historical examples. "
                f"The model prediction is supplemented with similar historical events "
                f"(k-NN analog lookup) for more reliable estimation. "
                f"Treat the model duration estimate as rough — the similar events "
                f"below provide better context."
            )
        else:
            result["method"] = "model"
            result["model_accuracy_note"] = (
                f"Severity model: accuracy={self.severity_metrics['accuracy']}, "
                f"F1(weighted)={self.severity_metrics['f1_weighted']}. "
                f"Duration model: MAE={self.duration_metrics['mae_min']}min, "
                f"R²={self.duration_metrics['r2_score']}."
            )
        
        # --- k-NN similar events ---
        knn_input = {
            "event_cause": event_input.get("event_cause", "others"),
            "corridor": event_input.get("corridor", "Non-corridor"),
            "hour_of_day": int(event_input.get("hour_of_day", 12)),
            "day_of_week": int(event_input.get("day_of_week", 2)),
            "requires_road_closure_int": int(event_input.get("requires_road_closure", 0)),
            "is_weekend": int(event_input.get("is_weekend", 0)),
        }
        knn_df = pd.DataFrame([knn_input])
        knn_encoded = self.knn_preprocessor.transform(knn_df)
        
        k = min(5, len(self.knn_ref_data))
        distances, indices = self.knn_model.kneighbors(knn_encoded, n_neighbors=k)
        
        similar_events = []
        for dist, idx in zip(distances[0], indices[0]):
            ref_row = self.knn_ref_data.iloc[idx]
            similar_events.append({
                "id": ref_row["id"],
                "event_cause": ref_row["event_cause"],
                "corridor": ref_row["corridor"],
                "hour_of_day": int(ref_row["hour_of_day"]) if pd.notna(ref_row["hour_of_day"]) else None,
                "day_of_week": int(ref_row["day_of_week"]) if pd.notna(ref_row["day_of_week"]) else None,
                "duration_min": round(float(ref_row["duration_to_close_min"]), 1),
                "severity_tier": ref_row["severity_tier"],
                "requires_road_closure": bool(ref_row["requires_road_closure"]),
                "similarity_score": round(1 - float(dist), 3),
                "description": str(ref_row["description"])[:100] if pd.notna(ref_row["description"]) else None,
            })
        
        result["similar_past_events"] = similar_events
        
        # For k-NN fallback, also compute analog-based duration estimate
        if is_rare_planned and similar_events:
            analog_durations = [e["duration_min"] for e in similar_events]
            result["analog_duration_median_min"] = round(float(np.median(analog_durations)), 1)
            result["analog_duration_range_min"] = [
                round(float(min(analog_durations)), 1),
                round(float(max(analog_durations)), 1),
            ]
        
        return result


# ─── Save / Load ─────────────────────────────────────────────────────────────

def save_models(engine: ForecastingEngine, severity_result: dict, duration_result: dict):
    """Save trained model components and metrics.
    
    We save individual components rather than the full ForecastingEngine object
    to avoid pickle module-reference issues (__main__ vs backend.forecasting).
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save individual components (avoids __main__ pickle issue)
    components = {
        "severity_pipeline": engine.severity_pipeline,
        "severity_le": engine.severity_le,
        "severity_metrics": engine.severity_metrics,
        "duration_pipeline": engine.duration_pipeline,
        "duration_metrics": engine.duration_metrics,
        "knn_model": engine.knn_model,
        "knn_preprocessor": engine.knn_preprocessor,
        "knn_ref_data": engine.knn_ref_data,
        "knn_features": engine.knn_features,
    }
    with open(MODEL_DIR / "forecasting_engine.pkl", "wb") as f:
        pickle.dump(components, f)
    
    # Save metrics as JSON for easy inspection
    metrics = {
        "severity_classifier": severity_result["metrics"],
        "duration_regressor": duration_result["metrics"],
    }
    with open(MODEL_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[SAVE] Models saved to {MODEL_DIR}/")
    print(f"  forecasting_engine.pkl")
    print(f"  model_metrics.json")


def load_engine() -> ForecastingEngine:
    """Load a saved forecasting engine from component dict."""
    with open(MODEL_DIR / "forecasting_engine.pkl", "rb") as f:
        components = pickle.load(f)
    
    # Reconstruct the engine from components
    severity_model = {
        "pipeline": components["severity_pipeline"],
        "label_encoder": components["severity_le"],
        "metrics": components["severity_metrics"],
    }
    duration_model = {
        "pipeline": components["duration_pipeline"],
        "metrics": components["duration_metrics"],
    }
    knn_index = {
        "knn_model": components["knn_model"],
        "preprocessor": components["knn_preprocessor"],
        "reference_data": components["knn_ref_data"],
        "knn_features": components["knn_features"],
    }
    
    return ForecastingEngine(severity_model, duration_model, knn_index)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if "--test" in sys.argv:
        # Load saved engine and run sample forecasts
        run_sample_forecasts()
        return
    
    print("=" * 70)
    print("PHASE 2: Impact Forecasting Engine")
    print("=" * 70)
    
    # Load data
    df = load_and_prepare_data()
    
    # Train models
    severity_result = train_severity_classifier(df)
    duration_result = train_duration_regressor(df)
    knn_index = build_knn_analog_index(df)
    
    # Build engine
    engine = ForecastingEngine(severity_result, duration_result, knn_index)
    
    # Save
    save_models(engine, severity_result, duration_result)
    
    # Run sample forecasts
    print("\n" + "=" * 70)
    print("SAMPLE FORECASTS")
    print("=" * 70)
    run_forecasts_with_engine(engine)


def run_sample_forecasts():
    """Load engine and run sample forecasts."""
    print("Loading saved forecasting engine...")
    engine = load_engine()
    run_forecasts_with_engine(engine)


def run_forecasts_with_engine(engine: ForecastingEngine):
    """Run sample forecasts on representative scenarios."""
    
    test_events = [
        {
            "name": "1. Vehicle breakdown on Mysore Road (common event)",
            "input": {
                "event_cause": "vehicle_breakdown",
                "corridor": "Mysore Road",
                "time_period": "morning_rush",
                "hour_of_day": 8,
                "day_of_week": 1,  # Tuesday
                "is_weekend": 0,
                "requires_road_closure": 0,
                "veh_type": "heavy_vehicle",
            }
        },
        {
            "name": "2. Planned procession on Bellary Road (rare planned event)",
            "input": {
                "event_cause": "procession",
                "corridor": "Bellary Road 1",
                "time_period": "morning_rush",
                "hour_of_day": 9,
                "day_of_week": 6,  # Sunday
                "is_weekend": 1,
                "requires_road_closure": 1,
                "veh_type": "none",
            }
        },
        {
            "name": "3. Construction on ORR East (common planned event)",
            "input": {
                "event_cause": "construction",
                "corridor": "ORR East 2",
                "time_period": "night",
                "hour_of_day": 23,
                "day_of_week": 3,  # Thursday
                "is_weekend": 0,
                "requires_road_closure": 1,
                "veh_type": "none",
            }
        },
        {
            "name": "4. Public event (rally) in CBD — peak hours",
            "input": {
                "event_cause": "public_event",
                "corridor": "CBD 2",
                "time_period": "midday",
                "hour_of_day": 11,
                "day_of_week": 5,  # Saturday
                "is_weekend": 1,
                "requires_road_closure": 1,
                "veh_type": "none",
            }
        },
        {
            "name": "5. Water logging on Hosur Road (monsoon scenario)",
            "input": {
                "event_cause": "water_logging",
                "corridor": "Hosur Road",
                "time_period": "evening_rush",
                "hour_of_day": 17,
                "day_of_week": 4,  # Friday
                "is_weekend": 0,
                "requires_road_closure": 0,
                "veh_type": "none",
            }
        },
    ]
    
    for test in test_events:
        print(f"\n{'─' * 60}")
        print(f"  Scenario: {test['name']}")
        print(f"{'─' * 60}")
        
        result = engine.forecast(test["input"])
        
        print(f"  Severity Tier:     {result['severity_tier']} "
              f"(confidence: {result['severity_confidence']:.1%})")
        print(f"  Probabilities:     {result['severity_probabilities']}")
        print(f"  Expected Duration: {result['expected_duration_min']} min")
        print(f"  Method:            {result['method']}")
        
        if "analog_duration_median_min" in result:
            print(f"  Analog Duration:   median={result['analog_duration_median_min']}min, "
                  f"range={result['analog_duration_range_min']}")
        
        print(f"  Note: {result['model_accuracy_note']}")
        
        print(f"  Similar Past Events:")
        for i, event in enumerate(result["similar_past_events"][:3], 1):
            print(f"    [{i}] {event['event_cause']} on {event['corridor']} — "
                  f"lasted {event['duration_min']}min (severity: {event['severity_tier']}, "
                  f"similarity: {event['similarity_score']:.2f})")


if __name__ == "__main__":
    main()
