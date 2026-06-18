"""
Impact Forecasting Engine
=========================
Phase 9: Real Astram Data Integration.
Trains models for High/Low severity classification and duration-to-clear regression.
Includes TF-IDF NLP text processing for descriptions and k-NN fallback.

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
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "events_clean.csv"
MODEL_DIR = PROCESSED_DIR / "models"

# ─── Feature Definitions ─────────────────────────────────────────────────────

CATEGORICAL_FEATURES = [
    "event_cause", "event_type", "corridor", "zone", 
    "police_station", "direction", "veh_type"
]
NUMERICAL_FEATURES = ["hour_of_day", "day_of_week", "is_weekend", "requires_road_closure_int"]
TEXT_FEATURE = "description"

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TEXT_FEATURE]

# Event-driven causes (for k-NN fallback logic)
EVENT_DRIVEN_CAUSES = ["public_event", "procession", "vip_movement", "protest", "construction", "planned"]

# ─── Data Preparation ────────────────────────────────────────────────────────

def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_CSV)
    
    df["requires_road_closure_int"] = df["requires_road_closure"].astype(int)
    
    # Fill missing with safe defaults
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)
        
    df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna("").astype(str)
    
    # Drop rows missing critical numeric features
    before = len(df)
    df = df.dropna(subset=["hour_of_day", "day_of_week", "severity_tier", "duration_to_close_min"])
    after = len(df)
    if before != after:
        print(f"[DATA] Dropped {before - after} rows with missing numeric targets/features")
    
    df["hour_of_day"] = df["hour_of_day"].astype(int)
    df["day_of_week"] = df["day_of_week"].astype(int)
    
    print(f"[DATA] Prepared {len(df)} rows with {len(ALL_FEATURES)} features")
    return df

def build_preprocessor():
    """Build a sklearn ColumnTransformer for feature encoding & NLP."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("text", TfidfVectorizer(max_features=300, min_df=2, stop_words="english"), TEXT_FEATURE)
        ],
        remainder="drop"
    )
    return preprocessor

# ─── Model 1: Severity Tier Classifier ───────────────────────────────────────

def train_severity_classifier(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("TRAINING: Severity Classifier (High/Low)")
    print("=" * 60)
    
    # Drop 'corridor' from training to prevent data leakage, as it administratively dictates Priority
    severity_features = [f for f in ALL_FEATURES if f != "corridor"]
    
    X = df[severity_features].copy()
    y = df["severity_tier"].copy()
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Class distribution: {dict(zip(le.classes_, np.bincount(y_encoded)))}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Build preprocessor dynamically without corridor
    cat_features_sev = [f for f in CATEGORICAL_FEATURES if f != "corridor"]
    preprocessor_sev = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features_sev),
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("text", TfidfVectorizer(max_features=300, min_df=2, stop_words="english"), TEXT_FEATURE)
        ],
        remainder="drop"
    )
    
    clf = Pipeline([
        ("preprocessor", preprocessor_sev),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, 
            min_samples_leaf=5, subsample=0.8, random_state=42
        ))
    ])
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    print(f"\n  --- Test Set Metrics ---")
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))
    
    metrics = {
        "model": "GradientBoostingClassifier",
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1_weighted, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "classes": list(le.classes_),
    }
    
    return {"pipeline": clf, "label_encoder": le, "metrics": metrics}

# ─── Model 2: Duration-to-Clear Regressor ────────────────────────────────────

def train_duration_regressor(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("TRAINING: Duration-to-Clear Regressor")
    print("=" * 60)
    
    X = df[ALL_FEATURES].copy()
    y = df["duration_to_close_min"].copy()
    
    y_log = np.log1p(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    y_test_original = np.expm1(y_test)
    
    preprocessor = build_preprocessor()
    reg = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, 
            min_samples_leaf=5, subsample=0.8, random_state=42
        ))
    ])
    
    reg.fit(X_train, y_train)
    
    y_pred = np.clip(np.expm1(reg.predict(X_test)), 1, None)
    
    mae = mean_absolute_error(y_test_original, y_pred)
    medae = median_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    
    print(f"\n  --- Test Set Metrics ---")
    print(f"  MAE:            {mae:.1f} min")
    print(f"  Median AE:      {medae:.1f} min")
    print(f"  R² Score:       {r2:.4f}")
    
    metrics = {
        "model": "GradientBoostingRegressor",
        "mae_min": round(mae, 1),
        "median_ae_min": round(medae, 1),
        "r2_score": round(r2, 4),
    }
    
    return {"pipeline": reg, "metrics": metrics}

# ─── k-NN Historical Analog Finder ──────────────────────────────────────────

def build_knn_analog_index(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("BUILDING: k-NN Historical Analog Index")
    print("=" * 60)
    
    knn_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    df_ref = df.copy()
    
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERICAL_FEATURES),
    ])
    
    X_ref = preprocessor.fit_transform(df_ref[knn_features])
    knn = NearestNeighbors(n_neighbors=min(10, len(df_ref)), metric="cosine")
    knn.fit(X_ref)
    
    ref_data = df_ref.reset_index(drop=True)
    return {
        "knn_model": knn,
        "preprocessor": preprocessor,
        "reference_data": ref_data,
        "knn_features": knn_features,
    }

# ─── Forecast Function ──────────────────────────────────────────────────────

class ForecastingEngine:
    def __init__(self, severity_model: dict, duration_model: dict, knn_index: dict):
        self.severity_pipeline = severity_model["pipeline"]
        self.severity_le = severity_model["label_encoder"]
        self.severity_metrics = severity_model["metrics"]
        
        self.duration_pipeline = duration_model["pipeline"]
        self.duration_metrics = duration_model["metrics"]
        
        self.knn_model = knn_index["knn_model"]
        self.knn_preprocessor = knn_index["preprocessor"]
        self.knn_ref_data = knn_index["reference_data"]
    
    def forecast(self, event_input: dict) -> dict:
        input_df = pd.DataFrame([event_input])
        
        # Ensure exact schema and defaults
        for col in CATEGORICAL_FEATURES:
            input_df[col] = str(event_input.get(col, "unknown"))
            
        input_df["requires_road_closure_int"] = int(event_input.get("requires_road_closure", 0))
        input_df["hour_of_day"] = int(event_input.get("hour_of_day", 12))
        input_df["day_of_week"] = int(event_input.get("day_of_week", 2))
        input_df["is_weekend"] = int(event_input.get("is_weekend", 0))
        input_df[TEXT_FEATURE] = str(event_input.get(TEXT_FEATURE, ""))
        
        result = {}
        
        # Severity
        severity_features = [f for f in ALL_FEATURES if f != "corridor"]
        severity_pred = self.severity_pipeline.predict(input_df[severity_features])[0]
        severity_proba = self.severity_pipeline.predict_proba(input_df[severity_features])[0]
        severity_label = self.severity_le.inverse_transform([severity_pred])[0]
        
        result["severity_tier"] = severity_label
        result["severity_confidence"] = round(float(max(severity_proba)), 3)
        result["severity_probabilities"] = {
            cls: round(float(p), 3) for cls, p in zip(self.severity_le.classes_, severity_proba)
        }
        
        # Duration
        dur_pred_log = self.duration_pipeline.predict(input_df[ALL_FEATURES])[0]
        dur_pred = max(1.0, float(np.expm1(dur_pred_log)))
        result["expected_duration_min"] = round(dur_pred, 1)
        
        # Method / KNN Fallback
        cause = event_input.get("event_cause", "")
        if cause in EVENT_DRIVEN_CAUSES:
            result["method"] = "knn_analog_fallback"
        else:
            result["method"] = "model"
            
        # KNN Lookup
        knn_df = input_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
        knn_encoded = self.knn_preprocessor.transform(knn_df)
        distances, indices = self.knn_model.kneighbors(knn_encoded, n_neighbors=min(5, len(self.knn_ref_data)))
        
        similar_events = []
        for dist, idx in zip(distances[0], indices[0]):
            ref_row = self.knn_ref_data.iloc[idx]
            similar_events.append({
                "id": ref_row["id"],
                "event_cause": ref_row["event_cause"],
                "corridor": ref_row["corridor"],
                "zone": ref_row["zone"],
                "duration_min": round(float(ref_row["duration_to_close_min"]), 1),
                "severity_tier": ref_row["severity_tier"],
                "similarity_score": round(1 - float(dist), 3),
                "description": str(ref_row["description"])[:100],
            })
        result["similar_past_events"] = similar_events
        
        return result

# ─── Save / Load ─────────────────────────────────────────────────────────────

def save_models(engine: ForecastingEngine, severity_result: dict, duration_result: dict):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(MODEL_DIR / "severity_model.pkl", "wb") as f:
        pickle.dump(severity_result, f)
        
    with open(MODEL_DIR / "duration_model.pkl", "wb") as f:
        pickle.dump(duration_result, f)
        
    knn_index = {
        "knn_model": engine.knn_model,
        "preprocessor": engine.knn_preprocessor,
        "reference_data": engine.knn_ref_data,
        "knn_features": CATEGORICAL_FEATURES + NUMERICAL_FEATURES,
    }
    with open(MODEL_DIR / "knn_index.pkl", "wb") as f:
        pickle.dump(knn_index, f)
        
    with open(MODEL_DIR / "model_metrics.json", "w") as f:
        json.dump({
            "severity": severity_result["metrics"],
            "duration": duration_result["metrics"]
        }, f, indent=2)
        
    print(f"\n[SAVE] Models saved to {MODEL_DIR}/")

def load_engine() -> ForecastingEngine:
    if not (MODEL_DIR / "severity_model.pkl").exists():
        raise FileNotFoundError("Models not found. Run forecasting.py first.")
        
    with open(MODEL_DIR / "severity_model.pkl", "rb") as f:
        sev = pickle.load(f)
    with open(MODEL_DIR / "duration_model.pkl", "rb") as f:
        dur = pickle.load(f)
    with open(MODEL_DIR / "knn_index.pkl", "rb") as f:
        knn = pickle.load(f)
        
    return ForecastingEngine(sev, dur, knn)

# ─── Execution ───────────────────────────────────────────────────────────────

def run_training_pipeline():
    print("=" * 70)
    print("PHASE 9: ML Model Training Pipeline (Real Data + NLP)")
    print("=" * 70)
    
    df = load_and_prepare_data()
    
    sev_result = train_severity_classifier(df)
    dur_result = train_duration_regressor(df)
    knn_result = build_knn_analog_index(df)
    
    engine = ForecastingEngine(sev_result, dur_result, knn_result)
    save_models(engine, sev_result, dur_result)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
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
        "day_of_week": 0, # Monday
        "is_weekend": 0,
        "description": "truck axle broken blocking left lane"
    }
    
    print("\n[TEST] Running Sample Forecast")
    print(json.dumps(sample_input, indent=2))
    
    result = engine.forecast(sample_input)
    print("\n[RESULT]")
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_sample_forecast()
    else:
        run_training_pipeline()
