"""
Data Cleaning & Feature Engineering Pipeline
=============================================
Phase 9: Loads real Astram CSV, cleans data, extracts extended features
(zone, police_station, direction, description NLP prep).

Usage:
    python3 backend/data_cleaning.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "astram_events.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "events_clean.csv"
CORRIDOR_CENTROIDS_CSV = PROCESSED_DIR / "corridor_centroids.csv"
CORRIDOR_ADJACENCY_CSV = PROCESSED_DIR / "corridor_adjacency.csv"
EDA_DIR = PROCESSED_DIR / "eda_charts"

# ─── Step 1: Load & Basic Cleaning ───────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df = df.replace(["NULL", "null", ""], np.nan)
    print(f"[LOAD] {len(df)} rows × {len(df.columns)} columns loaded from real Astram data")
    return df

def clean_event_cause(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cause_map = {
        "Debris": "debris",
        "Fog / Low Visibility": "fog_low_visibility",
    }
    df["event_cause"] = df["event_cause"].replace(cause_map)
    # Fill nulls in event cause
    df["event_cause"] = df["event_cause"].fillna("others")
    return df

def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt_cols = ["start_datetime", "end_datetime", "closed_datetime", "resolved_datetime"]
    for col in dt_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df

def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["latitude", "longitude", "endlatitude", "endlongitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == 0, col] = np.nan
    return df

def clean_boolean_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # requires_road_closure
    if "requires_road_closure" in df.columns:
        if df["requires_road_closure"].dtype == object:
            df["requires_road_closure"] = df["requires_road_closure"].map(
                {"TRUE": True, "FALSE": False, "True": True, "False": False, "yes": True, "no": False}
            ).fillna(False)
        df["requires_road_closure"] = df["requires_road_closure"].astype(bool)
        
    # priority (Target for Severity)
    df["priority"] = df["priority"].fillna("Low")
    
    # Corridor, Zone, Police Station, Direction
    df["corridor"] = df["corridor"].fillna("Non-corridor")
    df["zone"] = df["zone"].fillna("Unknown")
    df["police_station"] = df["police_station"].fillna("Unknown")
    df["direction"] = df["direction"].fillna("unknown")
    df["veh_type"] = df["veh_type"].fillna("none")
    df["event_type"] = df["event_type"].fillna("unplanned")
    
    # Description (for NLP)
    df["description"] = df["description"].fillna("").astype(str).str.lower()
    
    return df

# ─── Step 2: Feature Engineering ──────────────────────────────────────────────

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Drop rows without start_datetime
    df = df.dropna(subset=["start_datetime"])
    
    start_ist = df["start_datetime"].dt.tz_convert("Asia/Kolkata")
    df["hour_of_day"] = start_ist.dt.hour
    df["day_of_week"] = start_ist.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = start_ist.dt.month
    
    return df

def engineer_duration_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # We use closed_datetime or resolved_datetime
    end_time = df["closed_datetime"].combine_first(df["resolved_datetime"])
    
    duration = (end_time - df["start_datetime"]).dt.total_seconds() / 60
    
    MAX_DURATION_MIN = 10080  # 7 days
    valid_mask = (duration > 0) & (duration <= MAX_DURATION_MIN)
    df["duration_to_close_min"] = np.where(valid_mask, duration, np.nan)
    
    valid_count = df["duration_to_close_min"].notna().sum()
    print(f"[FEAT] duration_to_close_min: {valid_count} valid values")
    
    return df

def engineer_severity_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Map priority directly to severity_tier as requested."""
    df = df.copy()
    df["severity_tier"] = df["priority"]
    return df

# ─── Step 3: Corridor Centroids & Adjacency ─────────────────────────────────

def compute_corridor_centroids(df: pd.DataFrame) -> pd.DataFrame:
    centroids = df.groupby("corridor").agg(
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean"),
        event_count=("id", "count"),
    ).reset_index()
    
    centroids.to_csv(CORRIDOR_CENTROIDS_CSV, index=False)
    return centroids

# ─── Step 4: Select Final Columns ─────────────────────────────────────────────

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "id", "event_type", "event_cause", "status",
        "latitude", "longitude", 
        "corridor", "zone", "police_station", "direction",
        "start_datetime", "hour_of_day", "day_of_week", "is_weekend", "month",
        "requires_road_closure", "priority", "severity_tier", "veh_type",
        "description", "duration_to_close_min"
    ]
    
    actual_cols = [c for c in keep_cols if c in df.columns]
    df = df[actual_cols]
    
    # Drop rows without duration (can't train models on them)
    df = df.dropna(subset=["duration_to_close_min"])
    
    print(f"[SELECT] Keeping {len(actual_cols)} columns. Final shape: {df.shape}")
    return df

# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 70)
    print("PHASE 9: Data Cleaning & Feature Engineering (Real Data)")
    print("=" * 70)
    
    df = load_raw_data()
    
    print("\n--- Cleaning ---")
    df = clean_event_cause(df)
    df = parse_datetimes(df)
    df = clean_coordinates(df)
    df = clean_boolean_categorical(df)
    
    print("\n--- Feature Engineering ---")
    df = engineer_time_features(df)
    df = engineer_duration_feature(df)
    df = engineer_severity_tier(df)
    
    print("\n--- Corridor Geography ---")
    compute_corridor_centroids(df)
    
    print("\n--- Column Selection ---")
    df = select_columns(df)
    
    print("\n--- Save ---")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"[SAVE] Cleaned dataset: {CLEAN_CSV}")
    
    print("=" * 70)
    return df

if __name__ == "__main__":
    run_pipeline()
