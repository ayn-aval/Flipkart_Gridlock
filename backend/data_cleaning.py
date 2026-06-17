"""
Data Cleaning & Feature Engineering Pipeline
=============================================
Phase 1: Loads raw CSV, cleans data, engineers features,
computes corridor centroids/adjacency, produces EDA charts.

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
    """Load raw CSV and replace string 'NULL' with NaN."""
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df = df.replace("NULL", np.nan)
    print(f"[LOAD] {len(df)} rows × {len(df.columns)} columns loaded")
    return df


def clean_event_cause(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize event_cause: fix case inconsistencies, merge rare categories."""
    df = df.copy()
    # Fix case: 'Debris' -> 'debris', 'Fog / Low Visibility' -> 'fog_low_visibility'
    cause_map = {
        "Debris": "debris",
        "Fog / Low Visibility": "fog_low_visibility",
    }
    df["event_cause"] = df["event_cause"].replace(cause_map)
    
    # Merge ultra-rare categories (< 5 occurrences) into 'others'
    cause_counts = df["event_cause"].value_counts()
    rare_causes = cause_counts[cause_counts < 5].index.tolist()
    if rare_causes:
        print(f"[CLEAN] Merging rare event_cause values into 'others': {rare_causes}")
        df.loc[df["event_cause"].isin(rare_causes), "event_cause"] = "others"
    
    return df


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse all datetime columns to proper datetime types (UTC)."""
    df = df.copy()
    dt_cols = [
        "start_datetime", "end_datetime", "closed_datetime",
        "resolved_datetime", "created_date", "modified_datetime"
    ]
    for col in dt_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            valid = df[col].notna().sum()
            print(f"[PARSE] {col}: {valid} valid datetimes")
    return df


def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Clean lat/lon: replace 0 with NaN for endlatitude/endlongitude,
    filter out obviously invalid endlatitude/endlongitude values."""
    df = df.copy()
    
    # endlatitude/endlongitude: 0 means missing
    for col in ["endlatitude", "endlongitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == 0, col] = np.nan
    
    # Filter out invalid end coordinates (e.g. endlatitude ~59.86, endlongitude ~62.7)
    # Valid Bengaluru range: lat 12.5-13.5, lon 77.0-78.0
    if "endlatitude" in df.columns:
        invalid_mask = (
            (df["endlatitude"].notna()) & 
            ((df["endlatitude"] < 12.0) | (df["endlatitude"] > 14.0) |
             (df["endlongitude"] < 76.0) | (df["endlongitude"] > 79.0))
        )
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            print(f"[CLEAN] Nullifying {invalid_count} rows with out-of-range end coordinates")
            df.loc[invalid_mask, ["endlatitude", "endlongitude"]] = np.nan
    
    return df


def clean_requires_road_closure(df: pd.DataFrame) -> pd.DataFrame:
    """Convert requires_road_closure to proper boolean."""
    df = df.copy()
    # It may be string 'TRUE'/'FALSE' or already bool
    if df["requires_road_closure"].dtype == object:
        df["requires_road_closure"] = df["requires_road_closure"].map(
            {"TRUE": True, "FALSE": False, "True": True, "False": False,
             True: True, False: False}
        )
    df["requires_road_closure"] = df["requires_road_closure"].astype(bool)
    return df


def clean_corridor(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing corridor values with 'Non-corridor'."""
    df = df.copy()
    null_count = df["corridor"].isna().sum()
    if null_count > 0:
        print(f"[CLEAN] Filling {null_count} null corridor values with 'Non-corridor'")
        df["corridor"] = df["corridor"].fillna("Non-corridor")
    return df


def clean_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing priority values with mode (High)."""
    df = df.copy()
    null_count = df["priority"].isna().sum()
    if null_count > 0:
        mode_val = df["priority"].mode()[0]
        print(f"[CLEAN] Filling {null_count} null priority values with '{mode_val}'")
        df["priority"] = df["priority"].fillna(mode_val)
    return df


# ─── Step 2: Feature Engineering ──────────────────────────────────────────────

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive time-based features from start_datetime (IST)."""
    df = df.copy()
    
    # Convert to IST for feature extraction
    start_ist = df["start_datetime"].dt.tz_convert("Asia/Kolkata")
    
    df["hour_of_day"] = start_ist.dt.hour
    df["day_of_week"] = start_ist.dt.dayofweek  # 0=Monday, 6=Sunday
    df["day_name"] = start_ist.dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = start_ist.dt.month
    df["date"] = start_ist.dt.date
    
    # Time period buckets
    def time_period(h):
        if 6 <= h < 10:
            return "morning_rush"
        elif 10 <= h < 16:
            return "midday"
        elif 16 <= h < 21:
            return "evening_rush"
        else:
            return "night"
    
    df["time_period"] = df["hour_of_day"].apply(time_period)
    
    print(f"[FEAT] Time features: hour_of_day, day_of_week, day_name, is_weekend, month, date, time_period")
    return df


def engineer_duration_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Compute duration_to_close_min with outlier filtering."""
    df = df.copy()
    
    # Compute raw duration in minutes
    duration = (df["closed_datetime"] - df["start_datetime"]).dt.total_seconds() / 60
    
    # Apply filters: must be positive and <= 7 days (10080 min)
    # The prompt's example values (e.g. median 41 min for vehicle_breakdown)
    # suggest most real resolutions are within hours, not days.
    # Records > 7 days are likely stale admin closures.
    MAX_DURATION_MIN = 10080  # 7 days
    
    valid_mask = (duration > 0) & (duration <= MAX_DURATION_MIN)
    df["duration_to_close_min"] = np.where(valid_mask, duration, np.nan)
    
    valid_count = df["duration_to_close_min"].notna().sum()
    print(f"[FEAT] duration_to_close_min: {valid_count} valid values "
          f"(filtered: >0 and ≤{MAX_DURATION_MIN} min)")
    print(f"       Median: {df['duration_to_close_min'].median():.1f} min, "
          f"Mean: {df['duration_to_close_min'].mean():.1f} min")
    
    return df


def engineer_severity_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a severity tier from priority + duration buckets.
    
    Tiers:
    - Low:    priority=Low AND duration <= 60 min (or missing)
    - Medium: priority=High AND duration <= 120 min, OR priority=Low AND duration > 60 min
    - High:   priority=High AND (duration > 120 min OR requires_road_closure=True)
    """
    df = df.copy()
    
    dur = df["duration_to_close_min"].fillna(0)
    priority = df["priority"]
    road_closure = df["requires_road_closure"]
    
    conditions = [
        # High severity
        (priority == "High") & ((dur > 120) | road_closure),
        # Medium severity
        ((priority == "High") & (dur <= 120) & (dur > 0)) |
        ((priority == "Low") & (dur > 60)),
        # Low severity (everything else)
    ]
    choices = ["High", "Medium"]
    
    df["severity_tier"] = np.select(conditions, choices, default="Low")
    
    tier_counts = df["severity_tier"].value_counts()
    print(f"[FEAT] severity_tier distribution:")
    for tier, count in tier_counts.items():
        print(f"       {tier}: {count}")
    
    return df


def engineer_duration_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket duration into categories for classification."""
    df = df.copy()
    
    def bucket(d):
        if pd.isna(d):
            return "unknown"
        elif d <= 30:
            return "quick (<30min)"
        elif d <= 120:
            return "moderate (30-120min)"
        elif d <= 480:
            return "extended (2-8hr)"
        else:
            return "prolonged (>8hr)"
    
    df["duration_bucket"] = df["duration_to_close_min"].apply(bucket)
    
    bucket_counts = df["duration_bucket"].value_counts()
    print(f"[FEAT] duration_bucket distribution:")
    for b, count in bucket_counts.items():
        print(f"       {b}: {count}")
    
    return df


def engineer_is_event_driven(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows that match the theme-relevant event causes."""
    df = df.copy()
    event_driven_causes = ["public_event", "procession", "vip_movement", "protest", "construction"]
    df["is_event_driven"] = df["event_cause"].isin(event_driven_causes).astype(int)
    print(f"[FEAT] is_event_driven: {df['is_event_driven'].sum()} rows flagged")
    return df


# ─── Step 3: Corridor Centroids & Adjacency ─────────────────────────────────

def compute_corridor_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean lat/lon per corridor as centroid."""
    centroids = df.groupby("corridor").agg(
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean"),
        event_count=("id", "count"),
    ).reset_index()
    
    centroids.to_csv(CORRIDOR_CENTROIDS_CSV, index=False)
    print(f"[GEO] Corridor centroids: {len(centroids)} corridors -> {CORRIDOR_CENTROIDS_CSV}")
    return centroids


def compute_corridor_adjacency(centroids: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """For each corridor, find the N nearest other corridors by centroid distance."""
    from math import radians, cos, sin, asin, sqrt
    
    def haversine(lat1, lon1, lat2, lon2):
        """Haversine distance in km."""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 2 * 6371 * asin(sqrt(a))
    
    rows = []
    for _, row in centroids.iterrows():
        corridor = row["corridor"]
        distances = []
        for _, other in centroids.iterrows():
            if other["corridor"] == corridor:
                continue
            dist = haversine(
                row["centroid_lat"], row["centroid_lon"],
                other["centroid_lat"], other["centroid_lon"]
            )
            distances.append((other["corridor"], round(dist, 2), other["event_count"]))
        
        # Sort by distance, take top N
        distances.sort(key=lambda x: x[1])
        for rank, (neighbor, dist, neighbor_count) in enumerate(distances[:top_n], 1):
            rows.append({
                "corridor": corridor,
                "neighbor_corridor": neighbor,
                "rank": rank,
                "distance_km": dist,
                "neighbor_event_count": neighbor_count,
            })
    
    adjacency = pd.DataFrame(rows)
    adjacency.to_csv(CORRIDOR_ADJACENCY_CSV, index=False)
    print(f"[GEO] Corridor adjacency: {len(adjacency)} entries -> {CORRIDOR_ADJACENCY_CSV}")
    return adjacency


# ─── Step 4: Select & Save Final Columns ─────────────────────────────────────

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select the columns relevant for modeling and the dashboard."""
    keep_cols = [
        # Identifiers
        "id", "event_type", "event_cause", "status",
        # Location
        "latitude", "longitude", "endlatitude", "endlongitude",
        "address", "corridor", "zone", "police_station", "junction",
        # Time
        "start_datetime", "closed_datetime",
        "hour_of_day", "day_of_week", "day_name", "is_weekend",
        "month", "date", "time_period",
        # Event properties
        "requires_road_closure", "priority", "veh_type",
        "description",
        # Engineered features
        "duration_to_close_min", "duration_bucket", "severity_tier",
        "is_event_driven",
    ]
    
    # Only keep columns that actually exist
    actual_cols = [c for c in keep_cols if c in df.columns]
    df = df[actual_cols]
    print(f"[SELECT] Keeping {len(actual_cols)} columns (dropped {46 - len(actual_cols)} unused columns)")
    return df


# ─── Step 5: EDA Charts ─────────────────────────────────────────────────────

def generate_eda_charts(df: pd.DataFrame):
    """Generate 6 EDA charts and save as images."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})
    
    # ── Chart 1: Events by Cause ──
    fig, ax = plt.subplots(figsize=(12, 6))
    cause_counts = df["event_cause"].value_counts()
    colors = sns.color_palette("viridis", len(cause_counts))
    bars = ax.barh(cause_counts.index[::-1], cause_counts.values[::-1], color=colors[::-1])
    ax.set_xlabel("Number of Events")
    ax.set_title("Events by Cause", fontsize=14, fontweight="bold")
    # Add count labels
    for bar, val in zip(bars, cause_counts.values[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9)
    plt.savefig(EDA_DIR / "01_events_by_cause.png")
    plt.close()
    print(f"[EDA] Saved 01_events_by_cause.png")
    
    # ── Chart 2: Events by Hour of Day (IST) ──
    fig, ax = plt.subplots(figsize=(12, 5))
    hour_counts = df["hour_of_day"].value_counts().sort_index()
    ax.bar(hour_counts.index, hour_counts.values, color=sns.color_palette("coolwarm", 24), width=0.8)
    ax.set_xlabel("Hour of Day (IST)")
    ax.set_ylabel("Number of Events")
    ax.set_title("Events by Hour of Day (IST) — Note: reflects reporting patterns, not true incident rate",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    plt.savefig(EDA_DIR / "02_events_by_hour.png")
    plt.close()
    print(f"[EDA] Saved 02_events_by_hour.png")
    
    # ── Chart 3: Events by Corridor (top 15) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    corridor_counts = df["corridor"].value_counts().head(15)
    colors = sns.color_palette("mako", len(corridor_counts))
    bars = ax.barh(corridor_counts.index[::-1], corridor_counts.values[::-1], color=colors[::-1])
    ax.set_xlabel("Number of Events")
    ax.set_title("Top 15 Corridors by Event Count", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, corridor_counts.values[::-1]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9)
    plt.savefig(EDA_DIR / "03_events_by_corridor.png")
    plt.close()
    print(f"[EDA] Saved 03_events_by_corridor.png")
    
    # ── Chart 4: Duration Distribution by Cause ──
    fig, ax = plt.subplots(figsize=(14, 7))
    # Only plot causes with enough data and valid duration
    dur_df = df[df["duration_to_close_min"].notna()].copy()
    # Cap at 500 min for readability
    dur_df["duration_capped"] = dur_df["duration_to_close_min"].clip(upper=500)
    
    # Order by median duration
    cause_order = dur_df.groupby("event_cause")["duration_to_close_min"].median().sort_values(ascending=False).index
    
    sns.boxplot(data=dur_df, y="event_cause", x="duration_capped",
                order=cause_order, palette="viridis", ax=ax,
                fliersize=2, linewidth=0.8)
    ax.set_xlabel("Duration to Close (min, capped at 500)")
    ax.set_ylabel("Event Cause")
    ax.set_title("Duration-to-Close Distribution by Cause (capped at 500 min for visibility)",
                 fontsize=11, fontweight="bold")
    plt.savefig(EDA_DIR / "04_duration_by_cause.png")
    plt.close()
    print(f"[EDA] Saved 04_duration_by_cause.png")
    
    # ── Chart 5: Severity Tier by Event Cause (stacked) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    ct = pd.crosstab(df["event_cause"], df["severity_tier"])
    # Reorder columns
    for col in ["High", "Medium", "Low"]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["High", "Medium", "Low"]]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=True).index]
    ct.plot(kind="barh", stacked=True, ax=ax,
            color={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"})
    ax.set_xlabel("Number of Events")
    ax.set_title("Severity Tier Distribution by Event Cause", fontsize=14, fontweight="bold")
    ax.legend(title="Severity Tier", loc="lower right")
    plt.savefig(EDA_DIR / "05_severity_by_cause.png")
    plt.close()
    print(f"[EDA] Saved 05_severity_by_cause.png")
    
    # ── Chart 6: Heatmap — Hour × Day of Week ──
    fig, ax = plt.subplots(figsize=(12, 5))
    heatmap_data = df.pivot_table(
        index="day_name", columns="hour_of_day", values="id",
        aggfunc="count", fill_value=0
    )
    # Reorder days
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = heatmap_data.reindex(day_order)
    
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=False, fmt="d",
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Hour of Day (IST)")
    ax.set_ylabel("Day of Week")
    ax.set_title("Event Density: Hour × Day of Week", fontsize=14, fontweight="bold")
    plt.savefig(EDA_DIR / "06_heatmap_hour_day.png")
    plt.close()
    print(f"[EDA] Saved 06_heatmap_hour_day.png")


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_pipeline():
    """Execute the full cleaning & feature engineering pipeline."""
    print("=" * 70)
    print("PHASE 1: Data Cleaning & Feature Engineering Pipeline")
    print("=" * 70)
    
    # Load
    df = load_raw_data()
    
    # Clean
    print("\n--- Cleaning ---")
    df = clean_event_cause(df)
    df = parse_datetimes(df)
    df = clean_coordinates(df)
    df = clean_requires_road_closure(df)
    df = clean_corridor(df)
    df = clean_priority(df)
    
    # Engineer features
    print("\n--- Feature Engineering ---")
    df = engineer_time_features(df)
    df = engineer_duration_feature(df)
    df = engineer_severity_tier(df)
    df = engineer_duration_bucket(df)
    df = engineer_is_event_driven(df)
    
    # Corridor geography
    print("\n--- Corridor Geography ---")
    centroids = compute_corridor_centroids(df)
    adjacency = compute_corridor_adjacency(centroids)
    
    # Select columns
    print("\n--- Column Selection ---")
    df = select_columns(df)
    
    # Save cleaned dataset
    print("\n--- Save ---")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"[SAVE] Cleaned dataset: {CLEAN_CSV} ({len(df)} rows × {len(df.columns)} columns)")
    
    # Generate EDA charts
    print("\n--- EDA Charts ---")
    generate_eda_charts(df)
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"  Cleaned dataset: {CLEAN_CSV}")
    print(f"  Corridor centroids: {CORRIDOR_CENTROIDS_CSV}")
    print(f"  Corridor adjacency: {CORRIDOR_ADJACENCY_CSV}")
    print(f"  EDA charts: {EDA_DIR}/")
    print(f"  Final shape: {df.shape}")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    run_pipeline()
