"""
Data Audit Script for Astram Event Data
========================================
Loads the raw CSV, prints schema, null-rates, and category breakdowns.
Saves a summary to data/processed/data_audit.md.

Usage:
    python backend/data_audit.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "astram_events.csv"
OUTPUT_MD = PROJECT_ROOT / "data" / "processed" / "data_audit.md"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV with basic type inference."""
    df = pd.read_csv(path, low_memory=False)
    # Replace string 'NULL' with actual NaN
    df = df.replace("NULL", np.nan)
    return df


def compute_null_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of column-level null counts and percentages."""
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    return pd.DataFrame({
        "column": null_counts.index,
        "null_count": null_counts.values,
        "null_pct": null_pct.values,
        "non_null_count": (len(df) - null_counts).values,
        "dtype": [str(df[col].dtype) for col in df.columns],
    })


def category_breakdown(df: pd.DataFrame, col: str, top_n: int = 30) -> str:
    """Return a markdown-formatted value-counts table for a column."""
    if col not in df.columns:
        return f"Column `{col}` not found.\n"
    vc = df[col].value_counts(dropna=False).head(top_n)
    lines = [f"| Value | Count |", f"|---|---|"]
    for val, count in vc.items():
        label = str(val) if pd.notna(val) else "(null)"
        lines.append(f"| {label} | {count} |")
    return "\n".join(lines)


def datetime_range(df: pd.DataFrame, col: str) -> str:
    """Try to parse a datetime column and return its range."""
    if col not in df.columns:
        return "N/A"
    try:
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        valid = parsed.dropna()
        if valid.empty:
            return "No valid datetimes"
        return f"{valid.min()} → {valid.max()} ({len(valid)} non-null)"
    except Exception as e:
        return f"Parse error: {e}"


def generate_audit_report(df: pd.DataFrame) -> str:
    """Generate the full audit report as a markdown string."""
    lines = []
    lines.append("# Data Audit Report — Astram Event Data")
    lines.append(f"\n**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n**Source:** `data/raw/astram_events.csv`")
    
    # --- Basic shape ---
    lines.append(f"\n## 1. Dataset Shape")
    lines.append(f"- **Rows:** {len(df):,}")
    lines.append(f"- **Columns:** {len(df.columns)}")
    lines.append(f"- **Column names:** {', '.join(df.columns.tolist())}")
    
    # --- Null rates ---
    lines.append(f"\n## 2. Null Rates Per Column")
    null_df = compute_null_rates(df)
    lines.append("| Column | Dtype | Non-Null | Null Count | Null % |")
    lines.append("|---|---|---|---|---|")
    for _, row in null_df.iterrows():
        lines.append(
            f"| {row['column']} | {row['dtype']} | {row['non_null_count']} | "
            f"{row['null_count']} | {row['null_pct']}% |"
        )
    
    # --- Key categorical breakdowns ---
    key_cats = [
        ("event_type", "Event Type"),
        ("event_cause", "Event Cause"),
        ("priority", "Priority"),
        ("status", "Status"),
        ("requires_road_closure", "Requires Road Closure"),
        ("corridor", "Corridor (top 30)"),
        ("zone", "Zone"),
        ("police_station", "Police Station (top 20)"),
        ("veh_type", "Vehicle Type"),
        ("direction", "Direction"),
    ]
    
    lines.append(f"\n## 3. Category Breakdowns")
    for col, title in key_cats:
        lines.append(f"\n### {title} (`{col}`)")
        unique_count = df[col].nunique() if col in df.columns else 0
        non_null = df[col].notna().sum() if col in df.columns else 0
        lines.append(f"Unique values: {unique_count} | Non-null: {non_null}")
        lines.append(category_breakdown(df, col))
    
    # --- Datetime ranges ---
    lines.append(f"\n## 4. Datetime Ranges")
    dt_cols = ["start_datetime", "end_datetime", "closed_datetime", 
               "resolved_datetime", "created_date", "modified_datetime"]
    for col in dt_cols:
        lines.append(f"- **{col}:** {datetime_range(df, col)}")
    
    # --- Geographic coverage ---
    lines.append(f"\n## 5. Geographic Coverage")
    for col in ["latitude", "longitude", "endlatitude", "endlongitude"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            valid = series[series != 0].dropna()
            lines.append(
                f"- **{col}:** min={valid.min():.6f}, max={valid.max():.6f}, "
                f"non-zero count={len(valid)}"
            )
    
    # --- Junction stats ---
    lines.append(f"\n## 6. Junction Coverage")
    if "junction" in df.columns:
        non_null = df["junction"].notna().sum()
        unique = df["junction"].nunique()
        lines.append(f"- Non-null: {non_null} / {len(df)} ({non_null/len(df)*100:.1f}%)")
        lines.append(f"- Unique junctions: {unique}")
    
    # --- Duration proxy analysis ---
    lines.append(f"\n## 7. Duration-to-Close Proxy (closed_datetime − start_datetime)")
    try:
        start = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
        closed = pd.to_datetime(df["closed_datetime"], errors="coerce", utc=True)
        duration = (closed - start).dt.total_seconds() / 60  # minutes
        valid_dur = duration[(duration > 0) & (duration < 60 * 24 * 30)]  # < 30 days
        lines.append(f"- Valid durations (>0 and <30 days): {len(valid_dur)}")
        lines.append(f"- Mean: {valid_dur.mean():.1f} min")
        lines.append(f"- Median: {valid_dur.median():.1f} min")
        lines.append(f"- Std: {valid_dur.std():.1f} min")
        lines.append(f"- 5th percentile: {valid_dur.quantile(0.05):.1f} min")
        lines.append(f"- 95th percentile: {valid_dur.quantile(0.95):.1f} min")
        
        # By cause
        df_dur = df.copy()
        df_dur["_duration_min"] = duration
        df_dur = df_dur[(df_dur["_duration_min"] > 0) & (df_dur["_duration_min"] < 60 * 24 * 30)]
        cause_dur = df_dur.groupby("event_cause")["_duration_min"].agg(["median", "mean", "count"])
        cause_dur = cause_dur.sort_values("median", ascending=False)
        lines.append(f"\n### Median Duration by Event Cause")
        lines.append("| Event Cause | Median (min) | Mean (min) | Count |")
        lines.append("|---|---|---|---|")
        for cause, row in cause_dur.iterrows():
            lines.append(f"| {cause} | {row['median']:.1f} | {row['mean']:.1f} | {int(row['count'])} |")
    except Exception as e:
        lines.append(f"Error computing durations: {e}")
    
    # --- Diurnal pattern ---
    lines.append(f"\n## 8. Diurnal Pattern (Hour of Start, IST = UTC+5:30)")
    try:
        start_utc = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
        # Convert to IST (UTC+5:30) for Bengaluru
        start_ist = start_utc.dt.tz_convert("Asia/Kolkata")
        hours = start_ist.dt.hour
        hour_counts = hours.value_counts().sort_index()
        lines.append("| Hour (IST) | Event Count |")
        lines.append("|---|---|")
        for h, c in hour_counts.items():
            lines.append(f"| {int(h):02d}:00 | {c} |")
    except Exception as e:
        lines.append(f"Error: {e}")
    
    # --- Planned event subset ---
    lines.append(f"\n## 9. Planned/Event-Driven Subset Analysis")
    event_driven_causes = ["public_event", "procession", "vip_movement", "protest", "construction"]
    subset = df[df["event_cause"].isin(event_driven_causes)]
    lines.append(f"- Total rows matching event-driven causes: {len(subset)}")
    lines.append(category_breakdown(subset, "event_cause"))
    lines.append(f"\n**By event_type in this subset:**")
    lines.append(category_breakdown(subset, "event_type"))
    
    return "\n".join(lines)


def main():
    print(f"Loading data from: {RAW_CSV}")
    if not RAW_CSV.exists():
        print(f"ERROR: File not found at {RAW_CSV}")
        sys.exit(1)
    
    df = load_data(RAW_CSV)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Generate report
    report = generate_audit_report(df)
    
    # Save to markdown
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nAudit report saved to: {OUTPUT_MD}")
    
    # Also print to stdout for visibility
    print("\n" + "=" * 80)
    print(report)


if __name__ == "__main__":
    main()
