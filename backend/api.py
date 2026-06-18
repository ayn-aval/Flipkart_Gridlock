"""
FastAPI Backend — Event-Driven Congestion Forecasting API
=========================================================
Phase 4: REST API exposing historical events, hotspots, forecasting,
recommendations, and feedback endpoints.

Usage:
    python3 -m uvicorn backend.api:app --reload --port 8000
    # or
    python3 backend/api.py
"""

import sys
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "events_clean.csv"
LEARNING_LOG = PROCESSED_DIR / "learning_log.csv"
CORRIDOR_CENTROIDS = PROCESSED_DIR / "corridor_centroids.csv"
CORRIDOR_ADJACENCY = PROCESSED_DIR / "corridor_adjacency.csv"
EDA_DIR = PROCESSED_DIR / "eda_charts"

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🚦 Gridlock — Event-Driven Congestion Forecasting API",
    description=(
        "Forecast event-related traffic impact and recommend optimal "
        "manpower, barricading, and diversion plans for Bengaluru."
    ),
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State (loaded once at startup) ────────────────────────────────────
events_df: pd.DataFrame = None
forecast_engine = None
rec_engine = None


def nan_safe_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to list of dicts with NaN replaced by None.
    
    Handles both Python float and numpy float64 NaN values.
    """
    records = df.to_dict(orient="records")
    for record in records:
        for key, val in record.items():
            if val is None:
                continue
            try:
                if isinstance(val, (float, np.floating)) and (math.isnan(val) or math.isinf(val)):
                    record[key] = None
            except (TypeError, ValueError):
                pass
    return records

@app.on_event("startup")
def startup_load():
    """Load data and models at startup."""
    global events_df, forecast_engine, rec_engine

    # Load events
    events_df = pd.read_csv(CLEAN_CSV)
    print(f"[API] Loaded {len(events_df)} events")

    # Load forecasting engine
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
    from forecasting import load_engine
    forecast_engine = load_engine()
    print("[API] Forecasting engine loaded")

    # Load recommendation engine
    from recommendations import RecommendationEngine
    rec_engine = RecommendationEngine()
    print("[API] Recommendation engine loaded")

    # Initialize learning log if it doesn't exist
    if not LEARNING_LOG.exists():
        with open(LEARNING_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "event_id", "predicted_severity", "predicted_duration_min",
                "actual_severity", "actual_duration_min", "feedback_notes",
            ])
        print("[API] Learning log initialized")


# ═══════════════════════════════════════════════════════════════════════════════
#  Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════════

class ForecastRequest(BaseModel):
    """Input for the /forecast endpoint."""
    event_cause: str = Field(..., description="Type of event cause", 
                             examples=["procession", "vehicle_breakdown", "construction"])
    corridor: str = Field("Non-corridor", description="Traffic corridor name",
                         examples=["Bellary Road 1", "Mysore Road"])
    hour_of_day: int = Field(12, ge=0, le=23, description="Hour in IST (0-23)")
    day_of_week: int = Field(2, ge=0, le=6, description="0=Monday, 6=Sunday")
    is_weekend: int = Field(0, ge=0, le=1, description="1 if Saturday/Sunday")
    requires_road_closure: int = Field(0, ge=0, le=1, description="1 if road closure needed")
    veh_type: str = Field("none", description="Vehicle type (for breakdowns)",
                          examples=["none", "heavy_vehicle", "bmtc_bus"])
    time_period: Optional[str] = Field(None, description="Auto-computed if not provided",
                                       examples=["morning_rush", "midday", "evening_rush", "night"])


class FeedbackRequest(BaseModel):
    """Input for the /feedback endpoint."""
    event_id: str = Field(..., description="ID of the event")
    actual_severity: Optional[str] = Field(None, description="Actual severity (Low/Medium/High)")
    actual_duration_min: Optional[float] = Field(None, description="Actual duration in minutes")
    feedback_notes: Optional[str] = Field(None, description="Free-text notes")


# ═══════════════════════════════════════════════════════════════════════════════
#  GET /events — Query Historical Events
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/events", tags=["Historical Data"])
def get_events(
    corridor: Optional[str] = Query(None, description="Filter by corridor name"),
    event_cause: Optional[str] = Query(None, description="Filter by event cause"),
    severity: Optional[str] = Query(None, description="Filter by severity tier (Low/Medium/High)"),
    status: Optional[str] = Query(None, description="Filter by status (closed/active/resolved)"),
    event_type: Optional[str] = Query(None, description="Filter by event type (planned/unplanned)"),
    is_event_driven: Optional[int] = Query(None, description="1 = event-driven causes only"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=5000, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """Query historical events with optional filters.
    
    Returns paginated event data for the map/dashboard.
    """
    df = events_df.copy()

    # Apply filters
    if corridor:
        df = df[df["corridor"] == corridor]
    if event_cause:
        df = df[df["event_cause"] == event_cause]
    if severity:
        df = df[df["severity_tier"] == severity]
    if status:
        df = df[df["status"] == status]
    if event_type:
        df = df[df["event_type"] == event_type]
    if is_event_driven is not None:
        df = df[df["is_event_driven"] == is_event_driven]
    if date_from:
        df = df[df["date"] >= date_from]
    if date_to:
        df = df[df["date"] <= date_to]

    total = len(df)
    df = df.iloc[offset:offset + limit]

    # Convert to records, replacing NaN with None for JSON safety
    records = nan_safe_records(df)

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "count": len(records),
        "events": records,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GET /events/summary — Aggregated Stats
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/events/summary", tags=["Historical Data"])
def get_events_summary():
    """Get aggregated summary statistics for the dashboard."""
    df = events_df

    return {
        "total_events": len(df),
        "by_event_type": df["event_type"].value_counts().to_dict(),
        "by_event_cause": df["event_cause"].value_counts().to_dict(),
        "by_severity": df["severity_tier"].value_counts().to_dict(),
        "by_status": df["status"].value_counts().to_dict(),
        "by_priority": df["priority"].value_counts().to_dict(),
        "event_driven_count": int(df["is_event_driven"].sum()),
        "road_closure_count": int(df["requires_road_closure"].sum()),
        "corridors": sorted(df["corridor"].dropna().unique().tolist()),
        "event_causes": sorted(df["event_cause"].unique().tolist()),
        "date_range": {
            "min": str(df["date"].dropna().min()) if df["date"].notna().any() else None,
            "max": str(df["date"].dropna().max()) if df["date"].notna().any() else None,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GET /hotspots — Heatmap Data
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/hotspots", tags=["Historical Data"])
def get_hotspots(
    group_by: str = Query("corridor", description="Group by: corridor, zone, police_station"),
    event_cause: Optional[str] = Query(None, description="Filter by event cause"),
    hour: Optional[int] = Query(None, ge=0, le=23, description="Filter by hour of day"),
):
    """Aggregated event counts for heatmap visualization.
    
    Returns counts grouped by corridor/zone/police_station with coordinates.
    """
    df = events_df.copy()

    if event_cause:
        df = df[df["event_cause"] == event_cause]
    if hour is not None:
        df = df[df["hour_of_day"] == hour]

    valid_groups = ["corridor", "zone", "police_station"]
    if group_by not in valid_groups:
        raise HTTPException(400, f"group_by must be one of {valid_groups}")

    # Group and compute stats
    grouped = df.groupby(group_by).agg(
        event_count=("id", "count"),
        avg_lat=("latitude", "mean"),
        avg_lon=("longitude", "mean"),
        high_severity_count=("severity_tier", lambda x: (x == "High").sum()),
        road_closure_count=("requires_road_closure", "sum"),
        avg_duration_min=("duration_to_close_min", "mean"),
    ).reset_index()

    # Sort by event count descending
    grouped = grouped.sort_values("event_count", ascending=False)

    # Clean NaN for JSON
    grouped = grouped.where(grouped.notna(), None)

    return {
        "group_by": group_by,
        "total_groups": len(grouped),
        "hotspots": nan_safe_records(grouped),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GET /hotspots/geo — GeoJSON-style Points for Map
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/hotspots/geo", tags=["Historical Data"])
def get_hotspots_geo(
    event_cause: Optional[str] = Query(None, description="Filter by cause"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    corridor: Optional[str] = Query(None, description="Filter by corridor"),
    limit: int = Query(2000, ge=1, le=8200, description="Max points"),
):
    """Return individual event points for map markers/clustering."""
    df = events_df.copy()

    if event_cause:
        df = df[df["event_cause"] == event_cause]
    if severity:
        df = df[df["severity_tier"] == severity]
    if corridor:
        df = df[df["corridor"] == corridor]

    df = df.head(limit)

    points = []
    for _, row in df.iterrows():
        points.append({
            "id": row["id"],
            "lat": row["latitude"],
            "lon": row["longitude"],
            "event_cause": row["event_cause"],
            "severity_tier": row["severity_tier"],
            "corridor": row["corridor"],
            "requires_road_closure": bool(row["requires_road_closure"]) if pd.notna(row["requires_road_closure"]) else False,
            "hour_of_day": int(row["hour_of_day"]) if pd.notna(row["hour_of_day"]) else None,
            "description": str(row["description"])[:150] if pd.notna(row["description"]) else None,
        })

    return {"count": len(points), "points": points}


# ═══════════════════════════════════════════════════════════════════════════════
#  GET /corridors — Corridor Data
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/corridors", tags=["Historical Data"])
def get_corridors():
    """Get corridor centroids and adjacency data."""
    centroids = pd.read_csv(CORRIDOR_CENTROIDS)
    adjacency = pd.read_csv(CORRIDOR_ADJACENCY)

    corridors = []
    for _, row in centroids.iterrows():
        name = row["corridor"]
        neighbors = nan_safe_records(adjacency[adjacency["corridor"] == name])
        corridors.append({
            "name": name,
            "centroid_lat": row["centroid_lat"],
            "centroid_lon": row["centroid_lon"],
            "event_count": int(row["event_count"]),
            "neighbors": neighbors,
        })

    return {"count": len(corridors), "corridors": corridors}


# ═══════════════════════════════════════════════════════════════════════════════
#  GET /eda — EDA Chart Metadata
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/eda", tags=["Historical Data"])
def get_eda_charts():
    """List available EDA charts."""
    charts = []
    if EDA_DIR.exists():
        for f in sorted(EDA_DIR.glob("*.png")):
            charts.append({
                "filename": f.name,
                "url": f"/eda/{f.name}",
                "title": f.stem.replace("_", " ").title(),
            })
    return {"count": len(charts), "charts": charts}


# ═══════════════════════════════════════════════════════════════════════════════
#  POST /forecast — Predict + Recommend
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/forecast", tags=["Forecasting"])
def forecast_event(request: ForecastRequest):
    """Given a new/planned event, return severity forecast, duration estimate,
    resource recommendations, and diversion suggestions.
    
    This combines Phase 2 (forecasting) and Phase 3 (recommendations) into
    a single response for the dashboard's simulation feature.
    """
    # Auto-compute time_period if not provided
    time_period = request.time_period
    if not time_period:
        h = request.hour_of_day
        if 6 <= h < 10:
            time_period = "morning_rush"
        elif 10 <= h < 16:
            time_period = "midday"
        elif 16 <= h < 21:
            time_period = "evening_rush"
        else:
            time_period = "night"

    # Validate event_cause
    valid_causes = events_df["event_cause"].unique().tolist()
    if request.event_cause not in valid_causes:
        raise HTTPException(
            400,
            f"Invalid event_cause '{request.event_cause}'. "
            f"Valid values: {sorted(valid_causes)}"
        )

    # Build event input dict
    event_input = {
        "event_cause": request.event_cause,
        "corridor": request.corridor,
        "time_period": time_period,
        "hour_of_day": request.hour_of_day,
        "day_of_week": request.day_of_week,
        "is_weekend": request.is_weekend,
        "requires_road_closure": request.requires_road_closure,
        "veh_type": request.veh_type,
    }

    # Get forecast
    forecast_result = forecast_engine.forecast(event_input)

    # Get recommendation
    recommendation = rec_engine.recommend(forecast_result, event_input)

    return {
        "input": event_input,
        "forecast": forecast_result,
        "recommendation": recommendation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  POST /feedback — Post-Event Learning Log
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/feedback", tags=["Learning Loop"])
def submit_feedback(request: FeedbackRequest):
    """Log actual event outcome for post-event learning.
    
    Stores predicted vs actual data in a learning log CSV.
    This feeds the accuracy-tracking panel in Phase 6.
    """
    # Look up event's prediction (if it was a known event)
    event_row = events_df[events_df["id"] == request.event_id]
    predicted_severity = None
    predicted_duration = None

    if len(event_row) > 0:
        predicted_severity = event_row.iloc[0].get("severity_tier")
        predicted_duration = event_row.iloc[0].get("duration_to_close_min")

    # Append to learning log
    timestamp = datetime.now().isoformat()
    with open(LEARNING_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            request.event_id,
            predicted_severity,
            predicted_duration,
            request.actual_severity,
            request.actual_duration_min,
            request.feedback_notes,
        ])

    return {
        "status": "logged",
        "event_id": request.event_id,
        "timestamp": timestamp,
        "predicted_severity": predicted_severity,
        "actual_severity": request.actual_severity,
        "predicted_duration_min": predicted_duration,
        "actual_duration_min": request.actual_duration_min,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GET /feedback/log — View Learning Log
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/feedback/log", tags=["Learning Loop"])
def get_feedback_log():
    """Retrieve the full learning log for accuracy tracking."""
    if not LEARNING_LOG.exists():
        return {"count": 0, "entries": []}

    df = pd.read_csv(LEARNING_LOG)
    df = df.where(df.notna(), None)

    return {
        "count": len(df),
        "entries": nan_safe_records(df),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GET /health — Health Check
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health_check():
    """API health check."""
    return {
        "status": "healthy",
        "events_loaded": len(events_df) if events_df is not None else 0,
        "forecasting_engine": forecast_engine is not None,
        "recommendation_engine": rec_engine is not None,
        "learning_log_exists": LEARNING_LOG.exists(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Static file serving for EDA charts
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

if EDA_DIR.exists():
    app.mount("/eda", StaticFiles(directory=str(EDA_DIR)), name="eda_charts")

# Serve frontend
FRONTEND_DIR = PROJECT_ROOT / "frontend"

@app.get("/", tags=["Frontend"])
def serve_frontend():
    """Serve the dashboard."""
    return FileResponse(str(FRONTEND_DIR / "index.html"))

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


# ─── Run directly ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
