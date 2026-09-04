"""
FastAPI Backend — Event-Driven Congestion Forecasting API
=========================================================
REST API exposing historical events, hotspots, forecasting, recommendations,
computer vision and the post-event learning loop.

Usage:
    python3 -m uvicorn backend.api:app --reload --port 7860
"""

import os

# Must run before pandas / scikit-learn / xgboost / torch are imported. xgboost and
# torch each ship their own OpenMP runtime; on macOS, loading one after the other and
# then entering a parallel region from a worker thread deadlocks the process with no
# traceback (the CCTV endpoints hang forever). Pinning the native pools to a single
# thread and tolerating the duplicate runtime avoids that. Inference here is one small
# frame at a time, so there is no measurable throughput cost.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import csv
import json
import sys
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ─── Paths (always resolved from the project root, never the CWD) ─────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "events_clean.csv"
LEARNING_LOG = PROCESSED_DIR / "learning_log.csv"
CORRIDOR_CENTROIDS = PROCESSED_DIR / "corridor_centroids.csv"
CORRIDOR_ADJACENCY = PROCESSED_DIR / "corridor_adjacency.csv"
MODEL_METRICS = PROCESSED_DIR / "models" / "model_metrics.json"
EDA_DIR = PROCESSED_DIR / "eda_charts"
CCTV_DIR = PROJECT_ROOT / "data" / "cctv_samples"
ALERTS_PATH = PROJECT_ROOT / "data" / "active_alerts.json"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

LEARNING_LOG_COLUMNS = [
    "timestamp", "event_id", "predicted_severity", "predicted_duration_min",
    "actual_severity", "actual_duration_min", "feedback_notes",
]

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB
MAX_ACTIVE_PREDICTIONS = 500


def _load_dotenv():
    """Minimal .env loader that tolerates comments, blanks and malformed lines.

    The previous version did a bare `split("=", 1)` on every non-comment line, so a
    single line without an '=' raised ValueError at import time and took the whole
    app down before it could serve anything.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

# ─── Global State ─────────────────────────────────────────────────────────────
events_df: Optional[pd.DataFrame] = None
forecast_engine = None
rec_engine = None
corridor_centroid_lookup: dict = {}
active_predictions: "OrderedDict[str, dict]" = OrderedDict()
retrain_state = {"running": False, "last_result": None, "last_finished": None}

_feedback_lock = threading.Lock()
_feedback_cache = {"mtime": None, "payload": None}


def _load_all():
    """Load data, models and lookups. Also used to hot-reload after a retrain."""
    global events_df, forecast_engine, rec_engine, corridor_centroid_lookup

    events_df = pd.read_csv(CLEAN_CSV)
    print(f"[API] Loaded {len(events_df)} events")

    # backend/ on sys.path so the sibling modules resolve consistently. Everything
    # imports as `backend.x` — importing the same module under two names created two
    # copies of the YOLO weights in memory.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from backend.forecasting import load_engine
    forecast_engine = load_engine()
    print("[API] Forecasting engine loaded")

    from backend.recommendations import RecommendationEngine
    rec_engine = RecommendationEngine()
    print("[API] Recommendation engine loaded")

    if CORRIDOR_CENTROIDS.exists():
        cdf = pd.read_csv(CORRIDOR_CENTROIDS).dropna(subset=["centroid_lat", "centroid_lon"])
        corridor_centroid_lookup = {
            r["corridor"]: (float(r["centroid_lat"]), float(r["centroid_lon"]))
            for _, r in cdf.iterrows()
        }
        print(f"[API] {len(corridor_centroid_lookup)} corridor centroids loaded")

    if not LEARNING_LOG.exists():
        LEARNING_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LEARNING_LOG, "w", newline="") as f:
            csv.writer(f).writerow(LEARNING_LOG_COLUMNS)
        print("[API] Learning log initialized")

    _feedback_cache["mtime"] = None  # invalidate


def _warm_vision_model():
    """Spin up the vision worker process and preload its weights.

    Runs in a daemon thread so it never blocks boot. torch is imported in the child,
    not here — see backend/cv_worker.py for why that separation matters.
    """
    from backend import cv_worker
    print("[API] Vision worker warm" if cv_worker.warm() else "[API] Vision worker warm-up failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all()
    threading.Thread(target=_warm_vision_model, daemon=True).start()
    yield
    from backend import cv_worker
    cv_worker.shutdown()


app = FastAPI(
    title="Namma Route — Event-Driven Congestion Forecasting API",
    description=(
        "Forecast event-related traffic impact and recommend manpower, barricading "
        "and diversion plans for Bengaluru."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# allow_credentials=True together with allow_origins=["*"] is rejected by browsers —
# the two cancel out. This API uses no cookies or auth headers, so the correct
# combination is a wildcard origin with credentials disabled.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def nan_safe_records(df: pd.DataFrame) -> list:
    """DataFrame -> list of dicts with NaN/Inf replaced by None for JSON safety."""
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


def _require_engine():
    if forecast_engine is None or rec_engine is None or events_df is None:
        raise HTTPException(503, "Models are still loading. Try again in a moment.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════════

class ForecastRequest(BaseModel):
    event_cause: str = Field(..., description="Type of event cause",
                             examples=["procession", "vehicle_breakdown", "construction"])
    event_type: str = Field("unplanned", description="Event type (planned/unplanned)")
    corridor: str = Field("Non-corridor", description="Traffic corridor name")
    zone: str = Field("Unknown", description="City zone")
    police_station: str = Field("Unknown", description="Police station jurisdiction")
    direction: str = Field("unknown", description="Traffic direction")
    hour_of_day: int = Field(12, ge=0, le=23, description="Hour in IST (0-23)")
    day_of_week: int = Field(2, ge=0, le=6, description="0=Monday, 6=Sunday")
    is_weekend: int = Field(0, ge=0, le=1, description="1 if Saturday/Sunday")
    requires_road_closure: int = Field(0, ge=0, le=1, description="1 if road closure needed")
    veh_type: str = Field("none", description="Vehicle type (for breakdowns)")
    description: str = Field("", description="Raw event description")
    # Location is a real model input. When omitted it is filled from the corridor
    # centroid rather than defaulting silently to the city centre for every request.
    latitude: Optional[float] = Field(None, ge=12.5, le=13.5, description="Event latitude")
    longitude: Optional[float] = Field(None, ge=77.0, le=78.2, description="Event longitude")


class FeedbackRequest(BaseModel):
    event_id: str = Field(..., min_length=1, description="ID of the event")
    actual_severity: Optional[str] = Field(None, description="Actual severity (Low/Medium/High)")
    actual_duration_min: Optional[float] = Field(None, ge=0, description="Actual duration in minutes")
    feedback_notes: Optional[str] = Field(None, description="Free-text notes")


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/config", tags=["Configuration"])
def get_config():
    """Client configuration.

    The Mappls key is a browser-side map key: it has to reach the client to render
    tiles at all. Restrict it by referrer/domain in the Mappls console — that, not
    hiding this endpoint, is what actually protects the quota.
    """
    return {"mappls_api_key": os.environ.get("MAPPLS_API_KEY", "")}


# ═══════════════════════════════════════════════════════════════════════════════
#  Historical Data
# ═══════════════════════════════════════════════════════════════════════════════

EVENT_DRIVEN_CAUSES = ["public_event", "procession", "vip_movement", "protest", "construction"]


@app.get("/events", tags=["Historical Data"])
def get_events(
    corridor: Optional[str] = Query(None),
    event_cause: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    is_event_driven: Optional[int] = Query(None),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    """Query historical events with optional filters."""
    _require_engine()
    df = events_df

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
        mask = df["event_cause"].isin(EVENT_DRIVEN_CAUSES)
        df = df[mask] if is_event_driven == 1 else df[~mask]
    if date_from or date_to:
        parsed = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
        if date_from:
            df = df[parsed >= pd.Timestamp(date_from, tz="UTC")]
        if date_to:
            df = df[parsed <= pd.Timestamp(date_to, tz="UTC")]

    total = len(df)
    page = df.iloc[offset:offset + limit]

    return {
        "total": total, "offset": offset, "limit": limit,
        "count": len(page), "events": nan_safe_records(page),
    }


@app.get("/events/summary", tags=["Historical Data"])
def get_events_summary():
    """Aggregated summary statistics plus the dropdown vocabularies the UI needs."""
    _require_engine()
    df = events_df

    station_to_zone, station_to_top_corridor = {}, {}
    if {"police_station", "zone"}.issubset(df.columns):
        valid = df[df["zone"] != "Unknown"].dropna(subset=["police_station", "zone"])
        station_to_zone = (valid.drop_duplicates(subset=["police_station"])
                                .set_index("police_station")["zone"].to_dict())
    if {"police_station", "corridor"}.issubset(df.columns):
        valid = df[~df["corridor"].isin(["Unknown", "Non-corridor"])].dropna(
            subset=["police_station", "corridor"])
        st_corr = valid.groupby("police_station")["corridor"].apply(
            lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()
        station_to_top_corridor = {k: v for k, v in st_corr.items() if v is not None}

    severity_counts = df["severity_tier"].value_counts(dropna=False).to_dict()
    severity_counts = {
        ("Unknown" if (isinstance(k, float) and math.isnan(k)) or k is None else str(k)): int(v)
        for k, v in severity_counts.items()
    }

    sev_metrics = forecast_engine.severity_metrics if forecast_engine else {}

    return {
        "total_events": len(df),
        "by_severity": severity_counts,
        "severity_labelled_count": int(df["severity_tier"].notna().sum()),
        "by_event_cause": df["event_cause"].value_counts().to_dict(),
        "by_priority_flag": df["priority"].value_counts().to_dict() if "priority" in df.columns else {},
        "event_driven_count": int(df["event_cause"].isin(EVENT_DRIVEN_CAUSES).sum()),
        "model_accuracy_pct": (sev_metrics.get("accuracy") or 0) * 100 if sev_metrics else None,
        "model_baseline_pct": (sev_metrics.get("majority_class_baseline") or 0) * 100 if sev_metrics else None,
        "road_closure_count": int(df["requires_road_closure"].sum()),
        "corridors": sorted(df["corridor"].dropna().unique().tolist()),
        "zones": sorted(df["zone"].dropna().unique().tolist()),
        "police_stations": sorted(df["police_station"].dropna().unique().tolist()),
        "station_to_zone": station_to_zone,
        "station_to_top_corridor": station_to_top_corridor,
        "directions": sorted(df["direction"].dropna().unique().tolist()),
        "event_causes": sorted(df["event_cause"].unique().tolist()),
        "event_types": sorted(df["event_type"].unique().tolist()),
        "veh_types": sorted(df["veh_type"].unique().tolist()),
        "date_range": {
            "min": str(df["start_datetime"].dropna().min()) if df["start_datetime"].notna().any() else None,
            "max": str(df["start_datetime"].dropna().max()) if df["start_datetime"].notna().any() else None,
        },
    }


@app.get("/events/distributions", tags=["Historical Data"])
def get_event_distributions():
    """Pre-aggregated chart series.

    The dashboard used to pull /events?limit=10000 twice — every column of every row,
    ~2 MB per page load — purely to compute an hourly histogram and a closure rate in
    the browser. Both aggregations belong here; the response is a couple of KB.
    """
    _require_engine()
    df = events_df

    hourly = df["hour_of_day"].dropna().astype(int).value_counts().sort_index()
    hourly_counts = [int(hourly.get(h, 0)) for h in range(24)]

    grouped = df.groupby("event_cause")["requires_road_closure"]
    closure = (grouped.agg(["sum", "count"])
                      .assign(rate_pct=lambda x: (x["sum"] / x["count"] * 100).round(1))
                      .sort_values("rate_pct", ascending=False))

    return {
        "hourly_counts": hourly_counts,
        "closure_rate_by_cause": [
            {"event_cause": cause, "rate_pct": float(r["rate_pct"]),
             "closures": int(r["sum"]), "events": int(r["count"])}
            for cause, r in closure.iterrows()
        ],
        "reporting_bias_note": (
            "Event counts by hour reflect patrol and reporting shifts, not incident "
            "rates. Most events are logged between 19:00 and 08:00 IST."
        ),
    }


@app.get("/hotspots", tags=["Historical Data"])
def get_hotspots(
    group_by: str = Query("corridor"),
    event_cause: Optional[str] = Query(None),
    hour: Optional[int] = Query(None, ge=0, le=23),
):
    """Aggregated event counts for heatmap visualisation."""
    _require_engine()
    valid_groups = ["corridor", "zone", "police_station"]
    if group_by not in valid_groups:
        raise HTTPException(400, f"group_by must be one of {valid_groups}")

    df = events_df
    if event_cause:
        df = df[df["event_cause"] == event_cause]
    if hour is not None:
        df = df[df["hour_of_day"] == hour]

    grouped = df.groupby(group_by).agg(
        event_count=("id", "count"),
        avg_lat=("latitude", "mean"),
        avg_lon=("longitude", "mean"),
        high_severity_count=("severity_tier", lambda x: int((x == "High").sum())),
        road_closure_count=("requires_road_closure", "sum"),
        avg_duration_min=("duration_to_close_min", "mean"),
    ).reset_index().sort_values("event_count", ascending=False)

    return {
        "group_by": group_by,
        "total_groups": len(grouped),
        "hotspots": nan_safe_records(grouped),
    }


@app.get("/hotspots/geo", tags=["Historical Data"])
def get_hotspots_geo(
    event_cause: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    corridor: Optional[str] = Query(None),
    limit: int = Query(2000, ge=1, le=20000),
):
    """Individual event points for map markers."""
    _require_engine()
    df = events_df
    if event_cause:
        df = df[df["event_cause"] == event_cause]
    if severity:
        df = df[df["severity_tier"] == severity]
    if corridor:
        df = df[df["corridor"] == corridor]

    df = df.head(limit)
    points = []
    for row in df.itertuples():
        if pd.isna(row.latitude) or pd.isna(row.longitude):
            continue
        points.append({
            "id": row.id,
            "lat": float(row.latitude),
            "lon": float(row.longitude),
            "event_cause": row.event_cause,
            "severity_tier": row.severity_tier if pd.notna(row.severity_tier) else "Unknown",
            "corridor": row.corridor,
            "requires_road_closure": bool(row.requires_road_closure),
            "hour_of_day": int(row.hour_of_day) if pd.notna(row.hour_of_day) else None,
            "duration_min": round(float(row.duration_to_close_min), 1)
                            if pd.notna(row.duration_to_close_min) else None,
            "description": str(row.description)[:150] if pd.notna(row.description) else None,
        })
    return {"count": len(points), "points": points}


@app.get("/corridors", tags=["Historical Data"])
def get_corridors():
    """Corridor centroids and their nearest-neighbour adjacency."""
    if not CORRIDOR_CENTROIDS.exists() or not CORRIDOR_ADJACENCY.exists():
        raise HTTPException(503, "Corridor geography not built. Run backend/data_cleaning.py.")

    centroids = pd.read_csv(CORRIDOR_CENTROIDS)
    adjacency = pd.read_csv(CORRIDOR_ADJACENCY)

    corridors = []
    for _, row in centroids.iterrows():
        name = row["corridor"]
        corridors.append({
            "name": name,
            "centroid_lat": row["centroid_lat"],
            "centroid_lon": row["centroid_lon"],
            "event_count": int(row["event_count"]),
            "neighbors": nan_safe_records(adjacency[adjacency["corridor"] == name]),
        })
    return {"count": len(corridors), "corridors": corridors}


@app.get("/eda", tags=["Historical Data"])
def get_eda_charts():
    """List available EDA chart images."""
    charts = []
    if EDA_DIR.exists():
        for f in sorted(EDA_DIR.glob("*.png")):
            charts.append({
                "filename": f.name,
                "url": f"/eda-charts/{f.name}",
                "title": f.stem.replace("_", " ").title(),
            })
    return {"count": len(charts), "charts": charts}


# ═══════════════════════════════════════════════════════════════════════════════
#  Forecasting
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/forecast", tags=["Forecasting"])
def forecast_event(request: ForecastRequest):
    """Severity forecast, duration interval, resource plan and diversion options."""
    _require_engine()

    lat, lon = request.latitude, request.longitude
    if lat is None or lon is None:
        centroid = corridor_centroid_lookup.get(request.corridor)
        if centroid:
            lat, lon = centroid
        else:
            lat, lon = 12.97, 77.59

    event_input = {
        "event_cause": request.event_cause,
        "event_type": request.event_type,
        "corridor": request.corridor,
        "zone": request.zone,
        "police_station": request.police_station,
        "direction": (request.direction or "unknown").strip().lower(),
        "hour_of_day": request.hour_of_day,
        "day_of_week": request.day_of_week,
        "is_weekend": request.is_weekend,
        "requires_road_closure": request.requires_road_closure,
        "veh_type": request.veh_type,
        "description": request.description,
        "latitude": lat,
        "longitude": lon,
    }

    forecast_result = forecast_engine.forecast(event_input)
    recommendation = rec_engine.recommend(forecast_result, event_input)

    sim_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"
    active_predictions[sim_id] = {
        "severity_tier": forecast_result["severity_tier"],
        "duration_to_close_min": forecast_result["expected_duration_min"],
        "created_at": datetime.now().isoformat(),
    }
    while len(active_predictions) > MAX_ACTIVE_PREDICTIONS:
        active_predictions.popitem(last=False)

    return {
        "event_id": sim_id,
        "input": event_input,
        "forecast": forecast_result,
        "recommendation": recommendation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Learning Loop
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/feedback", tags=["Learning Loop"])
def submit_feedback(request: FeedbackRequest):
    """Log an actual event outcome against what the model predicted.

    Predictions are looked up, never invented. The previous implementation fabricated
    a plausible-looking prediction for unrecognised IDs by perturbing the answer the
    user had just supplied, which guaranteed a flattering accuracy figure and crashed
    outright when the duration was omitted.
    """
    _require_engine()
    event_id = request.event_id.strip()
    if not event_id:
        raise HTTPException(400, "event_id must not be empty")
    if request.actual_severity is None and request.actual_duration_min is None:
        raise HTTPException(400, "Provide at least one of actual_severity or actual_duration_min")

    predicted_severity = None
    predicted_duration = None
    prediction_source = "none"

    event_row = events_df[events_df["id"].astype(str) == event_id]
    if len(event_row) > 0:
        row_dict = nan_safe_records(event_row)[0]
        f_res = forecast_engine.forecast(row_dict)
        predicted_severity = f_res["severity_tier"]
        predicted_duration = f_res["expected_duration_min"]
        prediction_source = "model_on_historical_event"
    elif event_id in active_predictions:
        stored = active_predictions[event_id]
        predicted_severity = stored["severity_tier"]
        predicted_duration = stored["duration_to_close_min"]
        prediction_source = "response_planner_session"
    else:
        raise HTTPException(
            404,
            f"No prediction on record for event '{event_id}'. Feedback can only be "
            f"logged against a historical event ID or a SIM- ID returned by /forecast.",
        )

    timestamp = datetime.now().isoformat()
    with _feedback_lock:
        write_header = not LEARNING_LOG.exists() or LEARNING_LOG.stat().st_size == 0
        with open(LEARNING_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(LEARNING_LOG_COLUMNS)
            writer.writerow([
                timestamp, event_id, predicted_severity, predicted_duration,
                request.actual_severity, request.actual_duration_min, request.feedback_notes,
            ])
        _feedback_cache["mtime"] = None

    return {
        "status": "logged",
        "event_id": event_id,
        "timestamp": timestamp,
        "prediction_source": prediction_source,
        "predicted_severity": predicted_severity,
        "actual_severity": request.actual_severity,
        "predicted_duration_min": predicted_duration,
        "actual_duration_min": request.actual_duration_min,
    }


def _compute_feedback_metrics(df: pd.DataFrame) -> dict:
    """Live accuracy over logged feedback, counting only rows that can be scored."""
    dur = df.dropna(subset=["actual_duration_min", "predicted_duration_min"])
    mae = float((dur["actual_duration_min"] - dur["predicted_duration_min"]).abs().mean()) if len(dur) else None
    medae = float((dur["actual_duration_min"] - dur["predicted_duration_min"]).abs().median()) if len(dur) else None

    sev = df.dropna(subset=["actual_severity", "predicted_severity"])
    acc = float((sev["actual_severity"] == sev["predicted_severity"]).mean() * 100) if len(sev) else None

    return {
        "mae_duration_min": round(mae, 1) if mae is not None else None,
        "median_ae_duration_min": round(medae, 1) if medae is not None else None,
        "accuracy_severity_pct": round(acc, 1) if acc is not None else None,
        "scored_duration_events": int(len(dur)),
        "scored_severity_events": int(len(sev)),
        "total_feedback_events": int(len(df)),
    }


@app.get("/feedback/log", tags=["Learning Loop"])
def get_feedback_log(limit: int = Query(50, ge=1, le=500)):
    """Learning log entries plus live accuracy metrics.

    Cached on the file's mtime: the dashboard polls this every few seconds and the
    log grows without bound, so re-parsing the whole CSV per request was pure waste.
    """
    if not LEARNING_LOG.exists():
        return {"count": 0, "metrics": None, "entries": []}

    mtime = LEARNING_LOG.stat().st_mtime
    if _feedback_cache["mtime"] != mtime:
        df = pd.read_csv(LEARNING_LOG)
        _feedback_cache["mtime"] = mtime
        _feedback_cache["payload"] = {
            "metrics": _compute_feedback_metrics(df) if len(df) else None,
            "df": df,
        }

    cached = _feedback_cache["payload"]
    df = cached["df"]
    if len(df) == 0:
        return {"count": 0, "metrics": None, "entries": []}

    recent = df.sort_values("timestamp", ascending=False).head(limit)
    return {
        "count": len(recent),
        "metrics": cached["metrics"],
        "entries": nan_safe_records(recent),
    }


def _run_retrain():
    try:
        from backend.forecasting import run_training_pipeline
        run_training_pipeline(include_feedback=True)
        _load_all()
        retrain_state["last_result"] = "success"
    except Exception as exc:  # surfaced through /models/retrain/status
        retrain_state["last_result"] = f"failed: {exc}"
    finally:
        retrain_state["running"] = False
        retrain_state["last_finished"] = datetime.now().isoformat()


@app.post("/models/retrain", tags=["Learning Loop"])
def retrain_models(background_tasks: BackgroundTasks):
    """Retrain both models with officer feedback folded into the training set.

    This is the step the "continuous learning loop" always described and never did.
    Feedback rows overwrite the outcome of the matching historical event, severity is
    recomputed from the corrected duration, and the models are rebuilt and hot-swapped
    without restarting the server.
    """
    if retrain_state["running"]:
        return {"status": "already_running"}
    retrain_state["running"] = True
    retrain_state["last_result"] = None
    background_tasks.add_task(_run_retrain)
    return {"status": "started", "note": "Retraining runs in the background; poll /models/retrain/status."}


@app.get("/models/retrain/status", tags=["Learning Loop"])
def retrain_status():
    return dict(retrain_state)


@app.get("/models/metrics", tags=["System"])
def get_model_metrics():
    """Offline evaluation metrics for the trained models, with baselines."""
    if not MODEL_METRICS.exists():
        raise HTTPException(503, "Model metrics not found. Run backend/forecasting.py.")
    with open(MODEL_METRICS) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  Computer Vision
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/vision/analyze", tags=["Computer Vision"])
async def analyze_vision(file: UploadFile = File(...)):
    """Analyse an uploaded CCTV frame with YOLOv8."""
    from backend import cv_worker

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(415, "Upload must be an image")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
    if not contents:
        raise HTTPException(400, "Empty upload")

    try:
        result = cv_worker.analyze(contents)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/vision/junction/{junction_id}", tags=["Computer Vision"])
def analyze_junction_feed(junction_id: str):
    """Run YOLOv8 over a stored still from a named junction camera."""
    from backend import cv_worker

    if not junction_id.replace("_", "").isalnum():
        raise HTTPException(400, "Invalid junction id")

    image_path = CCTV_DIR / f"cam_{junction_id}.png"
    if not image_path.exists():
        raise HTTPException(404, f"Camera feed unavailable for junction: {junction_id}")

    try:
        result = cv_worker.analyze(image_path.read_bytes())
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


def _read_alerts() -> list:
    if not ALERTS_PATH.exists():
        return []
    try:
        with open(ALERTS_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


@app.get("/alerts/live", tags=["Computer Vision"])
def get_live_alerts():
    """Latest autonomous CCTV alerts, WITHOUT the annotated frame.

    The frame is a ~880 KB base64 data URI. This endpoint is polled every few seconds,
    so shipping the image with every poll transferred roughly a gigabyte an hour to an
    idle tab. Clients fetch /alerts/live/{id}/image only when the alert id changes.
    """
    alerts = _read_alerts()
    return [{k: v for k, v in a.items() if k != "annotated_image"} for a in alerts]


@app.get("/alerts/live/{alert_id}/image", tags=["Computer Vision"])
def get_alert_image(alert_id: str):
    """Annotated frame for one alert, served as a real JPEG rather than a data URI."""
    import base64
    for alert in _read_alerts():
        if alert.get("id") == alert_id:
            data_uri = alert.get("annotated_image") or ""
            if "," not in data_uri:
                raise HTTPException(404, "Alert has no image")
            raw = base64.b64decode(data_uri.split(",", 1)[1])
            return Response(content=raw, media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=3600"})
    raise HTTPException(404, f"No alert with id {alert_id}")


# ═══════════════════════════════════════════════════════════════════════════════
#  System / Frontend
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy" if forecast_engine is not None else "loading",
        "events_loaded": len(events_df) if events_df is not None else 0,
        "forecasting_engine": forecast_engine is not None,
        "recommendation_engine": rec_engine is not None,
        "learning_log_exists": LEARNING_LOG.exists(),
        "retraining": retrain_state["running"],
    }


@app.get("/", tags=["Frontend"])
def serve_landing():
    return FileResponse(str(FRONTEND_DIR / "landing.html"))


@app.get("/dashboard", tags=["Frontend"])
def serve_dashboard():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# Mounted under a distinct prefix so it cannot collide with the GET /eda route above.
if EDA_DIR.exists():
    app.mount("/eda-charts", StaticFiles(directory=str(EDA_DIR)), name="eda_charts")
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 7860)), reload=True)
