# 🚦 Event-Driven Congestion Forecasting & Resource Recommendation System

**Flipkart Gridlock 2.0 — Round 2 Prototype**

## Problem Statement

Political rallies, festivals, sports events, construction activities, and sudden gatherings create localized traffic breakdowns in Bengaluru. Today:
- **Event impact is not quantified in advance** → addressed by our Forecasting Engine (Phase 2)
- **Resource deployment is experience-driven** → addressed by our Recommendation Engine (Phase 3)
- **No post-event learning system** → addressed by our Feedback Loop (Phase 6)

## Architecture

```
backend/        FastAPI app: data loading, feature pipeline, forecasting model, recommendation engine, learning-log
frontend/       Lightweight web dashboard (HTML/JS + Leaflet + Chart.js) consuming the API
data/raw/       Original CSV (Astram event data, anonymized)
data/processed/ Cleaned/feature-engineered data, trained model artifacts, audit reports
notebooks/      EDA (optional, not required for the running app)
README.md
ASSUMPTIONS.md
```

## Dataset

- **Source:** Anonymized Bengaluru traffic-event log from the Astram platform
- **Size:** ~8,200 rows × 46 columns
- **Time span:** November 2023 – April 2024
- **Placed at:** `data/raw/astram_events.csv`

See `data/processed/data_audit.md` for the full data audit report.

## Build Progress

| Phase | Description | Status |
|-------|------------|--------|
| 0 | Scaffolding & Data Audit | ✅ Complete |
| 1 | Cleaning & Feature Engineering | ✅ Complete |
| 2 | Impact Forecasting Engine | ✅ Complete |
| 3 | Resource Recommendation Engine | ✅ Complete |
| 4 | Backend API | ✅ Complete |
| 5 | Dashboard Frontend | ✅ Complete |
| 6 | Real-Time Simulation + Learning Loop | ✅ Complete |
| 7 | Polish & Demo Readiness | ✅ Complete |

## Quick Start (Demo Mode)

The easiest way to see the prototype in action is to use the demo script, which launches both the FastAPI backend and the Real-Time Event Simulator in the background:

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the prototype
./run_demo.sh
```

Then open your browser to **http://localhost:8000/**.

## Advanced Usage

If you prefer to run components individually or train models from scratch:

```bash
# Run data audit
python3 backend/data_audit.py

# Run cleaning & feature engineering
python3 backend/data_cleaning.py

# Train forecasting models
python3 backend/forecasting.py

# Run sample forecasts (after training)
python3 backend/forecasting.py --test

# Run the API manually
python3 -m uvicorn backend.api:app --port 8000
```

## Phase 1 Outputs

### Cleaning Operations
- Event cause normalization: fixed `Debris`→`debris`, merged ultra-rare causes (<5 events) into `others`
- Datetime parsing: all 6 datetime columns parsed to UTC
- Coordinate cleaning: zeroed end-coordinates set to NaN, 2 out-of-range rows fixed
- Missing corridor (20 rows) → `Non-corridor`, missing priority (2 rows) → `High`
- Boolean conversion for `requires_road_closure`

### Engineered Features
| Feature | Description |
|---------|-------------|
| `hour_of_day` | Hour in IST (0–23) |
| `day_of_week` | 0=Monday, 6=Sunday |
| `is_weekend` | Binary flag |
| `time_period` | morning_rush / midday / evening_rush / night |
| `duration_to_close_min` | `closed_datetime − start_datetime` in minutes (filtered: >0 and ≤7 days) |
| `severity_tier` | Low / Medium / High (from priority + duration + road_closure) |
| `duration_bucket` | quick / moderate / extended / prolonged / unknown |
| `is_event_driven` | Flag for theme-relevant causes |

### Corridor Geography
- `corridor_centroids.csv`: 22 corridor centroids (mean lat/lon)
- `corridor_adjacency.csv`: Top-5 nearest neighbors per corridor (haversine distance)

### EDA Charts (6 total, in `data/processed/eda_charts/`)
1. Events by cause
2. Events by hour of day (IST)
3. Top 15 corridors by event count
4. Duration-to-close distribution by cause
5. Severity tier distribution by cause
6. Event density heatmap (hour × day of week)

## Phase 2 — Forecasting Engine

### Models

| Model | Task | Key Metric | Notes |
|-------|------|------------|-------|
| GBT Severity Classifier | Low/Med/High tier | Accuracy 71.9%, F1(w) 0.67 | Trained on all 8,057 rows; `event_cause` and `requires_road_closure` are dominant features |
| GBT Duration Regressor | Minutes to clear | Median AE 34 min, MAE 366 min | Trained on 2,711 rows with valid durations; log-transformed target. High MAE driven by long-tail causes (water_logging, construction) |
| k-NN Analog Finder | Fallback for rare planned events | Cosine similarity | Returns 5 most similar past events for procession, public_event, vip_movement, protest |

### Sample Forecast (procession on Bellary Road, Sunday morning, road closure)
```
Severity: High (99.1% confidence)
Duration: 13.2 min (model) / 34.1 min median (analog)
Method: knn_analog_fallback
Analog range: 2.9–144.1 min from 5 similar past events
```

## Phase 3 — Resource Recommendation Engine

### Architecture
- **35 rule entries** in a documented lookup table: `(event_cause, severity_tier, requires_road_closure)` → officer range, barricade range, action checklist
- **Diversion engine** using corridor adjacency table — suggests 2–3 nearest alternate corridors with hour-aware rationale
- **Explicit disclaimer** on every recommendation: this is heuristic-based, not learned from historical resource data

### Sample Recommendation (procession on Bellary Road, Sunday, road closure)
```
👮 Officers: 8–15       🚧 Barriers: 10–20 units
✅ Actions: Full route closure, officers at every junction, escort with patrol
            vehicles, issue traffic advisory, phased reopening
🔀 Diversions: CBD 2 (3.8 km, 89% fewer events at 09:00),
               ORR North 2 (4.1 km, 76% fewer events at 09:00)
```

## Phase 4 — Backend API

### Running the API
```bash
python3 -m uvicorn backend.api:app --port 8000
# Swagger docs at http://localhost:8000/docs
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/events` | Query historical events (filterable by corridor, cause, severity, date) |
| GET | `/events/summary` | Aggregated stats for dashboard |
| GET | `/hotspots` | Grouped event counts by corridor/zone/station |
| GET | `/hotspots/geo` | Individual event points for map markers |
| GET | `/corridors` | Corridor centroids + adjacency data |
| GET | `/eda` | EDA chart listing + static file serving |
| POST | `/forecast` | Severity + duration forecast with resource recommendations |
| POST | `/feedback` | Log actual outcomes for learning loop |
| GET | `/feedback/log` | Retrieve learning log entries |

## Tech Stack

- **Backend:** Python, FastAPI, scikit-learn, pandas
- **Frontend:** HTML/JS, Leaflet.js, Chart.js
- **ML:** Gradient Boosted Trees (scikit-learn)
- **No paid APIs or cloud services required**
