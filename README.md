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
| 2 | Impact Forecasting Engine | ⬜ Pending |
| 3 | Resource Recommendation Engine | ⬜ Pending |
| 4 | Backend API | ⬜ Pending |
| 5 | Dashboard Frontend | ⬜ Pending |
| 6 | Real-Time Simulation + Learning Loop | ⬜ Pending |
| 7 | Polish & Demo Readiness | ⬜ Pending |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run data audit
python3 backend/data_audit.py

# Run cleaning & feature engineering
python3 backend/data_cleaning.py
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

## Tech Stack

- **Backend:** Python, FastAPI, scikit-learn, pandas
- **Frontend:** HTML/JS, Leaflet.js, Chart.js
- **ML:** Gradient Boosted Trees (scikit-learn)
- **No paid APIs or cloud services required**
