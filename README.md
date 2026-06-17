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
| 1 | Cleaning & Feature Engineering | ⬜ Pending |
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
python backend/data_audit.py
```

## Tech Stack

- **Backend:** Python, FastAPI, scikit-learn, pandas
- **Frontend:** HTML/JS, Leaflet.js, Chart.js
- **ML:** Gradient Boosted Trees (scikit-learn)
- **No paid APIs or cloud services required**
