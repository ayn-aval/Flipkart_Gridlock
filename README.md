---
title: Namma Route
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

<div align="center">
  <img src="frontend/assets/logo.png" width="300" alt="Namma Route Logo">

  # Namma Route
  ### Event-Driven Congestion Forecasting & Resource Recommendation System
</div>

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/aynaval2003/namma-route)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer_Vision-yellow?style=for-the-badge)](https://ultralytics.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine_Learning-green?style=for-the-badge)](#)

> **Flipkart Gridlock 2.0 Prototype** — Namma Route transforms municipal traffic management from a *reactive* operational model to a highly *predictive* Decision Support System.

---

##  Live Deployment

The system is deployed globally as a containerized web application on Hugging Face Spaces. 

👉 **[Access the Live Dashboard Here](https://huggingface.co/spaces/aynaval2003/namma-route)**

*(Note: The application requires a Mappls API key to render the high-performance geographic tiles. This is injected securely via Hugging Face Secrets).*

---

## System Architecture & Data Flow

The system consists of four deeply integrated layers that automate the entire lifecycle of a traffic event—from detection to resolution.

```mermaid
graph TD
    %% Define styles
    classDef core fill:#e7f5ff,stroke:#74c0fc,stroke-width:2px,color:#212529;
    classDef ml fill:#f3f0ff,stroke:#b197fc,stroke-width:2px,color:#212529;

    A[CCTV RTSP Feeds] -->|Live Feed| B(YOLOv8 Vision Engine):::core
    A2[Manual ASTRAM Entry] -->|Dispatcher| D(Event Ingestion Pipeline)
    
    B -->|Anomaly Detection| D
    
    D --> E{Impact Forecasting Engine}:::ml
    
    E -->|Duration Prediction| F[XGBoost Regressor]:::ml
    E -->|Severity Classification| G[GBT Classifier]:::ml
    E -->|Irregular Event Fallback| H[KNN Analog Finder]:::ml

    F --> I[Resource Recommendation Engine]:::core
    G --> I
    H --> I
    
    I -->|Deployment & Diversions| K[Traffic Control Dashboard]
    
    K -->|Post-Resolution Clearance Data| L(Continuous Learning Loop):::core
    L -.->|Automated Retraining| E
```

---

##  Core Features in Detail

### 1. Automated CCTV Watchtower (Computer Vision)
Instead of forcing human operators to monitor 10,000 city cameras, our integrated **YOLOv8** pipeline processes live junction feeds autonomously. It detects static anomalies (breakdowns, accidents) and calculates vehicle density, injecting high-priority alerts directly into the forecasting engine with zero latency.

### 2. Impact Forecasting Engine (Machine Learning)
When an event occurs, the system instantly predicts the fallout using models trained on 8,057 historical Bengaluru traffic events:
- **Severity Classifier (Gradient Boosted Trees):** Achieves **71.9% accuracy** in predicting Low/Medium/High severity tiers based on event cause, time of day, and road closure requirements.
- **Duration Regressor (XGBoost):** Predicts the exact minutes required to clear the incident. The target variable is log-transformed during training to account for the heavy right-tail distribution of prolonged events.
- **Analog Fallback (K-Nearest Neighbors):** For highly irregular planned events (e.g., massive political rallies), standard regression underperforms. The system falls back to a KNN model using Cosine Similarity to find the 5 most historically similar past events and returns their median clearance time.

### 3. Surgical Resource Recommendation Engine
A deterministic heuristic layer that translates ML outputs into actionable field operations.
- Cross-references predicted severity against a predefined matrix to allocate exact personnel count, barricades, and specialized vehicles (e.g., Heavy Tow Trucks).
- **Diversion Routing:** Utilizes a pre-computed Haversine spatial adjacency matrix to recommend the top 2 optimal alternate corridors based on historical traffic flow at that exact hour.

### 4. The Continuous Learning Loop
Traffic dynamics change constantly. The system features a "System Accuracy" tab where ground officers log the *actual* clearance time vs the *predicted* time. The system stores this telemetry data and automatically retrains its own models upon reboot, ensuring the intelligence adapts to evolving civic infrastructure.

---

##  Dataset & Exploratory Data Analysis (EDA)

The models are powered by an anonymized Bengaluru traffic-event log (November 2023 – April 2024).
- **Size:** 8,200 rows × 46 columns
- **Engineered Features:** Included `hour_of_day`, `is_weekend`, `duration_to_close_min`, `severity_tier`, and geographic centroid clustering.

*(Extensive EDA charts, including event density heatmaps and duration distributions, are available directly within the "Analytics" tab of the Live Dashboard).*

---

##  Local Installation & Usage

If you prefer to run the system locally for development or testing:

### 1. Prerequisites
- Python 3.9+
- A [Mappls (MapmyIndia) API Key](https://about.mappls.com/api/)

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/your-username/namma-route.git
cd namma-route

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
MAPPLS_API_KEY=your_actual_api_key_here
```

### 4. Launch the Application
The `run_demo.sh` script boots up both the FastAPI backend and the background Real-Time Event Simulator.
```bash
chmod +x run_demo.sh
./run_demo.sh
```
Access the dashboard at: **http://localhost:8000**

---

##  Production Deployment (Docker)

The application is fully containerized for zero-configuration cloud deployment.

The `Dockerfile`:
1. Installs necessary Linux graphics dependencies (`libgl1`, `libglib2.0-0`) for headless OpenCV execution.
2. Executes `python3 backend/forecasting.py` to compile and pickle the ML models locally *inside* the container, completely preventing cross-version Pandas unpickling errors.
3. Exposes port `7860` and initiates `start_prod.sh`.
