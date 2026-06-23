---
title: Namma Route
emoji: 🚦
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# 🚦 Namma Route — Event-Driven Congestion Forecasting & Resource Recommendation System

**A Paradigm Shift in Traffic Management from Reactive to Proactive**

![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Deployed-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer_Vision-yellow)

---

## 📖 The Problem

Today, city traffic management is largely **reactive**. An accident happens, a traffic jam builds up over 30 minutes, mapping applications turn red, and *then* the police dispatch a unit. By the time the unit arrives, the gridlock has already paralyzed the corridor. Furthermore, dispatching is guesswork—sending a standard patrol car to a flipped truck wastes 40 minutes because they eventually have to call a heavy crane anyway.

## 💡 Our Solution: Namma Route

**Namma Route** is a comprehensive, AI-powered Decision Support System built for Traffic Police Command Centers. Instead of waiting for gridlock, the system detects anomalies, instantly predicts the fallout, recommends surgical resource deployments, and learns from every incident.

### 🌟 Core Features (The 4 Pillars)

1. **Automated CCTV Watchtower (YOLOv8)**
   It is physically impossible for humans to monitor 10,000 city cameras 24/7. Our built-in Computer Vision engine automatically processes live junction feeds to detect accidents, breakdowns, and vehicle density. When an anomaly is spotted, it automatically dispatches an alert into the forecasting engine without human intervention.
   
2. **Impact Forecasting Engine (XGBoost/KNN)**
   The moment an event is logged, the Machine Learning models instantly predict the exact severity and how many minutes the congestion will last. This allows traffic controllers to proactively trigger diversions *before* the gridlock cascades.

3. **Surgical Resource Recommendation Engine**
   Instead of guessing what to send, the "Response Planner" tells dispatchers exactly what resources to deploy based on the ML severity prediction (e.g., Heavy Tow Truck + Medical Response Team). It also automatically calculates the optimal alternate diversion corridors based on historical traffic flow at that exact hour.

4. **Continuous Learning Loop**
   Traffic dynamics change constantly. Our system features a "Feedback Loop" where officers on the ground can input the actual clearance time vs the AI's predicted time. The system stores this feedback and automatically retrains its own models, getting smarter and more accurate every single week.

---

## 🚀 Live Demo (For Judges)

We have deployed a 24/7 live prototype on Hugging Face Spaces for instant evaluation.

**[Launch Live Dashboard on Hugging Face](https://huggingface.co/spaces/aynaval2003/namma-route)**

> **Important Note on Map Data:** The application uses Mappls (formerly MapmyIndia) for rendering the high-performance dark-mode map tiles. The backend fetches the required API key dynamically from the server environment.

---

## 🛠 System Architecture

```text
namma-route/
├── backend/
│   ├── api.py                 # Core FastAPI server
│   ├── cv_engine.py           # YOLOv8 Computer Vision module
│   ├── forecasting.py         # XGBoost & KNN Model training & inference
│   ├── recommendation.py      # Rule-based resource & diversion logic
│   └── simulate_stream.py     # Background worker firing real-time events
├── frontend/
│   ├── app.js                 # Dynamic dashboard logic & API polling
│   ├── index.html             # UI Layout (Vanilla CSS/HTML)
│   └── index.css              # Custom styling (Dark Mode glassmorphism)
├── data/
│   ├── raw/                   # Original dataset (Astram events)
│   ├── processed/             # Cleaned data, pickled ML models, learning logs
│   └── cctv_samples/          # Simulated video feeds for YOLO testing
├── Dockerfile                 # Production Docker image definition
├── start_prod.sh              # Entry point script (launches API + Simulator)
└── requirements.txt           # Python dependencies
```

---

## 💻 Local Installation & Deployment

If you prefer to run the system locally on your own machine, follow these steps:

### Prerequisites
- Python 3.9+
- A [Mappls (MapmyIndia) API Key](https://about.mappls.com/api/)

### 1. Clone & Install
```bash
git clone https://github.com/your-username/namma-route.git
cd namma-route

# Install all required Python packages (including OpenCV, FastAPI, and YOLO)
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory and add your API key:
```env
MAPPLS_API_KEY=your_actual_api_key_here
```

### 3. Launch the Prototype
You can use the built-in demo launcher, which spins up both the **FastAPI Backend Server** and the **Real-Time Event Simulator** concurrently.
```bash
chmod +x run_demo.sh
./run_demo.sh
```

**Access the Application:**
Open your browser and navigate to: **http://localhost:8000**

---

## 🧠 Machine Learning Details

### Model 1: Severity Classifier (Gradient Boosted Trees)
- **Task:** Classifies an event as Low, Medium, or High tier impact.
- **Features:** Event cause, time of day, day of week, requires road closure.
- **Performance:** 71.9% Accuracy on the historical dataset.

### Model 2: Duration Regressor (XGBoost)
- **Task:** Predicts the exact minutes required to clear the incident.
- **Handling Outliers:** The target variable is log-transformed due to the heavy right-tail of prolonged events (like major water-logging or construction).

### Model 3: K-Nearest Neighbors (Analog Fallback)
- **Task:** For highly irregular planned events (like massive political rallies or VIP movements), standard regression often fails. The system falls back to a KNN model using Cosine Similarity to find the 5 most historically similar past events and returns their median clearance time as a highly reliable benchmark.

---

## 🔒 Production Deployment (Hugging Face / Docker)

This application is fully Dockerized for zero-configuration cloud deployment.

To deploy on Hugging Face Spaces:
1. Create a new **Docker** Space.
2. Upload this repository directly.
3. Go to Space Settings -> Variables and secrets.
4. Add a Secret with the Name: `MAPPLS_API_KEY` and Value: `your_api_key`.
5. The `Dockerfile` will automatically handle installing Linux dependencies (like `libgl1` for OpenCV), retraining the models to match the container's Python environment, and launching the application on port `7860`.

---
*Developed for Flipkart Gridlock 2.0*
