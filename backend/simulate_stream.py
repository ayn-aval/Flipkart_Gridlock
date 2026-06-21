"""
Gridlock Phase 6 — Real-Time Simulation Stream
================================================

Simulates a continuous stream of incoming traffic events to test
the dashboard and post-event learning loop in real-time.

It runs an infinite loop that:
1. Picks a random historical event
2. Sends a /forecast request to get predictions
3. Generates "actual" outcomes by applying slight variation to the true data
4. Sends a /feedback request to log the outcome
"""

import time
import random
import requests
import pandas as pd
from pathlib import Path

# Paths & URLs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "events_clean.csv"
API_BASE_URL = "http://127.0.0.1:8000"

def load_data():
    """Load the processed event data."""
    print(f"Loading data from {PROCESSED_DATA_PATH}...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        # Filter to events that have valid duration and severity
        df = df[df["duration_to_close_min"].notna() & df["severity_tier"].notna()]
        print(f"Loaded {len(df)} valid events for simulation.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

def run_simulation(df):
    """Run the continuous simulation loop."""
    print("\n🚀 Starting Real-Time Simulation Stream...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            # ────────────────────────────────────────────────────────
            # 1. Autonomous CCTV AI Loop (20% chance per cycle)
            # ────────────────────────────────────────────────────────
            cctv_samples_dir = PROJECT_ROOT / "data" / "cctv_samples"
            if random.random() < 0.20 and cctv_samples_dir.exists():
                images = list(cctv_samples_dir.glob("*.png")) + list(cctv_samples_dir.glob("*.jpg"))
                if images:
                    random_image = random.choice(images)
                    print(f"📷 [CCTV AUTO-POLL] Analyzing {random_image.name}...")
                    
                    try:
                        with open(random_image, "rb") as f:
                            files = {"file": (random_image.name, f, "image/jpeg")}
                            res_cv = requests.post(f"{API_BASE_URL}/vision/analyze", files=files)
                        
                        if res_cv.status_code == 200:
                            cv_data = res_cv.json()
                            if cv_data["severity"] == "High":
                                print(f"🚨 [CCTV AI ALERT] {cv_data['status']} detected! Triggering ML Forecaster...")
                                
                                # Call XGBoost with a generic payload representing the CCTV detection
                                cctv_payload = {
                                    "event_cause": "congestion" if "Congestion" in cv_data["status"] else "accident",
                                    "event_type": "camera_detection",
                                    "corridor": "ORR South 1", # Mock location
                                    "zone": "South",
                                    "police_station": "Madiwala",
                                    "direction": "south",
                                    "hour_of_day": 14,
                                    "day_of_week": 3,
                                    "is_weekend": 0,
                                    "requires_road_closure": 0,
                                    "veh_type": "none",
                                    "description": f"AI Auto-detected: {cv_data['status']}. Vehicles: {cv_data['total_vehicles']}"
                                }
                                
                                res_fc = requests.post(f"{API_BASE_URL}/forecast", json=cctv_payload)
                                if res_fc.status_code == 200:
                                    fc_data = res_fc.json()["forecast"]
                                    
                                    # Save to active alerts JSON
                                    import json
                                    from datetime import datetime
                                    alert_path = PROJECT_ROOT / "data" / "active_alerts.json"
                                    
                                    # Generate context-aware recommendations in simple language
                                    recs = [
                                        f"Send patrol car from {cctv_payload['police_station']} station immediately.",
                                        "Update display boards to divert incoming traffic.",
                                        "Dispatch heavy towing crane." if cv_data["severity"] == "High" else "Monitor live feed for next 15 mins."
                                    ]
                                    
                                    alert_record = {
                                        "id": f"CCTV-{int(time.time())}",
                                        "timestamp": datetime.now().isoformat(),
                                        "cv_status": cv_data["status"],
                                        "total_vehicles": cv_data["total_vehicles"],
                                        "predicted_severity": fc_data["severity_tier"],
                                        "predicted_duration_min": round(fc_data["expected_duration_min"], 1),
                                        "location": f"{cctv_payload['corridor']} ({cctv_payload['direction'].title()}bound)",
                                        "recommendations": recs,
                                        "annotated_image": cv_data["annotated_image"]
                                    }
                                    
                                    # Write as list
                                    with open(alert_path, "w") as f:
                                        json.dump([alert_record], f)
                                        
                                    print(f"✅ [CCTV PIPELINE] Alert dispatched! Expected clear time: {fc_data['expected_duration_min']:.1f}m\n")
                    except Exception as e:
                        print(f"  ❌ CCTV Pipeline Error: {e}")

            # Wait briefly so CCTV logs stand out
            time.sleep(1)

            # ────────────────────────────────────────────────────────
            # 2. Historical Event Stream Simulation (Original Logic)
            # ────────────────────────────────────────────────────────
            event = df.sample(1).iloc[0]
            event_id = event["id"]
            
            # Map required fields for forecast request
            corridor = event["corridor"] if pd.notna(event["corridor"]) else "Non-corridor"
            
            forecast_payload = {
                "event_cause": event["event_cause"],
                "event_type": event["event_type"],
                "corridor": corridor,
                "zone": event["zone"] if pd.notna(event["zone"]) else "Unknown",
                "police_station": event["police_station"] if pd.notna(event["police_station"]) else "Unknown",
                "direction": event["direction"] if pd.notna(event["direction"]) else "unknown",
                "hour_of_day": int(event["hour_of_day"]),
                "day_of_week": int(event["day_of_week"]),
                "is_weekend": int(event["is_weekend"]),
                "requires_road_closure": int(event["requires_road_closure"]),
                "veh_type": event["veh_type"] if pd.notna(event["veh_type"]) else "none",
                "description": str(event["description"]) if pd.notna(event["description"]) else ""
            }

            print(f"📡 Dispatching Event: {event_id} ({event['event_cause']} on {corridor})")
            
            # 2. Get Forecast
            try:
                res = requests.post(f"{API_BASE_URL}/forecast", json=forecast_payload)
                if res.status_code != 200:
                    print(f"  ❌ Forecast API Error: {res.text}")
                    time.sleep(2)
                    continue
                
                forecast_data = res.json()["forecast"]
                predicted_duration = forecast_data["expected_duration_min"]
                predicted_severity = forecast_data["severity_tier"]
            except Exception as e:
                print(f"  ❌ Failed to reach API: {e}")
                time.sleep(5)
                continue

            print(f"  🔮 Forecast: {predicted_severity} severity | {predicted_duration:.1f} min")

            # 3. Simulate "Actuals" (true value + random jitter)
            true_duration = event["duration_to_close_min"]
            true_severity = event["severity_tier"]
            
            # Apply some random jitter to duration (-10% to +10%)
            jitter = random.uniform(0.9, 1.1)
            actual_duration = round(true_duration * jitter, 1)
            
            # Usually severity is same as true, sometimes simulate a surprise
            actual_severity = true_severity
            if random.random() < 0.05:  # 5% chance of anomalous severity
                tiers = ["Low", "High"]
                if true_severity in tiers:
                    tiers.remove(true_severity)
                actual_severity = random.choice(tiers) if tiers else true_severity

            print(f"  🕒 Actuals Logged: {actual_severity} severity | {actual_duration:.1f} min")

            # 4. Submit Feedback
            feedback_payload = {
                "event_id": event_id,
                "actual_severity": actual_severity,
                "actual_duration_min": actual_duration,
                "feedback_notes": f"Auto-simulated outcome (True: {true_duration:.1f}m)"
            }

            try:
                res = requests.post(f"{API_BASE_URL}/feedback", json=feedback_payload)
                if res.status_code == 200:
                    print("  ✅ Feedback recorded successfully.\n")
                else:
                    print(f"  ❌ Feedback API Error: {res.text}\n")
            except Exception as e:
                print(f"  ❌ Failed to send feedback: {e}\n")

            # Wait before next event
            time.sleep(random.uniform(5.0, 8.0))

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")

if __name__ == "__main__":
    df = load_data()
    run_simulation(df)
