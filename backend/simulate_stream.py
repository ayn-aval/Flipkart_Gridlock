"""
Real-Time Simulation Stream
===========================
Replays historical events against the live API so the dashboard has something moving
during a demo, and exercises the post-event learning loop end to end.

OFF BY DEFAULT. Set ENABLE_SIMULATOR=1 to run it.

It used to be started unconditionally by start_prod.sh, which meant the public
deployment continuously manufactured events and appended synthetic "officer feedback"
to the learning log — inflating the very counters the dashboard presented as evidence
of real use. Simulated rows are now clearly tagged so they can never be mistaken for
real feedback, and the whole process has to be switched on deliberately.

Usage:
    ENABLE_SIMULATOR=1 python3 backend/simulate_stream.py
"""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "events_clean.csv"
CCTV_SAMPLES_DIR = PROJECT_ROOT / "data" / "cctv_samples"
ALERTS_PATH = PROJECT_ROOT / "data" / "active_alerts.json"

PORT = os.environ.get("PORT", "7860")
API_BASE_URL = f"http://127.0.0.1:{PORT}"

SIMULATED_NOTE = "SIMULATED — not officer-reported"


def wait_for_api(timeout: int = 90) -> bool:
    """Block until the API reports healthy. The simulator used to start racing uvicorn
    and burn its first several requests on connection errors."""
    print(f"Waiting for API at {API_BASE_URL} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            res = requests.get(f"{API_BASE_URL}/health", timeout=3)
            if res.ok and res.json().get("status") == "healthy":
                print("API is healthy — starting simulation.\n")
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    print("API did not become healthy in time; giving up.")
    return False


def load_data() -> pd.DataFrame:
    print(f"Loading data from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df = df[df["duration_to_close_min"].notna() & df["severity_tier"].notna()]
    print(f"Loaded {len(df)} events with measured outcomes for simulation.")
    return df


def run_cctv_cycle():
    """Analyse a random sample frame and, if it reads as congested, raise an alert."""
    images = list(CCTV_SAMPLES_DIR.glob("*.png")) + list(CCTV_SAMPLES_DIR.glob("*.jpg"))
    if not images:
        return

    image = random.choice(images)
    print(f"[CCTV] Analysing {image.name} ...")
    try:
        with open(image, "rb") as f:
            res_cv = requests.post(f"{API_BASE_URL}/vision/analyze",
                                   files={"file": (image.name, f, "image/png")}, timeout=60)
        if not res_cv.ok:
            return
        cv_data = res_cv.json()
        if cv_data.get("severity") != "High":
            print(f"[CCTV] {cv_data.get('status')} — no alert raised.")
            return

        print(f"[CCTV] {cv_data['status']} — running forecast ...")
        payload = {
            "event_cause": "congestion",
            "event_type": "unplanned",
            "corridor": "ORR South 1",
            "zone": "South Zone 1",
            "police_station": "Madiwala",
            "direction": "south",
            "hour_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "is_weekend": int(datetime.now().weekday() >= 5),
            "requires_road_closure": 0,
            "veh_type": "none",
            "description": (f"camera detected {cv_data['status'].lower()}; "
                            f"{cv_data['total_vehicles']} vehicles in frame"),
        }
        res_fc = requests.post(f"{API_BASE_URL}/forecast", json=payload, timeout=60)
        if not res_fc.ok:
            print(f"[CCTV] Forecast failed: {res_fc.text[:200]}")
            return

        fc = res_fc.json()["forecast"]
        rec = res_fc.json()["recommendation"]
        rng = fc.get("duration_range_min", {})

        alert = {
            "id": f"CCTV-{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "cv_status": cv_data["status"],
            "total_vehicles": cv_data["total_vehicles"],
            "density_per_megapixel": cv_data.get("density_per_megapixel"),
            "predicted_severity": fc["severity_tier"],
            "predicted_duration_min": fc["expected_duration_min"],
            "duration_range_min": {"p10": rng.get("p10"), "p90": rng.get("p90")},
            "duration_basis": fc.get("duration_basis", {}).get("description"),
            "location": f"{payload['corridor']} ({payload['direction'].title()}bound)",
            "recommendations": rec["action_checklist"][:3],
            "source": "simulated_cctv_poll",
            "annotated_image": cv_data["annotated_image"],
        }
        ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ALERTS_PATH.write_text(json.dumps([alert]))
        print(f"[CCTV] Alert raised — expected clearance ~{fc['expected_duration_min']:.0f} min\n")
    except requests.RequestException as e:
        print(f"[CCTV] Pipeline error: {e}")


def run_event_cycle(df: pd.DataFrame):
    """Replay one historical event: forecast it, then report its true outcome back."""
    event = df.sample(1).iloc[0]
    payload = {
        "event_cause": event["event_cause"],
        "event_type": event["event_type"],
        "corridor": event["corridor"] if pd.notna(event["corridor"]) else "Non-corridor",
        "zone": event["zone"] if pd.notna(event["zone"]) else "Unknown",
        "police_station": event["police_station"] if pd.notna(event["police_station"]) else "Unknown",
        "direction": str(event["direction"]).lower() if pd.notna(event["direction"]) else "unknown",
        "hour_of_day": int(event["hour_of_day"]),
        "day_of_week": int(event["day_of_week"]),
        "is_weekend": int(event["is_weekend"]),
        "requires_road_closure": int(bool(event["requires_road_closure"])),
        "veh_type": event["veh_type"] if pd.notna(event["veh_type"]) else "none",
        "description": str(event["description"]) if pd.notna(event["description"]) else "",
    }
    if pd.notna(event["latitude"]) and pd.notna(event["longitude"]):
        payload["latitude"] = float(event["latitude"])
        payload["longitude"] = float(event["longitude"])

    print(f"[EVENT] {event['id']} ({event['event_cause']} on {payload['corridor']})")
    try:
        res = requests.post(f"{API_BASE_URL}/forecast", json=payload, timeout=60)
        if not res.ok:
            print(f"  Forecast error: {res.text[:200]}")
            return
        fc = res.json()["forecast"]
        print(f"  Forecast: {fc['severity_tier']} severity | "
              f"~{fc['expected_duration_min']:.0f} min "
              f"({fc['duration_range_min']['p10']:.0f}-{fc['duration_range_min']['p90']:.0f} range)")

        # The "actual" is the event's real recorded outcome with a small measurement
        # jitter — never derived from the prediction.
        true_duration = float(event["duration_to_close_min"])
        actual_duration = round(true_duration * random.uniform(0.92, 1.08), 1)
        actual_severity = event["severity_tier"]

        feedback = {
            "event_id": str(event["id"]),
            "actual_severity": actual_severity,
            "actual_duration_min": actual_duration,
            "feedback_notes": f"{SIMULATED_NOTE} (recorded outcome {true_duration:.0f} min)",
        }
        res = requests.post(f"{API_BASE_URL}/feedback", json=feedback, timeout=30)
        if res.ok:
            print(f"  Logged actual: {actual_severity} | {actual_duration:.0f} min\n")
        else:
            print(f"  Feedback error: {res.text[:200]}\n")
    except requests.RequestException as e:
        print(f"  API unreachable: {e}\n")
        time.sleep(5)


def run_simulation(df: pd.DataFrame):
    print("Starting simulation stream. Press Ctrl+C to stop.\n")
    try:
        while True:
            if random.random() < 0.20 and CCTV_SAMPLES_DIR.exists():
                run_cctv_cycle()
                time.sleep(1)
            run_event_cycle(df)
            time.sleep(random.uniform(5.0, 8.0))
    except KeyboardInterrupt:
        print("\nSimulation stopped.")


if __name__ == "__main__":
    if os.environ.get("ENABLE_SIMULATOR") != "1":
        print("Simulator is disabled. Set ENABLE_SIMULATOR=1 to run it.")
        raise SystemExit(0)
    if wait_for_api():
        run_simulation(load_data())
