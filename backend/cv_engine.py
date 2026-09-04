"""
CCTV Vision Engine
==================
Counts vehicles in a still frame with YOLOv8 and converts the count into a
congestion reading.

What this is honest about: it is a per-frame vehicle counter with a density
threshold, not an accident detector. A single frame cannot distinguish stopped
traffic from moving traffic, so "Potential Blockage" is a density signal that
warrants a human look, not a classification. Detecting an actual incident needs
either stationarity across consecutive frames or bounding-box overlap analysis;
both are noted below as the upgrade path.
"""

import base64
import os
import threading
from pathlib import Path

# Constrain the native thread pools BEFORE torch/OpenCV are imported. FastAPI runs
# sync endpoints on a worker thread, and multi-threaded OpenMP inference off the main
# thread deadlocks on macOS — the request hangs forever with no traceback. Inference
# here is one small frame at a time, so single-threaded costs nothing measurable.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

torch.set_num_threads(1)
cv2.setNumThreads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Resolved from the project root, not the working directory: the previous relative
# path only worked when the server happened to be started from the repo root.
WEIGHTS_PATH = PROJECT_ROOT / "yolov8n.pt"

_model = None
_model_lock = threading.Lock()

# COCO classes relevant to road traffic
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Vehicles per megapixel of frame. Counting raw boxes made the threshold a function
# of camera zoom and resolution: a wide shot of free-flowing traffic tripped it while
# a tight shot of a real jam did not. Normalising by frame area makes the number mean
# roughly the same thing across cameras.
DENSITY_MODERATE = 12.0
DENSITY_HIGH = 22.0

# Fraction of frame area covered by vehicle boxes — a second, independent signal that
# rises when traffic is packed rather than merely numerous.
COVERAGE_HIGH = 0.28


def get_yolo_model():
    """Lazy-load the network so the API boots fast on constrained hardware.

    Double-checked locking: the startup warm-up thread and an early request could
    otherwise both find `_model is None` and construct the network concurrently,
    which loaded the weights twice and could wedge the native runtime.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print("[CV] Lazy-loading YOLOv8 model into memory...")
                _model = YOLO(str(WEIGHTS_PATH))
    return _model


def analyze_cctv_frame(image_bytes: bytes) -> dict:
    """Detect vehicles in a frame and return counts, a density reading and an overlay."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid or unreadable image format"}

    height, width = img.shape[:2]
    megapixels = max((height * width) / 1_000_000.0, 0.01)

    model = get_yolo_model()
    results = model(img, classes=list(VEHICLE_CLASSES.keys()), conf=0.25, verbose=False)
    result = results[0]
    boxes = result.boxes

    vehicle_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    covered_area = 0.0

    for box in boxes:
        cls_name = VEHICLE_CLASSES.get(int(box.cls[0]))
        if cls_name:
            vehicle_counts[cls_name] += 1
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        covered_area += max(0.0, x2 - x1) * max(0.0, y2 - y1)

    total_vehicles = int(sum(vehicle_counts.values()))
    density = total_vehicles / megapixels
    coverage = covered_area / float(height * width) if height and width else 0.0

    if density >= DENSITY_HIGH or coverage >= COVERAGE_HIGH:
        status, severity = "Heavy congestion", "High"
    elif density >= DENSITY_MODERATE:
        status, severity = "Moderate traffic", "Medium"
    else:
        status, severity = "Free flowing", "Low"

    # Heavy-vehicle presence in dense traffic raises the operational priority of a
    # look, because a stopped truck blocks proportionally more of the carriageway.
    heavy = vehicle_counts["truck"] + vehicle_counts["bus"]
    needs_review = severity == "High" and heavy >= 2
    if needs_review:
        status = "Heavy congestion with large vehicles — review feed"

    annotated_img = result.plot()
    ok, buffer = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return {"error": "Failed to encode annotated frame"}
    base64_string = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    return {
        "status": status,
        "severity": severity,
        "total_vehicles": total_vehicles,
        "breakdown": vehicle_counts,
        "density_per_megapixel": round(density, 1),
        "frame_coverage_pct": round(coverage * 100, 1),
        "needs_human_review": needs_review,
        "annotated_image": base64_string,
        "method_note": (
            "Single-frame vehicle density. This flags congestion, not incidents — "
            "distinguishing a jam from an accident requires stationarity across "
            "consecutive frames, which a still image cannot provide."
        ),
    }
