import io
import base64
import numpy as np
import cv2
from ultralytics import YOLO

# Load the YOLOv8 Nano model (will download ~6MB weights on first run)
model = YOLO('yolov8n.pt')

# COCO Dataset classes we care about for traffic
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def analyze_cctv_frame(image_bytes: bytes) -> dict:
    """
    Takes raw image bytes, runs YOLOv8 vehicle detection,
    and returns a base64 annotated image and insights.
    """
    # Convert bytes to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image format"}

    # Run YOLOv8 inference
    results = model(img, classes=list(VEHICLE_CLASSES.keys()), conf=0.25)
    
    # Extract results
    result = results[0]
    boxes = result.boxes
    
    vehicle_counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }
    
    total_vehicles = len(boxes)
    
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = VEHICLE_CLASSES.get(cls_id, "unknown")
        if cls_name in vehicle_counts:
            vehicle_counts[cls_name] += 1
            
    # Draw bounding boxes automatically using ultralytics plot
    annotated_img = result.plot()
    
    # Convert back to base64 for frontend
    _, buffer = cv2.imencode('.jpg', annotated_img)
    base64_encoded = base64.b64encode(buffer).decode('utf-8')
    base64_string = f"data:image/jpeg;base64,{base64_encoded}"
    
    # Basic logic for congestion detection
    status = "Clear"
    confidence = 0.95
    severity = "Low"
    
    if total_vehicles > 15:
        status = "High Congestion Detected"
        severity = "High"
    elif total_vehicles > 8:
        status = "Moderate Traffic"
        severity = "Medium"
        
    # Fake accident logic: if a lot of trucks and cars are densely packed 
    # (in reality we'd measure IOU overlapping or stopped frames over time)
    if vehicle_counts["truck"] >= 2 and total_vehicles > 10:
        status = "Potential Accident / Blockage"
        severity = "High"

    return {
        "status": status,
        "severity": severity,
        "total_vehicles": total_vehicles,
        "breakdown": vehicle_counts,
        "annotated_image": base64_string
    }
