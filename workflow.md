# Namma Route: Operational Workflow

This document outlines the step-by-step operational flow of the **Namma Route** platform from the perspective of a Bengaluru Traffic Police (BTP) Dispatcher at the Command Center.

The system handles two primary workflows: **Unplanned Events** (e.g., sudden accidents, breakdowns) and **Planned Events** (e.g., scheduled protests, VIP movements, festivals).

---

## Workflow A: Unplanned Events (Autonomous Detection)
*This flow triggers when a sudden event occurs, bypassing the need for manual civilian reporting.*

### Step 1: Autonomous Monitoring
- The **YOLOv8 Computer Vision Engine** continuously processes live CCTV feeds from major city junctions.
- It scans for vehicle density (cars, trucks, buses, motorcycles).

### Step 2: AI Detection & Alerting
- A truck breaks down in the middle of the Outer Ring Road (ORR), causing a sudden buildup of vehicles.
- The YOLOv8 model detects that the density threshold (>15 vehicles packed densely) has been breached.
- The system immediately fires an autonomous JSON alert to the backend.

### Step 3: Dashboard Notification
- Inside the Command Center, the **Namma Route Dashboard** flashes a high-priority "Active Alert."
- The dispatcher sees an annotated image of the camera feed (with bounding boxes showing the exact bottleneck).

### Step 4: Instant Forecasting
- Without the dispatcher needing to type anything, the system routes the alert data into the **XGBoost ML Engine**.
- The system instantly predicts:
  1. The event will reach **High Severity**.
  2. It will take an estimated **45 minutes** to clear based on historical analogs.

### Step 5: Resource Dispatch
- The dispatcher clicks the alert and views the **Response Planner**.
- The system recommends deploying **1 Heavy Tow Truck**, **5-10 Officers**, and lists the best adjacent corridors to divert incoming traffic to.
- The dispatcher radios the ground team with exact, data-backed instructions.

---

## Workflow B: Planned Events (Proactive Intelligence)
*This flow is used by dispatchers days or hours in advance of a known scheduled event to prevent gridlock before it starts.*

### Step 1: Event Registration
- The BTP is notified that a major **Political Procession** is scheduled for Friday evening at 5:00 PM near MG Road.

### Step 2: Predictive Simulation
- The dispatcher opens the **Response Planner** tab in the Namma Route dashboard.
- They input the hypothetical parameters:
  - Cause: `Procession`
  - Corridor: `MG Road`
  - Time: `17:00 (5 PM)`
  - Day: `Friday`
  - Road Closure Required: `Yes`

### Step 3: ML Output Generation
- The dispatcher clicks **Run Forecast**.
- The **XGBoost Model** analyzes 8,200 past events and outputs:
  - A **High Severity** warning with 94% confidence.
  - An estimated duration of **120 minutes** for the congestion to clear after the event starts.

### Step 4: Proactive Deployment
- The system generates a deterministic checklist for the dispatcher:
  1. Deploy **20-25 Traffic Officers** at critical chokepoints 2 hours prior.
  2. Send **50 Barricades** to block the specific intersections recommended by the system.
  3. Update digital traffic boards to advise civilians to use the recommended alternate routes.

---

## Workflow C: The Continuous Learning Loop
*This flow occurs after BOTH planned and unplanned events are resolved, ensuring the system gets smarter over time.*

### Step 1: Ground Truth Logging
- The event is completely cleared and traffic returns to normal.
- The dispatcher navigates to the **Post-Event Learning** tab.

### Step 2: Data Entry
- The dispatcher inputs the *actual* outcome of the event:
  - Actual Severity: `Medium` (Because the proactive barricading worked well!)
  - Actual Duration: `90 minutes` (Cleared 30 minutes faster than predicted).

### Step 3: Model Retraining
- The system records this new data point into the `learning_log.csv`.
- It calculates the delta (error) between what it predicted and what actually happened.
- On the next system cycle, the ML model ingests this new ground-truth data, adjusting its weights so that future predictions for MG Road processions are even more accurate.
