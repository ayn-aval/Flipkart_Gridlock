"""
Resource Recommendation Engine
===============================
Phase 3: Rule-based manpower, barricading, and diversion recommendation layer.
Consumes Phase 2's forecasting output and produces explainable recommendations.

IMPORTANT: This is a heuristic/rule-based system, NOT learned from historical
resource deployment data (which doesn't exist in the dataset). All rules are
documented with their reasoning. See ASSUMPTIONS.md for full details.

Usage:
    python3 backend/recommendations.py          # Run sample recommendations
    python3 backend/recommendations.py --table  # Print the full rule lookup table
"""

import sys
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CORRIDOR_ADJACENCY_CSV = PROCESSED_DIR / "corridor_adjacency.csv"
CLEAN_CSV = PROCESSED_DIR / "events_clean.csv"


# ═══════════════════════════════════════════════════════════════════════════════
#  MANPOWER & BARRICADE LOOKUP TABLE
# ═══════════════════════════════════════════════════════════════════════════════
#
#  This table is NOT learned from data — the dataset has no ground truth for
#  resources deployed. Instead, it is a transparent heuristic anchored to:
#
#  1. Road closure frequency by cause (from data audit):
#     - vip_movement: 80% need road closure → high resource needs
#     - public_event: 46% need road closure → high resource needs
#     - Protest - 40% need road closure → high resource needs
#     - tree_fall: 39% need road closure → moderate (physical obstruction)
#     - Construction work - 26% need road closure → moderate-high (planned)
#     - Procession - 26% need road closure → moderate-high (moving event)
#
#  2. Duration tiers (from Phase 1):
#     - vehicle_breakdown: usual clearance time is ~41 min → quick response
#     - Accident - usual clearance time is ~40 min → quick response + investigation
#     - Construction work - usual clearance time is ~296 min → sustained presence
#     - water_logging: usual clearance time is variable → area management
#
#  3. Severity tier (from Phase 2 model output):
#     - High → more officers, more barricades
#     - Medium → standard deployment
#     - Low → minimal/monitoring only
#
#  The officer and barricade counts are anchored to reasonable assumptions:
#  - A typical Bengaluru junction has 2-4 traffic officers
#  - A road closure at a major corridor might need 6-10 officers
#  - Major events (VIP, large rallies) historically get 15-30+ officers
#  - Barricade counts assume standard Bengaluru traffic barrier units
# ═══════════════════════════════════════════════════════════════════════════════

# Key: (event_cause, severity_tier, requires_road_closure)
# Value: dict with officer_range, barricade_range, actions
# Where road_closure is not a discriminator, we use None as wildcard

RESOURCE_RULES = {
    # ─── Vehicle Breakdown ──────────────────────────────────────────────
    ("vehicle_breakdown", "Low", False): {
        "officers_min": 1, "officers_max": 2,
        "barricades_min": 2, "barricades_max": 4,
        "actions": [
            "Deploy traffic cones around the breakdown site",
            "Direct traffic to available lanes",
            "Coordinate with tow/mechanic service",
            "Monitor for secondary congestion",
        ],
        "basis": "Minor Impact: Vehicle breakdown - minimal lane blockage, "
                 "typically cleared within 30-60 min. Road closure rate for "
                 "vehicle breakdowns is only 4.3% historically."
    },
    ("vehicle_breakdown", "Medium", False): {
        "officers_min": 2, "officers_max": 4,
        "barricades_min": 4, "barricades_max": 6,
        "actions": [
            "Deploy traffic cones and warning signs",
            "Set up lane diversion around the breakdown",
            "Coordinate with tow/mechanic service (priority)",
            "Monitor upstream junctions for traffic jam spreading backwards",
            "Alert nearby patrol units if congestion builds",
        ],
        "basis": "Moderate Impact: Vehicle breakdown - partial lane blockage on a "
                 "major corridor, expected 30-120 min to clear."
    },
    ("vehicle_breakdown", "High", True): {
        "officers_min": 4, "officers_max": 6,
        "barricades_min": 6, "barricades_max": 10,
        "actions": [
            "Implement partial/full road closure",
            "Deploy barricades at closure points",
            "Set up traffic diversion signs at approach roads",
            "Station officers at diversion points",
            "Coordinate emergency tow service",
            "Alert control room for main road traffic management",
        ],
        "basis": "Major Impact: vehicle breakdown requiring road closure: likely a "
                 "heavy vehicle or multi-vehicle situation on a key corridor."
    },

    # ─── Accident ───────────────────────────────────────────────────────
    ("accident", "Low", False): {
        "officers_min": 2, "officers_max": 3,
        "barricades_min": 4, "barricades_max": 6,
        "actions": [
            "Secure the accident scene",
            "Direct traffic around the site",
            "Document the scene (photos, measurements)",
            "Coordinate ambulance if injuries reported",
            "File FIR if required",
        ],
        "basis": "Minor Impact: Accident - minor incident, no major road blockage. "
                 "Usual clearance time is accident about 40 min."
    },
    ("accident", "Medium", False): {
        "officers_min": 3, "officers_max": 5,
        "barricades_min": 6, "barricades_max": 8,
        "actions": [
            "Secure the accident scene with barricades",
            "Direct traffic to alternate lanes",
            "Coordinate ambulance/medical response",
            "Document and investigate the scene",
            "Alert upstream junctions",
            "File FIR and coordinate with investigating officer",
        ],
        "basis": "Moderate Impact: accident on a corridor: partial lane blockage, "
                 "possible injuries, 30-120 min expected."
    },
    ("accident", "High", True): {
        "officers_min": 5, "officers_max": 8,
        "barricades_min": 8, "barricades_max": 12,
        "actions": [
            "Full scene lockdown with barricades",
            "Road closure at nearest junctions",
            "Deploy officers at all diversion points",
            "Coordinate ambulance, fire service if needed",
            "Set up traffic diversion route",
            "Alert hospital trauma centers",
            "Preserve scene for investigation",
            "Update control room with situation reports",
        ],
        "basis": "Major Impact: accident requiring road closure: major incident, "
                 "likely multi-vehicle or involving heavy vehicles."
    },

    # ─── Construction ───────────────────────────────────────────────────
    ("construction", "Low", False): {
        "officers_min": 1, "officers_max": 2,
        "barricades_min": 4, "barricades_max": 6,
        "actions": [
            "Monitor construction zone for traffic impact",
            "Ensure contractor has placed warning signs",
            "Verify construction permit and timeline",
            "Periodic patrol check",
        ],
        "basis": "Minor Impact: Construction work - minor works (e.g., utility repair), "
                 "no major lane closure. 26% of construction events need road closure."
    },
    ("construction", "Medium", False): {
        "officers_min": 2, "officers_max": 4,
        "barricades_min": 6, "barricades_max": 10,
        "actions": [
            "Verify construction permit and schedule",
            "Set up lane-closure signage at approach points",
            "Station officers at construction zone entry/exit",
            "Monitor for traffic buildup at peak hours",
            "Coordinate with construction contractor on timeline",
            "Ensure night-time reflective markers are placed",
        ],
        "basis": "Moderate Impact: construction on a corridor: lane closure but "
                 "not full road closure. Usual clearance time is construction about 296 min."
    },
    ("construction", "High", True): {
        "officers_min": 4, "officers_max": 8,
        "barricades_min": 10, "barricades_max": 20,
        "actions": [
            "Implement full road closure at construction zone",
            "Deploy barricades at all closure points",
            "Set up diversion route with signage",
            "Station officers at all diversion junctions",
            "Coordinate with construction contractor on strict timeline",
            "Alert public via traffic advisory (radio/social media)",
            "Plan for extended deployment (construction may last days)",
            "Ensure emergency vehicle access is maintained",
        ],
        "basis": "Major Impact: construction with road closure: major infrastructure "
                 "work (metro, flyover, pipeline). Usually lasts 5+ hours."
    },

    # ─── Public Event ───────────────────────────────────────────────────
    ("public_event", "Low", False): {
        "officers_min": 3, "officers_max": 5,
        "barricades_min": 6, "barricades_max": 10,
        "actions": [
            "Pre-position officers at event venue approaches",
            "Set up parking management zone",
            "Monitor crowd size and traffic impact",
            "Coordinate with event organizers on schedule",
            "Plan post-event traffic dispersal",
        ],
        "basis": "Minor Impact: Public event - small gathering (e.g., local ceremony), "
                 "limited traffic impact. 46% of public events need road closure."
    },
    ("public_event", "Medium", False): {
        "officers_min": 5, "officers_max": 10,
        "barricades_min": 10, "barricades_max": 15,
        "actions": [
            "Pre-position officers at all approach roads",
            "Set up crowd control barriers at venue",
            "Implement parking restrictions in the area",
            "Coordinate with event organizers on entry/exit flow",
            "Deploy traffic signal adjustments if possible",
            "Plan for staggered crowd dispersal",
            "Alert nearby hospitals for standby",
        ],
        "basis": "Moderate Impact: Public event - moderate crowd expected, "
                 "some traffic disruption. Historical events vary widely."
    },
    ("public_event", "High", True): {
        "officers_min": 10, "officers_max": 25,
        "barricades_min": 15, "barricades_max": 30,
        "actions": [
            "Full traffic management plan for the event zone",
            "Road closures at all approach roads",
            "Deploy officers at every junction within 1km radius",
            "Set up crowd control barriers and entry checkpoints",
            "Implement vehicle-free zone around venue",
            "Coordinate with event organizers, local government",
            "Deploy ambulance and fire service on standby",
            "Issue public traffic advisory 24+ hours in advance",
            "Arrange BMTC shuttle service if needed",
            "Plan multi-phase traffic restoration post-event",
        ],
        "basis": "Major Impact: public event with road closure: large gathering "
                 "(cricket match, festival, rally). Needs comprehensive traffic plan."
    },

    # ─── Procession ─────────────────────────────────────────────────────
    ("procession", "Low", False): {
        "officers_min": 3, "officers_max": 5,
        "barricades_min": 4, "barricades_max": 8,
        "actions": [
            "Escort the procession with patrol vehicle",
            "Pre-clear the route at upcoming junctions",
            "Alert intersecting corridors",
            "Monitor procession speed and route adherence",
        ],
        "basis": "Minor Impact: Procession - small group, no road closure. "
                 "Usual clearance time is procession about 37 min."
    },
    ("procession", "Medium", False): {
        "officers_min": 5, "officers_max": 8,
        "barricades_min": 6, "barricades_max": 12,
        "actions": [
            "Escort procession with multiple patrol vehicles",
            "Station officers at each junction along the route",
            "Implement rolling road closure ahead of procession",
            "Alert all corridors intersecting the route",
            "Coordinate with procession organizers on timing",
            "Re-open road segments behind the procession",
        ],
        "basis": "Moderate Impact: procession on a corridor: moderate-sized group "
                 "moving through traffic. Rolling closure needed."
    },
    ("procession", "High", True): {
        "officers_min": 8, "officers_max": 15,
        "barricades_min": 10, "barricades_max": 20,
        "actions": [
            "Full route closure ahead of procession",
            "Deploy officers at every junction on the route",
            "Set up diversion routes for intersecting traffic",
            "Coordinate with organizers on strict timing",
            "Escort with patrol vehicles (front and rear)",
            "Deploy crowd management at gathering points",
            "Issue traffic advisory for the route",
            "Plan phased road reopening behind procession",
        ],
        "basis": "Major Impact: procession with road closure: large procession "
                 "(religious, political) requiring full route management. "
                 "26% of processions usually need road closure."
    },

    # ─── VIP Movement ───────────────────────────────────────────────────
    ("vip_movement", "Low", False): {
        "officers_min": 4, "officers_max": 6,
        "barricades_min": 4, "barricades_max": 8,
        "actions": [
            "Pre-clear the VIP route",
            "Station officers at key junctions on the route",
            "Coordinate with VIP security team on timing",
            "Green-wave signal coordination if possible",
        ],
        "basis": "Minor Impact: VIP movement - standard VIP transit without road closure. "
                 "80% of VIP movements need road closure — even 'low severity' ones "
                 "get higher baseline resources."
    },
    ("vip_movement", "High", True): {
        "officers_min": 10, "officers_max": 25,
        "barricades_min": 10, "barricades_max": 25,
        "actions": [
            "Full route clearing and securing",
            "Road closure 15-30 min before VIP arrival",
            "Deploy officers at every junction on the route",
            "Anti-sabotage check of the route",
            "Coordinate with SPG/NSG/VIP security team",
            "Set up diversion routes for all intersecting traffic",
            "Deploy bomb disposal squad for route clearance",
            "Station ambulance and fire service on standby",
            "Issue traffic advisory with route and timing",
            "Phased reopening after VIP transit",
        ],
        "basis": "Major Impact: VIP movement with road closure: high-profile dignitary. "
                 "80% of VIP movements usually require road closure. "
                 "Highest resource intensity per event in the dataset."
    },

    # ─── Protest ────────────────────────────────────────────────────────
    ("protest", "Low", False): {
        "officers_min": 4, "officers_max": 8,
        "barricades_min": 6, "barricades_max": 10,
        "actions": [
            "Monitor protest size and mood",
            "Maintain safe distance buffer",
            "Set up barricades to contain protest to designated area",
            "Coordinate with protest organizers on duration",
            "Alert rapid response unit for standby",
            "Ensure emergency vehicle access",
        ],
        "basis": "Minor Impact: Protest - small, permitted demonstration. "
                 "40% of protests need road closure. Higher baseline resources "
                 "due to potential for escalation."
    },
    ("protest", "High", True): {
        "officers_min": 10, "officers_max": 30,
        "barricades_min": 15, "barricades_max": 30,
        "actions": [
            "Full perimeter barricading of protest zone",
            "Road closure on all approach roads",
            "Deploy rapid response unit",
            "Crowd management with entry/exit control",
            "Deploy water cannon on standby (if authorized)",
            "Coordinate with intelligence unit for situation assessment",
            "Station ambulance on standby",
            "Set up diversion routes for all traffic",
            "Issue public advisory to avoid the area",
            "Document and monitor via CCTV",
        ],
        "basis": "Major Impact: protest with road closure: large/unplanned protest "
                 "blocking a major road. 40% of protests need road closure. "
                 "Usual clearance time is protest about 25 min but can escalate."
    },

    # ─── Water Logging ──────────────────────────────────────────────────
    ("water_logging", "Low", False): {
        "officers_min": 1, "officers_max": 2,
        "barricades_min": 4, "barricades_max": 6,
        "actions": [
            "Place warning signs at waterlogged stretch",
            "Guide traffic to avoid deep-water areas",
            "Coordinate with BBMP for drainage clearance",
            "Monitor water level changes",
        ],
        "basis": "Minor Impact: Waterlogging - shallow, passable with caution."
    },
    ("water_logging", "Medium", False): {
        "officers_min": 2, "officers_max": 4,
        "barricades_min": 6, "barricades_max": 10,
        "actions": [
            "Barricade severely waterlogged lanes",
            "Direct traffic to higher-ground lanes",
            "Alert BBMP/fire service for drainage pumping",
            "Monitor for stranded vehicles",
            "Coordinate with nearby corridors for diversion",
        ],
        "basis": "Moderate Impact: Waterlogging - partial lane blockage, "
                 "variable duration depending on rainfall."
    },
    ("water_logging", "High", True): {
        "officers_min": 4, "officers_max": 8,
        "barricades_min": 8, "barricades_max": 15,
        "actions": [
            "Full road closure at waterlogged section",
            "Deploy barricades at approach roads",
            "Set up traffic diversion to higher-ground corridors",
            "Coordinate BBMP pumping operations",
            "Alert control room for area-wide management",
            "Assist stranded vehicles/pedestrians",
            "Monitor underpass water levels",
            "Issue weather-based traffic advisory",
        ],
        "basis": "Major Impact: waterlogging with road closure: deep flooding, "
                 "impassable road. Common at underpasses and low-lying corridors."
    },

    # ─── Tree Fall ──────────────────────────────────────────────────────
    ("tree_fall", "Low", False): {
        "officers_min": 1, "officers_max": 2,
        "barricades_min": 4, "barricades_max": 6,
        "actions": [
            "Barricade around the fallen tree/branch",
            "Direct traffic around the obstruction",
            "Coordinate with BBMP tree-clearing crew",
            "Check for downed power lines (alert BESCOM if found)",
        ],
        "basis": "Minor Impact: Tree fall - partial branch fall, navigable."
    },
    ("tree_fall", "High", True): {
        "officers_min": 3, "officers_max": 6,
        "barricades_min": 6, "barricades_max": 12,
        "actions": [
            "Road closure around the fallen tree",
            "Set up diversion route",
            "Coordinate BBMP/fire service for tree removal",
            "Check for and isolate downed power lines (BESCOM)",
            "Ensure no damage to vehicles/property underneath",
            "Clear debris after tree removal",
        ],
        "basis": "Major Impact: tree fall with road closure: full tree across road. "
                 "39% of tree falls need road closure. Usual clearance time is duration ~90 min."
    },

    # ─── Road Conditions ────────────────────────────────────────────────
    ("road_conditions", "Low", False): {
        "officers_min": 1, "officers_max": 2,
        "barricades_min": 2, "barricades_max": 4,
        "actions": [
            "Place warning signs at the hazard",
            "Alert BBMP for road repair",
            "Monitor for accident risk",
        ],
        "basis": "Minor Impact: road condition issue: minor hazard, manageable."
    },
    ("road_conditions", "High", True): {
        "officers_min": 2, "officers_max": 4,
        "barricades_min": 4, "barricades_max": 8,
        "actions": [
            "Barricade the damaged road section",
            "Implement lane diversion",
            "Coordinate with BBMP for emergency repair",
            "Monitor for worsening conditions",
        ],
        "basis": "Major Impact: road condition issue requiring closure."
    },

    # ─── Pot Holes ──────────────────────────────────────────────────────
    ("pot_holes", "Low", False): {
        "officers_min": 0, "officers_max": 1,
        "barricades_min": 2, "barricades_max": 4,
        "actions": [
            "Mark potholes with reflective cones/paint",
            "Report to BBMP for repair",
            "Monitor for accident risk at night",
        ],
        "basis": "Minor Impact: Pothole - road hazard, minimal traffic management needed."
    },
    ("pot_holes", "High", True): {
        "officers_min": 1, "officers_max": 3,
        "barricades_min": 4, "barricades_max": 6,
        "actions": [
            "Barricade the severely damaged section",
            "Direct traffic to safe lanes",
            "Priority report to BBMP for emergency repair",
            "Night-time illumination of the hazard",
        ],
        "basis": "Major Impact: pothole requiring lane closure."
    },

    # ─── Congestion ─────────────────────────────────────────────────────
    ("congestion", "Low", False): {
        "officers_min": 1, "officers_max": 2,
        "barricades_min": 0, "barricades_max": 2,
        "actions": [
            "Manual traffic signal management at the junction",
            "Clear any obstructions causing the congestion",
            "Monitor and report if congestion worsens",
        ],
        "basis": "Minor Impact: Traffic jam - minor traffic buildup."
    },
    ("congestion", "High", False): {
        "officers_min": 3, "officers_max": 6,
        "barricades_min": 2, "barricades_max": 4,
        "actions": [
            "Deploy officers at key junctions in the congested corridor",
            "Manual traffic signal management (green signal corridor)",
            "Implement temporary one-way or lane restrictions",
            "Clear any secondary obstructions",
            "Alert upstream corridors to divert traffic",
            "Coordinate with control room for signal timing adjustments",
        ],
        "basis": "Major Impact: congestion on a major corridor. "
                 "Usual clearance time is congestion about 72 min."
    },

    # ─── Debris ─────────────────────────────────────────────────────────
    ("debris", "Low", False): {
        "officers_min": 1, "officers_max": 2,
        "barricades_min": 2, "barricades_max": 4,
        "actions": [
            "Barricade around debris",
            "Coordinate clearing crew",
            "Direct traffic around obstruction",
        ],
        "basis": "Minor Impact: debris on road."
    },
    ("debris", "High", True): {
        "officers_min": 2, "officers_max": 4,
        "barricades_min": 4, "barricades_max": 8,
        "actions": [
            "Road closure at debris site",
            "Coordinate heavy equipment for clearing",
            "Set up diversion route",
            "Verify no hazardous material",
        ],
        "basis": "Major Impact: debris requiring road closure."
    },

    # ─── Others / Fallback ──────────────────────────────────────────────
    ("others", "Low", False): {
        "officers_min": 1, "officers_max": 2,
        "barricades_min": 2, "barricades_max": 4,
        "actions": [
            "Assess the situation on-site",
            "Deploy basic traffic management",
            "Report specifics to control room",
        ],
        "basis": "Catch-all for miscellaneous low-severity events."
    },
    ("others", "Medium", False): {
        "officers_min": 2, "officers_max": 4,
        "barricades_min": 4, "barricades_max": 6,
        "actions": [
            "Assess the situation on-site",
            "Set up lane diversion if needed",
            "Coordinate with relevant department",
            "Report to control room with updates",
        ],
        "basis": "Moderate Impact: miscellaneous event."
    },
    ("others", "High", True): {
        "officers_min": 3, "officers_max": 6,
        "barricades_min": 6, "barricades_max": 10,
        "actions": [
            "Assess and implement road closure if needed",
            "Set up traffic diversion",
            "Coordinate with relevant departments",
            "Update control room",
        ],
        "basis": "Major Impact: miscellaneous event with road closure."
    },
}


def _lookup_rule(event_cause: str, severity_tier: str, requires_road_closure: bool) -> dict:
    """Look up the best matching rule from the table.
    
    Tries exact match first, then falls back to less specific matches.
    """
    # Exact match
    key = (event_cause, severity_tier, requires_road_closure)
    if key in RESOURCE_RULES:
        return RESOURCE_RULES[key]
    
    # Try without road closure specificity
    for rc in [True, False]:
        key = (event_cause, severity_tier, rc)
        if key in RESOURCE_RULES:
            return RESOURCE_RULES[key]
    
    # Try with just the cause (any severity, prefer high)
    for sev in ["High", "Medium", "Low"]:
        for rc in [True, False]:
            key = (event_cause, sev, rc)
            if key in RESOURCE_RULES:
                return RESOURCE_RULES[key]
    
    # Ultimate fallback
    if requires_road_closure:
        return RESOURCE_RULES[("others", "High", True)]
    elif severity_tier == "Medium":
        return RESOURCE_RULES[("others", "Medium", False)]
    else:
        return RESOURCE_RULES[("others", "Low", False)]


# ═══════════════════════════════════════════════════════════════════════════════
#  DIVERSION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class DiversionEngine:
    """Suggests alternate corridors based on the adjacency table from Phase 1."""
    
    def __init__(self):
        self.adjacency = pd.read_csv(CORRIDOR_ADJACENCY_CSV)
        self.event_data = pd.read_csv(CLEAN_CSV)
        
        # Pre-compute hourly event frequency per corridor
        self._corridor_hour_freq = (
            self.event_data
            .groupby(["corridor", "hour_of_day"])
            .size()
            .reset_index(name="event_count")
        )
    
    def suggest_diversions(self, corridor: str, hour_of_day: int = None,
                           top_n: int = 3) -> list:
        """Suggest alternate corridors for diversion.
        
        Returns a list of dicts with corridor name, distance, and rationale.
        """
        if corridor == "Non-corridor" or corridor not in self.adjacency["corridor"].values:
            return [{
                "corridor": "Unable to suggest — event is not on a named corridor",
                "distance_km": None,
                "rationale": "The event location is not on a monitored traffic corridor. "
                             "Local diversion based on on-ground assessment is recommended.",
            }]
        
        # Get nearest corridors
        neighbors = self.adjacency[self.adjacency["corridor"] == corridor].head(top_n)
        
        suggestions = []
        for _, row in neighbors.iterrows():
            neighbor = row["neighbor_corridor"]
            distance = row["distance_km"]
            
            # Skip Non-corridor as a diversion target
            if neighbor == "Non-corridor":
                continue
            
            # Build rationale
            rationale_parts = [f"Nearest alternate route (approx. {distance:.1f} km away)"]
            
            # Check if the alternate has lower event frequency at this hour
            if hour_of_day is not None:
                orig_freq = self._get_hourly_freq(corridor, hour_of_day)
                alt_freq = self._get_hourly_freq(neighbor, hour_of_day)
                
                if alt_freq < orig_freq:
                    rationale_parts.append(
                        f"historically {int(((orig_freq - alt_freq) / max(orig_freq, 1)) * 100)}% "
                        f"fewer events at {hour_of_day:02d}:00 IST"
                    )
                elif alt_freq == orig_freq:
                    rationale_parts.append(f"similar event frequency at {hour_of_day:02d}:00 IST")
                else:
                    rationale_parts.append(
                        f"note: historically busier at {hour_of_day:02d}:00 IST — "
                        f"consider traffic conditions before diverting"
                    )
            
            suggestions.append({
                "corridor": neighbor,
                "distance_km": distance,
                "event_frequency": int(row["neighbor_event_count"]),
                "rationale": "; ".join(rationale_parts),
            })
        
        return suggestions
    
    def _get_hourly_freq(self, corridor: str, hour: int) -> int:
        """Get event count for a corridor at a specific hour."""
        mask = (
            (self._corridor_hour_freq["corridor"] == corridor) &
            (self._corridor_hour_freq["hour_of_day"] == hour)
        )
        matches = self._corridor_hour_freq[mask]["event_count"]
        return int(matches.iloc[0]) if len(matches) > 0 else 0


# ═══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE (combines rules + diversion)
# ═══════════════════════════════════════════════════════════════════════════════

class RecommendationEngine:
    """Combines the resource lookup table with the diversion engine."""
    
    def __init__(self):
        self.diversion_engine = DiversionEngine()
    
    def recommend(self, forecast_result: dict, event_input: dict) -> dict:
        """Generate a full recommendation given forecast output and event input.
        
        Args:
            forecast_result: Output from ForecastingEngine.forecast()
            event_input: Original event input dict
        
        Returns:
            {
                "manpower": {"officers_min": 4, "officers_max": 8},
                "barricading": {"barricades_min": 6, "barricades_max": 10},
                "action_checklist": [...],
                "diversion_suggestions": [...],
                "basis": "...",
                "recommendation_method": "rule_based_heuristic",
                "disclaimer": "...",
            }
        """
        severity = forecast_result.get("severity_tier", "Low")
        event_cause = event_input.get("event_cause", "others")
        road_closure = bool(event_input.get("requires_road_closure", False))
        corridor = event_input.get("corridor", "Non-corridor")
        hour = event_input.get("hour_of_day", None)
        
        # Look up resource rules
        rule = _lookup_rule(event_cause, severity, road_closure)
        
        # Get diversion suggestions
        diversions = self.diversion_engine.suggest_diversions(
            corridor, hour_of_day=hour, top_n=3
        )
        
        # Build recommendation
        recommendation = {
            "manpower": {
                "officers_min": rule["officers_min"],
                "officers_max": rule["officers_max"],
                "unit": "traffic police officers",
            },
            "barricading": {
                "barricades_min": rule["barricades_min"],
                "barricades_max": rule["barricades_max"],
                "unit": "standard traffic barrier units",
            },
            "action_checklist": rule["actions"],
            "diversion_suggestions": diversions,
            "basis": rule["basis"],
            "recommendation_method": "rule_based_heuristic",
            "severity_used": severity,
            "cause_used": event_cause,
            "road_closure_used": road_closure,
            "disclaimer": (
                "These recommendations are generated by a rule-based heuristic system, "
                "NOT learned from historical resource deployment data (which is not "
                "available in the dataset). Officer and barricade counts are anchored "
                "to road-closure frequency, event duration patterns, and standard "
                "traffic management practices. On-ground officers should adjust based "
                "on actual conditions."
            ),
        }
        
        return recommendation


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI / DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def print_rule_table():
    """Print the full rule lookup table for inspection."""
    print("=" * 80)
    print("FULL RESOURCE RECOMMENDATION LOOKUP TABLE")
    print("=" * 80)
    print(f"\n{'Cause':<20} {'Severity':<10} {'Closure':<8} {'Officers':<12} {'Barricades':<12}")
    print("-" * 80)
    
    for (cause, severity, closure), rule in sorted(RESOURCE_RULES.items()):
        officers = f"{rule['officers_min']}-{rule['officers_max']}"
        barricades = f"{rule['barricades_min']}-{rule['barricades_max']}"
        closure_str = "Yes" if closure else "No"
        print(f"{cause:<20} {severity:<10} {closure_str:<8} {officers:<12} {barricades:<12}")
    
    print(f"\nTotal rules: {len(RESOURCE_RULES)}")


def run_demo():
    """Run recommendation demo with forecasting engine."""
    # Import and load forecasting engine
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
    from forecasting import load_engine
    
    forecast_engine = load_engine()
    rec_engine = RecommendationEngine()
    
    test_scenarios = [
        {
            "name": "Vehicle breakdown on Mysore Road (Tuesday morning rush)",
            "input": {
                "event_cause": "vehicle_breakdown",
                "corridor": "Mysore Road",
                "time_period": "morning_rush",
                "hour_of_day": 8,
                "day_of_week": 1,
                "is_weekend": 0,
                "requires_road_closure": 0,
                "veh_type": "heavy_vehicle",
            }
        },
        {
            "name": "Planned procession on Bellary Road (Sunday, road closure)",
            "input": {
                "event_cause": "procession",
                "corridor": "Bellary Road 1",
                "time_period": "morning_rush",
                "hour_of_day": 9,
                "day_of_week": 6,
                "is_weekend": 1,
                "requires_road_closure": 1,
                "veh_type": "none",
            }
        },
        {
            "name": "VIP movement on CBD 2 (weekday, road closure)",
            "input": {
                "event_cause": "vip_movement",
                "corridor": "CBD 2",
                "time_period": "midday",
                "hour_of_day": 11,
                "day_of_week": 2,
                "is_weekend": 0,
                "requires_road_closure": 1,
                "veh_type": "none",
            }
        },
        {
            "name": "Water logging on Hosur Road (Friday evening rush)",
            "input": {
                "event_cause": "water_logging",
                "corridor": "Hosur Road",
                "time_period": "evening_rush",
                "hour_of_day": 17,
                "day_of_week": 4,
                "is_weekend": 0,
                "requires_road_closure": 0,
                "veh_type": "none",
            }
        },
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'═' * 70}")
        print(f"  SCENARIO: {scenario['name']}")
        print(f"{'═' * 70}")
        
        # Get forecast
        forecast = forecast_engine.forecast(scenario["input"])
        
        # Get recommendation
        rec = rec_engine.recommend(forecast, scenario["input"])
        
        # Print forecast
        print(f"\n  📊 FORECAST")
        print(f"     Severity: {forecast['severity_tier']} ({forecast['severity_confidence']:.0%} confidence)")
        print(f"     Duration: ~{forecast['expected_duration_min']} min")
        if "analog_duration_median_min" in forecast:
            print(f"     Analog:   ~{forecast['analog_duration_median_min']} min (usual clearance time is of similar events)")
        
        # Print recommendation
        print(f"\n  👮 MANPOWER")
        print(f"     Officers: {rec['manpower']['officers_min']}–{rec['manpower']['officers_max']} officers")
        
        print(f"\n  🚧 BARRICADING")
        print(f"     Barriers: {rec['barricading']['barricades_min']}–{rec['barricading']['barricades_max']} units")
        
        print(f"\n  ✅ ACTION CHECKLIST")
        for i, action in enumerate(rec["action_checklist"], 1):
            print(f"     {i}. {action}")
        
        print(f"\n  🔀 DIVERSION SUGGESTIONS")
        for i, div in enumerate(rec["diversion_suggestions"], 1):
            print(f"     {i}. {div['corridor']} — {div['rationale']}")
        
        print(f"\n  📝 BASIS: {rec['basis']}")


if __name__ == "__main__":
    if "--table" in sys.argv:
        print_rule_table()
    else:
        run_demo()
