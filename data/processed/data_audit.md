# Data Audit Report — Astram Event Data

**Generated:** 2026-06-17 22:21:40

**Source:** `data/raw/astram_events.csv`

## 1. Dataset Shape
- **Rows:** 8,173
- **Columns:** 46
- **Column names:** id, event_type, latitude, longitude, endlatitude, endlongitude, address, end_address, event_cause, requires_road_closure, start_datetime, end_datetime, status, authenticated, modified_datetime, map_file, direction, description, veh_type, veh_no, corridor, priority, cargo_material, reason_breakdown, age_of_truck, created_date, route_path, client_id, created_by_id, last_modified_by_id, assigned_to_police_id, citizen_accident_id, comment, police_station, meta_data, kgid, resolved_at_address, resolved_at_latitude, resolved_at_longitude, closed_by_id, closed_datetime, resolved_by_id, resolved_datetime, gba_identifier, zone, junction

## 2. Null Rates Per Column
| Column | Dtype | Non-Null | Null Count | Null % |
|---|---|---|---|---|
| id | str | 8173 | 0 | 0.0% |
| event_type | str | 8173 | 0 | 0.0% |
| latitude | float64 | 8173 | 0 | 0.0% |
| longitude | float64 | 8173 | 0 | 0.0% |
| endlatitude | float64 | 8004 | 169 | 2.07% |
| endlongitude | float64 | 8004 | 169 | 2.07% |
| address | str | 8170 | 3 | 0.04% |
| end_address | str | 687 | 7486 | 91.59% |
| event_cause | str | 8173 | 0 | 0.0% |
| requires_road_closure | bool | 8173 | 0 | 0.0% |
| start_datetime | str | 8173 | 0 | 0.0% |
| end_datetime | str | 490 | 7683 | 94.0% |
| status | str | 8173 | 0 | 0.0% |
| authenticated | str | 8173 | 0 | 0.0% |
| modified_datetime | str | 8173 | 0 | 0.0% |
| map_file | float64 | 0 | 8173 | 100.0% |
| direction | str | 43 | 8130 | 99.47% |
| description | str | 6813 | 1360 | 16.64% |
| veh_type | str | 4887 | 3286 | 40.21% |
| veh_no | str | 4886 | 3287 | 40.22% |
| corridor | str | 8153 | 20 | 0.24% |
| priority | str | 8171 | 2 | 0.02% |
| cargo_material | str | 276 | 7897 | 96.62% |
| reason_breakdown | str | 276 | 7897 | 96.62% |
| age_of_truck | float64 | 276 | 7897 | 96.62% |
| created_date | str | 8173 | 0 | 0.0% |
| route_path | str | 137 | 8036 | 98.32% |
| client_id | int64 | 8173 | 0 | 0.0% |
| created_by_id | str | 8171 | 2 | 0.02% |
| last_modified_by_id | str | 8170 | 3 | 0.04% |
| assigned_to_police_id | str | 128 | 8045 | 98.43% |
| citizen_accident_id | str | 128 | 8045 | 98.43% |
| comment | float64 | 0 | 8173 | 100.0% |
| police_station | str | 8173 | 0 | 0.0% |
| meta_data | float64 | 0 | 8173 | 100.0% |
| kgid | str | 7914 | 259 | 3.17% |
| resolved_at_address | str | 74 | 8099 | 99.09% |
| resolved_at_latitude | float64 | 74 | 8099 | 99.09% |
| resolved_at_longitude | float64 | 74 | 8099 | 99.09% |
| closed_by_id | str | 3141 | 5032 | 61.57% |
| closed_datetime | str | 3141 | 5032 | 61.57% |
| resolved_by_id | str | 74 | 8099 | 99.09% |
| resolved_datetime | str | 74 | 8099 | 99.09% |
| gba_identifier | str | 3444 | 4729 | 57.86% |
| zone | str | 3444 | 4729 | 57.86% |
| junction | str | 2510 | 5663 | 69.29% |

## 3. Category Breakdowns

### Event Type (`event_type`)
Unique values: 2 | Non-null: 8173
| Value | Count |
|---|---|
| unplanned | 7706 |
| planned | 467 |

### Event Cause (`event_cause`)
Unique values: 17 | Non-null: 8173
| Value | Count |
|---|---|
| vehicle_breakdown | 4896 |
| others | 638 |
| pot_holes | 537 |
| construction | 480 |
| water_logging | 458 |
| accident | 365 |
| tree_fall | 284 |
| road_conditions | 170 |
| congestion | 136 |
| public_event | 84 |
| procession | 72 |
| vip_movement | 20 |
| protest | 15 |
| Debris | 12 |
| test_demo | 3 |
| Fog / Low Visibility | 2 |
| debris | 1 |

### Priority (`priority`)
Unique values: 2 | Non-null: 8171
| Value | Count |
|---|---|
| High | 5030 |
| Low | 3141 |
| (null) | 2 |

### Status (`status`)
Unique values: 3 | Non-null: 8173
| Value | Count |
|---|---|
| closed | 7095 |
| active | 1007 |
| resolved | 71 |

### Requires Road Closure (`requires_road_closure`)
Unique values: 2 | Non-null: 8173
| Value | Count |
|---|---|
| False | 7497 |
| True | 676 |

### Corridor (top 30) (`corridor`)
Unique values: 22 | Non-null: 8153
| Value | Count |
|---|---|
| Non-corridor | 3124 |
| Mysore Road | 743 |
| Bellary Road 1 | 610 |
| Tumkur Road | 458 |
| Bellary Road 2 | 379 |
| Hosur Road | 298 |
| ORR North 1 | 275 |
| Old Madras Road | 263 |
| Magadi Road | 245 |
| ORR East 1 | 244 |
| ORR North 2 | 235 |
| Bannerghata Road | 209 |
| ORR East 2 | 187 |
| West of Chord Road | 174 |
| ORR West 1 | 168 |
| CBD 2 | 104 |
| Hennur Main Road | 96 |
| IRR(Thanisandra road) | 95 |
| Varthur Road | 77 |
| Old Airport Road | 76 |
| Airport New South Road | 67 |
| CBD 1 | 26 |
| (null) | 20 |

### Zone (`zone`)
Unique values: 10 | Non-null: 3444
| Value | Count |
|---|---|
| (null) | 4729 |
| Central Zone 2 | 623 |
| West Zone 1 | 433 |
| North Zone 2 | 413 |
| West Zone 2 | 358 |
| South Zone 2 | 354 |
| North Zone 1 | 318 |
| Central Zone 1 | 269 |
| East Zone 1 | 253 |
| South Zone 1 | 233 |
| East Zone 2 | 190 |

### Police Station (top 20) (`police_station`)
Unique values: 54 | Non-null: 8173
| Value | Count |
|---|---|
| Yelahanka | 377 |
| HAL Old Airport | 361 |
| Sadashivanagar | 302 |
| Byatarayanapura | 297 |
| Halasuru Gate | 297 |
| Yeshwanthpura | 280 |
| Hennuru | 276 |
| Kodigehalli | 272 |
| Banaswadi | 245 |
| K.R. Pura | 228 |
| Kamakshipalya | 224 |
| No Police Station | 219 |
| Cubbon Park | 212 |
| Jalahalli | 197 |
| Chamarajpet | 192 |
| High ground | 185 |
| Madiwala | 184 |
| Whitefield | 181 |
| Peenya | 178 |
| Jayanagara | 178 |
| Ashok Nagar | 171 |
| Magadi Road | 142 |
| Hebbala | 135 |
| Jnanabharathi | 134 |
| Jeevanbheemanagar | 127 |
| R.T. Nagar | 127 |
| J.P. Nagar | 124 |
| Electronic City | 124 |
| Sheshadripuram | 123 |
| Mahadevapura | 120 |

### Vehicle Type (`veh_type`)
Unique values: 10 | Non-null: 4887
| Value | Count |
|---|---|
| (null) | 3286 |
| bmtc_bus | 1466 |
| heavy_vehicle | 965 |
| lcv | 678 |
| others | 449 |
| private_bus | 359 |
| private_car | 345 |
| truck | 276 |
| ksrtc_bus | 217 |
| taxi | 95 |
| auto | 37 |

### Direction (`direction`)
Unique values: 8 | Non-null: 43
| Value | Count |
|---|---|
| (null) | 8130 |
| south_west | 12 |
| north_west | 10 |
| west | 8 |
| south | 7 |
| north | 2 |
| north_east | 2 |
| east | 1 |
| south_east | 1 |

## 4. Datetime Ranges
- **start_datetime:** 2023-11-09 19:24:48.154000+00:00 → 2024-04-08 17:11:42.780000+00:00 (8057 non-null)
- **end_datetime:** 2023-11-12 02:05:46+00:00 → 2027-11-09 11:35:46+00:00 (475 non-null)
- **closed_datetime:** 2023-11-09 22:48:37.836256+00:00 → 2024-04-20 06:16:08.118135+00:00 (3141 non-null)
- **resolved_datetime:** 2023-11-10 07:21:41.463359+00:00 → 2024-04-02 22:33:50.469479+00:00 (74 non-null)
- **created_date:** 2023-09-29 23:38:19.342539+00:00 → 2024-04-08 17:22:58.849385+00:00 (8171 non-null)
- **modified_datetime:** 2023-11-09 20:35:47.789399+00:00 → 2024-04-20 06:16:08.264418+00:00 (8173 non-null)

## 5. Geographic Coverage
- **latitude:** min=12.801041, max=13.267510, non-zero count=8173
- **longitude:** min=77.308731, max=77.769403, non-zero count=8173
- **endlatitude:** min=12.839346, max=59.860133, non-zero count=689
- **endlongitude:** min=62.712878, max=80.720691, non-zero count=689

## 6. Junction Coverage
- Non-null: 2510 / 8173 (30.7%)
- Unique junctions: 294

## 7. Duration-to-Close Proxy (closed_datetime − start_datetime)
- Valid durations (>0 and <30 days): 2983
- Mean: 2488.3 min
- Median: 61.2 min
- Std: 6933.2 min
- 5th percentile: 7.9 min
- 95th percentile: 19188.7 min

### Median Duration by Event Cause
| Event Cause | Median (min) | Mean (min) | Count |
|---|---|---|---|
| Debris | 22978.2 | 22978.2 | 2 |
| road_conditions | 6228.6 | 10324.5 | 76 |
| pot_holes | 5784.1 | 10394.8 | 118 |
| water_logging | 2756.5 | 7032.1 | 223 |
| construction | 1897.9 | 7081.2 | 107 |
| tree_fall | 625.2 | 4764.6 | 167 |
| others | 242.2 | 5614.2 | 380 |
| congestion | 71.5 | 74.7 | 22 |
| vehicle_breakdown | 40.7 | 58.2 | 1784 |
| accident | 40.0 | 47.9 | 87 |
| procession | 36.5 | 54.5 | 13 |
| protest | 24.5 | 24.5 | 2 |
| test_demo | 1.8 | 1.8 | 2 |

## 8. Diurnal Pattern (Hour of Start, IST = UTC+5:30)
| Hour (IST) | Event Count |
|---|---|
| 00:00 | 478 |
| 01:00 | 549 |
| 02:00 | 839 |
| 03:00 | 645 |
| 04:00 | 566 |
| 05:00 | 442 |
| 06:00 | 394 |
| 07:00 | 374 |
| 08:00 | 349 |
| 09:00 | 436 |
| 10:00 | 653 |
| 11:00 | 671 |
| 12:00 | 615 |
| 13:00 | 361 |
| 14:00 | 256 |
| 15:00 | 93 |
| 16:00 | 71 |
| 17:00 | 65 |
| 18:00 | 52 |
| 19:00 | 21 |
| 20:00 | 8 |
| 21:00 | 8 |
| 22:00 | 22 |
| 23:00 | 89 |

## 9. Planned/Event-Driven Subset Analysis
- Total rows matching event-driven causes: 671
| Value | Count |
|---|---|
| construction | 480 |
| public_event | 84 |
| procession | 72 |
| vip_movement | 20 |
| protest | 15 |

**By event_type in this subset:**
| Value | Count |
|---|---|
| planned | 461 |
| unplanned | 210 |