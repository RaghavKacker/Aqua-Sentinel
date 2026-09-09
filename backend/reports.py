"""
AquaSentinel AI - Deliverable & Report Exporter
Generates standard RFC 7946 GeoJSON, tabular hydrographic CSVs, and structured JSON reports.
"""

import csv
import io
import json
from typing import List, Dict, Any

def export_geojson(detections: List[Dict[str, Any]], survey_name: str) -> Dict[str, Any]:
    """
    Constructs an RFC 7946 GeoJSON FeatureCollection from detected targets.
    Targets with WGS84 coordinates are exported as Point geometries.
    """
    features = []
    for d in detections:
        lat = d.get("latitude")
        lon = d.get("longitude")

        geometry = None
        if lat is not None and lon is not None:
            geometry = {
                "type": "Point",
                "coordinates": [lon, lat]
            }

        properties = {
            "target_id": d["id"],
            "survey_id": d["survey_id"],
            "class_name": d["class_name"],
            "confidence_ai": d["confidence_ai"],
            "confidence_final": d["confidence_final"],
            "evidence_status": d["evidence_status"],
            "review_status": d["review_status"],
            "relief_height_m": d.get("relief_height_m"),
            "dimensions_m": {
                "length": d.get("length_m"),
                "width": d.get("width_m"),
                "area": d.get("area_m2")
            },
            "risk_points": d["risk_points"],
            "priority_level": d["priority_level"],
            "coordinate_frame": d.get("coordinate_frame")
        }

        features.append({
            "type": "Feature",
            "id": d["id"],
            "geometry": geometry,
            "properties": properties
        })

    return {
        "type": "FeatureCollection",
        "survey_name": survey_name,
        "features": features
    }


def export_csv(detections: List[Dict[str, Any]]) -> str:
    """
    Constructs a tabular CSV hydrographic target catalog.
    """
    output = io.StringIO()
    fieldnames = [
        "target_id", "class_name", "confidence_final", "evidence_status", "review_status",
        "priority_level", "risk_points", "latitude", "longitude", "local_x_m", "local_y_m",
        "relief_height_m", "length_m", "width_m", "area_m2", "x_min", "y_min", "x_max", "y_max"
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for d in detections:
        row = {
            "target_id": d.get("id"),
            "class_name": d.get("class_name"),
            "confidence_final": d.get("confidence_final"),
            "evidence_status": d.get("evidence_status"),
            "review_status": d.get("review_status"),
            "priority_level": d.get("priority_level"),
            "risk_points": d.get("risk_points"),
            "latitude": d.get("latitude"),
            "longitude": d.get("longitude"),
            "local_x_m": d.get("local_x_m"),
            "local_y_m": d.get("local_y_m"),
            "relief_height_m": d.get("relief_height_m"),
            "length_m": d.get("length_m"),
            "width_m": d.get("width_m"),
            "area_m2": d.get("area_m2"),
            "x_min": d.get("x_min"),
            "y_min": d.get("y_min"),
            "x_max": d.get("x_max"),
            "y_max": d.get("y_max")
        }
        writer.writerow(row)

    return output.getvalue()
