"""
End-to-end integration test for AquaSentinel AI FastAPI backend pipeline.
"""

import io
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_system_status():
    res = client.get("/api/v1/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ONLINE"
    assert data["offline_ready"] is True
    assert data["compute_profile"] == "NORMAL"


def test_process_survey_e2e():
    """Run full pipeline: upload -> preprocess -> tile -> detect -> shadow -> geo -> db -> export."""
    # Generate realistic SSS patch with ghost gear target
    from ml.prepare_dataset import generate_synthetic_sss_patch
    img, _ = generate_synthetic_sss_patch(width=640, height=640, has_target=True, target_class=1)

    _, encoded = cv2.imencode('.png', img)
    sonar_bytes = io.BytesIO(encoded.tobytes())

    # Sample navigation CSV
    nav_csv = (
        "timestamp,latitude,longitude,altitude,heading\n"
        "1693569300.0,24.8912,54.9212,8.5,180.0\n"
    )
    nav_bytes = io.BytesIO(nav_csv.encode('utf-8'))

    # Mock inference to return a deterministic detection for testing full downstream pipeline
    from unittest.mock import patch
    mock_tile_detection = [{
        "class_id": 0,
        "class_name": "crab_pot",
        "confidence_ai": 0.85,
        "x_min": 150,
        "y_min": 200,
        "x_max": 230,
        "y_max": 280,
        "polygon": [150, 200, 230, 200, 230, 280, 150, 280]
    }]

    with patch.object(client.app.state.inference_engine if hasattr(client.app.state, 'inference_engine') else None or __import__('backend.main', fromlist=['inference_engine']).inference_engine, "predict_tile", return_value=mock_tile_detection):
        response = client.post(
            "/api/v1/surveys/process",
            files={
                "sonar_file": ("test_survey_line.png", sonar_bytes, "image/png"),
                "nav_file": ("nav_telemetry.csv", nav_bytes, "text/csv")
            },
            data={
                "apply_slant_range": "false",
                "apply_clahe": "false",
                "confidence_threshold": "0.15"
            }
        )

    assert response.status_code == 200
    data = response.json()
    assert "survey_id" in data
    assert data["degradation_tier"] == "FULL"
    assert data["tiles_processed"] >= 1
    assert data["detections_count"] >= 1

    det = data["detections"][0]
    assert "confidence_final" in det
    assert "evidence_status" in det
    assert "review_status" in det
    assert "priority_level" in det
    assert det["coordinate_frame"] == "WGS84"
    assert det["latitude"] is not None

    survey_id = data["survey_id"]

    # Test GeoJSON export
    geo_res = client.get(f"/api/v1/surveys/{survey_id}/export/geojson")
    assert geo_res.status_code == 200
    geo_json = geo_res.json()
    assert geo_json["type"] == "FeatureCollection"
    assert len(geo_json["features"]) >= 1

    # Test CSV export
    csv_res = client.get(f"/api/v1/surveys/{survey_id}/export/csv")
    assert csv_res.status_code == 200
    assert "target_id,class_name" in csv_res.text

    # Test Dashboard KPIs
    stats_res = client.get("/api/v1/system/stats")
    assert stats_res.status_code == 200
    assert stats_res.json()["total_surveys"] >= 1
