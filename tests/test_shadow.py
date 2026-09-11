"""
Tests for Acoustic Shadow Evidence Verification and Geolocation Projection.
"""

import pytest
import numpy as np
from backend.pipeline.shadow_gate import verify_acoustic_shadow
from backend.pipeline.geolocation import project_detection_geolocation, compute_risk_score

def test_shadow_verification_supporting_evidence():
    """Test target with distinct down-range acoustic shadow -> SUPPORTING evidence."""
    # 500x640 synthetic image with nadir in center (x=320)
    img = np.full((300, 640), fill_value=100, dtype=np.uint8)

    # Highlight at starboard (x=360 to 400)
    img[100:150, 360:400] = 230
    # Shadow down-range (x=400 to 460) with low backscatter (10 intensity)
    img[100:150, 400:460] = 10

    detection = {
        "x_min": 360,
        "y_min": 100,
        "x_max": 400,
        "y_max": 150,
        "confidence_ai": 0.80,
        "class_name": "mine_cylinder"
    }

    result = verify_acoustic_shadow(
        image=img,
        detection=detection,
        nadir_x=320,
        altitude_m=8.0,
        meters_per_pixel=0.05
    )

    assert result["evidence"] == "SUPPORTING"
    assert result["review_status"] == "VERIFIED"
    assert result["confidence_adjusted"] > 0.80
    assert result["relief_height_m"] is not None
    assert result["relief_height_m"] > 0.1


def test_shadow_verification_never_discards_when_absent():
    """Test that missing shadow applies soft penalty and flags for review, never discards."""
    # Uniform seabed
    img = np.full((300, 640), fill_value=100, dtype=np.uint8)
    # Bright spot without shadow
    img[100:150, 360:400] = 230

    detection = {
        "x_min": 360,
        "y_min": 100,
        "x_max": 400,
        "y_max": 150,
        "confidence_ai": 0.75,
        "class_name": "ghost_net"
    }

    result = verify_acoustic_shadow(
        image=img,
        detection=detection,
        nadir_x=320,
        altitude_m=8.0,
        meters_per_pixel=0.05
    )

    # Should not be discarded! Should be flagged for review with soft penalty
    assert result["evidence"] in ["ABSENT", "NEUTRAL"]
    assert result["review_status"] in ["FLAGGED_FOR_REVIEW", "UNVERIFIED"]
    assert result["confidence_adjusted"] > 0.20


def test_wgs84_geolocation_projection():
    """Test target projection to WGS84 coordinates."""
    detection = {"x_min": 400, "y_min": 100, "x_max": 440, "y_max": 120}
    platform_nav = {
        "latitude": 24.500000,
        "longitude": 54.500000,
        "heading_deg": 90.0,  # Heading East
        "altitude_m": 8.0
    }

    # Nadir at 320 -> target center at 420 -> +100px starboard (towards South if heading East)
    geo = project_detection_geolocation(
        detection=detection,
        nadir_x=320,
        meters_per_pixel=0.05,
        platform_nav=platform_nav
    )

    assert geo["coordinate_frame"] == "WGS84"
    assert geo["latitude"] is not None
    assert geo["longitude"] is not None
    assert geo["latitude"] < 24.500000  # Starboard (+90 deg from East) is South -> Lat decreases


def test_risk_score_calculation():
    """Test risk scoring urgency tiers."""
    # Critical hazard: mine with supporting shadow
    crit = compute_risk_score(
        class_name="mine_cylinder",
        confidence=0.90,
        evidence_status="SUPPORTING",
        relief_height_m=1.2,
        area_m2=6.0
    )
    assert crit["priority_level"] == "CRITICAL"
    assert crit["points"] >= 75
