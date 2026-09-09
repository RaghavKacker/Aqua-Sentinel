"""
Tests for AquaSentinel AI Input Normalizer and Graceful Degradation Ladder.
"""

import pytest
import numpy as np
import cv2
from backend.input_normalizer import normalize_input_sources, _parse_nav_csv

def test_normalize_raw_image_no_metadata():
    """Test Level 1: No metadata -> Detection and confidence only; tier is NONE."""
    dummy_img = np.zeros((500, 640), dtype=np.uint8)
    _, encoded = cv2.imencode('.png', dummy_img)
    img_bytes = encoded.tobytes()

    result = normalize_input_sources(
        image_bytes_or_path=img_bytes,
        filename="test_swath.png",
        nav_file_bytes=None
    )

    assert result.degradation_tier == "NONE"
    assert result.image.shape == (500, 640)
    assert len(result.navigation_fixes) == 0
    assert result.get_mean_altitude() is None


def test_normalize_partial_metadata():
    """Test Level 2: Partial metadata -> Altitude only, no lat/lon."""
    dummy_img = np.zeros((500, 640), dtype=np.uint8)
    _, encoded = cv2.imencode('.png', dummy_img)
    img_bytes = encoded.tobytes()

    result = normalize_input_sources(
        image_bytes_or_path=img_bytes,
        filename="test_swath.png",
        nav_file_bytes=None,
        metadata_json={"altitude_m": 8.5}
    )

    assert result.degradation_tier == "PARTIAL"
    assert result.get_mean_altitude() == 8.5


def test_normalize_full_metadata_csv():
    """Test Level 3: Full metadata -> Sonar + CSV nav log with Lat/Lon & Altitude."""
    dummy_img = np.zeros((500, 640), dtype=np.uint8)
    _, encoded = cv2.imencode('.png', dummy_img)
    img_bytes = encoded.tobytes()

    csv_content = b"""timestamp,latitude,longitude,altitude,heading
1693569300.0,24.891234,54.921234,7.2,180.5
1693569301.0,24.891250,54.921240,7.1,180.5
"""
    result = normalize_input_sources(
        image_bytes_or_path=img_bytes,
        filename="test_survey.png",
        nav_file_bytes=csv_content
    )

    assert result.degradation_tier == "FULL"
    assert len(result.navigation_fixes) == 2
    assert pytest.approx(result.get_mean_altitude(), 0.1) == 7.15
    assert result.summary["has_geolocation"] is True


def test_corrupted_metadata_never_crashes():
    """Test that garbage metadata never raises fatal exception."""
    dummy_img = np.zeros((100, 100), dtype=np.uint8)
    _, encoded = cv2.imencode('.png', dummy_img)
    img_bytes = encoded.tobytes()

    garbage_csv = b"garbage\xff\xfe\x00data\r\nnot,a,real,csv"
    result = normalize_input_sources(
        image_bytes_or_path=img_bytes,
        filename="corrupted.png",
        nav_file_bytes=garbage_csv
    )

    assert result.image is not None
    assert result.degradation_tier in ["NONE", "PARTIAL"]
