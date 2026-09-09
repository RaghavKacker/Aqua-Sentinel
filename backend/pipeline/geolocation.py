"""
AquaSentinel AI - Metadata-Aware Geolocation & Risk Engine
Projects sonar detections to WGS84 coordinates, local metric odometry, or untagged state.
Computes target dimensions and operational risk priority.
"""

import math
from typing import Dict, Any, Optional, Tuple

WGS84_EARTH_RADIUS_M = 6378137.0

# Base class hazard points [0 - 50]
CLASS_RISK_POINTS = {
    "mine_cylinder": 50,
    "ghost_gear": 45,
    "crab_pot": 35,
    "debris_anomaly": 25,
    "submarine_pipeline": 40,
    "shipwreck": 30
}

def project_detection_geolocation(
    detection: Dict[str, Any],
    nadir_x: int,
    meters_per_pixel: float = 0.05,
    platform_nav: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Translates cross-track pixel offset to real-world coordinates.
    Graceful degradation: WGS84 -> LOCAL_METRIC_ODOMETRY -> UNTAGGED.
    """
    x_min = detection["x_min"]
    x_max = detection["x_max"]
    target_cx = (x_min + x_max) / 2.0

    # Cross-track ground range in meters from nadir center
    cross_track_offset_px = target_cx - nadir_x
    cross_track_m = abs(cross_track_offset_px) * meters_per_pixel
    is_starboard = (cross_track_offset_px >= 0)

    if not platform_nav:
        return {
            "coordinate_frame": "UNTAGGED",
            "latitude": None,
            "longitude": None,
            "local_x_m": None,
            "local_y_m": None,
            "cross_track_m": round(cross_track_m, 2)
        }

    # 1. Check if WGS84 coordinates are provided
    lat0 = platform_nav.get("latitude")
    lon0 = platform_nav.get("longitude")
    heading_deg = platform_nav.get("heading_deg", 0.0)

    if lat0 is not None and lon0 is not None:
        # Cross-track bearing is perpendicular to heading: +90 for starboard, -90 for port
        bearing_offset = 90.0 if is_starboard else -90.0
        target_bearing_rad = math.radians((heading_deg + bearing_offset) % 360.0)

        # Great-circle forward projection
        d_div_r = cross_track_m / WGS84_EARTH_RADIUS_M
        lat0_rad = math.radians(lat0)
        lon0_rad = math.radians(lon0)

        target_lat_rad = math.asin(
            math.sin(lat0_rad) * math.cos(d_div_r) +
            math.cos(lat0_rad) * math.sin(d_div_r) * math.cos(target_bearing_rad)
        )
        target_lon_rad = lon0_rad + math.atan2(
            math.sin(target_bearing_rad) * math.sin(d_div_r) * math.cos(lat0_rad),
            math.cos(d_div_r) - math.sin(lat0_rad) * math.sin(target_lat_rad)
        )

        return {
            "coordinate_frame": "WGS84",
            "latitude": round(math.degrees(target_lat_rad), 7),
            "longitude": round(math.degrees(target_lon_rad), 7),
            "local_x_m": None,
            "local_y_m": None,
            "cross_track_m": round(cross_track_m, 2)
        }

    # 2. Check if local metric odometry (AUV dead-reckoning) is provided
    local_x = platform_nav.get("local_x_m")
    local_y = platform_nav.get("local_y_m")
    yaw_rad = platform_nav.get("yaw_rad", math.radians(heading_deg))

    if local_x is not None and local_y is not None:
        sign = 1.0 if is_starboard else -1.0
        # Starboard is +Y in right-handed AUV body frame (X=forward, Y=starboard)
        target_lx = local_x - sign * cross_track_m * math.sin(yaw_rad)
        target_ly = local_y + sign * cross_track_m * math.cos(yaw_rad)

        return {
            "coordinate_frame": "LOCAL_METRIC_ODOMETRY",
            "latitude": None,
            "longitude": None,
            "local_x_m": round(target_lx, 2),
            "local_y_m": round(target_ly, 2),
            "cross_track_m": round(cross_track_m, 2)
        }

    return {
        "coordinate_frame": "UNTAGGED",
        "latitude": None,
        "longitude": None,
        "local_x_m": None,
        "local_y_m": None,
        "cross_track_m": round(cross_track_m, 2)
    }


def compute_target_dimensions(
    detection: Dict[str, Any],
    meters_per_pixel: float = 0.05,
    vessel_speed_mps: float = 1.5,
    ping_rate_hz: float = 10.0
) -> Dict[str, Optional[float]]:
    """
    Computes length, width, and footprint area in physical meters.
    """
    w_px = max(1, detection["x_max"] - detection["x_min"])
    h_px = max(1, detection["y_max"] - detection["y_min"])

    cross_track_width_m = w_px * meters_per_pixel
    # Along-track length: pings * (vessel_speed / ping_rate)
    meters_per_ping = vessel_speed_mps / max(ping_rate_hz, 0.1)
    along_track_length_m = h_px * meters_per_ping

    area_m2 = cross_track_width_m * along_track_length_m

    return {
        "length_m": round(along_track_length_m, 2),
        "width_m": round(cross_track_width_m, 2),
        "area_m2": round(area_m2, 2)
    }


def compute_risk_score(
    class_name: str,
    confidence: float,
    evidence_status: str,
    relief_height_m: Optional[float],
    area_m2: Optional[float]
) -> Dict[str, Any]:
    """
    Calculates operational risk priority [0 - 100 points].
    Confidence = likelihood detection is correct.
    Risk = importance/urgency of the detected object.
    """
    base_points = CLASS_RISK_POINTS.get(class_name, 25)

    # Size and 3D relief bonus
    size_bonus = 0
    if area_m2 is not None and area_m2 > 5.0:
        size_bonus += 15
    elif area_m2 is not None and area_m2 > 1.5:
        size_bonus += 8

    relief_bonus = 0
    if relief_height_m is not None and relief_height_m > 0.8:
        relief_bonus += 15
    elif relief_height_m is not None and relief_height_m > 0.3:
        relief_bonus += 8

    # Acoustic evidence multiplier
    evidence_mult = 1.0
    if evidence_status == "SUPPORTING":
        evidence_mult = 1.15
    elif evidence_status == "ABSENT":
        evidence_mult = 0.80

    total_points = min(100, int((base_points + size_bonus + relief_bonus) * evidence_mult * confidence))

    if total_points >= 75:
        priority_level = "CRITICAL"
    elif total_points >= 50:
        priority_level = "HIGH"
    elif total_points >= 30:
        priority_level = "MEDIUM"
    else:
        priority_level = "LOW"

    return {
        "points": total_points,
        "priority_level": priority_level
    }
