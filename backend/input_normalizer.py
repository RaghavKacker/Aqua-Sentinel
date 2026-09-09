"""
AquaSentinel AI - Input Normalizer
Normalizes diverse Side-Scan Sonar imagery, vehicle navigation logs, and survey metadata.
Guarantees graceful degradation: missing metadata never crashes the system.
"""

import os
import csv
import io
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2

class NormalizedSurveyInput:
    def __init__(
        self,
        image: np.ndarray,
        survey_name: str,
        image_path: Optional[str] = None,
        navigation_fixes: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.image = image
        self.survey_name = survey_name
        self.image_path = image_path
        self.navigation_fixes = navigation_fixes or []
        self.metadata = metadata or {}

        # Determine degradation tier
        self.degradation_tier = self._determine_tier()

    def _determine_tier(self) -> str:
        has_nav = len(self.navigation_fixes) > 0
        has_geo = any(f.get("latitude") is not None for f in self.navigation_fixes)
        has_alt = any(f.get("altitude_m") is not None for f in self.navigation_fixes) or self.metadata.get("altitude_m") is not None

        if has_nav and (has_geo or any(f.get("local_x_m") is not None for f in self.navigation_fixes)) and has_alt:
            return "FULL"
        elif has_nav or has_alt:
            return "PARTIAL"
        return "NONE"

    @property
    def summary(self) -> Dict[str, Any]:
        h, w = self.image.shape[:2]
        return {
            "survey_name": self.survey_name,
            "dimensions": [w, h],
            "degradation_tier": self.degradation_tier,
            "nav_records_count": len(self.navigation_fixes),
            "altitude_available": self.get_mean_altitude() is not None,
            "has_geolocation": any(f.get("latitude") is not None for f in self.navigation_fixes)
        }

    def get_mean_altitude(self) -> Optional[float]:
        if "altitude_m" in self.metadata and self.metadata["altitude_m"] is not None:
            return float(self.metadata["altitude_m"])

        altitudes = [f["altitude_m"] for f in self.navigation_fixes if f.get("altitude_m") is not None]
        if altitudes:
            return float(np.mean(altitudes))
        return None

    def get_nav_at_row(self, row_idx: int) -> Optional[Dict[str, Any]]:
        if not self.navigation_fixes:
            return None
        # Nearest ping interpolation
        h = self.image.shape[0]
        n_nav = len(self.navigation_fixes)
        if n_nav == 1:
            return self.navigation_fixes[0]

        target_idx = int((row_idx / max(1, h)) * n_nav)
        target_idx = min(n_nav - 1, max(0, target_idx))
        return self.navigation_fixes[target_idx]


def normalize_input_sources(
    image_bytes_or_path: Any,
    filename: str,
    nav_file_bytes: Optional[bytes] = None,
    metadata_json: Optional[Dict[str, Any]] = None
) -> NormalizedSurveyInput:
    """
    Ingests raw file streams, normalizes imagery, parses navigation CSVs, and wraps them.
    Missing or corrupted metadata gracefully degrades without raising fatal exceptions.
    """
    # 1. Image loading
    if isinstance(image_bytes_or_path, (str, Path)):
        img_path = str(image_bytes_or_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Fallback for Netpbm PBM/PGM or unhandled formats
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            img = _decode_image_bytes(img_bytes)
    elif isinstance(image_bytes_or_path, bytes):
        img_path = None
        img = _decode_image_bytes(image_bytes_or_path)
    else:
        raise ValueError("Invalid image input type: must be path or bytes")

    if img is None:
        raise ValueError(f"Failed to decode sonar image: {filename}")

    # Standardize to single channel grayscale or 3-channel RGB
    if len(img.shape) == 2:
        # Grayscale
        normalized_img = img
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            normalized_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            normalized_img = img
    else:
        normalized_img = img

    # 2. Navigation parsing
    nav_fixes = []
    if nav_file_bytes:
        try:
            nav_fixes = _parse_nav_csv(nav_file_bytes)
        except Exception as e:
            print(f"[NORMALIZER WARNING] Corrupted or non-standard navigation log: {e}. Gracefully continuing without nav.")

    survey_name = Path(filename).stem
    return NormalizedSurveyInput(
        image=normalized_img,
        survey_name=survey_name,
        image_path=img_path,
        navigation_fixes=nav_fixes,
        metadata=metadata_json or {}
    )


def _decode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)


def _parse_nav_csv(data: bytes) -> List[Dict[str, Any]]:
    """
    Robust CSV parser supporting both SubPipeMiniSSS format (EstimatedState.csv)
    and standard topside GPS/NMEA tabular exports.
    """
    text = data.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    fixes = []

    for row in reader:
        # Normalize column names to lowercase stripped
        low_row = {k.strip().lower(): v.strip() for k, v in row.items() if k}

        # Timestamp
        ts = _safe_float(low_row.get("timestamp") or low_row.get("time") or low_row.get("t"))

        # WGS84 Geolocation
        lat = _safe_float(low_row.get("latitude") or low_row.get("lat"))
        lon = _safe_float(low_row.get("longitude") or low_row.get("lon") or low_row.get("long"))

        # Local AUV Odometry (e.g. SubPipeMiniSSS x, y, z)
        local_x = _safe_float(low_row.get("x") or low_row.get("pos_x") or low_row.get("northing"))
        local_y = _safe_float(low_row.get("y") or low_row.get("pos_y") or low_row.get("easting"))

        # Altitude
        alt = _safe_float(low_row.get("altitude") or low_row.get("alt") or low_row.get("altitude_m"))

        # Heading / Attitude
        heading = _safe_float(low_row.get("heading") or low_row.get("yaw") or low_row.get("psi"))
        if heading is not None and "yaw" in low_row and abs(heading) <= 6.28:
            # Likely in radians (like SubPipeMiniSSS)
            heading_deg = np.degrees(heading) % 360.0
        elif heading is not None:
            heading_deg = heading % 360.0
        else:
            heading_deg = 0.0

        fixes.append({
            "timestamp": ts,
            "latitude": lat,
            "longitude": lon,
            "local_x_m": local_x,
            "local_y_m": local_y,
            "altitude_m": alt,
            "heading_deg": round(heading_deg, 2)
        })

    return fixes


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
