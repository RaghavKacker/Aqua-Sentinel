"""
AquaSentinel AI - Acoustic Shadow Verification Engine
Inspects expected down-range acoustic shadow region in the sonar propagation direction.
Never blindly rejects detections; acts as an evidence rater.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

def verify_acoustic_shadow(
    image: np.ndarray,
    detection: Dict[str, Any],
    nadir_x: Optional[int] = None,
    altitude_m: Optional[float] = None,
    meters_per_pixel: float = 0.05
) -> Dict[str, Any]:
    """
    Evaluates acoustic shadow evidence behind a candidate target.

    Evidence States:
    - SUPPORTING: Shadow distinctly detected down-range -> boosts confidence & calculates relief height.
    - NEUTRAL: Shadow region ambiguous or noisy -> retains visual confidence.
    - ABSENT: Clear seafloor highlight where shadow is expected but missing -> soft confidence penalty, flagged for review.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    if nadir_x is None:
        nadir_x = w // 2

    x_min = max(0, detection["x_min"])
    y_min = max(0, detection["y_min"])
    x_max = min(w, detection["x_max"])
    y_max = min(h, detection["y_max"])
    target_w = max(1, x_max - x_min)
    target_h = max(1, y_max - y_min)

    # 1. Determine acoustic propagation direction
    # If target is right of nadir, sound travels left-to-right (shadow casts to the right)
    # If target is left of nadir, sound travels right-to-left (shadow casts to the left)
    target_cx = (x_min + x_max) // 2
    is_starboard = (target_cx >= nadir_x)

    # Search window length down-range (usually 1.5x to 3x object width)
    max_shadow_search_px = min(w // 4, max(40, target_w * 3))

    if is_starboard:
        shadow_x1 = x_max
        shadow_x2 = min(w, x_max + max_shadow_search_px)
    else:
        shadow_x1 = max(0, x_min - max_shadow_search_px)
        shadow_x2 = x_min

    shadow_y1 = max(0, y_min - 5)
    shadow_y2 = min(h, y_max + 5)

    shadow_crop = gray[shadow_y1:shadow_y2, shadow_x1:shadow_x2]
    target_crop = gray[y_min:y_max, x_min:x_max]

    # Reference seabed backscatter in adjacent area
    ref_y1 = max(0, y_min - 20)
    ref_y2 = min(h, y_max + 20)
    ref_x1 = max(0, x_min - 20)
    ref_x2 = min(w, x_max + 20)
    ref_seabed = gray[ref_y1:ref_y2, ref_x1:ref_x2]

    mean_seabed = float(np.mean(ref_seabed)) + 1e-5
    mean_target = float(np.mean(target_crop))

    if shadow_crop.size == 0:
        return {
            "evidence": "NEUTRAL",
            "review_status": "UNVERIFIED",
            "shadow_contrast": 0.0,
            "relief_height_m": None,
            "shadow_bbox": None,
            "confidence_adjusted": detection["confidence_ai"]
        }

    mean_shadow = float(np.mean(shadow_crop))

    # Shadow contrast ratio C = 1.0 - (mean_shadow / mean_seabed)
    shadow_contrast = max(0.0, min(1.0, 1.0 - (mean_shadow / mean_seabed)))

    # Estimate physical shadow length (pixels with intensity < 0.5 * mean_seabed)
    shadow_mask = (shadow_crop < (mean_seabed * 0.55))
    col_has_shadow = np.any(shadow_mask, axis=0)
    if not is_starboard:
        col_has_shadow = col_has_shadow[::-1]  # from object outward

    shadow_length_px = 0
    for has_shadow in col_has_shadow:
        if has_shadow:
            shadow_length_px += 1
        else:
            if shadow_length_px > 5:
                break  # shadow ended

    # Physical relief height estimation: h = (H * L_shadow) / R_slant
    relief_height_m = None
    if altitude_m is not None and altitude_m > 0:
        slant_range_px = max(10, abs(target_cx - nadir_x))
        slant_range_m = slant_range_px * meters_per_pixel
        shadow_length_m = shadow_length_px * meters_per_pixel
        if slant_range_m > 0:
            relief_height_m = round((altitude_m * shadow_length_m) / slant_range_m, 2)

    # Evidence categorization & Confidence adjustment (NEVER automatically reject)
    raw_conf = detection["confidence_ai"]

    if shadow_contrast >= 0.35 and shadow_length_px >= 5:
        # Supporting evidence: Boost confidence
        evidence = "SUPPORTING"
        review_status = "VERIFIED"
        confidence_adjusted = min(1.0, raw_conf + 0.12 * shadow_contrast)
    elif shadow_contrast < 0.15 and raw_conf > 0.65:
        # Expected shadow absent: Apply soft penalty and flag for human review
        evidence = "ABSENT"
        review_status = "FLAGGED_FOR_REVIEW"
        confidence_adjusted = max(0.20, raw_conf * 0.85)
    else:
        # Neutral evidence: Indeterminate seabed
        evidence = "NEUTRAL"
        review_status = "UNVERIFIED"
        confidence_adjusted = raw_conf

    # Shadow bounding box for UI overlay
    if is_starboard:
        actual_shadow_x2 = min(w, x_max + shadow_length_px)
        shadow_bbox = [x_max, shadow_y1, actual_shadow_x2, shadow_y2]
    else:
        actual_shadow_x1 = max(0, x_min - shadow_length_px)
        shadow_bbox = [actual_shadow_x1, shadow_y1, x_min, shadow_y2]

    return {
        "evidence": evidence,
        "review_status": review_status,
        "shadow_contrast": round(shadow_contrast, 3),
        "shadow_length_px": shadow_length_px,
        "relief_height_m": relief_height_m,
        "shadow_bbox": shadow_bbox if shadow_length_px >= 5 else None,
        "confidence_adjusted": round(confidence_adjusted, 4)
    }
