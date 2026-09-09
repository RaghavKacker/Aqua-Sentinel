"""
AquaSentinel AI - Slant-Range Ground Correction
Applies Pythagorean slant-range to ground-range projection:
R_ground = sqrt(R_slant^2 - H^2)
"""

from typing import Tuple, Optional
import numpy as np
import cv2

def apply_slant_range_correction(
    image: np.ndarray,
    altitude_m: Optional[float],
    meters_per_pixel: float = 0.05,
    is_dual_swath: bool = True
) -> Tuple[np.ndarray, bool]:
    """
    Performs Pythagorean slant-range correction.
    If altitude_m is None or <= 0, returns the original image without modification (graceful degradation).
    """
    if altitude_m is None or altitude_m <= 0.0:
        return image, False

    height_px, width_px = image.shape[:2]
    altitude_px = altitude_m / meters_per_pixel

    corrected = np.zeros_like(image)

    if is_dual_swath:
        mid_x = width_px // 2
        # Port channel (left of nadir)
        port_half = image[:, :mid_x]
        port_corrected = _correct_half_swath(port_half, altitude_px, is_port=True)
        # Starboard channel (right of nadir)
        starboard_half = image[:, mid_x:]
        starboard_corrected = _correct_half_swath(starboard_half, altitude_px, is_port=False)

        corrected = np.hstack([port_corrected, starboard_corrected])
    else:
        corrected = _correct_half_swath(image, altitude_px, is_port=False)

    return corrected, True


def _correct_half_swath(half_img: np.ndarray, altitude_px: float, is_port: bool) -> np.ndarray:
    h, w = half_img.shape[:2]
    out = np.zeros_like(half_img)

    # For each ground-range pixel x_g, compute corresponding slant-range pixel x_s
    # x_s = sqrt(x_g^2 + altitude_px^2)
    x_g = np.arange(w, dtype=np.float32)
    x_s = np.sqrt(x_g**2 + altitude_px**2)

    # Map indices
    if is_port:
        # For port channel, nadir is at the right edge of the half swath
        # Reverse mapping
        map_x = (w - 1) - x_s
        map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
        map_x = map_x[::-1]  # flip back to match left-to-right
    else:
        # For starboard channel, nadir is at the left edge
        map_x = np.clip(x_s, 0, w - 1).astype(np.float32)

    map_y = np.arange(h, dtype=np.float32)
    map_xx, map_yy = np.meshgrid(map_x, map_y)

    out = cv2.remap(half_img, map_xx, map_yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out
