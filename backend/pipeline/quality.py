"""
AquaSentinel AI - Image Quality Assessment
Computes sonar signal metrics: SNR, dynamic range, mean intensity, and contrast.
"""

from typing import Dict, Any
import numpy as np
import cv2

def assess_image_quality(image: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes an 8-bit grayscale sonar image or RGB image and returns hydrographic quality indices.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    dynamic_range = max_val - min_val

    # Contrast metric (Michelson contrast)
    if (max_val + min_val) > 0:
        michelson_contrast = float((max_val - min_val) / (max_val + min_val))
    else:
        michelson_contrast = 0.0

    # Approximate acoustic Signal-to-Noise Ratio (SNR) in dB
    # Uses mean as acoustic signal estimate and std as acoustic reverberation/speckle noise
    if std_val > 0.001:
        snr_db = float(20.0 * np.log10(mean_val / std_val))
    else:
        snr_db = 0.0

    # Qualitative rating
    if dynamic_range > 150 and snr_db > 5.0:
        quality_rating = "EXCELLENT"
    elif dynamic_range > 80 and snr_db > 2.0:
        quality_rating = "GOOD"
    elif dynamic_range > 40:
        quality_rating = "FAIR"
    else:
        quality_rating = "POOR"

    return {
        "mean_intensity": round(mean_val, 2),
        "std_intensity": round(std_val, 2),
        "dynamic_range": dynamic_range,
        "michelson_contrast": round(michelson_contrast, 3),
        "snr_db": round(snr_db, 2),
        "rating": quality_rating
    }
