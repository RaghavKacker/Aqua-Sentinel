"""
AquaSentinel AI - Dynamic Waterfall Tiling & Coordinate Reconstruction
Extracts overlapping tiles (~640x640 with ~20% overlap) and merges tile detections back to full swath coordinates.
"""

from typing import List, Tuple, Dict, Any, Generator
import numpy as np

def generate_tiles(
    image: np.ndarray,
    tile_size: int = 640,
    overlap_ratio: float = 0.20
) -> Generator[Tuple[np.ndarray, int, int, int, int], None, None]:
    """
    Generator yielding (tile, x_offset, y_offset, tile_width, tile_height).
    Memory efficient: avoids pre-allocating all slices simultaneously.
    """
    h, w = image.shape[:2]
    step = int(tile_size * (1.0 - overlap_ratio))
    step = max(step, 1)

    y_points = list(range(0, h, step))
    x_points = list(range(0, w, step))

    # Adjust last points to touch the edge
    if len(y_points) > 1 and (y_points[-1] + tile_size) > h and (h - tile_size) >= 0:
        y_points[-1] = h - tile_size
    if len(x_points) > 1 and (x_points[-1] + tile_size) > w and (w - tile_size) >= 0:
        x_points[-1] = w - tile_size

    for y in y_points:
        y_end = min(y + tile_size, h)
        cur_h = y_end - y
        for x in x_points:
            x_end = min(x + tile_size, w)
            cur_w = x_end - x

            tile = image[y:y_end, x:x_end]

            # Pad tile to tile_size x tile_size if along boundaries
            if cur_h < tile_size or cur_w < tile_size:
                if len(image.shape) == 3:
                    padded = np.zeros((tile_size, tile_size, image.shape[2]), dtype=image.dtype)
                else:
                    padded = np.zeros((tile_size, tile_size), dtype=image.dtype)
                padded[:cur_h, :cur_w] = tile
                yield padded, x, y, cur_w, cur_h
            else:
                yield tile, x, y, cur_w, cur_h


def map_detections_to_global(
    tile_detections: List[Dict[str, Any]],
    x_offset: int,
    y_offset: int
) -> List[Dict[str, Any]]:
    """
    Translates local tile bounding box and polygon coordinates into global waterfall coordinates.
    """
    global_dets = []
    for det in tile_detections:
        d = det.copy()
        # Bounding box translation
        d["x_min"] = int(det["x_min"] + x_offset)
        d["y_min"] = int(det["y_min"] + y_offset)
        d["x_max"] = int(det["x_max"] + x_offset)
        d["y_max"] = int(det["y_max"] + y_offset)

        # Polygon coordinates translation [x1, y1, x2, y2, ...]
        if "polygon" in det and det["polygon"]:
            global_poly = []
            poly = det["polygon"]
            for i in range(0, len(poly), 2):
                gx = poly[i] + x_offset
                gy = poly[i + 1] + y_offset
                global_poly.extend([round(float(gx), 2), round(float(gy), 2)])
            d["polygon"] = global_poly

        global_dets.append(d)
    return global_dets


def apply_cross_tile_nms(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.40
) -> List[Dict[str, Any]]:
    """
    Performs standard Non-Maximum Suppression (NMS) across overlapping global detections.
    """
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence_ai", 0.0), reverse=True)
    kept = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        kept.append(best)

        remaining = []
        for candidate in sorted_dets:
            # Check class match
            if candidate.get("class_name") != best.get("class_name"):
                remaining.append(candidate)
                continue

            iou = _compute_iou(best, candidate)
            if iou < iou_threshold:
                remaining.append(candidate)

        sorted_dets = remaining

    return kept


def _compute_iou(boxA: Dict[str, Any], boxB: Dict[str, Any]) -> float:
    xA = max(boxA["x_min"], boxB["x_min"])
    yA = max(boxA["y_min"], boxB["y_min"])
    xB = min(boxA["x_max"], boxB["x_max"])
    yB = min(boxA["y_max"], boxB["y_max"])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = max(1, (boxA["x_max"] - boxA["x_min"]) * (boxA["y_max"] - boxA["y_min"]))
    areaB = max(1, (boxB["x_max"] - boxB["x_min"]) * (boxB["y_max"] - boxB["y_min"]))

    return interArea / float(areaA + areaB - interArea)
