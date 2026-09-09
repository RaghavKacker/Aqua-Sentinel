"""
AquaSentinel AI - Adaptive YOLO-Seg Inference Engine
Handles hardware-aware AI inference with automatic CUDA/CPU fallback and segmentation polygon extraction.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

CLASS_LABELS = {
    0: "crab_pot",
    1: "ghost_gear",
    2: "mine_cylinder",
    3: "debris_anomaly"
}

class AdaptiveInferenceEngine:
    def __init__(self, model_path: Optional[str] = None, compute_profile: str = "NORMAL"):
        self.compute_profile = compute_profile.upper()
        self.model = None
        self.device = "cpu"
        self.has_cuda = False
        self.model_loaded = False

        self._initialize_hardware()
        self._load_model(model_path)

    def _initialize_hardware(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.has_cuda = True
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"[AI] Detected CUDA GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

                if self.compute_profile in ["HIGH", "NORMAL"]:
                    self.device = "cuda:0"
                else:
                    self.device = "cpu"
            else:
                self.device = "cpu"
                print("[AI] CUDA unavailable; running in CPU fallback mode.")
        except ImportError:
            self.device = "cpu"
            print("[AI] PyTorch not available; CPU fallback mode active.")

    def _load_model(self, model_path: Optional[str]):
        if not model_path:
            model_path = os.getenv("MODEL_WEIGHTS_PATH", "./models/best.pt")

        target_file = Path(model_path)
        if target_file.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(target_file))
                self.model_loaded = True
                print(f"[AI] Loaded fine-tuned YOLO-Seg model: {target_file}")
            except Exception as e:
                print(f"[AI WARNING] Could not load YOLO weights from {target_file}: {e}")
                self._load_fallback_yolo()
        else:
            self._load_fallback_yolo()

    def _load_fallback_yolo(self):
        try:
            from ultralytics import YOLO
            # Try to load base lightweight YOLO-seg model
            self.model = YOLO("yolo11n-seg.pt")
            self.model_loaded = True
            print("[AI] Loaded base yolo11n-seg.pt for inference.")
        except Exception as e:
            print(f"[AI INFO] Ultralytics base model not cached or offline: {e}. Using deterministic acoustic detector fallback.")
            self.model_loaded = False

    def predict_tile(
        self,
        tile: np.ndarray,
        confidence_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Runs YOLO-Seg on a single tile (~640x640) and returns candidate detections with segmentation masks.
        """
        h, w = tile.shape[:2]
        detections = []

        if self.model_loaded and self.model is not None:
            try:
                results = self.model.predict(
                    source=tile,
                    conf=confidence_threshold,
                    device=self.device,
                    verbose=False,
                    imgsz=640
                )
                for r in results:
                    boxes = r.boxes
                    masks = r.masks

                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy()
                        x_min, y_min, x_max, y_max = [int(v) for v in xyxy]

                        class_name = CLASS_LABELS.get(cls_id, f"debris_{cls_id}")

                        # Extract segmentation polygon if present
                        polygon = []
                        if masks is not None and i < len(masks):
                            poly_xy = masks[i].xy[0]  # Array of (x, y) coordinates
                            polygon = poly_xy.flatten().tolist()
                        else:
                            # Fallback to 4-corner box polygon
                            polygon = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]

                        detections.append({
                            "class_id": cls_id,
                            "class_name": class_name,
                            "confidence_ai": round(conf, 4),
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                            "polygon": polygon
                        })
                return detections
            except Exception as e:
                print(f"[AI ERROR] YOLO inference failed: {e}. Executing heuristic detector.")

        # Heuristic Sonar Feature Fallback (ensures offline operation even before Colab training is finished)
        return self._heuristic_sonar_detector(tile, confidence_threshold)

    def _heuristic_sonar_detector(self, tile: np.ndarray, conf_thresh: float) -> List[Dict[str, Any]]:
        """
        Deterministic acoustic anomaly detector detecting high-backscatter highlight blobs.
        """
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile.copy()

        h, w = gray.shape
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        # Highlight threshold: mean + 2.2 * std
        thresh_val = min(250, int(mean_val + 2.2 * std_val))
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter tiny speckles and huge full-tile swaths
            if 150 < area < (w * h * 0.25):
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / max(bh, 1)

                # Heuristic confidence based on intensity highlight
                roi = gray[y:y+bh, x:x+bw]
                roi_mean = np.mean(roi)
                conf = min(0.95, max(0.35, float((roi_mean - mean_val) / 100.0)))

                if conf < conf_thresh:
                    continue

                # Heuristic class assignment based on aspect ratio
                if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                    cls_id = 0  # crab_pot / elongate trap
                elif 0.8 <= aspect_ratio <= 1.2 and area < 800:
                    cls_id = 2  # mine_cylinder
                elif area > 2000:
                    cls_id = 1  # ghost_gear
                else:
                    cls_id = 3  # debris_anomaly

                poly = cnt.reshape(-1, 2).flatten().tolist()

                dets.append({
                    "class_id": cls_id,
                    "class_name": CLASS_LABELS[cls_id],
                    "confidence_ai": round(conf, 4),
                    "x_min": x,
                    "y_min": y,
                    "x_max": x + bw,
                    "y_max": y + bh,
                    "polygon": poly
                })

        return dets
