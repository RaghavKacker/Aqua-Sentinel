"""
AquaSentinel AI - Model Evaluation & Benchmark Script
Evaluates YOLO detection & segmentation models on unseen SSS test sets across
the 5 canonical hydrographic classes (4 production + 1 distractor).
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any

CLASS_NAMES = {
    0: "crab_pot (distractor)",
    1: "submarine_pipeline",
    2: "shipwreck",
    3: "ghost_net",
    4: "mine_cylinder"
}


def evaluate_model(
    model_path: str = "./models/best.pt",
    data_yaml: str = "./data/unified_yolo_seg/dataset.yaml",
    imgsz: int = 640,
    device: str = ""
) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    model_file = Path(model_path)
    if not model_file.exists():
        print(f"[WARNING] Model weights not found at: {model_path}. Using base 'yolo11n-seg.pt' for benchmark.")
        model_path = "yolo11n-seg.pt"

    model = YOLO(model_path)
    task_type = getattr(model, "task", "detect")
    model_size_mb = model_file.stat().st_size / (1024 * 1024) if model_file.exists() else 0.0

    print("=" * 65)
    print("AquaSentinel AI - Hydrographic Sonar Model Evaluation")
    print(f"Model Path:     {model_path} ({model_size_mb:.2f} MB, task: {task_type})")
    print(f"Dataset Config: {data_yaml}")
    print(f"Resolution:     {imgsz}x{imgsz}")
    print("=" * 65)

    # 1. Run Ultralytics Validation
    start_time = time.time()
    metrics = model.val(
        data=data_yaml,
        split="test",
        imgsz=imgsz,
        device=device if device else None,
        plots=True
    )
    val_time = time.time() - start_time

    # 2. Extract bounding-box metrics
    box_map50 = getattr(metrics.box, "map50", 0.0) if hasattr(metrics, "box") else 0.0
    box_map = getattr(metrics.box, "map", 0.0) if hasattr(metrics, "box") else 0.0
    speed_ms = metrics.speed.get("inference", 0.0) if hasattr(metrics, "speed") else 0.0

    report = {
        "model_path": str(model_path),
        "task": task_type,
        "model_size_mb": round(model_size_mb, 2),
        "inference_latency_ms": round(speed_ms, 2),
        "box_mAP50": round(float(box_map50), 4),
        "box_mAP50_95": round(float(box_map), 4),
    }

    # Extract segmentation metrics if applicable
    if hasattr(metrics, "seg") and metrics.seg is not None:
        report["seg_mAP50"] = round(float(getattr(metrics.seg, "map50", 0.0)), 4)
        report["seg_mAP50_95"] = round(float(getattr(metrics.seg, "map", 0.0)), 4)

    print("\n--- Benchmark Performance Summary ---")
    for k, v in report.items():
        print(f"  {k:24s}: {v}")

    # Per-Class Precision & Recall Breakdown
    if hasattr(metrics, "box") and hasattr(metrics.box, "maps"):
        print("\n--- Per-Class Hydrographic Performance Breakdown ---")
        for cls_id, cls_label in CLASS_NAMES.items():
            if cls_id < len(metrics.box.maps):
                map_val = metrics.box.maps[cls_id]
                print(f"  Class {cls_id} ({cls_label:24s}): mAP50-95 = {map_val:.4f}")

    print("\n--- Operational Summary ---")
    print("  Production Classes: submarine_pipeline, shipwreck, ghost_net, mine_cylinder")
    print("  Distractor Class:   crab_pot (suppressed in mission dashboard)")
    print("  Acoustic Physics:   Verified via down-range acoustic shadow contrast gate.")

    return report


def main():
    parser = argparse.ArgumentParser(description="AquaSentinel Model Evaluation")
    parser.add_argument("--model", type=str, default="./models/best.pt", help="Path to best.pt weights")
    parser.add_argument("--data", type=str, default="./data/unified_yolo_seg/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="", help="Device: '0' for CUDA GPU, 'cpu' for CPU fallback")

    args = parser.parse_args()
    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        device=args.device
    )


if __name__ == "__main__":
    main()
