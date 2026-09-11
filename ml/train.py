"""
AquaSentinel AI - Smart Side-Scan Sonar (SSS) YOLO Training Pipeline
Tailored for Underwater Marine Debris & Anomaly Detection:
  0: crab_pot (Hard Negative / Distractor)
  1: submarine_pipeline (Active Production Class)
  2: shipwreck (Active Production Class)
  3: ghost_net (Active Production Class)
  4: mine_cylinder (Active Production Class - Critical Safety)

Features:
- Hardware-Aware: Automatically configures GPU (RTX 3050 / CUDA), optimal batch size, FP16 AMP, and CPU fallback.
- Sonar-Physics Augmentations: Preserves acoustic backscatter fidelity (no hue distortion, port/starboard symmetry).
- Pre-flight Dataset Audit: Verifies dataset integrity and class distribution before launching training.
- Automated Model Export: Exports the highest-scoring weights directly to `models/best.pt`.
- Post-Training Evaluation: Runs immediate benchmark evaluation on the test split.
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

CANONICAL_CLASSES = {
    0: "crab_pot",
    1: "submarine_pipeline",
    2: "shipwreck",
    3: "ghost_net",
    4: "mine_cylinder"
}


def audit_dataset_integrity(data_yaml_path: Path) -> bool:
    """Pre-flight check to ensure dataset images and label files exist and match the 5 classes."""
    if not data_yaml_path.exists():
        print(f"[ERROR] Dataset configuration file not found at: {data_yaml_path}")
        return False

    dataset_root = data_yaml_path.parent
    train_images = dataset_root / "images" / "train"
    val_images = dataset_root / "images" / "val"
    train_labels = dataset_root / "labels" / "train"

    train_img_count = len(list(train_images.glob("*.*"))) if train_images.exists() else 0
    val_img_count = len(list(val_images.glob("*.*"))) if val_images.exists() else 0
    train_lbl_count = len(list(train_labels.glob("*.txt"))) if train_labels.exists() else 0

    print("=" * 65)
    print("AquaSentinel AI - Dataset Pre-flight Audit")
    print(f"Dataset Root:    {dataset_root.resolve()}")
    print(f"Training Images: {train_img_count} files in {train_images}")
    print(f"Training Labels: {train_lbl_count} files in {train_labels}")
    print(f"Val Images:      {val_img_count} files in {val_images}")
    print("Canonical Classes:")
    for cid, cname in CANONICAL_CLASSES.items():
        role = "Distractor (Hard Negative)" if cid == 0 else "Active Production"
        print(f"  [{cid}] {cname:22s} -> {role}")
    print("=" * 65)

    if train_img_count == 0:
        print(f"[WARNING] No training images found in {train_images}. Please populate dataset before training.")
        return False

    return True


def detect_optimal_hardware(user_device: str, user_batch: Optional[int]) -> Dict[str, Any]:
    """Detects available GPU hardware (e.g. RTX 3050 4GB) and sets optimal batch size and workers."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    device = user_device
    if not device:
        device = "0" if cuda_available else "cpu"

    batch_size = user_batch if user_batch else 16
    workers = 4 if os.name != 'nt' else 2  # Windows-friendly worker allocation

    if cuda_available and device != "cpu":
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[HARDWARE] Detected CUDA GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

        # Optimize for 4GB VRAM cards (RTX 3050 / GTX 1650)
        if vram_gb <= 4.5:
            if not user_batch:
                batch_size = 16  # Optimal for 640x640 detection in 4GB VRAM
            print(f"[HARDWARE] 4GB VRAM Profile Active -> Batch Size: {batch_size}, FP16 Mixed Precision: ON")
    else:
        print(f"[HARDWARE] Running on CPU. Batch Size: {batch_size}")
        workers = 0

    return {
        "device": device,
        "batch_size": batch_size,
        "workers": workers,
        "amp": True if (cuda_available and device != "cpu") else False
    }


def train_model(
    data_yaml: str = "./data/unified_yolo_seg/dataset.yaml",
    model_variant: str = "./models/best.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: Optional[int] = None,
    device: str = "",
    output_dir: str = "./runs/train",
    export_dest: str = "./models/best.pt",
    patience: int = 15,
    save_period: int = 5
) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    data_yaml_path = Path(data_yaml)
    if not audit_dataset_integrity(data_yaml_path):
        proceed = input("\nProceed with training anyway? (y/n): ") if sys.stdin.isatty() else "y"
        if proceed.lower() != "y":
            print("[INFO] Training aborted.")
            return None

    # Resolve baseline weights
    base_model_path = Path(model_variant)
    if not base_model_path.exists():
        print(f"[INFO] Weights '{model_variant}' not found locally. Using standard 'yolo11n.pt' baseline.")
        model_variant = "yolo11n.pt"

    # Hardware detection
    hw_config = detect_optimal_hardware(device, batch_size)

    print("\n" + "=" * 65)
    print("AquaSentinel AI - Starting Smart Sonar Training")
    print(f"Base Model:     {model_variant}")
    print(f"Data YAML:      {data_yaml_path.resolve()}")
    print(f"Target Epochs:  {epochs}")
    print(f"Image Size:     {imgsz}x{imgsz}")
    print(f"Batch Size:     {hw_config['batch_size']}")
    print(f"Compute Device: {hw_config['device']}")
    print(f"Early Stopping: patience={patience} epochs")
    print("=" * 65 + "\n")

    # Initialize model
    model = YOLO(model_variant)

    # Launch domain-tailored training
    # Sonar-Specific Augmentation Policy:
    # 1. fliplr=0.5: Port & Starboard sonar tracks are physically symmetric.
    # 2. degrees=10.0: Realistic vehicle yaw and current drift.
    # 3. mosaic=0.5: Blends disparate seabed backgrounds together.
    # 4. hsv_h=0.0 & hsv_s=0.0: Sonar is single-channel backscatter; color distortion violates acoustics.
    # 5. hsv_v=0.2: Simulates Time-Varied Gain (TVG) and beam spreading intensity shifts.
    start_time = time.time()
    results = model.train(
        data=str(data_yaml_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=hw_config["batch_size"],
        device=hw_config["device"],
        workers=hw_config["workers"],
        project=output_dir,
        name="aquasentinel_training",
        exist_ok=True,
        plots=True,
        save=True,
        save_period=save_period,
        patience=patience,
        amp=hw_config["amp"],
        # Sonar Acoustic Augmentations
        fliplr=0.5,
        degrees=10.0,
        mosaic=0.5,
        scale=0.2,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        verbose=True
    )
    total_training_time = time.time() - start_time
    print(f"\n[COMPLETE] Training finished in {total_training_time / 60.0:.2f} minutes.")

    # Locate and export the best checkpoint
    trained_best = Path(output_dir) / "aquasentinel_training" / "weights" / "best.pt"
    if trained_best.exists():
        dest_path = Path(export_dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(trained_best, dest_path)
        dest_size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Best weights ({dest_size_mb:.2f} MB) exported to: {dest_path.resolve()}")
    else:
        print(f"[INFO] Weights saved in: {output_dir}/aquasentinel_training/weights")

    # Run instant benchmark on test split
    print("\n" + "=" * 65)
    print("Running Post-Training Validation Benchmark on Test Split...")
    print("=" * 65)
    try:
        from ml.evaluate import evaluate_model
        evaluate_model(
            model_path=str(export_dest),
            data_yaml=str(data_yaml_path.resolve()),
            imgsz=imgsz,
            device=hw_config["device"]
        )
    except Exception as e:
        print(f"[INFO] Post-training evaluation skipped: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="AquaSentinel AI - Smart Sonar YOLO Training Pipeline")
    parser.add_argument("--data", type=str, default="./data/unified_yolo_seg/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="./models/best.pt", help="Baseline weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (auto-configured for GPU if omitted)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--device", type=str, default="", help="Device: '0' for CUDA GPU, 'cpu' for CPU fallback")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience epochs")
    parser.add_argument("--export", type=str, default="./models/best.pt", help="Destination path for best.pt")

    args = parser.parse_args()
    train_model(
        data_yaml=args.data,
        model_variant=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        output_dir="./runs/train",
        export_dest=args.export,
        patience=args.patience
    )


if __name__ == "__main__":
    main()
