"""
AquaSentinel AI - Google Colab / Local YOLO-Seg Training Script
Trains an Ultralytics YOLO-Seg model for underwater Side-Scan Sonar segmentation.
"""

import os
import sys
import argparse
from pathlib import Path

def train_model(
    data_yaml: str = "./data/unified_yolo_seg/dataset.yaml",
    model_variant: str = "yolo11n-seg.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "",
    output_dir: str = "./runs/train",
    export_dest: str = "./models/best.pt"
):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print("AquaSentinel AI - Side-Scan Sonar YOLO-Seg Training")
    print(f"Data Config:    {data_yaml}")
    print(f"Model Baseline: {model_variant}")
    print(f"Epochs:         {epochs}")
    print(f"Image Size:     {imgsz}x{imgsz}")
    print(f"Batch Size:     {batch_size}")
    print("=" * 60)

    # Initialize YOLO-Seg model
    model = YOLO(model_variant)

    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device if device else None,
        project=output_dir,
        name="aquasentinel_seg",
        exist_ok=True,
        plots=True,
        save=True,
        mosaic=0.5,       # Moderate mosaic for sonar structures
        degrees=10.0,     # Small rotation for sonar orientation
        fliplr=0.5        # Horizontal flip (port/starboard symmetry)
    )

    # Export best model
    trained_best = Path(output_dir) / "aquasentinel_seg" / "weights" / "best.pt"
    if trained_best.exists():
        dest_path = Path(export_dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(trained_best, dest_path)
        print(f"\n[SUCCESS] Trained model exported to: {dest_path.resolve()}")
    else:
        print(f"\n[INFO] Training complete. Artifacts saved in: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="AquaSentinel AI YOLO-Seg Training")
    parser.add_argument("--data", type=str, default="./data/unified_yolo_seg/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt", help="YOLO-Seg baseline model (yolo11n-seg.pt or yolov8n-seg.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="", help="Device: '0' for CUDA GPU, 'cpu' for CPU fallback")
    parser.add_argument("--export", type=str, default="./models/best.pt", help="Destination path for best.pt")

    args = parser.parse_args()
    train_model(
        data_yaml=args.data,
        model_variant=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        export_dest=args.export
    )


if __name__ == "__main__":
    main()
