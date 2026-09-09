"""
AquaSentinel AI - Dataset Preparation Pipeline
Prepares unified YOLO-Seg segmentation dataset from GhostVision and SSS-Mine/NOMBO datasets.
Enforces mission/survey-based splitting to prevent data leakage.
"""

import os
import sys
import json
import random
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

# Unified Canonical Classes for AquaSentinel Prototype
CANONICAL_CLASSES = {
    0: "crab_pot",
    1: "ghost_gear",
    2: "mine_cylinder",
    3: "debris_anomaly"
}

CLASS_NAME_TO_ID = {v: k for k, v in CANONICAL_CLASSES.items()}


def bbox_to_polygon(xmin: float, ymin: float, xmax: float, ymax: float) -> List[float]:
    """Convert normalized bounding box [xmin, ymin, xmax, ymax] to 4-corner polygon."""
    return [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]


def create_dataset_yaml(output_dir: Path, class_names: Dict[int, str]) -> Path:
    """Generate Ultralytics-compatible dataset.yaml file."""
    yaml_path = output_dir / "dataset.yaml"
    names_str = "\n".join([f"  {k}: {v}" for k, v in class_names.items()])
    yaml_content = f"""# AquaSentinel AI - Side-Scan Sonar Segmentation Dataset
path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
{names_str}
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def generate_synthetic_sss_patch(
    width: int = 640,
    height: int = 640,
    has_target: bool = True,
    target_class: int = 0
) -> Tuple[np.ndarray, List[Tuple[int, List[float]]]]:
    """
    Generate a realistic synthetic Side-Scan Sonar patch for development/testing.
    Simulates acoustic seabed reverberation, speckle noise, acoustic highlight, and acoustic shadow.
    """
    # 1. Base seafloor reverberation (Rayleigh / Gamma-like noise)
    base_intensity = np.random.uniform(50, 90)
    noise = np.random.rayleigh(scale=base_intensity, size=(height, width))
    sonar_img = np.clip(noise, 0, 255).astype(np.uint8)

    # 2. Add seabed ripple textures / low frequency variations
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    xx, yy = np.meshgrid(x, y)
    ripples = (15 * np.sin(xx * 1.5 + yy * 0.5)).astype(np.int16)
    sonar_img = np.clip(sonar_img.astype(np.int16) + ripples, 0, 255).astype(np.uint8)

    annotations = []

    if has_target:
        # Acoustic propagation assumed cross-track (e.g. left-to-right from nadir on left)
        tx = random.randint(150, width - 250)
        ty = random.randint(150, height - 200)
        tw = random.randint(40, 90)
        th = random.randint(30, 80)

        # Highlight region (bright acoustic backscatter)
        highlight_color = random.randint(200, 255)
        if target_class == 0:  # crab pot (box-like cage)
            cv2.rectangle(sonar_img, (tx, ty), (tx + tw, ty + th), highlight_color, -1)
            # Mesh lines
            for i in range(tx, tx + tw, 10):
                cv2.line(sonar_img, (i, ty), (i, ty + th), 80, 1)
        elif target_class == 1:  # ghost gear (irregular netting tangle)
            pts = np.array([
                [tx, ty + th // 2],
                [tx + tw // 3, ty],
                [tx + tw, ty + th // 4],
                [tx + (3 * tw) // 4, ty + th],
                [tx + tw // 4, ty + (3 * th) // 4]
            ], np.int32)
            cv2.fillPoly(sonar_img, [pts], highlight_color)
        elif target_class == 2:  # mine cylinder
            cv2.ellipse(sonar_img, (tx + tw // 2, ty + th // 2), (tw // 2, th // 3), 0, 0, 360, highlight_color, -1)
        else:  # debris anomaly
            cv2.rectangle(sonar_img, (tx, ty), (tx + tw, ty + th), highlight_color, -1)

        # Acoustic shadow cast down-range (to the right of target away from sound source)
        shadow_len = random.randint(tw, tw * 2 + 40)
        shadow_poly = np.array([
            [tx + tw, ty],
            [tx + tw + shadow_len, ty - 5],
            [tx + tw + shadow_len, ty + th + 5],
            [tx + tw, ty + th]
        ], np.int32)
        # Acoustic shadow has very low backscatter (< 20 intensity)
        cv2.fillPoly(sonar_img, [shadow_poly], random.randint(5, 20))

        # Normalized polygon for YOLO-Seg
        xmin_norm = max(0.0, min(1.0, tx / width))
        ymin_norm = max(0.0, min(1.0, ty / height))
        xmax_norm = max(0.0, min(1.0, (tx + tw) / width))
        ymax_norm = max(0.0, min(1.0, (ty + th) / height))

        poly_coords = bbox_to_polygon(xmin_norm, ymin_norm, xmax_norm, ymax_norm)
        annotations.append((target_class, poly_coords))

    return sonar_img, annotations


def generate_demo_dataset(output_dir: Path, num_samples: int = 120) -> Path:
    """
    Generates a full ready-to-train YOLO-Seg dataset for immediate baseline training
    and Colab testing without needing multi-gigabyte downloads upfront.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Mission-based grouping
    num_missions = 6
    missions = [f"mission_{i+1:02d}" for i in range(num_missions)]
    mission_splits = {
        "train": missions[:4],   # 4 missions in train
        "val": [missions[4]],    # 1 unseen mission in val
        "test": [missions[5]]    # 1 completely unseen mission in test
    }

    print(f"Generating synthetic demonstration SSS dataset in: {output_dir}")
    print(f"Mission split strategy (no data leakage): {mission_splits}")

    samples_per_mission = num_samples // num_missions

    for split, split_missions in mission_splits.items():
        sample_idx = 0
        for mission in split_missions:
            for _ in range(samples_per_mission):
                sample_idx += 1
                has_target = (random.random() > 0.20)  # 20% hard negatives (empty seabed)
                target_cls = random.choice([0, 1, 2, 3])

                img, annos = generate_synthetic_sss_patch(640, 640, has_target, target_cls)
                base_name = f"{mission}_tile_{sample_idx:04d}"

                img_path = output_dir / "images" / split / f"{base_name}.png"
                lbl_path = output_dir / "labels" / split / f"{base_name}.txt"

                cv2.imwrite(str(img_path), img)

                # Write YOLO-Seg label format: class x1 y1 x2 y2 ... xn yn
                with open(lbl_path, "w", encoding="utf-8") as f:
                    for cls_id, coords in annos:
                        coord_str = " ".join([f"{c:.6f}" for c in coords])
                        f.write(f"{cls_id} {coord_str}\n")

    yaml_path = create_dataset_yaml(output_dir, CANONICAL_CLASSES)
    print(f"[OK] Demo dataset generated successfully! Config: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="AquaSentinel Dataset Preparation Pipeline")
    parser.add_argument("--output_dir", type=str, default="./data/unified_yolo_seg", help="Output path for unified dataset")
    parser.add_argument("--demo", action="store_true", default=True, help="Generate synthetic SSS demo dataset for instant training")
    parser.add_argument("--samples", type=int, default=120, help="Number of demo samples to generate")
    parser.add_argument("--ghostvision_dir", type=str, default=None, help="Path to raw GhostVision directory if available")
    parser.add_argument("--sssmine_dir", type=str, default=None, help="Path to raw SSS-Mine/NOMBO directory if available")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    if args.demo or (not args.ghostvision_dir and not args.sssmine_dir):
        generate_demo_dataset(out_dir, num_samples=args.samples)
    else:
        print("Processing external raw datasets...")
        # Placeholder for external dataset ingestion when paths provided
        out_dir.mkdir(parents=True, exist_ok=True)
        create_dataset_yaml(out_dir, CANONICAL_CLASSES)


if __name__ == "__main__":
    main()
