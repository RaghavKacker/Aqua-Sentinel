# AquaSentinel AI - Model Training Guide

Comprehensive guide for organizing your dataset and training the **AquaSentinel AI** Side-Scan Sonar (SSS) target detection model locally on your laptop.

---

## 1. How to Start Training Locally

The new smart training script automatically detects your **NVIDIA RTX 3050 GPU (4GB VRAM)**, enables **FP16 Mixed Precision (`amp=True`)**, sets optimal batch sizes, applies sonar-physics augmentations, and exports the top-performing weights to `models/best.pt`.

### Standard GPU Training Command (Recommended for RTX 3050)
```powershell
.\venv\Scripts\python ml/train.py --data ./data/unified_yolo_seg/dataset.yaml --model ./models/best.pt --epochs 50 --batch 16 --device 0
```

### Fast Fine-Tuning / Quick Test Run (10–15 Epochs)
```powershell
.\venv\Scripts\python ml/train.py --epochs 15 --batch 16 --device 0
```

### Train from Scratch (Using base YOLO11 weights)
```powershell
.\venv\Scripts\python ml/train.py --model yolo11n.pt --epochs 60 --batch 16 --device 0
```

### CPU Fallback (If training on battery or without GPU)
```powershell
.\venv\Scripts\python ml/train.py --epochs 20 --batch 8 --device cpu
```

---

## 2. Where to Store the Training Data

Store all images and annotation files inside [`data/unified_yolo_seg/`](file:///d:/Projects/COPYPASTE/Aqua/data/unified_yolo_seg):

```text
Aqua/
├── data/
│   └── unified_yolo_seg/
│       ├── dataset.yaml               <-- Dataset configuration & class definitions
│       │
│       ├── images/                    <-- Sonar imagery (.jpg / .png)
│       │   ├── train/                 <-- 70% of your sonar dataset (Training)
│       │   ├── val/                   <-- 20% of your sonar dataset (Validation)
│       │   └── test/                  <-- 10% of your sonar dataset (Unseen Benchmark)
│       │
│       └── labels/                    <-- YOLO format annotation files (.txt)
│           ├── train/                 <-- .txt files matching images/train/
│           ├── val/                   <-- .txt files matching images/val/
│           └── test/                  <-- .txt files matching images/test/
│
├── models/
│   └── best.pt                        <-- Auto-updated with your newly trained weights
│
└── runs/train/aquasentinel_training/  <-- Loss curves, confusion matrix & checkpoint weights
```

> [!IMPORTANT]
> Every image in `images/train/sonar_001.png` must have an identical filename `.txt` label in `labels/train/sonar_001.txt`.
> For background tiles with no objects, keep an **empty `.txt` file** to teach the model to ignore flat seabeds.

---

## 3. What the Training Data Must Have

### A. The 5 Canonical Hydrographic Classes
Your label `.txt` files must use these exact Class IDs:

| Class ID | Class Label | Role | How to Annotate |
|:---|:---|:---|:---|
| `0` | **`crab_pot`** | **Hard Negative / Distractor** | Annotate crab traps, cages, and minor seabed fixtures. *(Trained to suppress false seabed alarms).* |
| `1` | **`submarine_pipeline`** | **Production Class** | Annotate linear subsea pipelines, exposed pipes, and umbilical cables. |
| `2` | **`shipwreck`** | **Production Class** | Annotate shipwreck hulls, cargo debris, plane wreckage, and sunken structures. |
| `3` | **`ghost_net`** | **Production Class** | Annotate tangled abandoned fishing gear, loose nets, and synthetic rope balls. |
| `4` | **`mine_cylinder`** | **Production Class (CRITICAL)** | Annotate cylindrical objects, discarded drums, and suspected underwater ordnance. |

---

### B. Label Annotation Format (YOLO Format)
Each row in a label file `.txt` represents one target:

#### 1. Bounding Box Format (YOLO Detect):
```text
<class_id> <x_center> <y_center> <width> <height>
```
*All coordinates are normalized between `0.0` and `1.0` relative to image dimensions.*

**Example (`labels/train/survey_pipe_001.txt`):**
```text
1 0.485200 0.512400 0.820000 0.145000
4 0.720100 0.315000 0.085000 0.075000
```

#### 2. Polygon Segmentation Format (Optional YOLO Seg):
```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ... <xn> <yn>
```

---

### C. Golden Rules for Sonar Training Data Quality

1. **Resolution & Tiling**:
   - Best resolution is **640×640 pixels**.
   - If you have long waterfall strips, tile them into 640×640 patches with ~20% overlap.

2. **Include Acoustic Highlights & Shadows**:
   - Box the **bright acoustic backscatter highlight** together with its **immediate dark acoustic shadow** behind it.
   - Genuine 3D underwater objects always cast acoustic shadows.

3. **Include Negative Background Tiles (~10% to 15%)**:
   - Include pure seafloor patches (sand ripples, muddy flats, natural rocks) with **empty `.txt` label files**.
   - This prevents the model from hallucinating debris on plain seabeds.

4. **Prevent Mission Data Leakage**:
   - Split your data by **Survey Mission / Corridor**, not random shuffling.
   - Example: Mission 1–4 in `train/`, Mission 5 in `val/`, Mission 6 in `test/`.

---

## 4. After Training Finishes

1. The script automatically exports the best weights to:
   ```text
   models/best.pt
   ```
2. Your local FastAPI backend ([http://127.0.0.1:8000](http://127.0.0.1:8000)) and React Dashboard ([http://localhost:5173](http://localhost:5173)) will immediately use the newly trained model on the next survey run.
3. Review training performance graphs:
   ```text
   runs/train/aquasentinel_training/
   ├── results.png              <-- Loss curves and mAP progression over epochs
   ├── confusion_matrix.png     <-- True positive vs False alarm matrix
   ├── val_batch0_pred.jpg      <-- Visual predictions on validation set
   └── PR_curve.png             <-- Precision-Recall curve
   ```
