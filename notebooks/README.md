# AquaSentinel AI - Training Notebooks

This folder contains Google Colab-ready Jupyter Notebooks for training and evaluating the **AquaSentinel AI** Side-Scan Sonar segmentation models.

## Available Notebooks

### 1. `AquaSentinel_Training.ipynb` (Primary Multi-Phase Pipeline)
* **Direct Colab Link:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RaghavKacker/Aqua-Sentinel/blob/main/notebooks/AquaSentinel_Training.ipynb)
* **Purpose:** The official 11-step production training notebook matching the AquaSentinel specification:
  1. Mount Google Drive
  2. Install dependencies
  3. Download GhostVision
  4. Download SSS-Mine
  5. Inspect both datasets (Directory trees, label formats, JSON/TXT samples)
  6. Convert annotations (Target class consolidation to YOLO-Seg polygon format)
  7. Create train/val/test splits (Survey/mission-isolated splitting)
  8. Train YOLO-Seg (Transfer learning from `yolo11n-seg.pt`)
  9. Validate
  10. Test & Visualize (Polygon overlays, acoustic shadow evidence)
  11. Compare Results & Export (`best.pt` deployment)

### 2. `aquasentinel_training_colab.ipynb` (Synthetic / Fast Benchmark)
* **Direct Colab Link:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RaghavKacker/Aqua-Sentinel/blob/main/notebooks/aquasentinel_training_colab.ipynb)
* **Purpose:** Quick-validation benchmark notebook with built-in synthetic SSS generator for pipeline verification.

## How to Run in Google Colab

1. Open the [AquaSentinel_Training Colab Notebook](https://colab.research.google.com/github/RaghavKacker/Aqua-Sentinel/blob/main/notebooks/AquaSentinel_Training.ipynb).
2. Set the hardware accelerator to **T4 GPU** (**Runtime** → **Change runtime type** → **T4 GPU**).
3. Run cells sequentially.
4. When training completes, download `best.pt` and place it inside your local project:
   ```text
   d:\Projects\AquaSentinel\models\best.pt
   ```

For detailed guidance, refer to [COLAB_TRAINING_WORKFLOW.md](file:///d:/Projects/AquaSentinel/COLAB_TRAINING_WORKFLOW.md).
