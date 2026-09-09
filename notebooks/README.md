# AquaSentinel AI - Training Notebooks

This folder contains Google Colab-ready Jupyter Notebooks for training and evaluating the **AquaSentinel AI** Side-Scan Sonar segmentation models.

## Available Notebooks

### 1. `AquaSentinel_YOLO_Seg_Colab_Training.ipynb`
* **Purpose:** End-to-end training and evaluation of the **Ultralytics YOLO-Seg** model on Google Colab GPU.
* **Key Steps:**
  1. GPU Acceleration verification (`nvidia-smi`)
  2. Dependencies installation (`ultralytics`, `opencv-python-headless`)
  3. Dataset preparation (GhostVision + SSS-Mine or built-in demo generator)
  4. Survey-isolated training (prevents acoustic ping leakage between splits)
  5. Validation on unseen survey missions
  6. Prediction visualization with segmentation masks
  7. One-click export and download of `best.pt` for local offline deployment

## How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload `AquaSentinel_YOLO_Seg_Colab_Training.ipynb`.
3. Set the hardware accelerator to **T4 GPU** (**Runtime** → **Change runtime type** → **T4 GPU**).
4. Run all cells.
5. When finished, move the downloaded `best.pt` to:
   ```text
   d:\Projects\AquaSentinel\models\best.pt
   ```

For detailed guidance, refer to [COLAB_TRAINING_WORKFLOW.md](file:///d:/Projects/AquaSentinel/COLAB_TRAINING_WORKFLOW.md).
