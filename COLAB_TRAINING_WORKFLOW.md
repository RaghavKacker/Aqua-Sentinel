# AquaSentinel AI: Google Colab Training & Local Integration Workflow

This comprehensive guide explains step-by-step how to train the **YOLO-Seg** segmentation model in **Google Colab GPU** and integrate the trained weights (`best.pt`) with your local offline **AquaSentinel AI** system.

---

## 1. Architectural Overview & Workflow

```text
[Step 1: Dataset Prep]
GhostVision (Crab Pots/Gear) + SSS-Mine (AUV Targets/Clutter)
                        ↓
[Step 2: Google Colab GPU]
Upload / Mount to Colab → Train YOLO-Seg (T4/V100 GPU) → Validation on Unseen Mission
                        ↓
[Step 3: Export Model]
Download `best.pt` from Colab
                        ↓
[Step 4: Local Offline Deployment]
Place in `AquaSentinel/models/best.pt`
                        ↓
[Step 5: Run Application]
FastAPI Backend (Port 8000) + React Dashboard (Port 5173)
```

---

## 2. Why Use Google Colab for Training?

* **Hardware Efficiency:** Colab provides free access to high-performance NVIDIA GPUs (T4 with 15 GB VRAM or A100/V100 in Colab Pro).
* **Local Resource Protection:** Keeps your local workstation/laptop (e.g. RTX 3050 4GB VRAM) free for real-time offline inference, preprocessing, database storage, and UI visualization.
* **Faster Convergence:** Batch size 16 or 32 at 640×640 resolution trains in under 30–45 minutes on a Colab T4 GPU.

---

## 3. Step-by-Step Training Guide

### Step 3.1: Open the Notebook in Google Colab
1. In the repository, locate the training notebook:
   `notebooks/AquaSentinel_YOLO_Seg_Colab_Training.ipynb`
2. Go to [Google Colab](https://colab.research.google.com/).
3. Click **Upload** and select `AquaSentinel_YOLO_Seg_Colab_Training.ipynb`.
4. Set the runtime to GPU:
   * Navigate to **Runtime** → **Change runtime type**.
   * Under **Hardware accelerator**, select **T4 GPU** (or any available GPU).
   * Click **Save**.

---

### Step 3.2: Verify GPU Acceleration in Colab
Run the first cell in the notebook:
```python
!nvidia-smi
```
Verify that an NVIDIA GPU (such as Tesla T4 with ~15 GB memory) is detected and CUDA is available.

---

### Step 3.3: Install Ultralytics and Dependencies
```python
!pip install -q ultralytics opencv-python-headless pyyaml matplotlib
```

Verify that Ultralytics YOLO is accessible:
```python
import torch
import ultralytics
print(f"PyTorch Version: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
ultralytics.checks()
```

---

### Step 3.4: Dataset Acquisition & Preparation

You have two options in Colab:

#### Option A: Training on Real Datasets (GhostVision + SSS-Mine / NOMBO)
1. Mount your Google Drive where you have stored your datasets:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Copy or symlink your dataset folders to the Colab runtime environment:
   ```bash
   cp -r /content/drive/MyDrive/AquaSentinel_Data/GhostVision ./data/
   cp -r /content/drive/MyDrive/AquaSentinel_Data/SSS-Mine ./data/
   ```
3. Run the dataset preparation script:
   ```bash
   python ml/prepare_dataset.py --output_dir ./data/unified_yolo_seg
   ```
   > **Note on Data Splitting:** The script groups tiles by **survey / mission**. Training on missions 1–4, validating on mission 5, and testing on mission 6 prevents data leakage between overlapping acoustic waterfall pings.

#### Option B: Fast Demo Dataset Generation (No Downloads Required)
If you want to quickly train a working baseline model without downloading multi-gigabyte datasets:
```bash
python ml/prepare_dataset.py --demo --samples 300 --output_dir ./data/unified_yolo_seg
```
This generates 300 realistic synthetic Side-Scan Sonar patches with seabed ripple noise, acoustic highlights, and down-range acoustic shadows across the canonical classes:
* `0: crab_pot`
* `1: ghost_gear`
* `2: mine_cylinder`
* `3: debris_anomaly`

---

### Step 3.5: Run YOLO-Seg Model Training
Execute the training command with Ultralytics YOLO11-Seg (or YOLOv8-Seg):

```python
from ultralytics import YOLO

# 1. Load pretrained baseline segmentation weights
model = YOLO('yolo11n-seg.pt')

# 2. Train on unified sonar dataset
results = model.train(
    data='./data/unified_yolo_seg/dataset.yaml',
    epochs=40,
    imgsz=640,
    batch=16,
    device=0,               # Uses Colab GPU
    project='./runs/train',
    name='aquasentinel_seg',
    exist_ok=True,
    plots=True,
    save=True,
    mosaic=0.5,             # Sonar structure preservation
    fliplr=0.5              # Port/starboard swath symmetry
)
```

---

### Step 3.6: Evaluate on Unseen Test Missions
Benchmark model metrics on the isolated test split:
```python
metrics = model.val(
    data='./data/unified_yolo_seg/dataset.yaml',
    split='test',
    imgsz=640,
    plots=True
)

print(f"Box mAP50:    {metrics.box.map50:.4f}")
print(f"Box mAP50-95: {metrics.box.map:.4f}")
print(f"Mask mAP50:   {metrics.seg.map50:.4f}")
print(f"Mask mAP50-95:{metrics.seg.map:.4f}")
```

---

### Step 3.7: Export and Download `best.pt`
Once training finishes, download the optimal weights:
```python
from google.colab import files

# Download the best weights to your local computer
files.download('./runs/train/aquasentinel_seg/weights/best.pt')
```
Alternatively, save a backup directly to Google Drive:
```python
!cp ./runs/train/aquasentinel_seg/weights/best.pt /content/drive/MyDrive/AquaSentinel_Models/best.pt
```

---

## 4. Integrating Trained Weights into Local AquaSentinel

Once `best.pt` is downloaded to your computer:

1. **Move `best.pt` into the project's `models/` directory:**
   ```text
   d:\Projects\AquaSentinel\models\best.pt
   ```
2. **Verify that `.env` points to this weight file:**
   ```bash
   MODEL_WEIGHTS_PATH=./models/best.pt
   ```
3. **Start the local offline backend:**
   The backend automatically detects and loads `models/best.pt` during startup.
   ```powershell
   .\venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000
   ```
   You will see:
   ```text
   [AI] Loaded fine-tuned YOLO-Seg model: .\models\best.pt
   ```

---

## 5. End-to-End System Execution

With `models/best.pt` in place:
1. **Launch Backend:**
   ```powershell
   .\venv\Scripts\python -m uvicorn backend.main:app --port 8000
   ```
2. **Launch Frontend:**
   ```powershell
   cd frontend
   npm run dev
   ```
3. **Open Browser:** Navigate to `http://localhost:5173`
4. **Inspect Results:**
   * Upload an SSS image (e.g. from `SubPipeMiniSSS` or generated sample).
   * The pipeline will automatically run:
     $$\text{Image} \rightarrow \text{Normalizer} \rightarrow \text{Preprocessing} \rightarrow \text{YOLO-Seg} \rightarrow \text{Shadow Gate} \rightarrow \text{Geolocation} \rightarrow \text{Dashboard}$$
   * View segmentation masks, verify acoustic shadows, inspect target dimensions, and export GeoJSON/CSV reports!

---

## 6. Training Tips & Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **CUDA Out of Memory in Colab** | Batch size too large for GPU | Decrease `batch=8` or `imgsz=512`. |
| **Low Recall on Small Debris** | Tiles too downsampled | Ensure training resolution is at least `imgsz=640`. |
| **High False Positives on Rocks** | Dataset lacks negative background tiles | Ensure at least 15–20% of training tiles are empty seabed (hard negatives). |
| **Model Fails to Load Locally** | PyTorch version mismatch or missing weights | Ensure `models/best.pt` is in place. If missing, system falls back to base model or deterministic acoustic detector. |
