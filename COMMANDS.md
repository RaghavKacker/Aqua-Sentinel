# AquaSentinel AI - Command Reference & Testing Guide

This document contains all essential commands for running, testing, benchmarking, and evaluating the **AquaSentinel AI** Side-Scan Sonar (SSS) target detection system.

---

## 1. Starting the Application

### Launch FastAPI Backend
```powershell
.\venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
* **API URL**: http://127.0.0.1:8000
* **Interactive OpenAPI Swagger Docs**: http://127.0.0.1:8000/docs

### Launch React Frontend Dashboard
```powershell
cd frontend
npm run dev
```
* **Operations Dashboard**: http://localhost:5173

---

## 2. Detectable Classes Architecture

The model is trained on **5 canonical hydrographic classes** with **4 active production classes** and 1 distractor class:

| Class ID | Class Label | Description & Source | Production Role |
| :--- | :--- | :--- | :--- |
| `0` | **`crab_pot`** | Seabed traps / crab pots | **Hard Negative / Distractor** *(Trained to reject seabed false alarms; filtered out in UI)* |
| `1` | **`submarine_pipeline`** | Subsea pipelines, linear cables & structural conduits | **Active Production Class** *(High Priority)* |
| `2` | **`shipwreck`** | Sunken hulls, maritime vessels & wreckage anomalies | **Active Production Class** *(Navigational Hazard)* |
| `3` | **`ghost_net`** | Abandoned/lost fishing nets & synthetic drag debris | **Active Production Class** *(Ecological Hazard)* |
| `4` | **`mine_cylinder`** | Cylindrical objects, suspected underwater munitions | **Active Production Class** *(CRITICAL Safety Hazard)* |

---

## 3. Running System & Physics Tests (PyTest)

Executes all 10 unit and integration tests (validates Input Normalizer, Acoustic Shadow Verification physics, WGS84 great-circle geolocation projection math, and end-to-end survey processing):

```powershell
.\venv\Scripts\pytest tests/ -v
```

---

## 4. Model Evaluation & Benchmarking

### Benchmark on Test Split (`mAP50`, `mAP50-95`, Latency)
Evaluates `models/best.pt` across all classes on the test dataset split:

```powershell
.\venv\Scripts\python ml/evaluate.py --model ./models/best.pt --data ./data/unified_yolo_seg/dataset.yaml
```

### Visual Inference on Unlabelled Test Images
Runs detection/segmentation inference across test sonar patches and outputs images with overlaid boxes & confidence labels:

```powershell
.\venv\Scripts\yolo detect predict model=./models/best.pt source=data/unified_yolo_seg/images/test save=True project=outputs name=test_predictions
```
*(For segmentation models, replace `detect` with `segment`)*.

---

## 5. Dataset Preparation & Local Model Training

### Generate Synthetic Sonar Dataset (Acoustic Backscatter & Shadow Simulation)
```powershell
.\venv\Scripts\python ml/prepare_dataset.py --output ./data/unified_yolo_seg --synthetic --samples 100
```

### Train / Fine-Tune YOLO Model Locally
```powershell
.\venv\Scripts\python ml/train.py --data ./data/unified_yolo_seg/dataset.yaml --epochs 30 --batch 16 --imgsz 640
```

---

## 6. Cleaning Temporary Output Artifacts
To wipe all generated survey outputs, database records, and inference run files:
```powershell
if (Test-Path outputs) { Remove-Item -Recurse -Force outputs }
if (Test-Path runs) { Remove-Item -Recurse -Force runs }
```
