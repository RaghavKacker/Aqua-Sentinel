"""
AquaSentinel AI - FastAPI Application Entry Point
Orchestrates the offline-first Side-Scan Sonar target detection and verification pipeline.
"""

import os
import sys
import uuid
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from pydantic import BaseModel
import cv2
import numpy as np

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from input_normalizer import normalize_input_sources, NormalizedSurveyInput
from database import Database
from reports import export_geojson, export_csv
from pipeline.quality import assess_image_quality
from pipeline.slant_range import apply_slant_range_correction
from pipeline.preprocess import preprocess_sonar_image
from pipeline.tiling import generate_tiles, map_detections_to_global, apply_cross_tile_nms
from pipeline.inference import AdaptiveInferenceEngine
from pipeline.shadow_gate import verify_acoustic_shadow
from pipeline.geolocation import project_detection_geolocation, compute_target_dimensions, compute_risk_score

app = FastAPI(
    title="AquaSentinel AI API",
    description="Offline-first Side-Scan Sonar Marine Debris Detection & Acoustic Verification Engine",
    version="1.0.0"
)

# Enable CORS for local React/Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent output storage setup
OUTPUT_DIR = Path(os.getenv("OUTPUT_STORAGE_PATH", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SURVEYS_DIR = OUTPUT_DIR / "surveys"
SURVEYS_DIR.mkdir(parents=True, exist_ok=True)

# Mount outputs for serving crops and preprocessed images
app.mount("/static/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Initialize database & inference engine
db = Database(str(OUTPUT_DIR / "aquasentinel.db"))
inference_engine = AdaptiveInferenceEngine(
    model_path=os.getenv("MODEL_WEIGHTS_PATH", "./models/best.pt"),
    compute_profile=os.getenv("COMPUTE_PROFILE", "NORMAL")
)


@app.get("/api/v1/system/status")
def get_system_status():
    """Returns local compute profile, hardware acceleration telemetry, and offline status."""
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None"
    except ImportError:
        cuda_avail = False
        gpu_name = "None"

    return {
        "status": "ONLINE",
        "offline_ready": True,
        "compute_profile": inference_engine.compute_profile,
        "device": inference_engine.device,
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "yolo_model_loaded": inference_engine.model_loaded
    }


@app.get("/api/v1/system/stats")
def get_dashboard_stats():
    """Returns summary KPIs for the operations dashboard."""
    return db.get_stats()


@app.get("/api/v1/surveys")
def list_surveys():
    """Returns all ingested and processed surveys."""
    return db.get_surveys()


@app.get("/api/v1/surveys/{survey_id}")
def get_survey_details(survey_id: str):
    """Returns manifest and processing summary for a survey."""
    survey = db.get_survey(survey_id)
    if not survey:
        raise HTTPException(status_code=404, detail="Survey not found")
    detections = db.get_detections_for_survey(survey_id)
    return {
        "survey": survey,
        "detections_count": len(detections),
        "detections": detections
    }


@app.get("/api/v1/surveys/{survey_id}/detections")
def get_survey_detections(survey_id: str):
    """Returns all detected targets with segmentation masks and shadow evidence."""
    return db.get_detections_for_survey(survey_id)


@app.get("/api/v1/surveys/{survey_id}/export/{export_format}")
def export_survey_deliverables(survey_id: str, export_format: str):
    """Exports target records in RFC 7946 GeoJSON or CSV format."""
    survey = db.get_survey(survey_id)
    if not survey:
        raise HTTPException(status_code=404, detail="Survey not found")
    detections = db.get_detections_for_survey(survey_id)

    if export_format.lower() == "geojson":
        geojson_data = export_geojson(detections, survey["name"])
        return JSONResponse(content=geojson_data)
    elif export_format.lower() == "csv":
        csv_text = export_csv(detections)
        return PlainTextResponse(
            content=csv_text,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={survey['name']}_targets.csv"}
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid format: supported formats are 'geojson' or 'csv'")


@app.post("/api/v1/surveys/process")
async def process_survey(
    sonar_file: UploadFile = File(...),
    nav_file: Optional[UploadFile] = File(None),
    apply_slant_range: bool = Form(True),
    apply_clahe: bool = Form(True),
    confidence_threshold: float = Form(0.25)
):
    """
    Primary processing pipeline:
    Input Normalizer -> Preprocessing -> Slant-Range -> Tiling -> YOLO-Seg ->
    Acoustic Shadow Gate -> Geolocation -> Risk Scoring -> SQLite Database.
    """
    survey_id = f"srv_{uuid.uuid4().hex[:10]}"
    survey_folder = SURVEYS_DIR / survey_id
    survey_folder.mkdir(parents=True, exist_ok=True)
    crops_folder = survey_folder / "crops"
    crops_folder.mkdir(parents=True, exist_ok=True)

    sonar_bytes = await sonar_file.read()
    nav_bytes = await nav_file.read() if nav_file else None

    # 1. Input Normalizer (Graceful Degradation)
    try:
        normalized_input = normalize_input_sources(
            image_bytes_or_path=sonar_bytes,
            filename=sonar_file.filename,
            nav_file_bytes=nav_bytes
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Input normalization failed: {e}")

    raw_image = normalized_input.image
    h, w = raw_image.shape[:2]
    nadir_x = w // 2

    # Save raw image locally
    raw_img_path = survey_folder / f"raw_{sonar_file.filename}"
    cv2.imwrite(str(raw_img_path), raw_image)

    # 2. Quality Assessment
    quality_metrics = assess_image_quality(raw_image)

    # 3. Slant-Range Correction
    altitude_m = normalized_input.get_mean_altitude()
    if apply_slant_range and altitude_m is not None:
        processed_image, slant_applied = apply_slant_range_correction(
            raw_image, altitude_m=altitude_m, meters_per_pixel=0.05
        )
    else:
        processed_image = raw_image.copy()
        slant_applied = False

    # 4. Sonar Preprocessing (Nadir blanking, EGN, CLAHE)
    processed_image = preprocess_sonar_image(
        processed_image,
        remove_water_column=True,
        nadir_width_ratio=0.04,
        apply_clahe=apply_clahe
    )

    # Save preprocessed image
    processed_img_path = survey_folder / "preprocessed.png"
    cv2.imwrite(str(processed_img_path), processed_image)

    # 5. Dynamic Waterfall Tiling (~640x640 with ~20% overlap)
    raw_detections = []
    tile_count = 0

    for tile, x_off, y_off, cur_w, cur_h in generate_tiles(processed_image, tile_size=640, overlap_ratio=0.20):
        tile_count += 1
        # 6. Adaptive YOLO-Seg Inference
        t_dets = inference_engine.predict_tile(tile, confidence_threshold=confidence_threshold)
        if t_dets:
            # Filter detections to valid unpadded area
            valid_t_dets = [d for d in t_dets if d["x_min"] < cur_w and d["y_min"] < cur_h]
            global_t_dets = map_detections_to_global(valid_t_dets, x_off, y_off)
            raw_detections.extend(global_t_dets)

    # 7. Cross-tile Non-Maximum Suppression (NMS)
    filtered_detections = apply_cross_tile_nms(raw_detections, iou_threshold=0.35)

    final_detections = []
    det_idx = 0

    # 8. Acoustic Verification, Geolocation & Risk Scoring
    for d in filtered_detections:
        det_idx += 1
        target_id = f"TGT-{survey_id[-4:].upper()}-{det_idx:03d}"

        # Acoustic shadow evaluation (evidence rater, never blindly discards)
        shadow_result = verify_acoustic_shadow(
            processed_image,
            detection=d,
            nadir_x=nadir_x,
            altitude_m=altitude_m,
            meters_per_pixel=0.05
        )

        # Geolocation projection (graceful degradation)
        center_y = (d["y_min"] + d["y_max"]) // 2
        nav_fix = normalized_input.get_nav_at_row(center_y)
        geo_result = project_detection_geolocation(
            detection=d,
            nadir_x=nadir_x,
            meters_per_pixel=0.05,
            platform_nav=nav_fix
        )

        # Physical dimension estimation
        dim_result = compute_target_dimensions(detection=d, meters_per_pixel=0.05)

        # Operational risk priority calculation
        risk_result = compute_risk_score(
            class_name=d["class_name"],
            confidence=shadow_result["confidence_adjusted"],
            evidence_status=shadow_result["evidence"],
            relief_height_m=shadow_result["relief_height_m"],
            area_m2=dim_result["area_m2"]
        )

        # Save acoustic crop thumbnail
        crop_rel_path = None
        crop_y1 = max(0, d["y_min"] - 15)
        crop_y2 = min(h, d["y_max"] + 15)
        crop_x1 = max(0, d["x_min"] - 15)
        crop_x2 = min(w, d["x_max"] + 15)
        crop_img = processed_image[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop_img.size > 0:
            crop_filename = f"{target_id}_crop.png"
            cv2.imwrite(str(crops_folder / crop_filename), crop_img)
            crop_rel_path = f"/static/outputs/surveys/{survey_id}/crops/{crop_filename}"

        final_d = {
            "id": target_id,
            "survey_id": survey_id,
            "class_name": d["class_name"],
            "confidence_ai": d["confidence_ai"],
            "confidence_final": shadow_result["confidence_adjusted"],
            "evidence_status": shadow_result["evidence"],
            "review_status": shadow_result["review_status"],
            "shadow_contrast": shadow_result["shadow_contrast"],
            "shadow_bbox": shadow_result["shadow_bbox"],
            "relief_height_m": shadow_result["relief_height_m"],
            "length_m": dim_result["length_m"],
            "width_m": dim_result["width_m"],
            "area_m2": dim_result["area_m2"],
            "coordinate_frame": geo_result["coordinate_frame"],
            "latitude": geo_result["latitude"],
            "longitude": geo_result["longitude"],
            "local_x_m": geo_result["local_x_m"],
            "local_y_m": geo_result["local_y_m"],
            "risk_points": risk_result["points"],
            "priority_level": risk_result["priority_level"],
            "x_min": d["x_min"],
            "y_min": d["y_min"],
            "x_max": d["x_max"],
            "y_max": d["y_max"],
            "polygon": d.get("polygon", []),
            "crop_path": crop_rel_path
        }
        final_detections.append(final_d)

    # 9. Store in SQLite Database
    db.insert_survey(
        survey_id=survey_id,
        name=normalized_input.survey_name,
        sonar_path=f"/static/outputs/surveys/{survey_id}/preprocessed.png",
        width=w,
        height=h,
        degradation_tier=normalized_input.degradation_tier
    )
    db.insert_detections(final_detections)

    return {
        "survey_id": survey_id,
        "survey_name": normalized_input.survey_name,
        "degradation_tier": normalized_input.degradation_tier,
        "image_dimensions": [w, h],
        "quality": quality_metrics,
        "slant_range_applied": slant_applied,
        "tiles_processed": tile_count,
        "detections_count": len(final_detections),
        "raw_image_url": f"/static/outputs/surveys/{survey_id}/raw_{sonar_file.filename}",
        "preprocessed_image_url": f"/static/outputs/surveys/{survey_id}/preprocessed.png",
        "detections": final_detections
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
