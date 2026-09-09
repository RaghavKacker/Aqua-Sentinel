"""
AquaSentinel AI - SQLite Database Layer
Stores survey records, processing jobs, target detections, and telemetry.
"""

import os
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.getenv("SQLITE_DB_PATH", "./outputs/aquasentinel.db")

class Database:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self.get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS surveys (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                sonar_path TEXT,
                status TEXT NOT NULL,
                width_px INTEGER NOT NULL,
                height_px INTEGER NOT NULL,
                degradation_tier TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                survey_id TEXT NOT NULL REFERENCES surveys(id),
                class_name TEXT NOT NULL,
                confidence_ai REAL NOT NULL,
                confidence_final REAL NOT NULL,
                evidence_status TEXT NOT NULL,
                review_status TEXT NOT NULL,
                x_min INTEGER NOT NULL,
                y_min INTEGER NOT NULL,
                x_max INTEGER NOT NULL,
                y_max INTEGER NOT NULL,
                polygon_json TEXT,
                shadow_bbox_json TEXT,
                relief_height_m REAL,
                length_m REAL,
                width_m REAL,
                area_m2 REAL,
                coordinate_frame TEXT,
                latitude REAL,
                longitude REAL,
                local_x_m REAL,
                local_y_m REAL,
                risk_points INTEGER NOT NULL,
                priority_level TEXT NOT NULL,
                crop_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.commit()

    def insert_survey(self, survey_id: str, name: str, sonar_path: str, width: int, height: int, degradation_tier: str, status: str = "COMPLETED"):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO surveys (id, name, sonar_path, status, width_px, height_px, degradation_tier)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (survey_id, name, sonar_path, status, width, height, degradation_tier))
            conn.commit()

    def insert_detections(self, detections: List[Dict[str, Any]]):
        if not detections:
            return
        with self.get_connection() as conn:
            for d in detections:
                conn.execute("""
                INSERT OR REPLACE INTO detections (
                    id, survey_id, class_name, confidence_ai, confidence_final, evidence_status,
                    review_status, x_min, y_min, x_max, y_max, polygon_json, shadow_bbox_json,
                    relief_height_m, length_m, width_m, area_m2, coordinate_frame, latitude,
                    longitude, local_x_m, local_y_m, risk_points, priority_level, crop_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    d["id"], d["survey_id"], d["class_name"], d["confidence_ai"], d["confidence_final"],
                    d["evidence_status"], d["review_status"], d["x_min"], d["y_min"], d["x_max"], d["y_max"],
                    json.dumps(d.get("polygon", [])),
                    json.dumps(d.get("shadow_bbox", [])) if d.get("shadow_bbox") else None,
                    d.get("relief_height_m"), d.get("length_m"), d.get("width_m"), d.get("area_m2"),
                    d.get("coordinate_frame"), d.get("latitude"), d.get("longitude"),
                    d.get("local_x_m"), d.get("local_y_m"), d["risk_points"], d["priority_level"],
                    d.get("crop_path")
                ))
            conn.commit()

    def get_surveys(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM surveys ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_survey(self, survey_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM surveys WHERE id = ?", (survey_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_detections_for_survey(self, survey_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM detections WHERE survey_id = ? ORDER BY risk_points DESC", (survey_id,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["polygon"] = json.loads(d["polygon_json"]) if d.get("polygon_json") else []
                d["shadow_bbox"] = json.loads(d["shadow_bbox_json"]) if d.get("shadow_bbox_json") else None
                results.append(d)
            return results

    def get_stats(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            n_surveys = conn.execute("SELECT COUNT(*) FROM surveys").fetchone()[0]
            n_targets = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            n_critical = conn.execute("SELECT COUNT(*) FROM detections WHERE priority_level IN ('CRITICAL', 'HIGH')").fetchone()[0]
            n_verified = conn.execute("SELECT COUNT(*) FROM detections WHERE review_status = 'VERIFIED'").fetchone()[0]
            return {
                "total_surveys": n_surveys,
                "total_detections": n_targets,
                "high_critical_risks": n_critical,
                "verified_3d_objects": n_verified
            }
