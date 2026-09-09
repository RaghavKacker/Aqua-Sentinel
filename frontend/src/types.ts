export type DegradationTier = 'FULL' | 'PARTIAL' | 'NONE';

export type EvidenceStatus = 'SUPPORTING' | 'NEUTRAL' | 'ABSENT';

export type ReviewStatus = 'VERIFIED' | 'UNVERIFIED' | 'FLAGGED_FOR_REVIEW';

export type PriorityLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface Detection {
  id: string;
  survey_id: string;
  class_name: string;
  confidence_ai: number;
  confidence_final: number;
  evidence_status: EvidenceStatus;
  review_status: ReviewStatus;
  shadow_contrast: number;
  shadow_bbox?: [number, number, number, number] | null;
  relief_height_m?: number | null;
  length_m?: number | null;
  width_m?: number | null;
  area_m2?: number | null;
  coordinate_frame?: 'WGS84' | 'LOCAL_METRIC_ODOMETRY' | 'UNTAGGED';
  latitude?: number | null;
  longitude?: number | null;
  local_x_m?: number | null;
  local_y_m?: number | null;
  risk_points: number;
  priority_level: PriorityLevel;
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  polygon?: number[];
  crop_path?: string | null;
}

export interface Survey {
  id: string;
  name: string;
  sonar_path: string;
  status: string;
  width_px: number;
  height_px: number;
  degradation_tier: DegradationTier;
  created_at: string;
}

export interface SystemStatus {
  status: string;
  offline_ready: boolean;
  compute_profile: string;
  device: string;
  cuda_available: boolean;
  gpu_name: string;
  yolo_model_loaded: boolean;
}

export interface DashboardStats {
  total_surveys: number;
  total_detections: number;
  high_critical_risks: number;
  verified_3d_objects: number;
}
