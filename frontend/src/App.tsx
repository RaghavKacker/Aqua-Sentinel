import React, { useState, useEffect, useRef } from 'react';
import { 
  Shield, Activity, Cpu, HardDrive, Download, Eye, Layers, 
  MapPin, AlertTriangle, CheckCircle, RefreshCw, ZoomIn, ZoomOut, 
  Sliders, Navigation, Database, Compass, ChevronRight, FileText,
  SlidersHorizontal, Check
} from 'lucide-react';
import { Detection, Survey, SystemStatus, DashboardStats } from './types';

export default function App() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    status: 'ONLINE',
    offline_ready: true,
    compute_profile: 'NORMAL',
    device: 'cpu',
    cuda_available: false,
    gpu_name: 'Checking...',
    yolo_model_loaded: false
  });

  const [stats, setStats] = useState<DashboardStats>({
    total_surveys: 0,
    total_detections: 0,
    high_critical_risks: 0,
    verified_3d_objects: 0
  });

  const [surveys, setSurveys] = useState<Survey[]>([]);
  const [selectedSurvey, setSelectedSurvey] = useState<Survey | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null);

  // Form State
  const [sonarFile, setSonarFile] = useState<File | null>(null);
  const [navFile, setNavFile] = useState<File | null>(null);
  const [applySlantRange, setApplySlantRange] = useState(true);
  const [applyClahe, setApplyClahe] = useState(true);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Viewer State
  const [activeTab, setActiveTab] = useState<'waterfall' | 'map'>('waterfall');
  const [viewMode, setViewMode] = useState<'preprocessed' | 'raw'>('preprocessed');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showMasks, setShowMasks] = useState(true);
  const [showShadows, setShowShadows] = useState(true);
  const [currentRawUrl, setCurrentRawUrl] = useState<string | null>(null);
  const [currentPreprocessedUrl, setCurrentPreprocessedUrl] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Fetch initial telemetry
  useEffect(() => {
    fetchTelemetry();
    fetchSurveys();
    const interval = setInterval(fetchTelemetry, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchTelemetry = async () => {
    try {
      const res = await fetch('/api/v1/system/status');
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(data);
      }
      const statsRes = await fetch('/api/v1/system/stats');
      if (statsRes.ok) {
        const sData = await statsRes.json();
        setStats(sData);
      }
    } catch (e) {
      console.log("Telemetry check offline or waiting for backend");
    }
  };

  const fetchSurveys = async () => {
    try {
      const res = await fetch('/api/v1/surveys');
      if (res.ok) {
        const data = await res.json();
        setSurveys(data);
        if (data.length > 0 && !selectedSurvey) {
          loadSurveyDetails(data[0].id);
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  const loadSurveyDetails = async (surveyId: string) => {
    try {
      const res = await fetch(`/api/v1/surveys/${surveyId}`);
      if (res.ok) {
        const data = await res.json();
        setSelectedSurvey(data.survey);
        setDetections(data.detections);
        setCurrentPreprocessedUrl(data.survey.sonar_path);
        if (data.detections.length > 0) {
          setSelectedDetection(data.detections[0]);
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleProcessSurvey = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sonarFile) {
      setErrorMessage("Please select a sonar image file first.");
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append('sonar_file', sonarFile);
    if (navFile) {
      formData.append('nav_file', navFile);
    }
    formData.append('apply_slant_range', String(applySlantRange));
    formData.append('apply_clahe', String(applyClahe));
    formData.append('confidence_threshold', String(confidenceThreshold));

    try {
      const res = await fetch('/api/v1/surveys/process', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Processing failed");
      }

      const result = await res.json();
      setCurrentRawUrl(result.raw_image_url);
      setCurrentPreprocessedUrl(result.preprocessed_image_url);
      setDetections(result.detections);
      if (result.detections.length > 0) {
        setSelectedDetection(result.detections[0]);
      }
      fetchSurveys();
      fetchTelemetry();
    } catch (err: any) {
      setErrorMessage(err.message || "An unexpected error occurred during processing");
    } finally {
      setIsProcessing(false);
    }
  };

  // Draw sonar overlays onto canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imgUrl = viewMode === 'raw' && currentRawUrl ? currentRawUrl : currentPreprocessedUrl;
    if (!imgUrl) return;

    const img = new Image();
    img.src = imgUrl;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      // Nadir guide line
      const nadirX = canvas.width / 2;
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.25)';
      ctx.setLineDash([6, 6]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(nadirX, 0);
      ctx.lineTo(nadirX, canvas.height);
      ctx.stroke();
      ctx.setLineDash([]);

      // Overlays
      detections.forEach((d) => {
        const isSelected = selectedDetection?.id === d.id;

        // 1. Acoustic Shadow Box
        if (showShadows && d.shadow_bbox) {
          const [sx1, sy1, sx2, sy2] = d.shadow_bbox;
          ctx.fillStyle = 'rgba(5, 12, 26, 0.65)';
          ctx.strokeStyle = 'rgba(0, 240, 255, 0.5)';
          ctx.lineWidth = 1;
          ctx.fillRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
          ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
        }

        // 2. Segmentation Mask (YOLO-Seg Polygon)
        if (showMasks && d.polygon && d.polygon.length >= 6) {
          ctx.beginPath();
          ctx.moveTo(d.polygon[0], d.polygon[1]);
          for (let i = 2; i < d.polygon.length; i += 2) {
            ctx.lineTo(d.polygon[i], d.polygon[i + 1]);
          }
          ctx.closePath();
          ctx.fillStyle = d.priority_level === 'CRITICAL' ? 'rgba(255, 77, 109, 0.35)' :
                          d.priority_level === 'HIGH' ? 'rgba(247, 127, 0, 0.35)' :
                          d.priority_level === 'MEDIUM' ? 'rgba(255, 209, 102, 0.35)' :
                          'rgba(6, 214, 160, 0.35)';
          ctx.fill();
        }

        // 3. Bounding Box
        if (showBoxes) {
          ctx.strokeStyle = isSelected ? '#00f0ff' :
                            d.priority_level === 'CRITICAL' ? '#ff4d6d' :
                            d.priority_level === 'HIGH' ? '#f77f00' :
                            d.priority_level === 'MEDIUM' ? '#ffd166' : '#06d6a0';
          ctx.lineWidth = isSelected ? 3 : 2;
          ctx.strokeRect(d.x_min, d.y_min, d.x_max - d.x_min, d.y_max - d.y_min);

          // Label pill
          ctx.fillStyle = ctx.strokeStyle;
          ctx.font = '12px Courier New';
          const label = `${d.class_name} (${Math.round(d.confidence_final * 100)}%)`;
          const textWidth = ctx.measureText(label).width;
          ctx.fillRect(d.x_min, Math.max(0, d.y_min - 16), textWidth + 8, 16);

          ctx.fillStyle = '#030a16';
          ctx.fillText(label, d.x_min + 4, Math.max(12, d.y_min - 4));
        }
      });
    };
  }, [currentPreprocessedUrl, currentRawUrl, viewMode, detections, showBoxes, showMasks, showShadows, selectedDetection]);

  const handleDownloadDeliverable = (format: 'geojson' | 'csv') => {
    if (!selectedSurvey) return;
    window.open(`/api/v1/surveys/${selectedSurvey.id}/export/${format}`, '_blank');
  };

  return (
    <div className="min-h-screen flex flex-col bg-ocean-950 text-slate-100">
      {/* Top Telemetry Bar */}
      <header className="border-b border-ocean-800 bg-ocean-900/90 backdrop-blur sticky top-0 z-50 px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <Shield className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-lg tracking-wider text-slate-100 font-mono">AQUASENTINEL AI</h1>
                <span className="text-xs bg-cyan-500/20 text-cyan-300 px-2 py-0.5 rounded font-mono border border-cyan-500/30">V1.0</span>
              </div>
              <p className="text-xs text-slate-400">Offline-First Side-Scan Sonar Debris & Anomaly System</p>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs font-mono">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span>AIR-GAPPED / OFFLINE</span>
            </div>

            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-ocean-800 border border-ocean-700 text-slate-300">
              <Cpu className="w-4 h-4 text-cyan-400" />
              <span>PROFILE: {systemStatus.compute_profile}</span>
            </div>

            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-ocean-800 border border-ocean-700 text-slate-300">
              <Activity className="w-4 h-4 text-amber-400" />
              <span>{systemStatus.cuda_available ? systemStatus.gpu_name : 'AMD RYZEN CPU (FALLBACK)'}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <div className="flex-1 grid grid-cols-12 gap-4 p-4 overflow-hidden">
        {/* Left Column: Upload & Survey Index (3 cols) */}
        <div className="col-span-12 lg:col-span-3 flex flex-col gap-4">
          {/* KPI Mini-Cards */}
          <div className="grid grid-cols-2 gap-2">
            <div className="p-3 bg-ocean-900 border border-ocean-800 rounded-lg">
              <div className="text-xs text-slate-400">SURVEYS</div>
              <div className="text-2xl font-bold font-mono text-cyan-400">{stats.total_surveys}</div>
            </div>
            <div className="p-3 bg-ocean-900 border border-ocean-800 rounded-lg">
              <div className="text-xs text-slate-400">TARGETS</div>
              <div className="text-2xl font-bold font-mono text-slate-100">{stats.total_detections}</div>
            </div>
            <div className="p-3 bg-ocean-900 border border-ocean-800 rounded-lg">
              <div className="text-xs text-slate-400">VERIFIED 3D</div>
              <div className="text-2xl font-bold font-mono text-emerald-400">{stats.verified_3d_objects}</div>
            </div>
            <div className="p-3 bg-ocean-900 border border-ocean-800 rounded-lg">
              <div className="text-xs text-slate-400">CRITICAL / HIGH</div>
              <div className="text-2xl font-bold font-mono text-rose-400">{stats.high_critical_risks}</div>
            </div>
          </div>

          {/* Survey Upload & Processing Form */}
          <div className="p-4 bg-ocean-900 border border-ocean-800 rounded-lg flex flex-col gap-3">
            <h2 className="text-sm font-semibold font-mono text-cyan-400 flex items-center gap-2">
              <HardDrive className="w-4 h-4" /> INGEST & PROCESS SURVEY
            </h2>

            <form onSubmit={handleProcessSurvey} className="flex flex-col gap-3 text-xs">
              <div>
                <label className="block text-slate-400 mb-1">Side-Scan Sonar Waterfall (PBM, PNG, JPG)</label>
                <input 
                  type="file" 
                  accept=".png,.jpg,.jpeg,.pbm,.tif,.tiff"
                  onChange={(e) => setSonarFile(e.target.files?.[0] || null)}
                  className="w-full text-xs text-slate-300 file:mr-2 file:py-1.5 file:px-3 file:rounded file:border-0 file:text-xs file:bg-cyan-500/20 file:text-cyan-300 hover:file:bg-cyan-500/30 cursor-pointer bg-ocean-950 border border-ocean-800 rounded p-1"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Synchronous Navigation Telemetry (Optional CSV)</label>
                <input 
                  type="file" 
                  accept=".csv,.txt"
                  onChange={(e) => setNavFile(e.target.files?.[0] || null)}
                  className="w-full text-xs text-slate-300 file:mr-2 file:py-1.5 file:px-3 file:rounded file:border-0 file:text-xs file:bg-ocean-800 file:text-slate-300 hover:file:bg-ocean-700 cursor-pointer bg-ocean-950 border border-ocean-800 rounded p-1"
                />
                <span className="text-[10px] text-slate-500">Supports SubPipeMiniSSS or WGS84 GPS logs</span>
              </div>

              <div className="pt-2 border-t border-ocean-800 space-y-2">
                <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                  <input 
                    type="checkbox" 
                    checked={applySlantRange} 
                    onChange={(e) => setApplySlantRange(e.target.checked)}
                    className="rounded border-ocean-700 text-cyan-500 focus:ring-0"
                  />
                  <span>Slant-Range Pythagorean Correction</span>
                </label>

                <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                  <input 
                    type="checkbox" 
                    checked={applyClahe} 
                    onChange={(e) => setApplyClahe(e.target.checked)}
                    className="rounded border-ocean-700 text-cyan-500 focus:ring-0"
                  />
                  <span>Swath EGN & CLAHE Equalization</span>
                </label>

                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>AI Confidence Threshold</span>
                    <span className="font-mono text-cyan-400">{Math.round(confidenceThreshold * 100)}%</span>
                  </div>
                  <input 
                    type="range" 
                    min="0.10" 
                    max="0.90" 
                    step="0.05"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>
              </div>

              {errorMessage && (
                <div className="p-2 bg-rose-500/10 border border-rose-500/30 rounded text-rose-300 text-xs">
                  {errorMessage}
                </div>
              )}

              <button
                type="submit"
                disabled={isProcessing}
                className={`w-full py-2.5 rounded font-mono font-bold flex items-center justify-center gap-2 transition-all ${
                  isProcessing 
                    ? 'bg-ocean-800 text-slate-500 cursor-not-allowed' 
                    : 'bg-cyan-500 hover:bg-cyan-400 text-ocean-950 shadow-lg shadow-cyan-500/20'
                }`}
              >
                {isProcessing ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />
                    <span>RUNNING PIPELINE...</span>
                  </>
                ) : (
                  <>
                    <Activity className="w-4 h-4" />
                    <span>START AI DETECTION</span>
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Ingested Surveys Catalog */}
          <div className="flex-1 p-3 bg-ocean-900 border border-ocean-800 rounded-lg flex flex-col gap-2 overflow-y-auto">
            <h3 className="text-xs font-mono text-slate-400 flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5" /> PREVIOUS SURVEY RUNS
            </h3>
            {surveys.length === 0 ? (
              <p className="text-xs text-slate-500 italic p-2">No surveys analyzed yet.</p>
            ) : (
              <div className="space-y-1.5">
                {surveys.map((srv) => (
                  <button
                    key={srv.id}
                    onClick={() => loadSurveyDetails(srv.id)}
                    className={`w-full text-left p-2 rounded text-xs transition flex items-center justify-between ${
                      selectedSurvey?.id === srv.id 
                        ? 'bg-cyan-500/10 border border-cyan-500/40 text-cyan-300' 
                        : 'bg-ocean-950/60 hover:bg-ocean-800 border border-transparent text-slate-300'
                    }`}
                  >
                    <div className="truncate">
                      <div className="font-semibold truncate">{srv.name}</div>
                      <div className="text-[10px] text-slate-500 font-mono">
                        {srv.width_px}x{srv.height_px}px | Tier: {srv.degradation_tier}
                      </div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-slate-500 flex-shrink-0" />
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Center Column: Dual Waterfall Sonar Viewer (6 cols) */}
        <div className="col-span-12 lg:col-span-6 flex flex-col gap-3">
          {/* Controls Bar */}
          <div className="flex items-center justify-between bg-ocean-900 border border-ocean-800 rounded-lg px-4 py-2 text-xs">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setViewMode('preprocessed')}
                className={`px-3 py-1 rounded font-mono font-semibold transition ${
                  viewMode === 'preprocessed' ? 'bg-cyan-500 text-ocean-950' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                EQUALIZED WATERFALL
              </button>
              <button
                onClick={() => setViewMode('raw')}
                className={`px-3 py-1 rounded font-mono font-semibold transition ${
                  viewMode === 'raw' ? 'bg-cyan-500 text-ocean-950' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                RAW ACOUSTIC
              </button>
            </div>

            <div className="flex items-center gap-3">
              <label className="flex items-center gap-1 cursor-pointer text-slate-300">
                <input 
                  type="checkbox" 
                  checked={showBoxes} 
                  onChange={(e) => setShowBoxes(e.target.checked)}
                  className="rounded text-cyan-500 focus:ring-0"
                />
                <span>Boxes</span>
              </label>

              <label className="flex items-center gap-1 cursor-pointer text-slate-300">
                <input 
                  type="checkbox" 
                  checked={showMasks} 
                  onChange={(e) => setShowMasks(e.target.checked)}
                  className="rounded text-cyan-500 focus:ring-0"
                />
                <span>Masks</span>
              </label>

              <label className="flex items-center gap-1 cursor-pointer text-slate-300">
                <input 
                  type="checkbox" 
                  checked={showShadows} 
                  onChange={(e) => setShowShadows(e.target.checked)}
                  className="rounded text-cyan-500 focus:ring-0"
                />
                <span>Shadows</span>
              </label>

              <div className="h-4 w-[1px] bg-ocean-700"></div>

              <div className="flex items-center gap-1">
                <button 
                  onClick={() => setZoomLevel((z) => Math.max(0.5, z - 0.25))}
                  className="p-1 hover:bg-ocean-800 rounded text-slate-400 hover:text-slate-100"
                >
                  <ZoomOut className="w-4 h-4" />
                </button>
                <span className="font-mono text-[11px] text-cyan-400 w-10 text-center">
                  {Math.round(zoomLevel * 100)}%
                </span>
                <button 
                  onClick={() => setZoomLevel((z) => Math.min(3, z + 0.25))}
                  className="p-1 hover:bg-ocean-800 rounded text-slate-400 hover:text-slate-100"
                >
                  <ZoomIn className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Interactive Waterfall Viewport */}
          <div className="flex-1 bg-ocean-950 border border-ocean-800 rounded-lg overflow-auto p-2 relative flex items-center justify-center min-h-[500px]">
            {currentPreprocessedUrl ? (
              <div 
                style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'top center' }} 
                className="transition-transform duration-100"
              >
                <canvas ref={canvasRef} className="border border-ocean-800 shadow-2xl rounded" />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center text-slate-500 gap-3">
                <Compass className="w-12 h-12 text-ocean-800 animate-pulse" />
                <p className="text-xs font-mono">Upload a sonar swath or select a survey to inspect waterfall overlays.</p>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Target Details & Deliverable Export (3 cols) */}
        <div className="col-span-12 lg:col-span-3 flex flex-col gap-4">
          {/* Target Inspector Card */}
          <div className="p-4 bg-ocean-900 border border-ocean-800 rounded-lg flex flex-col gap-3">
            <div className="flex items-center justify-between border-b border-ocean-800 pb-2">
              <h2 className="text-sm font-semibold font-mono text-cyan-400 flex items-center gap-1.5">
                <Eye className="w-4 h-4" /> TARGET INSPECTOR
              </h2>
              {selectedDetection && (
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold ${
                  selectedDetection.priority_level === 'CRITICAL' ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' :
                  selectedDetection.priority_level === 'HIGH' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                  selectedDetection.priority_level === 'MEDIUM' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30' :
                  'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                }`}>
                  PRIORITY: {selectedDetection.priority_level}
                </span>
              )}
            </div>

            {selectedDetection ? (
              <div className="space-y-3 text-xs">
                {/* Crop Patch */}
                <div className="flex gap-3">
                  <div className="w-24 h-24 bg-ocean-950 border border-ocean-700 rounded overflow-hidden flex-shrink-0 flex items-center justify-center">
                    {selectedDetection.crop_path ? (
                      <img 
                        src={selectedDetection.crop_path} 
                        alt={selectedDetection.id} 
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <span className="text-[10px] text-slate-600">No Crop</span>
                    )}
                  </div>
                  <div className="flex-1 space-y-1">
                    <div className="font-bold text-slate-100 font-mono text-sm">{selectedDetection.id}</div>
                    <div className="text-cyan-400 font-semibold uppercase">{selectedDetection.class_name.replace('_', ' ')}</div>
                    <div className="text-[11px] text-slate-400">
                      Score: <span className="font-mono text-slate-200">{Math.round(selectedDetection.confidence_final * 100)}%</span>
                      <span className="text-[10px] text-slate-500 ml-1">(AI: {Math.round(selectedDetection.confidence_ai * 100)}%)</span>
                    </div>
                    <div className="text-[11px] text-slate-400">
                      Risk Points: <span className="font-mono text-amber-400 font-bold">{selectedDetection.risk_points}/100</span>
                    </div>
                  </div>
                </div>

                {/* Acoustic Shadow Evidence */}
                <div className="p-2.5 bg-ocean-950 border border-ocean-800 rounded space-y-1.5">
                  <div className="font-mono text-slate-400 font-semibold flex items-center justify-between text-[11px]">
                    <span>ACOUSTIC SHADOW</span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                      selectedDetection.evidence_status === 'SUPPORTING' ? 'bg-emerald-500/20 text-emerald-300' :
                      selectedDetection.evidence_status === 'ABSENT' ? 'bg-rose-500/20 text-rose-300' :
                      'bg-slate-700 text-slate-300'
                    }`}>
                      {selectedDetection.evidence_status}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-300">
                    <div>Contrast: <span className="font-mono text-cyan-400">{selectedDetection.shadow_contrast}</span></div>
                    <div>Relief: <span className="font-mono text-emerald-400">{selectedDetection.relief_height_m ? `${selectedDetection.relief_height_m}m` : 'N/A'}</span></div>
                  </div>

                  <div className="text-[10px] text-slate-500">
                    Review Status: <span className="text-slate-300 font-mono">{selectedDetection.review_status}</span>
                  </div>
                </div>

                {/* Geolocation & Metric Coordinates */}
                <div className="p-2.5 bg-ocean-950 border border-ocean-800 rounded space-y-1 text-[11px]">
                  <div className="font-mono text-slate-400 font-semibold text-[11px] flex items-center gap-1">
                    <MapPin className="w-3 h-3 text-cyan-400" /> GEOLOCATION ({selectedDetection.coordinate_frame || 'UNTAGGED'})
                  </div>
                  {selectedDetection.coordinate_frame === 'WGS84' ? (
                    <div className="font-mono text-slate-200 text-[10px]">
                      <div>LAT: {selectedDetection.latitude}</div>
                      <div>LON: {selectedDetection.longitude}</div>
                    </div>
                  ) : selectedDetection.coordinate_frame === 'LOCAL_METRIC_ODOMETRY' ? (
                    <div className="font-mono text-slate-200 text-[10px]">
                      <div>X: {selectedDetection.local_x_m}m</div>
                      <div>Y: {selectedDetection.local_y_m}m</div>
                    </div>
                  ) : (
                    <div className="text-[10px] text-slate-500 italic">No vehicle navigation telemetry supplied.</div>
                  )}
                </div>

                {/* Dimensions */}
                <div className="p-2.5 bg-ocean-950 border border-ocean-800 rounded text-[11px] space-y-1">
                  <div className="font-mono text-slate-400 font-semibold text-[11px]">PHYSICAL SIZE</div>
                  <div className="grid grid-cols-3 gap-1 text-[10px] font-mono text-slate-300">
                    <div>L: {selectedDetection.length_m ? `${selectedDetection.length_m}m` : '-'}</div>
                    <div>W: {selectedDetection.width_m ? `${selectedDetection.width_m}m` : '-'}</div>
                    <div>Area: {selectedDetection.area_m2 ? `${selectedDetection.area_m2}m²` : '-'}</div>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500 italic">Select a target from the list below to inspect physical metrics.</p>
            )}
          </div>

          {/* Detections List for Active Survey */}
          <div className="flex-1 p-3 bg-ocean-900 border border-ocean-800 rounded-lg flex flex-col gap-2 overflow-y-auto max-h-[300px]">
            <div className="text-xs font-mono text-slate-400 flex items-center justify-between">
              <span>DETECTED TARGETS ({detections.length})</span>
            </div>
            <div className="space-y-1">
              {detections.map((d) => (
                <button
                  key={d.id}
                  onClick={() => setSelectedDetection(d)}
                  className={`w-full text-left p-2 rounded text-xs transition flex items-center justify-between ${
                    selectedDetection?.id === d.id
                      ? 'bg-cyan-500/20 border border-cyan-500 text-cyan-200'
                      : 'bg-ocean-950 hover:bg-ocean-800/80 border border-ocean-800 text-slate-300'
                  }`}
                >
                  <div>
                    <div className="font-bold font-mono">{d.id}</div>
                    <div className="text-[10px] text-slate-400">{d.class_name} • {Math.round(d.confidence_final * 100)}%</div>
                  </div>
                  <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold font-mono ${
                    d.priority_level === 'CRITICAL' ? 'bg-rose-500/30 text-rose-300' :
                    d.priority_level === 'HIGH' ? 'bg-amber-500/30 text-amber-300' :
                    d.priority_level === 'MEDIUM' ? 'bg-yellow-500/30 text-yellow-300' :
                    'bg-emerald-500/30 text-emerald-300'
                  }`}>
                    {d.priority_level}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Export Deliverables Bar */}
          <div className="p-3 bg-ocean-900 border border-ocean-800 rounded-lg flex flex-col gap-2">
            <h3 className="text-xs font-mono text-slate-400 flex items-center gap-1.5">
              <Download className="w-3.5 h-3.5" /> EXPORT DELIVERABLES
            </h3>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <button
                onClick={() => handleDownloadDeliverable('geojson')}
                disabled={!selectedSurvey}
                className="py-2 px-3 rounded bg-ocean-800 hover:bg-ocean-700 text-cyan-300 border border-ocean-700 flex items-center justify-center gap-1.5 transition disabled:opacity-50"
              >
                <MapPin className="w-3.5 h-3.5" /> GeoJSON
              </button>
              <button
                onClick={() => handleDownloadDeliverable('csv')}
                disabled={!selectedSurvey}
                className="py-2 px-3 rounded bg-ocean-800 hover:bg-ocean-700 text-emerald-300 border border-ocean-700 flex items-center justify-center gap-1.5 transition disabled:opacity-50"
              >
                <FileText className="w-3.5 h-3.5" /> CSV Catalog
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
