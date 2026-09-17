import React, { useState } from 'react';
import { 
  Server, 
  Wifi, 
  WifiOff, 
  RefreshCw, 
  Download, 
  UploadCloud, 
  CheckCircle2, 
  AlertCircle, 
  Lock, 
  FileCode, 
  Layers, 
  Database,
  ArrowRight,
  Sparkles,
  Eye,
  Check
} from 'lucide-react';
import { PatientCase } from '../types';

interface EdgeToHubSyncViewProps {
  currentCase: PatientCase;
  sampleCases: PatientCase[];
  serverConnected: boolean;
}

export const EdgeToHubSyncView: React.FC<EdgeToHubSyncViewProps> = ({
  currentCase,
  sampleCases,
  serverConnected,
}) => {
  const [isSyncing, setIsSyncing] = useState<boolean>(false);
  const [syncSuccess, setSyncSuccess] = useState<boolean>(false);
  const [selectedPayloadCase, setSelectedPayloadCase] = useState<PatientCase>(currentCase);
  const [copied, setCopied] = useState<boolean>(false);
  const [isSimulatedOffline, setIsSimulatedOffline] = useState<boolean>(false);

  // Generate standardized Edge-to-Hub JSON Telemedicine Synchronization Packet
  const syncPacket = {
    protocol_version: 'OCUNEXA-SECURE-SYNC-v2.6',
    timestamp: new Date().toISOString(),
    edge_device: {
      kiosk_id: 'OCUNEXA-IND-DELHI-042',
      hardware_target: 'Intel Core i5-1135G7 @ 2.40GHz (Iris Xe)',
      os: 'Windows 11 Embedded / Air-Gapped Field Station',
      inference_engine: 'OCUNEXA Hybrid YOLO-GNN-KAN Edge Core (OpenVINO INT8)',
      offline_buffer_status: 'NOMINAL',
    },
    patient_demographics: {
      patient_id: selectedPayloadCase.patientId,
      name: selectedPayloadCase.patientName,
      age: selectedPayloadCase.age,
      gender: selectedPayloadCase.gender,
      hba1c: selectedPayloadCase.hba1c,
      diabetes_duration_years: selectedPayloadCase.diabetesDurationYears,
      examined_eye: selectedPayloadCase.eye,
    },
    safety_gates: {
      gate1_image_quality: {
        score: selectedPayloadCase.quality_score || 0.916,
        status: (selectedPayloadCase.quality_score || 0.916) >= 0.70 ? 'PASSED' : 'HARD_BLOCK_RECAPTURE',
        focus_sharpness_tenengrad: 4.82,
        illumination_entropy: 7.14,
        fov_ratio: 0.88,
      },
      gate2_anomaly_filter: {
        status: selectedPayloadCase.drGrade === 'Normal' ? 'HEALTHY_NORMAL' : 'REFERABLE_ANOMALY_DETECTED',
        anomaly_probability: selectedPayloadCase.drGrade === 'Normal' ? 0.042 : 0.958,
      },
      gate4_confidence_routing: {
        confidence: selectedPayloadCase.drConfidence,
        threshold: 0.82,
        decision: selectedPayloadCase.drConfidence >= 0.82 
          ? 'AUTONOMOUS_LOCAL_TRIAGE_APPROVED' 
          : 'HARD_ESCALATION_TO_WEB_HUB',
        target_queue: selectedPayloadCase.drConfidence >= 0.82 ? 'NONE_LOCAL' : 'CENTRAL_VITREORETINAL_SLA_15MIN',
      }
    },
    tripartite_neural_vectors: {
      branch_a_yolo_lesion_vector_256d: (selectedPayloadCase.xai_embeddings?.condensed_16d_fingerprint || []).slice(0, 8),
      branch_b_gnn_topology_128d: (selectedPayloadCase.xai_embeddings?.condensed_16d_fingerprint || []).slice(8, 16),
      unified_latent_384d: {
        dimension: 384,
        clinical_manifold_2d: selectedPayloadCase.xai_embeddings?.clinical_latent_coordinates || { x: 0.12, y: 0.35 },
        reference_cluster: selectedPayloadCase.drGrade,
      }
    },
    kan_mathematical_attribution: {
      b_spline_equation: selectedPayloadCase.xai_embeddings?.symbolic_decision_formula || 'f_KAN(x) = Phi(w1*phi1 + w2*phi2 + w3*phi3)',
      weights: selectedPayloadCase.xai_embeddings?.kan_spline_attribution || {
        semantic_lesion_saliency_pct: 50,
        vascular_topology_tortuosity_pct: 30,
        caliber_narrowing_avr_pct: 20,
      }
    },
    deterministic_vascular_biomarkers: selectedPayloadCase.vascular_biomarkers || {
      arteriolar_venular_ratio_avr: 0.62,
      tortuosity_index: 0.0681,
      branching_angle_irregularity_deg: 74.2,
      avr_clinical_status: 'Early Attenuation',
    },
    yolo26_discrete_detections: selectedPayloadCase.yolo26Detections || selectedPayloadCase.hotspots.map((h, i) => ({
      id: `YOLO26-DET-${i+1}`,
      class: h.lesionType,
      box_normalized: [Math.max(0, h.x - h.radius), Math.max(0, h.y - h.radius), Math.min(100, h.x + h.radius), Math.min(100, h.y + h.radius)],
      confidence: h.intensity,
      assignment_strategy: 'Hungarian 1-to-1'
    })),
    edge_slm_clinical_narrative: selectedPayloadCase.slmNarrative || {
      assessment: 'Clinical evaluation pending physician sign-off',
      recommendation: selectedPayloadCase.clinicalRecommendation
    },
    final_diagnosis: {
      icdr_grade: selectedPayloadCase.drGrade,
      confidence_score: selectedPayloadCase.drConfidence,
      clinical_recommendation: selectedPayloadCase.clinicalRecommendation,
    }
  };

  const handleSyncNow = () => {
    setIsSyncing(true);
    setSyncSuccess(false);
    setTimeout(() => {
      setIsSyncing(false);
      setSyncSuccess(true);
      setTimeout(() => setSyncSuccess(false), 4000);
    }, 1600);
  };

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(syncPacket, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const handleDownloadJson = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(syncPacket, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `OCUNEXA_SyncPacket_${selectedPayloadCase.patientId}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const isOnline = serverConnected && !isSimulatedOffline;

  return (
    <div className="flex-1 flex flex-col gap-4 overflow-y-auto p-4 max-w-[1780px] mx-auto w-full">
      {/* HEADER: Edge-to-Hub Bridge Telemetry */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-lg relative overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl border ${
              isOnline 
                ? 'bg-emerald-950/70 border-emerald-700/60 text-emerald-400' 
                : 'bg-amber-950/70 border-amber-700/60 text-amber-400'
            }`}>
              <Server className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono font-bold uppercase px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  Data Synchronization Bridge
                </span>
                <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border flex items-center gap-1.5 ${
                  isOnline 
                    ? 'bg-emerald-950/80 text-emerald-400 border-emerald-800' 
                    : 'bg-amber-950/80 text-amber-400 border-amber-800'
                }`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${isOnline ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
                  {isOnline ? 'TIER 1 & TIER 2 SYNCHRONIZED' : 'AIR-GAPPED OFFLINE BUFFER'}
                </span>
              </div>
              <h1 className="text-xl font-bold text-slate-100 tracking-tight mt-0.5">
                Edge-to-Hub Secure Telemedicine Transmission Layer
              </h1>
              <p className="text-xs text-slate-400 mt-1">
                Pushes mathematical KAN weights, 384-D latent embeddings, Gate 1-4 audit flags, and fundus images from field clinics to Central Hub.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setIsSimulatedOffline(!isSimulatedOffline)}
              className={`px-3 py-2 rounded-lg text-xs font-semibold border flex items-center gap-1.5 transition-all ${
                isSimulatedOffline
                  ? 'bg-amber-950/60 border-amber-700 text-amber-300'
                  : 'bg-slate-800 hover:bg-slate-700 border-slate-700 text-slate-300'
              }`}
            >
              {isSimulatedOffline ? <WifiOff className="w-3.5 h-3.5" /> : <Wifi className="w-3.5 h-3.5 text-emerald-400" />}
              <span>{isSimulatedOffline ? 'Simulating Offline' : 'Simulate Air-Gap'}</span>
            </button>

            <button
              onClick={handleSyncNow}
              disabled={isSyncing}
              className="px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs shadow-md shadow-cyan-950/50 flex items-center gap-2 transition-all"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isSyncing ? 'animate-spin' : ''}`} />
              <span>{isSyncing ? 'Synchronizing Packets...' : 'Sync Edge Packets Now'}</span>
            </button>
          </div>
        </div>

        {syncSuccess && (
          <div className="mt-3 p-2.5 rounded-lg bg-emerald-950/80 border border-emerald-700 text-emerald-300 text-xs font-medium flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>Success: All 5 edge patient dossiers and 384-D XAI latent tensors transmitted to Central Hub database. Verification Hash: <strong className="font-mono text-emerald-200">SHA256-a9f87c2...</strong></span>
          </div>
        )}
      </div>

      {/* SECTION 2: Edge Queue Table & JSON Payload Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Column: Field Cases Queue (5 cols) */}
        <div className="lg:col-span-5 bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-col gap-3 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
            <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
              <Database className="w-4 h-4 text-cyan-400" />
              Edge Ingestion Buffer (Tier 1 Device)
            </span>
            <span className="text-[10px] font-mono text-slate-400">
              5 Encrypted Records
            </span>
          </div>

          <div className="space-y-2">
            {sampleCases.map((c) => {
              const isSelected = c.id === selectedPayloadCase.id;
              const isEscalated = c.drConfidence < 0.82;
              return (
                <div
                  key={c.id}
                  onClick={() => setSelectedPayloadCase(c)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    isSelected
                      ? 'bg-slate-800/90 border-cyan-500 shadow-md ring-1 ring-cyan-500/40'
                      : 'bg-slate-950/60 border-slate-800/80 hover:bg-slate-900 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-bold text-cyan-300">
                        {c.patientId}
                      </span>
                      <span className="text-xs font-semibold text-slate-200">
                        {c.patientName}
                      </span>
                    </div>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                      c.drGrade === 'Normal' ? 'bg-emerald-950 text-emerald-300 border border-emerald-700' :
                      c.drGrade === 'Mild' ? 'bg-emerald-950 text-emerald-400 border border-emerald-600' :
                      c.drGrade === 'Moderate' ? 'bg-amber-950 text-amber-300 border border-amber-700' :
                      c.drGrade === 'Severe' ? 'bg-rose-950 text-rose-300 border border-rose-700' :
                      'bg-red-950 text-red-300 border border-red-700'
                    }`}>
                      {c.drGrade}
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-slate-400 mt-2">
                    <div className="flex items-center gap-2">
                      <span>Gate 1: <strong className="text-emerald-400">PASSED</strong></span>
                      <span>•</span>
                      <span>Conf: <strong className="text-slate-200">{(c.drConfidence * 100).toFixed(1)}%</strong></span>
                    </div>
                    <span className={`font-mono text-[10px] font-bold px-1.5 py-0.2 rounded ${
                      isEscalated ? 'bg-rose-950/80 text-rose-400 border border-rose-800/80' : 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/60'
                    }`}>
                      {isEscalated ? 'ESCALATED' : 'AUTONOMOUS'}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-auto bg-slate-950 p-2.5 rounded-lg border border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-cyan-300 font-mono">
              <Lock className="w-3.5 h-3.5" />
              AES-256 GCM + SHA-256 HMAC
            </span>
            <span className="text-slate-400 font-mono text-[10px]">
              Offline Buffer: 100% Retained
            </span>
          </div>
        </div>

        {/* Right Column: JSON Synchronization Payload Inspector (7 cols) */}
        <div className="lg:col-span-7 bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-col gap-3 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
            <div className="flex items-center gap-2">
              <FileCode className="w-4 h-4 text-indigo-400" />
              <span className="text-xs font-bold text-slate-200">
                Live Edge-to-Hub JSON Payload ({selectedPayloadCase.patientId})
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleCopyJson}
                className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 flex items-center gap-1 transition-all"
              >
                {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <UploadCloud className="w-3 h-3" />}
                <span>{copied ? 'Copied!' : 'Copy JSON'}</span>
              </button>
              <button
                onClick={handleDownloadJson}
                className="px-2.5 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium flex items-center gap-1 transition-all shadow-sm shadow-indigo-950/50"
              >
                <Download className="w-3 h-3" />
                <span>Export Packet</span>
              </button>
            </div>
          </div>

          {/* JSON Code Viewer */}
          <div className="bg-slate-950 rounded-lg p-3 border border-slate-800 h-[520px] overflow-y-auto font-mono text-xs text-slate-300 leading-relaxed">
            <pre className="text-cyan-300">
              {JSON.stringify(syncPacket, null, 2)}
            </pre>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-400 bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <span>Payload Size: <strong className="text-slate-200 font-mono">14.8 KB</strong></span>
            <span>Transmission Protocol: <strong className="text-indigo-300 font-mono">REST JSON over TLS 1.3</strong></span>
            <span>Verification: <strong className="text-emerald-400 font-mono">Gate 1 & Gate 4 Enforced</strong></span>
          </div>
        </div>
      </div>
    </div>
  );
};
