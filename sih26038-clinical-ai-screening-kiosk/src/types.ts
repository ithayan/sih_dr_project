export type DRSeverityGrade = 'Normal' | 'Mild' | 'Moderate' | 'Severe' | 'Proliferative';

export type SystemicRiskLevel = 'Low' | 'Medium' | 'High';

export interface LesionTypology {
  microaneurysms: boolean;
  hemorrhages: boolean;
  hardExudates: boolean;
  cottonWoolSpots: boolean;
  neovascularization: boolean;
  countMA: number;
  countHE: number;
  countEX: number;
}

export interface GeneToken {
  gene: string;
  rank: number;
  expressionScore: number;
  attentionWeight: number; // 0.0 - 1.0 (Extracted from TransformerEncoderLayer)
  pathway: string;
  clinicalSignificance: string;
}

export interface GradCAMHotspot {
  id: string;
  x: number; // percentage 0-100
  y: number; // percentage 0-100
  radius: number; // percentage 0-100
  intensity: number; // 0.0 - 1.0
  lesionType: 'Microaneurysm' | 'Hemorrhage' | 'Hard Exudate' | 'Cotton Wool Spot' | 'Neovascular Loop';
  etdrsZone: string;
}

export interface YOLO26Detection {
  detection_id: string;
  lesion_type: string;
  box: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    width: number;
    height: number;
  };
  polygon_mask?: Array<{ x: number; y: number }>;
  confidence: number;
  etdrs_zone: string;
  assignment_mode: string;
}

export interface SLMNarrativeReport {
  title: string;
  clinical_grade: string;
  stage_numeric: number;
  calibrated_confidence: number;
  assessment: string;
  objective_findings: string;
  plan_and_recommendation: string;
  gate4_triage_routing: string;
  telemedicine_escalation_required: boolean;
  digital_signature: string;
}

export interface XAIEmbeddingData {
  embedding_dimension: number;
  yolo_dimension?: number;
  swin_dimension?: number;
  gnn_dimension: number;
  kan_latent_dimension: number;
  clinical_latent_coordinates: { x: number; y: number };
  reference_cluster_centroids: Array<{ grade: number; name: string; x: number; y: number }>;
  kan_spline_attribution: {
    semantic_lesion_saliency_pct: number;
    vascular_topology_tortuosity_pct: number;
    caliber_narrowing_avr_pct: number;
  };
  top_contributing_dimensions: Array<{
    dim: number;
    branch: string;
    feature: string;
    weight: number;
  }>;
  condensed_16d_fingerprint: number[];
  symbolic_decision_formula: string;
}

export interface PatientCase {
  id: string;
  patientId: string;
  patientName: string;
  age: number;
  gender: 'Male' | 'Female';
  hba1c: number;
  diabetesDurationYears: number;
  eye: 'OD (Right Eye)' | 'OS (Left Eye)';
  cameraModel: string;
  imageQuality: 'Gradable (High)' | 'Gradable (Acceptable)' | 'Ungradable';
  sampleDescription: string;
  imageSrc?: string;
  drGrade: DRSeverityGrade;
  drConfidence: number; // e.g., 0.94
  systemicRisk: SystemicRiskLevel;
  systemicConfidence: number; // e.g., 0.89
  biDirectionalConcordance: 'Concordant' | 'Discordant - Microvascular Precursor' | 'Discordant - Systemic Dominant';
  foveaCoordinates: { x: number; y: number }; // percentage 0-100
  opticDiscCoordinates: { x: number; y: number };
  lesions: LesionTypology;
  hotspots: GradCAMHotspot[];
  genes: GeneToken[];
  rawGeneSentence: string;
  clinicalRecommendation: string;
  xai_embeddings?: XAIEmbeddingData;
  yolo26Detections?: YOLO26Detection[];
  slmNarrative?: SLMNarrativeReport;
  vascular_biomarkers?: {
    arteriolar_venular_ratio_avr: number;
    tortuosity_index: number;
    branching_angle_irregularity_deg: number;
    avr_clinical_status: string;
  };
  gate1_passed?: boolean;
  quality_score?: number;
  triage_routing?: string;
  etdrsSubfieldRisks: {
    central1mm: number;
    innerSuperior: number;
    innerNasal: number;
    innerInferior: number;
    innerTemporal: number;
    outerSuperior: number;
    outerNasal: number;
    outerInferior: number;
    outerTemporal: number;
  };
}

export interface InferenceStep {
  id: number;
  name: string;
  architecture: string;
  dimensions: string;
  status: 'idle' | 'running' | 'completed';
  latencyMs: number;
  vectorSummary: string;
  details: string;
}

export interface KioskTelemetry {
  device: string;
  cpu: string;
  gpu: string;
  ramUsage: string;
  totalLatencyMs: number;
  kioskId: string;
  firmware: string;
  isAirGapped: boolean;
}
