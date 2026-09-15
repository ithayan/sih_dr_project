import React, { useState } from 'react';
import { 
  Network, 
  Activity, 
  AlertTriangle, 
  CheckCircle2, 
  FileText, 
  ArrowRight, 
  HelpCircle,
  TrendingUp,
  ShieldAlert,
  BarChart3,
  Cpu,
  Layers,
  Compass,
  GitFork,
  Sparkles,
  Edit3,
  Check,
  Award,
  Sliders,
  History,
  Clock
} from 'lucide-react';
import { PatientCase, DRSeverityGrade } from '../types';

interface SystemicTriageColumnProps {
  currentCase: PatientCase;
  onOpenReport: () => void;
}

export const SystemicTriageColumn: React.FC<SystemicTriageColumnProps> = ({
  currentCase,
  onOpenReport,
}) => {
  const [activeTab, setActiveTab] = useState<'manifold' | 'attribution' | 'slm_narrative'>('manifold');
  
  // Longitudinal Disease Progression Tracker state
  const [showLongitudinal, setShowLongitudinal] = useState<boolean>(true);

  // KAN Interactive Attribution Sliders state
  const initialAttribution = currentCase.xai_embeddings?.kan_spline_attribution || {
    semantic_lesion_saliency_pct: 50,
    vascular_topology_tortuosity_pct: 32,
    caliber_narrowing_avr_pct: 18,
  };

  const [lesionWeight, setLesionWeight] = useState<number>(initialAttribution.semantic_lesion_saliency_pct);
  const [topoWeight, setTopoWeight] = useState<number>(initialAttribution.vascular_topology_tortuosity_pct);
  const [avrWeight, setAvrWeight] = useState<number>(initialAttribution.caliber_narrowing_avr_pct);

  // Editable SLM Narrative State
  const defaultSlmNarrative = `Patient ${currentCase.patientName} (${currentCase.age}y, HbA1c ${currentCase.hba1c}%) exhibits clinical findings consistent with ${currentCase.drGrade} Diabetic Retinopathy. Branch A Swin-Tiny transformer localized ${currentCase.lesions.countMA} microaneurysms and ${currentCase.lesions.countHE} focal intraretinal hemorrhages. Branch B vascular GAT measured an Arteriolar-to-Venular Ratio (AVR) of ${currentCase.vascular_biomarkers?.arteriolar_venular_ratio_avr.toFixed(3) || '0.620'} and vessel tortuosity of ${currentCase.vascular_biomarkers?.tortuosity_index.toFixed(4) || '0.0681'}. ${
    currentCase.drGrade === 'Severe' || currentCase.drGrade === 'Proliferative' 
      ? 'URGENT: Hard escalation to Vitreoretinal clinic within 2 weeks indicated.' 
      : 'RECOMMENDATION: Routine follow-up exam in 6-12 months with glycemic stabilization.'
  }`;

  const [slmNarrativeText, setSlmNarrativeText] = useState<string>(defaultSlmNarrative);
  const [isEditingSlm, setIsEditingSlm] = useState<boolean>(false);
  const [doctorSignedOff, setDoctorSignedOff] = useState<boolean>(false);
  const [signOffTimestamp, setSignOffTimestamp] = useState<string>('');

  const xai = currentCase.xai_embeddings || {
    embedding_dimension: 384,
    swin_dimension: 256,
    gnn_dimension: 128,
    kan_latent_dimension: 64,
    clinical_latent_coordinates: { x: 0.12, y: 0.35 },
    reference_cluster_centroids: [
      { grade: 0, name: 'Normal', x: -0.70, y: 0.12 },
      { grade: 1, name: 'Mild NPDR', x: -0.32, y: 0.40 },
      { grade: 2, name: 'Moderate NPDR', x: 0.12, y: 0.35 },
      { grade: 3, name: 'Severe NPDR', x: 0.55, y: -0.22 },
      { grade: 4, name: 'Proliferative DR', x: 0.85, y: -0.65 }
    ],
    kan_spline_attribution: initialAttribution,
    top_contributing_dimensions: [
      { dim: 42, branch: 'Branch A (Swin)', feature: 'Microaneurysm Saliency', weight: 0.42 },
      { dim: 108, branch: 'Branch A (Swin)', feature: 'Retinal Blot Hemorrhages', weight: 0.38 },
      { dim: 268, branch: 'Branch B (GNN)', feature: 'Arteriolar Caliber (AVR)', weight: 0.34 },
      { dim: 312, branch: 'Branch B (GNN)', feature: 'Vascular Tortuosity Index', weight: 0.29 },
      { dim: 376, branch: 'Branch C (KAN)', feature: 'B-Spline Decision Activation', weight: 0.25 }
    ],
    condensed_16d_fingerprint: [0.12, 0.45, 0.78, 0.34, 0.89, 0.23, 0.67, 0.91, 0.43, 0.55, 0.72, 0.38, 0.61, 0.84, 0.29, 0.76],
    symbolic_decision_formula: 'f_KAN(x) = Phi_out(0.50 * phi_Swin(x_A) + 0.32 * phi_GNN(x_B) + 0.18 * phi_AVR(x_C))'
  };

  const coords = xai.clinical_latent_coordinates;
  const vasc = currentCase.vascular_biomarkers || {
    arteriolar_venular_ratio_avr: 0.62,
    tortuosity_index: 0.0681,
    branching_angle_irregularity_deg: 74.2,
    avr_clinical_status: 'Normal Caliber'
  };

  const getDRGradeBadgeStyle = (grade: DRSeverityGrade) => {
    switch (grade) {
      case 'Normal':
        return { bg: 'bg-emerald-950/80', border: 'border-emerald-500', text: 'text-emerald-400', badge: 'bg-emerald-500 text-slate-950' };
      case 'Mild':
        return { bg: 'bg-emerald-950/40', border: 'border-emerald-600/70', text: 'text-emerald-300', badge: 'bg-emerald-600 text-white' };
      case 'Moderate':
        return { bg: 'bg-amber-950/80', border: 'border-amber-500', text: 'text-amber-400', badge: 'bg-amber-500 text-slate-950' };
      case 'Severe':
        return { bg: 'bg-rose-950/80', border: 'border-rose-500', text: 'text-rose-400', badge: 'bg-rose-500 text-white' };
      case 'Proliferative':
        return { bg: 'bg-red-950/90', border: 'border-red-500 ring-2 ring-red-500/40 animate-pulse', text: 'text-red-400', badge: 'bg-red-600 text-white' };
    }
  };

  const gradeStyle = getDRGradeBadgeStyle(currentCase.drGrade);

  // Map [-1, 1] coordinate to SVG [10, 90]%
  const toSvgX = (val: number) => Math.round(50 + (val * 42));
  const toSvgY = (val: number) => Math.round(50 - (val * 42));

  // Dynamic symbolic equation updated from interactive sliders
  const totalW = lesionWeight + topoWeight + avrWeight || 1;
  const normLesionW = ((lesionWeight / totalW) * 100).toFixed(0);
  const normTopoW = ((topoWeight / totalW) * 100).toFixed(0);
  const normAvrW = ((avrWeight / totalW) * 100).toFixed(0);
  const dynamicFormula = `f_KAN(x) = Phi_out(${(+normLesionW / 100).toFixed(2)} * phi_Swin(x_A) + ${(+normTopoW / 100).toFixed(2)} * phi_GNN(x_B) + ${(+normAvrW / 100).toFixed(2)} * phi_AVR(x_C))`;

  // Longitudinal trajectory points simulating disease progression over 12 months
  const longitudinalPoints = [
    { label: 'Visit 1 (t-12m)', x: coords.x - 0.28, y: coords.y + 0.12, grade: 'Normal/Mild' },
    { label: 'Visit 2 (t-6m)', x: coords.x - 0.14, y: coords.y + 0.06, grade: 'Mild/Mod' },
    { label: 'Current Scan (t_now)', x: coords.x, y: coords.y, grade: currentCase.drGrade },
  ];

  const handleSignOff = () => {
    setDoctorSignedOff(true);
    setSignOffTimestamp(new Date().toLocaleTimeString() + ', ' + new Date().toLocaleDateString());
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-sm flex flex-col h-full gap-3 overflow-y-auto">
      {/* HEADER: Title & Status */}
      <div className="border-b border-slate-800 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-400"></span>
            <h2 className="text-sm font-bold text-slate-100 tracking-tight">
              Advanced XAI Suite & Clinical Triage
            </h2>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-700/60 flex items-center gap-1">
            <Network className="w-3 h-3" />
            384-D Tripartite Space
          </span>
        </div>
        <p className="text-[11px] text-slate-400 mt-0.5">
          Intrinsically explainable Kolmogorov-Arnold Network (KAN) B-splines & SLM narrative synthesis
        </p>
      </div>

      {/* CLASSIFICATION SUMMARY TILE */}
      <div className={`p-3 rounded-lg border ${gradeStyle.bg} ${gradeStyle.border} transition-all`}>
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono font-bold tracking-wider text-slate-400 uppercase">
                Gate 4 Severity Staging
              </span>
              <span className="text-[10px] font-mono text-cyan-300 bg-cyan-950/80 px-2 py-0.2 rounded border border-cyan-800/60">
                {currentCase.drConfidence >= 0.82 ? 'Autonomous Triage' : 'Escalated to Hub'}
              </span>
            </div>
            <div className="text-base font-extrabold text-white flex items-center gap-2 mt-0.5">
              <span>{currentCase.drGrade} DR</span>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${gradeStyle.badge}`}>
                {(currentCase.drConfidence * 100).toFixed(1)}% Conf
              </span>
            </div>
          </div>
          <button
            onClick={onOpenReport}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-all shadow-md shadow-indigo-950/50"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Clinical Dossier</span>
          </button>
        </div>
      </div>

      {/* DETERMINISTIC VASCULAR METRICS (BRANCH B GNN) */}
      <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-2.5">
        <div className="flex items-center justify-between text-[11px] font-bold text-slate-300 mb-2">
          <span className="flex items-center gap-1 text-cyan-400">
            <GitFork className="w-3.5 h-3.5" />
            Topological Vascular Biomarkers
          </span>
          <span className="text-[10px] font-mono text-slate-400">Branch B GAT</span>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
            <div className="text-[10px] text-slate-400">AVR Caliber</div>
            <div className="text-xs font-mono font-bold text-cyan-300 mt-0.5">
              {vasc.arteriolar_venular_ratio_avr.toFixed(3)}
            </div>
            <div className="text-[8px] text-slate-400 mt-0.5">Ref: 0.67-0.75</div>
          </div>
          <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
            <div className="text-[10px] text-slate-400">Tortuosity Index</div>
            <div className="text-xs font-mono font-bold text-amber-300 mt-0.5">
              {vasc.tortuosity_index.toFixed(4)}
            </div>
            <div className="text-[8px] text-slate-400 mt-0.5">Arc-Chord Curv</div>
          </div>
          <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
            <div className="text-[10px] text-slate-400">Branching Angle</div>
            <div className="text-xs font-mono font-bold text-emerald-300 mt-0.5">
              {vasc.branching_angle_irregularity_deg.toFixed(1)}°
            </div>
            <div className="text-[8px] text-slate-400 mt-0.5">Bifurcation Geom</div>
          </div>
        </div>
      </div>

      {/* TAB NAVIGATION: Manifold vs KAN Attribution vs SLM Narrative */}
      <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-[11px] font-bold">
        <button
          onClick={() => setActiveTab('manifold')}
          className={`flex-1 py-1 rounded transition-all flex items-center justify-center gap-1 ${
            activeTab === 'manifold'
              ? 'bg-indigo-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Compass className="w-3 h-3" />
          <span>2D Manifold</span>
        </button>
        <button
          onClick={() => setActiveTab('attribution')}
          className={`flex-1 py-1 rounded transition-all flex items-center justify-center gap-1 ${
            activeTab === 'attribution'
              ? 'bg-indigo-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Sliders className="w-3 h-3" />
          <span>KAN Sliders</span>
        </button>
        <button
          onClick={() => setActiveTab('slm_narrative')}
          className={`flex-1 py-1 rounded transition-all flex items-center justify-center gap-1 ${
            activeTab === 'slm_narrative'
              ? 'bg-indigo-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Edit3 className="w-3 h-3" />
          <span>SLM Narrative</span>
        </button>
      </div>

      {/* TAB CONTENT 1: INTERACTIVE 2D CLINICAL MANIFOLD SCATTER */}
      {activeTab === 'manifold' && (
        <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-3 flex flex-col gap-2">
          <div className="flex items-center justify-between text-[11px]">
            <span className="font-bold text-slate-300">2D Clinical Embedding Space</span>
            <span className="text-[10px] font-mono text-indigo-400">
              Pos: ({coords.x.toFixed(3)}, {coords.y.toFixed(3)})
            </span>
          </div>

          <div className="relative w-full h-44 bg-slate-900 rounded-lg border border-slate-800/80 overflow-hidden">
            {/* Background Grid Lines */}
            <div className="absolute inset-0 grid grid-cols-4 grid-rows-4 opacity-15 pointer-events-none">
              <div className="border-r border-b border-slate-600"></div>
              <div className="border-r border-b border-slate-600"></div>
              <div className="border-r border-b border-slate-600"></div>
              <div className="border-b border-slate-600"></div>
              <div className="border-r border-b border-slate-600"></div>
              <div className="border-r border-b border-slate-600"></div>
              <div className="border-r border-b border-slate-600"></div>
              <div className="border-b border-slate-600"></div>
              <div className="border-r border-b border-slate-600"></div>
              <div className="border-r border-b border-slate-600"></div>
              <div className="border-r border-b border-slate-600"></div>
              <div className="border-b border-slate-600"></div>
            </div>

            {/* SVG Elements: Longitudinal Trajectory & Clusters */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              {/* Reference Cluster Centroid Ellipses */}
              {xai.reference_cluster_centroids.map((c) => {
                const cx = toSvgX(c.x);
                const cy = toSvgY(c.y);
                const colors = ['#10b981', '#34d399', '#f59e0b', '#f43f5e', '#ef4444'];
                return (
                  <g key={c.grade} opacity={0.35}>
                    <ellipse cx={`${cx}%`} cy={`${cy}%`} rx="12%" ry="10%" fill={colors[c.grade]} />
                    <text x={`${cx}%`} y={`${cy - 6}%`} fill={colors[c.grade]} fontSize="8" textAnchor="middle" fontWeight="bold">
                      G{c.grade}: {c.name}
                    </text>
                  </g>
                );
              })}

              {/* Longitudinal Vector Trajectory (t-12m -> t-6m -> Current) */}
              {showLongitudinal && (
                <>
                  <line 
                    x1={`${toSvgX(longitudinalPoints[0].x)}%`} 
                    y1={`${toSvgY(longitudinalPoints[0].y)}%`}
                    x2={`${toSvgX(longitudinalPoints[1].x)}%`} 
                    y2={`${toSvgY(longitudinalPoints[1].y)}%`}
                    stroke="#38bdf8" 
                    strokeWidth="1.5" 
                    strokeDasharray="3 2" 
                  />
                  <line 
                    x1={`${toSvgX(longitudinalPoints[1].x)}%`} 
                    y1={`${toSvgY(longitudinalPoints[1].y)}%`}
                    x2={`${toSvgX(longitudinalPoints[2].x)}%`} 
                    y2={`${toSvgY(longitudinalPoints[2].y)}%`}
                    stroke="#38bdf8" 
                    strokeWidth="2" 
                  />
                  {/* Historical Dots */}
                  <circle cx={`${toSvgX(longitudinalPoints[0].x)}%`} cy={`${toSvgY(longitudinalPoints[0].y)}%`} r="3" fill="#64748b" />
                  <circle cx={`${toSvgX(longitudinalPoints[1].x)}%`} cy={`${toSvgY(longitudinalPoints[1].y)}%`} r="4" fill="#38bdf8" />
                </>
              )}
            </svg>

            {/* Current Patient Coordinate Dot */}
            <div
              className="absolute w-4 h-4 -ml-2 -mt-2 rounded-full bg-cyan-400 border-2 border-white shadow-lg shadow-cyan-400/80 animate-pulse z-20 flex items-center justify-center"
              style={{ left: `${toSvgX(coords.x)}%`, top: `${toSvgY(coords.y)}%` }}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-slate-950"></span>
            </div>

            {/* Trajectory Toggle Button in bottom corner */}
            <div className="absolute bottom-2 left-2 z-20">
              <button
                onClick={() => setShowLongitudinal(!showLongitudinal)}
                className="px-2 py-0.5 rounded bg-slate-950/80 border border-slate-700 text-[9px] font-mono text-cyan-300 flex items-center gap-1 hover:bg-slate-800 transition-all"
              >
                <History className="w-2.5 h-2.5" />
                <span>{showLongitudinal ? 'Hide Trajectory' : 'Show 12m Trajectory'}</span>
              </button>
            </div>
          </div>

          <div className="text-[10px] text-slate-400 flex items-center justify-between">
            <span>Projection: <strong className="text-slate-300">R^384 &rarr; R^2</strong></span>
            <span>Centroid Dist: <strong className="text-emerald-400">0.084 (Congruent)</strong></span>
          </div>
        </div>
      )}

      {/* TAB CONTENT 2: KAN ATTRIBUTION SLIDERS & B-SPLINE BASIS */}
      {activeTab === 'attribution' && (
        <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-3 flex flex-col gap-3">
          <div className="flex items-center justify-between text-[11px]">
            <span className="font-bold text-slate-300">KAN B-Spline Mathematical Decomposition</span>
            <span className="text-[10px] font-mono text-cyan-400">Interactive</span>
          </div>

          {/* Slider 1: Semantic Lesions */}
          <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
            <div className="flex justify-between text-[11px]">
              <span className="text-slate-300 font-medium">Lesion Saliency &phi;_Swin(x_A):</span>
              <span className="font-mono font-bold text-rose-400">{normLesionW}%</span>
            </div>
            <input 
              type="range" 
              min="10" 
              max="80" 
              value={lesionWeight}
              onChange={(e) => setLesionWeight(parseInt(e.target.value))}
              className="w-full mt-1.5 accent-rose-400 cursor-pointer h-1.5 bg-slate-800 rounded"
            />
            <span className="text-[9px] text-slate-500 mt-0.5 block">Microaneurysms, hemorrhages, hard exudates</span>
          </div>

          {/* Slider 2: Vascular Topology */}
          <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
            <div className="flex justify-between text-[11px]">
              <span className="text-slate-300 font-medium">Vessel Topology &phi;_GNN(x_B):</span>
              <span className="font-mono font-bold text-indigo-400">{normTopoW}%</span>
            </div>
            <input 
              type="range" 
              min="10" 
              max="60" 
              value={topoWeight}
              onChange={(e) => setTopoWeight(parseInt(e.target.value))}
              className="w-full mt-1.5 accent-indigo-400 cursor-pointer h-1.5 bg-slate-800 rounded"
            />
            <span className="text-[9px] text-slate-500 mt-0.5 block">GAT bifurcation graph & vessel tortuosity</span>
          </div>

          {/* Slider 3: Caliber Attenuation (AVR) */}
          <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
            <div className="flex justify-between text-[11px]">
              <span className="text-slate-300 font-medium">Caliber Ratio &phi;_AVR(x_C):</span>
              <span className="font-mono font-bold text-cyan-400">{normAvrW}%</span>
            </div>
            <input 
              type="range" 
              min="5" 
              max="40" 
              value={avrWeight}
              onChange={(e) => setAvrWeight(parseInt(e.target.value))}
              className="w-full mt-1.5 accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded"
            />
            <span className="text-[9px] text-slate-500 mt-0.5 block">Arteriolar-to-venular narrowing factor</span>
          </div>

          {/* Live Symbolic Decision Formula */}
          <div className="bg-slate-900 p-2.5 rounded border border-slate-800 font-mono text-[10px] text-slate-300 overflow-x-auto">
            <div className="text-[9px] text-slate-500 uppercase mb-1">Dynamically Derived KAN Closed-Form Equation:</div>
            <div className="text-cyan-300 font-bold">{dynamicFormula}</div>
          </div>
        </div>
      )}

      {/* TAB CONTENT 3: EDITABLE SLM NARRATIVE & DOCTOR SIGN-OFF */}
      {activeTab === 'slm_narrative' && (
        <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-3 flex flex-col gap-3">
          <div className="flex items-center justify-between text-[11px]">
            <span className="font-bold text-slate-300 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
              Generative Edge SLM Clinical Narrative
            </span>
            <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-800/60">
              Editable
            </span>
          </div>

          {/* Editable Text Box */}
          <div className="relative">
            <textarea
              rows={5}
              value={slmNarrativeText}
              onChange={(e) => {
                setSlmNarrativeText(e.target.value);
                setIsEditingSlm(true);
              }}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 leading-relaxed focus:outline-none focus:border-cyan-500 font-sans"
              placeholder="Doctor's clinical notes and modifications..."
            />
            {isEditingSlm && (
              <span className="absolute bottom-2 right-2 text-[9px] font-mono text-amber-400 bg-amber-950/80 px-1.5 py-0.2 rounded border border-amber-800/80">
                Doctor Modified
              </span>
            )}
          </div>

          {/* Doctor Sign-Off Action Card */}
          <div className={`p-3 rounded-lg border transition-all ${
            doctorSignedOff 
              ? 'bg-emerald-950/70 border-emerald-500' 
              : 'bg-slate-900 border-slate-800'
          }`}>
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-1.5">
                  <Award className={`w-4 h-4 ${doctorSignedOff ? 'text-emerald-400' : 'text-slate-400'}`} />
                  <span className="text-xs font-bold text-slate-200">
                    {doctorSignedOff ? 'Certified Ophthalmologist Sign-Off' : 'Pending Specialist Sign-Off'}
                  </span>
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  {doctorSignedOff ? (
                    <span className="text-emerald-300">
                      Dr. Arvind Sharma, MS, FRCSI (Reg: MCI-74892-VR) • {signOffTimestamp}
                    </span>
                  ) : (
                    <span>Review findings, edit narrative if necessary, and certify triage.</span>
                  )}
                </div>
              </div>

              {!doctorSignedOff ? (
                <button
                  onClick={handleSignOff}
                  className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs flex items-center gap-1.5 transition-all shadow-md shadow-emerald-950/50"
                >
                  <Check className="w-3.5 h-3.5" />
                  <span>Certify & Sign Off</span>
                </button>
              ) : (
                <span className="text-[10px] font-mono font-bold px-2 py-1 rounded bg-emerald-900/80 text-emerald-300 border border-emerald-600 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  SIGNED & VERIFIED
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
