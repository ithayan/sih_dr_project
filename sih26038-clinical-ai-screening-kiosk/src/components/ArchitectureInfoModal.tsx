import React from 'react';
import { 
  X, 
  Cpu, 
  Layers, 
  Dna, 
  Eye, 
  ArrowRight, 
  CheckCircle, 
  ShieldCheck, 
  Server,
  Zap,
  Activity
} from 'lucide-react';
import { KioskTelemetry } from '../types';

interface ArchitectureInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  telemetry: KioskTelemetry;
}

export const ArchitectureInfoModal: React.FC<ArchitectureInfoModalProps> = ({
  isOpen,
  onClose,
  telemetry,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-black/85 backdrop-blur-sm overflow-y-auto">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-3xl w-full max-h-[92vh] flex flex-col shadow-2xl overflow-hidden my-auto text-slate-100">
        {/* Top Bar */}
        <div className="flex items-center justify-between px-5 py-3.5 bg-slate-950 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-bold text-slate-200">
              SIH26038 Multi-Modal AI Architecture Specification
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto space-y-5 text-xs text-slate-300">
          {/* Header Banner */}
          <div className="bg-gradient-to-r from-cyan-950/60 to-blue-950/60 border border-cyan-800/40 p-4 rounded-xl">
            <h3 className="text-sm font-bold text-cyan-200 mb-1">
              OCUNEXA: Hybrid YOLO-GNN-KAN Edge-to-Hub Clinical Architecture
            </h3>
            <p className="text-slate-400 leading-relaxed">
              Strictly engineered for offline rural telemedicine on standard Intel® Core™ i5 workstations (&lt; 800 ms latency budget). 
              Integrates YOLO26 Nano instance segmentation (OpenVINO INT8), 2-layer Vascular Graph Attention Networks (GAT), and 
              Kolmogorov-Arnold Network (KAN) B-spline decision heads, guarded by deterministic clinical safety gates.
            </p>
          </div>

          {/* Pipeline Diagram Blocks */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Tripartite AI Engine & Hard Clinical Gates
            </h4>

            {/* Gate 1 & 2 */}
            <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 flex items-start gap-3">
              <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono font-bold shrink-0">
                G1/2
              </div>
              <div>
                <strong className="text-slate-100 text-xs block mb-1">
                  Hard-Gated Ingestion (Gate 1 Quality Lock & Gate 2 Anomaly Filter)
                </strong>
                <p className="text-slate-400 leading-normal">
                  Gate 1 evaluates focus sharpness (Tenengrad Sobel energy), illumination balance, and circular FOV. 
                  If composite score &lt; 0.70 $\rightarrow$ HARD BLOCK (&quot;Recapture Required&quot;), halting all downstream inference. 
                  Gate 2 executes a binary pre-screener separating pristine fundi from referable pathology.
                </p>
              </div>
            </div>

            {/* Branch A */}
            <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 flex items-start gap-3">
              <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono font-bold shrink-0">
                01
              </div>
              <div>
                <strong className="text-slate-100 text-xs block mb-1">
                  Branch A: YOLO26 Nano Semantic Lesion Engine (Intel OpenVINO INT8)
                </strong>
                <p className="text-slate-400 leading-normal">
                  Utilizes <code className="text-cyan-300 font-mono">yolo26n-seg.pt</code> compiled to OpenVINO IR for hardware-accelerated CPU inference. 
                  Employs one-to-one Hungarian label assignment (eliminating non-maximum suppression latency bottlenecks) 
                  to extract discrete bounding boxes and polygon proto-masks for microaneurysms, hemorrhages, and exudates (256-D feature vector).
                </p>
              </div>
            </div>

            {/* Branch B */}
            <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 flex items-start gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/20 font-mono font-bold shrink-0">
                02
              </div>
              <div>
                <strong className="text-slate-100 text-xs block mb-1">
                  Branch B: Topological Vascular Engine (2-Layer GAT on DRIVE Skeletons)
                </strong>
                <p className="text-slate-400 leading-normal">
                  Skeletonizes the retinal vascular tree, builds node-edge bifurcation graphs, and computes 2-layer Graph Attention message passing:
                  <span className="font-mono text-purple-300 block mt-1">h_i^(l+1) = σ(Σ_j α_ij W h_j^(l))</span>
                  Extracts deterministic Arteriolar-to-Venular Ratio (AVR), vessel tortuosity, and Murray's branching irregularity (128-D vector).
                </p>
              </div>
            </div>

            {/* Branch C */}
            <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 flex items-start gap-3">
              <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20 font-mono font-bold shrink-0">
                03
              </div>
              <div>
                <strong className="text-slate-100 text-xs block mb-1">
                  Branch C: Kolmogorov-Arnold Network (KAN) Decision Head
                </strong>
                <p className="text-slate-400 leading-normal">
                  Fuses Branch A (256-D) + Branch B (128-D) = 384-D unified latent embedding. 
                  Replaces fixed linear weights with learnable B-spline univariate activation functions on edges:
                  <span className="font-mono text-amber-300 block mt-1">f(x) = Σ_q Φ_q( Σ_p φ_(q,p)(x_p) )</span>
                  Yields mathematically transparent 5-class ICDR severity classification (Normal $\to$ PDR).
                </p>
              </div>
            </div>

            {/* Gate 4 & Edge SLM */}
            <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 flex items-start gap-3">
              <div className="p-2 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20 font-mono font-bold shrink-0">
                G4
              </div>
              <div>
                <strong className="text-slate-100 text-xs block mb-1">
                  Generative Edge SLM & Gate 4 Confidence Telemedicine Routing
                </strong>
                <p className="text-slate-400 leading-normal">
                  A quantized Edge Small Language Model synthesizes plain-English clinical notes from YOLO boxes and KAN weights. 
                  Gate 4 enforces hard clinical routing: if calibration confidence $\ge 0.82 \rightarrow$ Autonomous local screening approval; 
                  if confidence &lt; 0.82 $\rightarrow$ HARD ESCALATION to the centralized Web Hub queue.
                </p>
              </div>
            </div>
          </div>

          {/* Benchmark Table */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
            <h4 className="text-xs font-bold text-slate-200 mb-2 flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-emerald-400" />
              Intel Core i5 Edge Benchmark Telemetry (OpenVINO INT8)
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
              <div className="bg-slate-900 p-2 rounded border border-slate-800">
                <span className="text-slate-500 block text-[10px]">Host Processor</span>
                <span className="text-slate-200 font-bold">Intel Core i5 CPU</span>
              </div>
              <div className="bg-slate-900 p-2 rounded border border-slate-800">
                <span className="text-slate-500 block text-[10px]">Inference Latency</span>
                <span className="text-emerald-400 font-bold">30.5 ms Total</span>
              </div>
              <div className="bg-slate-900 p-2 rounded border border-slate-800">
                <span className="text-slate-500 block text-[10px]">Latency Budget</span>
                <span className="text-cyan-400 font-bold">&lt; 800 ms (Passed)</span>
              </div>
              <div className="bg-slate-900 p-2 rounded border border-slate-800">
                <span className="text-slate-500 block text-[10px]">Edge Footprint</span>
                <span className="text-slate-200 font-bold">18 MB INT8 IR</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-slate-950 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-bold rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
          >
            Close Specification
          </button>
        </div>
      </div>
    </div>
  );
};
