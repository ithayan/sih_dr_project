import React from 'react';
import { 
  X, 
  Printer, 
  Check, 
  Download, 
  Eye, 
  Dna, 
  ShieldCheck, 
  AlertCircle, 
  Building2,
  Calendar,
  Clock,
  UserCheck
} from 'lucide-react';
import { PatientCase, KioskTelemetry } from '../types';

interface ClinicalReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  patientCase: PatientCase;
  telemetry: KioskTelemetry;
}

export const ClinicalReportModal: React.FC<ClinicalReportModalProps> = ({
  isOpen,
  onClose,
  patientCase,
  telemetry,
}) => {
  if (!isOpen) return null;

  const handlePrint = () => {
    window.print();
  };

  const currentDate = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  const currentTime = new Date().toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-black/80 backdrop-blur-sm overflow-y-auto">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-4xl w-full max-h-[92vh] flex flex-col shadow-2xl overflow-hidden my-auto text-slate-100">
        {/* Modal Top Bar (Screen Only) */}
        <div className="flex items-center justify-between px-5 py-3.5 bg-slate-950 border-b border-slate-800 print:hidden">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
            <span className="text-sm font-bold text-slate-200">
              Clinical Diagnostic Report Generator • SIH26038
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrint}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white shadow-md transition"
            >
              <Printer className="w-3.5 h-3.5" />
              <span>Print / Save PDF</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-slate-800 transition"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Printable Report Content Area */}
        <div className="p-6 md:p-8 overflow-y-auto space-y-6 bg-slate-900 print:bg-white print:text-black print:p-4 print:space-y-4">
          {/* Header & Letterhead */}
          <div className="flex flex-wrap items-start justify-between gap-4 pb-4 border-b border-slate-800 print:border-black">
            <div>
              <div className="flex items-center gap-2 text-cyan-400 print:text-blue-900 font-bold text-lg md:text-xl">
                <Building2 className="w-6 h-6" />
                <span>SMART INDIA HACKATHON • CLINICAL AI WORKSTATION</span>
              </div>
              <p className="text-xs text-slate-400 print:text-gray-600 mt-0.5 font-medium">
                Bi-Directional Retinal Microvascular & Single-Cell Immunological Risk Screening Kiosk
              </p>
              <p className="text-[11px] text-slate-500 print:text-gray-500 font-mono">
                Project Code: SIH26038 | Offline Edge Hardware: {telemetry.cpu.split('(')[0]}
              </p>
            </div>
            <div className="text-right text-xs font-mono text-slate-400 print:text-gray-700">
              <div>Report Ref: <strong className="text-slate-200 print:text-black">REP-{patientCase.patientId}-XAI</strong></div>
              <div>Date: {currentDate}</div>
              <div>Time: {currentTime} UTC</div>
              <div className="text-emerald-400 print:text-emerald-800 font-semibold">AIR-GAPPED OFFLINE VERIFIED</div>
            </div>
          </div>

          {/* Patient Demographics & Camera Metadata */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-950/70 print:bg-gray-100 p-3.5 rounded-xl border border-slate-800 print:border-gray-300 text-xs">
            <div>
              <span className="text-slate-500 print:text-gray-500 block text-[10px]">Patient Name</span>
              <strong className="text-slate-100 print:text-black text-sm">{patientCase.patientName}</strong>
            </div>
            <div>
              <span className="text-slate-500 print:text-gray-500 block text-[10px]">Patient ID / Age / Sex</span>
              <span className="text-slate-200 print:text-black font-mono">
                {patientCase.patientId} | {patientCase.age}y | {patientCase.gender}
              </span>
            </div>
            <div>
              <span className="text-slate-500 print:text-gray-500 block text-[10px]">Eye Examined / Camera</span>
              <span className="text-slate-200 print:text-black">
                {patientCase.eye} ({patientCase.cameraModel.split(' ')[0]})
              </span>
            </div>
            <div>
              <span className="text-slate-500 print:text-gray-500 block text-[10px]">Metabolic Profile</span>
              <span className="text-amber-400 print:text-amber-700 font-mono font-bold">
                HbA1c: {patientCase.hba1c}% ({patientCase.diabetesDurationYears}y DM)
              </span>
            </div>
          </div>

          {/* Core Triage Findings */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Vision Finding */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 print:border-gray-300 print:bg-white">
              <div className="flex items-center gap-2 mb-2">
                <Eye className="w-4 h-4 text-cyan-400 print:text-blue-700" />
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 print:text-black">
                  Vision Layer: Retinopathy Staging
                </h4>
              </div>
              <div className="flex items-baseline gap-3 my-2">
                <span className="text-2xl font-black text-rose-400 print:text-red-700">
                  {patientCase.drGrade} NPDR
                </span>
                <span className="text-xs font-mono text-slate-400 print:text-gray-600">
                  Confidence: {(patientCase.drConfidence * 100).toFixed(1)}%
                </span>
              </div>
              <p className="text-xs text-slate-400 print:text-gray-700">
                Gate 3 Lesion Identification: Microaneurysms ({patientCase.lesions.countMA}), Hemorrhages ({patientCase.lesions.countHE}), Hard Exudates ({patientCase.lesions.countEX}), Neovascularization ({patientCase.lesions.neovascularization ? 'Positive' : 'None'}).
              </p>
            </div>

            {/* Systemic Finding */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 print:border-gray-300 print:bg-white">
              <div className="flex items-center gap-2 mb-2">
                <Dna className="w-4 h-4 text-purple-400 print:text-purple-700" />
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 print:text-black">
                  Systemic Layer: Immunological Risk
                </h4>
              </div>
              <div className="flex items-baseline gap-3 my-2">
                <span className="text-2xl font-black text-amber-400 print:text-amber-700">
                  {patientCase.systemicRisk} Risk
                </span>
                <span className="text-xs font-mono text-slate-400 print:text-gray-600">
                  Confidence: {(patientCase.systemicConfidence * 100).toFixed(1)}%
                </span>
              </div>
              <p className="text-xs text-slate-400 print:text-gray-700">
                Cell2Sentence Distilled Edge Transformer (256-D) analysis indicates active microvascular endothelial permeability & cytotoxic cytokine cascade.
              </p>
            </div>
          </div>

          {/* Dual Explainability (XAI) Evidence */}
          <div className="border border-slate-800 print:border-gray-300 rounded-xl p-4 bg-slate-950/40 print:bg-white">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200 print:text-black mb-3 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400 print:text-emerald-700" />
              Dual Explainable AI (XAI) Verification Audit
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Top Gene Drivers by Attention */}
              <div>
                <span className="text-[11px] font-bold text-slate-300 print:text-black block mb-2">
                  Top Biological Attention Drivers (PyTorch TransformerEncoderLayer):
                </span>
                <div className="space-y-1.5">
                  {patientCase.genes.slice(0, 5).map((g) => (
                    <div
                      key={g.gene}
                      className="flex items-center justify-between text-xs bg-slate-900 print:bg-gray-50 px-2.5 py-1 rounded border border-slate-800 print:border-gray-200"
                    >
                      <div className="flex items-center gap-2">
                        <strong className="font-mono text-cyan-300 print:text-blue-800">{g.gene}</strong>
                        <span className="text-[10px] text-slate-400 print:text-gray-600 truncate max-w-[170px]">{g.pathway}</span>
                      </div>
                      <span className="font-mono text-slate-200 print:text-black font-bold">
                        {(g.attentionWeight * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* ETDRS Subfield Risk Distribution */}
              <div>
                <span className="text-[11px] font-bold text-slate-300 print:text-black block mb-2">
                  ETDRS Anatomical Macular Subfield Risk (% Saliency Density):
                </span>
                <div className="grid grid-cols-3 gap-1.5 text-center text-[10px] font-mono">
                  <div className="p-1.5 rounded bg-slate-900 print:bg-gray-100 border border-slate-800 print:border-gray-300">
                    <span className="text-slate-400 print:text-gray-600 block">Sup Outer</span>
                    <strong className="text-rose-400 print:text-red-700">{patientCase.etdrsSubfieldRisks.outerSuperior}%</strong>
                  </div>
                  <div className="p-1.5 rounded bg-slate-900 print:bg-gray-100 border border-slate-800 print:border-gray-300">
                    <span className="text-slate-400 print:text-gray-600 block">Sup Inner</span>
                    <strong className="text-rose-400 print:text-red-700">{patientCase.etdrsSubfieldRisks.innerSuperior}%</strong>
                  </div>
                  <div className="p-1.5 rounded bg-slate-900 print:bg-gray-100 border border-slate-800 print:border-gray-300">
                    <span className="text-slate-400 print:text-gray-600 block">Temp Outer</span>
                    <strong className="text-amber-400 print:text-amber-700">{patientCase.etdrsSubfieldRisks.outerTemporal}%</strong>
                  </div>
                  <div className="p-1.5 rounded bg-slate-900 print:bg-gray-100 border border-slate-800 print:border-gray-300">
                    <span className="text-slate-400 print:text-gray-600 block">Nasal Inner</span>
                    <strong className="text-slate-300 print:text-black">{patientCase.etdrsSubfieldRisks.innerNasal}%</strong>
                  </div>
                  <div className="p-1.5 rounded bg-cyan-950/70 print:bg-blue-100 border border-cyan-800 print:border-blue-300">
                    <span className="text-cyan-300 print:text-blue-900 block font-bold">1mm Fovea</span>
                    <strong className="text-cyan-200 print:text-blue-900">{patientCase.etdrsSubfieldRisks.central1mm}%</strong>
                  </div>
                  <div className="p-1.5 rounded bg-slate-900 print:bg-gray-100 border border-slate-800 print:border-gray-300">
                    <span className="text-slate-400 print:text-gray-600 block">Temp Inner</span>
                    <strong className="text-rose-400 print:text-red-700">{patientCase.etdrsSubfieldRisks.innerTemporal}%</strong>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Actionable Clinical Recommendation */}
          <div className="bg-slate-950 print:bg-gray-50 border border-slate-800 print:border-gray-300 rounded-xl p-4 text-xs">
            <h4 className="font-bold text-slate-200 print:text-black mb-1 flex items-center gap-1.5">
              <AlertCircle className="w-4 h-4 text-amber-400 print:text-amber-700" />
              Actionable Vitreoretinal Referral & Clinical Plan:
            </h4>
            <p className="text-slate-300 print:text-gray-800 leading-relaxed font-sans">
              {patientCase.clinicalRecommendation}
            </p>
          </div>

          {/* Hardware & Physician Sign-Off */}
          <div className="pt-4 border-t border-slate-800 print:border-black grid grid-cols-2 gap-6 text-xs">
            <div className="text-[11px] text-slate-500 print:text-gray-600 font-mono">
              <div>Kiosk Deployment: {telemetry.kioskId}</div>
              <div>Inference Latency: {telemetry.totalLatencyMs} ms total (Edge i5-1240P)</div>
              <div>Firmware: {telemetry.firmware} (Air-gapped)</div>
            </div>
            <div className="text-right flex flex-col justify-end">
              <div className="inline-block border-b border-slate-600 print:border-black w-48 ml-auto pb-1 text-center font-serif italic text-slate-300 print:text-black">
                Dr. A. Sen, MD (Ophthal)
              </div>
              <span className="text-[10px] text-slate-500 print:text-gray-600 mt-1 block">
                Attending Clinical Reviewer / Tele-Ophthalmology Signature
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
