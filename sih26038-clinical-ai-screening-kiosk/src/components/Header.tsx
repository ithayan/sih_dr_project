import React from 'react';
import { 
  Activity, 
  Cpu, 
  ShieldCheck, 
  Eye, 
  FileText, 
  Info, 
  RotateCcw,
  Sparkles,
  Server,
  Building2,
  UploadCloud,
  Layers,
  UserCheck
} from 'lucide-react';
import { KioskTelemetry, PatientCase } from '../types';

export type KioskViewMode = 'workstation' | 'operations' | 'sync';

interface HeaderProps {
  telemetry: KioskTelemetry;
  selectedCase: PatientCase;
  sampleCases: PatientCase[];
  serverConnected?: boolean;
  serverEngine?: string;
  currentView: KioskViewMode;
  onSelectView: (view: KioskViewMode) => void;
  onSelectCase: (c: PatientCase) => void;
  onOpenReport: () => void;
  onOpenArchInfo: () => void;
  onResetInference: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  telemetry,
  selectedCase,
  sampleCases,
  serverConnected = false,
  serverEngine,
  currentView,
  onSelectView,
  onSelectCase,
  onOpenReport,
  onOpenArchInfo,
  onResetInference,
}) => {
  return (
    <header className="bg-slate-900/95 border-b border-slate-800 sticky top-0 z-30 backdrop-blur-md px-4 py-2.5">
      <div className="max-w-[1920px] mx-auto flex flex-wrap items-center justify-between gap-3">
        {/* Left: Project Branding & Status */}
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 text-white font-bold shadow-md shadow-cyan-950/50">
            <Eye className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800 tracking-wider">
                SIH26038
              </span>
              <h1 className="text-sm md:text-base font-bold text-slate-100 tracking-tight flex items-center gap-1.5">
                NetraX Telemedicine Hub
                <span className="text-xs font-normal text-slate-400 hidden sm:inline">
                  • Edge-to-Hub Architecture (YOLO-GNN-KAN)
                </span>
              </h1>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-400 mt-0.5">
              {serverConnected ? (
                <span className="flex items-center gap-1.5 text-emerald-400 font-medium bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/80">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                  AI SERVER ONLINE (PORT 8080)
                </span>
              ) : (
                <span className="flex items-center gap-1.5 text-amber-400 font-medium bg-amber-950/50 px-2 py-0.5 rounded border border-amber-800/60">
                  <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                  STANDALONE EDGE MODE
                </span>
              )}
              <span className="text-slate-600">•</span>
              <span className="hidden md:inline font-mono text-[11px] text-slate-300">
                {telemetry.cpu.split('(')[0]}
              </span>
              <span className="text-slate-600 hidden md:inline">•</span>
              <span className="hidden lg:inline text-[11px] text-slate-400">
                Kiosk: <strong className="text-slate-300 font-mono">{telemetry.kioskId}</strong>
              </span>
            </div>
          </div>
        </div>

        {/* Center: Primary Edge-to-Hub View Navigation Tabs */}
        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => onSelectView('workstation')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              currentView === 'workstation'
                ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-md shadow-cyan-950/50'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>Hub Diagnostic Workstation</span>
          </button>

          <button
            onClick={() => onSelectView('operations')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              currentView === 'operations'
                ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-md shadow-indigo-950/50'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Building2 className="w-3.5 h-3.5" />
            <span>District Operations (M/M/c)</span>
          </button>

          <button
            onClick={() => onSelectView('sync')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              currentView === 'sync'
                ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-md shadow-emerald-950/50'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <UploadCloud className="w-3.5 h-3.5" />
            <span>Edge-to-Hub Sync</span>
          </button>
        </div>

        {/* Right: Specialist Badge & Actions */}
        <div className="flex items-center gap-2">
          {/* Doctor On Duty Badge */}
          <div className="hidden xl:flex items-center gap-2 bg-slate-950/80 px-3 py-1.5 rounded-lg border border-slate-800">
            <UserCheck className="w-4 h-4 text-emerald-400" />
            <div className="text-left">
              <span className="text-[11px] font-bold text-slate-200 block leading-tight">Dr. Arvind Sharma</span>
              <span className="text-[9px] text-slate-400 font-mono block leading-tight">Senior Vitreoretinal Consultant</span>
            </div>
          </div>

          <button
            onClick={onOpenReport}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold transition-all border border-slate-700"
            title="Download/Print Clinical Dossier"
          >
            <FileText className="w-3.5 h-3.5 text-cyan-400" />
            <span className="hidden sm:inline">Dossier</span>
          </button>

          <button
            onClick={onOpenArchInfo}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold transition-all border border-slate-700"
            title="Swin-GNN-KAN Architectural Specs"
          >
            <Info className="w-3.5 h-3.5 text-indigo-400" />
            <span className="hidden sm:inline">Architecture</span>
          </button>
        </div>
      </div>
    </header>
  );
};
