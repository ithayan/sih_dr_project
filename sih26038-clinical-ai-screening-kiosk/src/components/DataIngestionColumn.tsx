import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  Dna, 
  Camera, 
  Cpu, 
  CheckCircle2, 
  Play, 
  RefreshCw, 
  Sliders, 
  Database,
  FileCode,
  Zap,
  Check,
  AlertCircle
} from 'lucide-react';
import { PatientCase, InferenceStep, KioskTelemetry } from '../types';

interface DataIngestionColumnProps {
  currentCase: PatientCase;
  inferenceSteps: InferenceStep[];
  isInferring: boolean;
  onRunInference: () => void;
  onUpdateGeneSentence: (text: string) => void;
  onCustomImageUpload: (dataUrl: string, fileName: string) => void;
  telemetry: KioskTelemetry;
}

export const DataIngestionColumn: React.FC<DataIngestionColumnProps> = ({
  currentCase,
  inferenceSteps,
  isInferring,
  onRunInference,
  onUpdateGeneSentence,
  onCustomImageUpload,
  telemetry,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'text' | 'tokens'>('tokens');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      processFile(file);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file: File) => {
    setUploadedFileName(file.name);
    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target?.result) {
        onCustomImageUpload(event.target.result as string, file.name);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleGeneMatrixFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        if (text) {
          // Clean CSV/TSV or lines into token sentence
          const tokens = text
            .replace(/[,\t\r\n]+/g, ' ')
            .trim();
          onUpdateGeneSentence(tokens);
        }
      };
      reader.readAsText(file);
    }
  };

  const tokenList = currentCase.rawGeneSentence.split(/\s+/).filter(Boolean);

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* SECTION 1: VISION INPUT */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5 shadow-sm">
        <div className="flex items-center justify-between mb-2.5">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <Camera className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">Vision Input</h2>
              <p className="text-[11px] text-slate-400">Portable Fundus Camera Image</p>
            </div>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-blue-300 border border-slate-700">
            {currentCase.eye}
          </span>
        </div>

        {/* Drag and Drop Zone */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-lg p-3 text-center cursor-pointer transition-all duration-200 ${
            dragActive
              ? 'border-cyan-400 bg-cyan-950/20'
              : 'border-slate-700 hover:border-slate-500 bg-slate-950/60'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileInputChange}
          />
          <div className="flex flex-col items-center justify-center gap-1.5 py-1">
            <UploadCloud className="w-6 h-6 text-cyan-400" />
            <p className="text-xs font-medium text-slate-200">
              {uploadedFileName ? (
                <span className="text-cyan-300 font-semibold">{uploadedFileName}</span>
              ) : (
                <>Drop Retinal Image here or <span className="text-cyan-400 underline">Browse</span></>
              )}
            </p>
            <p className="text-[10px] text-slate-400">
              TIFF, PNG, JPG from Remidio, Volk, Zeiss (45° Non-Mydriatic)
            </p>
          </div>
        </div>

        {/* Image Metadata Strip */}
        <div className="mt-2.5 grid grid-cols-2 gap-2 text-[11px] bg-slate-950/50 p-2 rounded-lg border border-slate-800/80">
          <div>
            <span className="text-slate-500 block text-[10px]">Camera System</span>
            <span className="text-slate-300 font-medium truncate block">{currentCase.cameraModel}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">Quality Gate 1</span>
            <span className="text-emerald-400 font-medium flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3" />
              {currentCase.imageQuality}
            </span>
          </div>
        </div>
      </div>

      {/* SECTION 2: SYSTEMIC INPUT */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5 shadow-sm">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Dna className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">Systemic Input</h2>
              <p className="text-[11px] text-slate-400">Single-Cell Gene Expression Matrix</p>
            </div>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
            C2S Top 128/256
          </span>
        </div>

        {/* Tab switcher: Tokens vs Text Input */}
        <div className="flex items-center justify-between text-xs mb-2">
          <div className="flex bg-slate-950 p-0.5 rounded-md border border-slate-800 text-[11px]">
            <button
              onClick={() => setActiveTab('tokens')}
              className={`px-2 py-0.5 rounded transition ${
                activeTab === 'tokens' ? 'bg-slate-800 text-white font-medium' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Token View ({tokenList.length})
            </button>
            <button
              onClick={() => setActiveTab('text')}
              className={`px-2 py-0.5 rounded transition ${
                activeTab === 'text' ? 'bg-slate-800 text-white font-medium' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Raw Edit
            </button>
          </div>

          <label className="text-[11px] text-cyan-400 hover:text-cyan-300 cursor-pointer flex items-center gap-1">
            <FileCode className="w-3 h-3" />
            <span>Load Matrix File</span>
            <input 
              type="file" 
              accept=".csv,.tsv,.txt" 
              className="hidden" 
              onChange={handleGeneMatrixFileUpload} 
            />
          </label>
        </div>

        {activeTab === 'tokens' ? (
          <div className="bg-slate-950/80 p-2 rounded-lg border border-slate-800 max-h-36 overflow-y-auto">
            <div className="flex flex-wrap gap-1">
              {tokenList.map((token, idx) => {
                const isHighDriver = idx < 4;
                const isMedDriver = idx >= 4 && idx < 8;
                return (
                  <span
                    key={`${token}-${idx}`}
                    className={`text-[10px] font-mono px-1.5 py-0.5 rounded border transition-colors ${
                      isHighDriver
                        ? 'bg-rose-950/70 border-rose-800/80 text-rose-300 font-bold'
                        : isMedDriver
                        ? 'bg-amber-950/50 border-amber-800/60 text-amber-300'
                        : 'bg-slate-900 border-slate-800 text-slate-300'
                    }`}
                    title={`Rank #${idx + 1}: ${token}`}
                  >
                    #{idx + 1} {token}
                  </span>
                );
              })}
            </div>
          </div>
        ) : (
          <div>
            <textarea
              value={currentCase.rawGeneSentence}
              onChange={(e) => onUpdateGeneSentence(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs font-mono text-slate-300 focus:outline-none focus:border-cyan-500 h-28 resize-none"
              placeholder="Enter rank-ordered gene tokens separated by space or commas (e.g. VEGFA IFNG IL6 CTLA4 PDCD1...)"
            />
            <p className="text-[10px] text-slate-500 mt-1">
              Cell2Sentence tokenization sorts single-cell counts into textual ranking.
            </p>
          </div>
        )}
      </div>

      {/* SECTION 3: PROCESSING STATUS (DYNAMIC STEPPER) */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5 shadow-sm flex-1 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                <Cpu className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">Processing Pipeline</h2>
                <p className="text-[11px] text-slate-400">Multi-Modal Edge Inference</p>
              </div>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
              ~{telemetry.totalLatencyMs} ms
            </span>
          </div>

          {/* Stepper list */}
          <div className="space-y-2.5">
            {inferenceSteps.map((step, index) => {
              const isDone = step.status === 'completed';
              const isRunning = step.status === 'running';

              return (
                <div
                  key={step.id}
                  className={`p-2.5 rounded-lg border transition-all ${
                    isRunning
                      ? 'bg-cyan-950/30 border-cyan-500/50 shadow-sm shadow-cyan-950'
                      : isDone
                      ? 'bg-slate-950/60 border-slate-800/80'
                      : 'bg-slate-950/30 border-slate-800/40 opacity-60'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                          isDone
                            ? 'bg-emerald-500 text-white'
                            : isRunning
                            ? 'bg-cyan-500 text-white animate-spin'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        {isDone ? (
                          <Check className="w-3 h-3" />
                        ) : isRunning ? (
                          <RefreshCw className="w-3 h-3" />
                        ) : (
                          index + 1
                        )}
                      </div>
                      <div>
                        <span className="text-xs font-bold text-slate-200 block">
                          {step.name}
                        </span>
                        <span className="text-[10px] text-slate-400 font-mono">
                          {step.architecture}
                        </span>
                      </div>
                    </div>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-900 text-cyan-300 border border-slate-800">
                      {step.dimensions}
                    </span>
                  </div>

                  {/* Latency and detail */}
                  <div className="mt-1.5 pt-1.5 border-t border-slate-800/60 flex items-center justify-between text-[10px] text-slate-400">
                    <span className="text-slate-500">{step.details}</span>
                    <span className="font-mono text-emerald-400">{step.latencyMs} ms</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Execute Button */}
        <div className="mt-4 pt-3 border-t border-slate-800">
          <button
            onClick={onRunInference}
            disabled={isInferring}
            className={`w-full py-2.5 px-3 rounded-lg font-bold text-xs flex items-center justify-center gap-2 transition-all shadow-md ${
              isInferring
                ? 'bg-slate-800 text-slate-400 cursor-not-allowed border border-slate-700'
                : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-cyan-950/50 hover:shadow-cyan-900/70 border border-cyan-500/30'
            }`}
          >
            {isInferring ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />
                <span>Executing Multi-Modal Fusion...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 text-cyan-300" />
                <span>Run Hierarchical Inference</span>
              </>
            )}
          </button>

          {/* Air-gapped edge reassurance note */}
          <div className="mt-2 flex items-center justify-between text-[10px] text-slate-500 font-mono">
            <span>RAM: {telemetry.ramUsage.split('(')[0]}</span>
            <span className="text-emerald-500/90 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
              100% Offline Edge Mode
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
