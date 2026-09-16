import React, { useState, useRef } from 'react';
import { 
  Eye, 
  Layers, 
  ZoomIn, 
  ZoomOut, 
  Maximize2, 
  Target, 
  Sliders, 
  Activity,
  Crosshair,
  Sparkles,
  Info
} from 'lucide-react';
import { PatientCase, GradCAMHotspot } from '../types';

interface RetinalVisualizerColumnProps {
  currentCase: PatientCase;
  customImageSrc?: string | null;
}

export const RetinalVisualizerColumn: React.FC<RetinalVisualizerColumnProps> = ({
  currentCase,
  customImageSrc,
}) => {
  const [showGradCam, setShowGradCam] = useState<boolean>(true);
  const [showYoloBoxes, setShowYoloBoxes] = useState<boolean>(true);
  const [showEtdrsGrid, setShowEtdrsGrid] = useState<boolean>(true);
  const [showVessels, setShowVessels] = useState<boolean>(false);
  const [showLesionTags, setShowLesionTags] = useState<boolean>(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.65);
  const [zoomLevel, setZoomLevel] = useState<number>(1.0);
  const [hoverCoord, setHoverCoord] = useState<{ x: number; y: number } | null>(null);
  const [activeHotspot, setActiveHotspot] = useState<GradCAMHotspot | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * 100);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * 100);
    setHoverCoord({ x: Math.max(0, Math.min(100, x)), y: Math.max(0, Math.min(100, y)) });

    // Check if hovering near a hotspot
    const found = currentCase.hotspots.find(
      (h) => Math.hypot(h.x - x, h.y - y) <= h.radius * 0.9
    );
    setActiveHotspot(found || null);
  };

  const handleMouseLeave = () => {
    setHoverCoord(null);
    setActiveHotspot(null);
  };

  const fovea = currentCase.foveaCoordinates;
  const disc = currentCase.opticDiscCoordinates;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-sm flex flex-col h-full">
      {/* HEADER: Title & Quick Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3 pb-3 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
            <h2 className="text-sm font-bold text-slate-100 tracking-tight">
              Vision Layer: YOLO26 Lesion Masks & Vascular Manifold
            </h2>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
              YOLO26 Nano (OpenVINO INT8) • 256-D
            </span>
          </div>
          <p className="text-[11px] text-slate-400 mt-0.5">
            One-to-one Hungarian label assignment (NMS-free) with ETDRS rings & GAT vessel topology
          </p>
        </div>

        {/* PRIMARY TOGGLE: Overlay Grad-CAM Heatmap */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowGradCam(!showGradCam)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold transition-all shadow-md ${
              showGradCam
                ? 'bg-gradient-to-r from-rose-600 via-amber-500 to-yellow-500 text-slate-950 font-extrabold shadow-rose-950/40 ring-1 ring-amber-300'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Grad-CAM Saliency</span>
            <span
              className={`text-[9px] px-1.5 py-0.2 rounded font-mono font-bold ${
                showGradCam ? 'bg-black/80 text-amber-300' : 'bg-slate-900 text-slate-400'
              }`}
            >
              {showGradCam ? 'ACTIVE' : 'OFF'}
            </span>
          </button>
        </div>
      </div>

      {/* SECONDARY TOOLBAR: ETDRS, YOLO Boxes, Vessels, Opacity & Zoom */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs bg-slate-950/70 p-2 rounded-lg border border-slate-800 mb-3">
        {/* Layer Checkboxes */}
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 cursor-pointer text-slate-300 hover:text-slate-100">
            <input
              type="checkbox"
              checked={showYoloBoxes}
              onChange={(e) => setShowYoloBoxes(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-emerald-500 focus:ring-0"
            />
            <span className="text-[11px] font-medium text-emerald-400 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
              YOLO26 Boxes (NMS-Free)
            </span>
          </label>

          <label className="flex items-center gap-1.5 cursor-pointer text-slate-300 hover:text-slate-100">
            <input
              type="checkbox"
              checked={showEtdrsGrid}
              onChange={(e) => setShowEtdrsGrid(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0"
            />
            <span className="text-[11px] font-medium">ETDRS Grid (1, 3, 6mm)</span>
          </label>

          <label className="flex items-center gap-1.5 cursor-pointer text-slate-300 hover:text-slate-100">
            <input
              type="checkbox"
              checked={showVessels}
              onChange={(e) => setShowVessels(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0"
            />
            <span className="text-[11px] font-medium text-cyan-400 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
              Neural & Vascular Tree
            </span>
          </label>
        </div>

        {/* Heatmap Opacity & Zoom Controls */}
        <div className="flex items-center gap-4">
          {showGradCam && (
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-slate-400">Opacity:</span>
              <input
                type="range"
                min="0.2"
                max="0.95"
                step="0.05"
                value={heatmapOpacity}
                onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                className="w-16 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-400"
              />
              <span className="text-[10px] font-mono text-amber-400 w-7">
                {Math.round(heatmapOpacity * 100)}%
              </span>
            </div>
          )}

          <div className="flex items-center bg-slate-900 rounded border border-slate-800 p-0.5">
            <button
              onClick={() => setZoomLevel((z) => Math.max(0.8, +(z - 0.2).toFixed(1)))}
              className="p-1 hover:bg-slate-800 text-slate-400 hover:text-slate-200 rounded"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-[10px] font-mono px-1.5 text-slate-300">
              {Math.round(zoomLevel * 100)}%
            </span>
            <button
              onClick={() => setZoomLevel((z) => Math.min(2.0, +(z + 0.2).toFixed(1)))}
              className="p-1 hover:bg-slate-800 text-slate-400 hover:text-slate-200 rounded"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setZoomLevel(1.0)}
              className="p-1 hover:bg-slate-800 text-slate-400 hover:text-slate-200 rounded ml-1"
              title="Reset Zoom"
            >
              <Maximize2 className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>

      {/* MAIN RETINAL FUNDUS CANVAS CONTAINER */}
      <div 
        className="relative flex-1 bg-black rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center select-none cursor-crosshair min-h-[440px]"
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <div 
          className="relative w-full h-full flex items-center justify-center transition-transform duration-100"
          style={{ transform: `scale(${zoomLevel})` }}
        >
          {/* Custom Uploaded Image Mode or Procedural Medical Retina Canvas */}
          {customImageSrc ? (
            <img
              src={customImageSrc}
              alt="Uploaded Fundus"
              className="w-full h-full object-contain max-h-[560px] pointer-events-none"
            />
          ) : (
            <svg
              viewBox="0 0 800 800"
              className="w-full h-full max-h-[580px] drop-shadow-2xl"
              preserveAspectRatio="xMidYMid meet"
            >
              <defs>
                {/* Retinal Fundus Gradient (Deep red-orange with peripheral vignetting) */}
                <radialGradient id="retinaFundusBg" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#d34e1e" />
                  <stop offset="65%" stopColor="#a8310f" />
                  <stop offset="85%" stopColor="#5f1305" />
                  <stop offset="100%" stopColor="#240502" />
                </radialGradient>

                {/* Optic Disc Gradient */}
                <radialGradient id="opticDiscGrad" cx="45%" cy="45%" r="50%">
                  <stop offset="0%" stopColor="#ffebb5" />
                  <stop offset="60%" stopColor="#fca570" />
                  <stop offset="100%" stopColor="#d95326" />
                </radialGradient>

                {/* Macular Lutea Avascular Zone */}
                <radialGradient id="foveaGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#450a0a" stopOpacity="0.8" />
                  <stop offset="60%" stopColor="#78190c" stopOpacity="0.4" />
                  <stop offset="100%" stopColor="#a8310f" stopOpacity="0" />
                </radialGradient>

                {/* Filter for Grad-CAM thermal blur */}
                <filter id="gradCamBlur" x="-30%" y="-30%" width="160%" height="160%">
                  <feGaussianBlur stdDeviation="18" result="blur" />
                  <feColorMatrix
                    type="matrix"
                    values="1 0 0 0 0
                            0 1 0 0 0
                            0 0 1 0 0
                            0 0 0 1.2 0"
                  />
                </filter>
              </defs>

              {/* Fundus Background Sphere */}
              <circle cx="400" cy="400" r="380" fill="url(#retinaFundusBg)" />

              {/* Micro-choroidal background texture lines */}
              <g opacity="0.08" stroke="#ff8f66" strokeWidth="1">
                {Array.from({ length: 18 }).map((_, i) => (
                  <path
                    key={`choroid-${i}`}
                    d={`M 150,${200 + i * 25} Q 400,${250 + i * 20} 650,${220 + i * 25}`}
                    fill="none"
                  />
                ))}
              </g>

              {/* Optic Disc (Presents at ~26% X, 48% Y for OD) */}
              <g id="optic-disc-landmark">
                {/* Scleral Ring */}
                <ellipse
                  cx={disc.x * 8}
                  cy={disc.y * 8}
                  rx="46"
                  ry="54"
                  fill="#ffdb99"
                  opacity="0.95"
                  filter="drop-shadow(0 0 8px rgba(255,180,100,0.5))"
                />
                {/* Physiological Cup */}
                <ellipse
                  cx={disc.x * 8 - 4}
                  cy={disc.y * 8}
                  rx="22"
                  ry="26"
                  fill="#fff7d6"
                  opacity="0.9"
                />
                {/* Central Retinal Artery / Vein trunk emergence */}
                <circle cx={disc.x * 8 - 4} cy={disc.y * 8} r="8" fill="#500707" />
              </g>

              {/* Macula & Fovea Centralis */}
              <g id="macula-fovea-landmark">
                <circle
                  cx={fovea.x * 8}
                  cy={fovea.y * 8}
                  r="70"
                  fill="url(#foveaGlow)"
                />
                {/* Foveola Pinpoint Umbo */}
                <circle
                  cx={fovea.x * 8}
                  cy={fovea.y * 8}
                  r="4"
                  fill="#ffedd5"
                  opacity="0.8"
                />
              </g>

              {/* Retinal Vascular Tree (Arterioles and Venules) */}
              <g id="retinal-vascular-tree" strokeLinecap="round">
                {/* Superior Temporal Arcade (Thick venule + arteriole) */}
                <path
                  d={`M ${disc.x * 8},${disc.y * 8} C 250,220 380,140 520,160 S 680,240 710,320`}
                  fill="none"
                  stroke="#4a0404"
                  strokeWidth={showVessels ? '7' : '5'}
                  opacity={showVessels ? '0.95' : '0.8'}
                />
                <path
                  d={`M ${disc.x * 8},${disc.y * 8} C 260,235 375,160 500,175 S 650,255 690,335`}
                  fill="none"
                  stroke="#bd1c1c"
                  strokeWidth={showVessels ? '4' : '2.5'}
                  opacity={showVessels ? '0.95' : '0.75'}
                />

                {/* Inferior Temporal Arcade */}
                <path
                  d={`M ${disc.x * 8},${disc.y * 8} C 240,560 380,660 540,640 S 680,560 720,460`}
                  fill="none"
                  stroke="#4a0404"
                  strokeWidth={showVessels ? '7' : '5'}
                  opacity={showVessels ? '0.95' : '0.8'}
                />
                <path
                  d={`M ${disc.x * 8},${disc.y * 8} C 250,545 375,640 520,620 S 650,545 695,450`}
                  fill="none"
                  stroke="#bd1c1c"
                  strokeWidth={showVessels ? '4' : '2.5'}
                  opacity={showVessels ? '0.95' : '0.75'}
                />

                {/* Nasal Arcades (Branching to the left side) */}
                <path
                  d={`M ${disc.x * 8},${disc.y * 8} C 160,320 110,280 60,260`}
                  fill="none"
                  stroke="#4a0404"
                  strokeWidth="3.5"
                  opacity="0.75"
                />
                <path
                  d={`M ${disc.x * 8},${disc.y * 8} C 160,460 110,520 70,550`}
                  fill="none"
                  stroke="#5a0808"
                  strokeWidth="3.5"
                  opacity="0.75"
                />

                {/* Macular and paramacular terminal capillary branches */}
                <path
                  d={`M 400,170 Q 430,280 ${fovea.x * 8 - 40},${fovea.y * 8 - 30}`}
                  fill="none"
                  stroke="#7f1212"
                  strokeWidth="1.8"
                  opacity="0.65"
                />
                <path
                  d={`M 420,630 Q 440,520 ${fovea.x * 8 - 30},${fovea.y * 8 + 35}`}
                  fill="none"
                  stroke="#7f1212"
                  strokeWidth="1.8"
                  opacity="0.65"
                />
                <path
                  d={`M 530,165 Q 560,300 ${fovea.x * 8 + 45},${fovea.y * 8 - 25}`}
                  fill="none"
                  stroke="#7f1212"
                  strokeWidth="1.8"
                  opacity="0.65"
                />
                <path
                  d={`M 540,635 Q 550,510 ${fovea.x * 8 + 45},${fovea.y * 8 + 30}`}
                  fill="none"
                  stroke="#7f1212"
                  strokeWidth="1.8"
                  opacity="0.65"
                />
              </g>

              {/* PATHOLOGICAL LESIONS (GROUND TRUTH / CLINICAL ANATOMY) */}
              <g id="pathology-lesions">
                {/* 1. Microaneurysms (Red pinpoint capillary swellings) */}
                {currentCase.lesions.microaneurysms && (
                  <g fill="#990000" stroke="#ff3b30" strokeWidth="0.8">
                    <circle cx="490" cy="350" r="3.2" />
                    <circle cx="510" cy="380" r="2.8" />
                    <circle cx="540" cy="420" r="3.5" />
                    <circle cx="420" cy="490" r="3.0" />
                    <circle cx="590" cy="430" r="2.5" />
                    <circle cx="530" cy="500" r="3.8" />
                    <circle cx="610" cy="320" r="3.0" />
                  </g>
                )}

                {/* 2. Retinal Hemorrhages (Blot and flame hemorrhages) */}
                {currentCase.lesions.hemorrhages && (
                  <g fill="#5c0000" stroke="#7a0000">
                    {/* Flame hemorrhage along nerve fiber */}
                    <path d="M 480,270 Q 520,290 550,305 Q 510,300 480,270" />
                    {/* Dot and blot hemorrhages */}
                    <ellipse cx="505" cy="345" rx="8" ry="6" />
                    <ellipse cx="585" cy="445" rx="10" ry="7" />
                    <ellipse cx="430" cy="510" rx="7" ry="5" />
                    <ellipse cx="610" cy="315" rx="9" ry="6" />
                  </g>
                )}

                {/* 3. Hard Exudates (Lipid leakage - Bright yellow crystalline deposits) */}
                {currentCase.lesions.hardExudates && (
                  <g fill="#fff59d" stroke="#fbc02d" strokeWidth="0.5">
                    <circle cx="535" cy="440" r="2.5" />
                    <circle cx="539" cy="443" r="2.0" />
                    <circle cx="543" cy="438" r="3.0" />
                    <circle cx="547" cy="442" r="2.2" />
                    <circle cx="552" cy="439" r="2.8" />
                    <circle cx="538" cy="448" r="2.4" />
                    {/* Circinate ring pattern */}
                    <circle cx="525" cy="435" r="2.2" />
                    <circle cx="530" cy="430" r="2.6" />
                    <circle cx="545" cy="428" r="2.0" />
                  </g>
                )}

                {/* 4. Cotton Wool Spots (Soft exudates / Axoplasmic stasis) */}
                {currentCase.lesions.cottonWoolSpots && (
                  <g fill="#ffffff" opacity="0.85" filter="drop-shadow(0 0 4px white)">
                    <ellipse cx="510" cy="330" rx="14" ry="10" />
                    <ellipse cx="580" cy="460" rx="16" ry="11" />
                  </g>
                )}

                {/* 5. Neovascularization Fronds (PDR severe sign) */}
                {currentCase.lesions.neovascularization && (
                  <g stroke="#e11d48" strokeWidth="1.8" fill="none" opacity="0.9">
                    {/* NVD (Neovascularization of Disc) */}
                    <path d={`M ${disc.x * 8 + 10},${disc.y * 8 - 20} q 10,-15 25,-10 t 15,-15 t -10,-10`} />
                    <path d={`M ${disc.x * 8 + 20},${disc.y * 8 - 10} q 15,-5 20,15 t 15,5`} />
                    {/* NVE (Neovascularization elsewhere) */}
                    <path d="M 590,440 q 15,-20 30,-5 t 20,15 t -10,20" />
                  </g>
                )}
              </g>

              {/* NEURAL & VASCULAR ARBORIZATION CONTINUITY MASK (When active) */}
              {showVessels && (
                currentCase.vesselTreeMaskBase64 ? (
                  <image
                    href={currentCase.vesselTreeMaskBase64.startsWith('data:') ? currentCase.vesselTreeMaskBase64 : `data:image/png;base64,${currentCase.vesselTreeMaskBase64}`}
                    x="0"
                    y="0"
                    width="800"
                    height="800"
                    preserveAspectRatio="none"
                    opacity={0.88}
                    style={{ mixBlendMode: 'screen', filter: 'drop-shadow(0 0 4px #00e5ff)' }}
                  />
                ) : (
                  <g id="geo-fno-mask" opacity="0.65" stroke="#38bdf8" strokeWidth="6" fill="none" filter="drop-shadow(0 0 6px #0284c7)">
                    <path d={`M ${disc.x * 8},${disc.y * 8} C 250,220 380,140 520,160 S 680,240 710,320`} />
                    <path d={`M ${disc.x * 8},${disc.y * 8} C 240,560 380,660 540,640 S 680,560 720,460`} />
                    <circle cx={disc.x * 8} cy={disc.y * 8} r="30" stroke="#0284c7" strokeWidth="4" />
                  </g>
                )
              )}
            </svg>
          )}

          {/* GRAD-CAM THERMAL SALIENCY OVERLAY */}
          {showGradCam && (
            <div 
              className="absolute inset-0 pointer-events-none transition-opacity duration-300"
              style={{ opacity: heatmapOpacity }}
            >
              <svg 
                viewBox="0 0 100 100" 
                className="w-full h-full"
                preserveAspectRatio="xMidYMid meet"
              >
                <defs>
                  {/* Radial thermal gradients for each hotspot */}
                  {currentCase.hotspots.map((h) => (
                    <radialGradient key={`grad-${h.id}`} id={`grad-${h.id}`} cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#dc2626" stopOpacity={h.intensity} />
                      <stop offset="35%" stopColor="#ea580c" stopOpacity={h.intensity * 0.9} />
                      <stop offset="65%" stopColor="#eab308" stopOpacity={h.intensity * 0.75} />
                      <stop offset="85%" stopColor="#06b6d4" stopOpacity={h.intensity * 0.35} />
                      <stop offset="100%" stopColor="#0284c7" stopOpacity="0" />
                    </radialGradient>
                  ))}
                </defs>

                {/* Render Salient Hotspot Gaussians */}
                {currentCase.hotspots.map((h) => (
                  <circle
                    key={`cam-blob-${h.id}`}
                    cx={h.x}
                    cy={h.y}
                    r={h.radius}
                    fill={`url(#grad-${h.id})`}
                    style={{ mixBlendMode: 'screen' }}
                  />
                ))}
              </svg>
            </div>
          )}

          {/* ETDRS GRID OVERLAY (Standard 1mm, 3mm, 6mm rings centered on Fovea) */}
          {showEtdrsGrid && (
            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
              <svg 
                viewBox="0 0 100 100" 
                className="w-full h-full"
                preserveAspectRatio="xMidYMid meet"
              >
                {/* Central Foveal Subfield (1mm) */}
                <circle
                  cx={fovea.x}
                  cy={fovea.y}
                  r="6.5"
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="0.4"
                  strokeDasharray="1.5, 0.8"
                  opacity="0.85"
                />

                {/* Inner Macular Ring (3mm) */}
                <circle
                  cx={fovea.x}
                  cy={fovea.y}
                  r="17"
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth="0.4"
                  strokeDasharray="2, 1"
                  opacity="0.8"
                />

                {/* Outer Macular Ring (6mm) */}
                <circle
                  cx={fovea.x}
                  cy={fovea.y}
                  r="32"
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="0.4"
                  strokeDasharray="2.5, 1"
                  opacity="0.75"
                />

                {/* Crosshairs dividing into Superior, Nasal, Inferior, Temporal */}
                <line
                  x1={fovea.x - 32}
                  y1={fovea.y - 32}
                  x2={fovea.x + 32}
                  y2={fovea.y + 32}
                  stroke="#94a3b8"
                  strokeWidth="0.25"
                  opacity="0.5"
                />
                <line
                  x1={fovea.x - 32}
                  y1={fovea.y + 32}
                  x2={fovea.x + 32}
                  y2={fovea.y - 32}
                  stroke="#94a3b8"
                  strokeWidth="0.25"
                  opacity="0.5"
                />

                {/* Foveal White Marker Cross */}
                <line
                  x1={fovea.x - 1.5}
                  y1={fovea.y}
                  x2={fovea.x + 1.5}
                  y2={fovea.y}
                  stroke="#ffffff"
                  strokeWidth="0.5"
                />
                <line
                  x1={fovea.x}
                  y1={fovea.y - 1.5}
                  x2={fovea.x}
                  y2={fovea.y + 1.5}
                  stroke="#ffffff"
                  strokeWidth="0.5"
                />

                {/* Ring Labels */}
                <text x={fovea.x + 7.5} y={fovea.y - 2} fill="#10b981" fontSize="1.8" fontFamily="monospace" fontWeight="bold">
                  1mm (Central)
                </text>
                <text x={fovea.x + 18} y={fovea.y - 2} fill="#f59e0b" fontSize="1.8" fontFamily="monospace" fontWeight="bold">
                  3mm (Inner)
                </text>
                <text x={fovea.x + 33} y={fovea.y - 2} fill="#ef4444" fontSize="1.8" fontFamily="monospace" fontWeight="bold">
                  6mm (Outer)
                </text>
              </svg>
            </div>
          )}

          {/* YOLO26 NANO INSTANCE SEGMENTATION MASKS & BOUNDING BOXES (NMS-FREE) */}
          {showYoloBoxes && (
            <div className="absolute inset-0 pointer-events-none">
              <svg viewBox="0 0 100 100" className="w-full h-full" preserveAspectRatio="xMidYMid meet">
                {/* Fallback to hotspots if yolo26Detections is not yet populated */}
                {(currentCase.yolo26Detections || currentCase.hotspots.map((h, i) => ({
                  detection_id: `YOLO26-DET-${String(i + 1).padStart(3, '0')}`,
                  lesion_type: h.lesionType,
                  box: {
                    x1: Math.max(0, h.x - h.radius * 0.9),
                    y1: Math.max(0, h.y - h.radius * 0.9),
                    x2: Math.min(100, h.x + h.radius * 0.9),
                    y2: Math.min(100, h.y + h.radius * 0.9),
                    width: h.radius * 1.8,
                    height: h.radius * 1.8
                  },
                  confidence: h.intensity,
                  etdrs_zone: h.etdrsZone,
                  assignment_mode: 'Hungarian 1-to-1'
                }))).map((det, idx) => {
                  const color = 
                    det.lesion_type.includes('Microaneurysm') ? '#ef4444' :
                    det.lesion_type.includes('Hemorrhage') ? '#dc2626' :
                    det.lesion_type.includes('Exudate') ? '#eab308' :
                    det.lesion_type.includes('Cotton') ? '#38bdf8' : '#a855f7';

                  const b = det.box;
                  return (
                    <g key={`yolo-box-${idx}`} className="transition-all duration-150">
                      {/* Bounding Box Rect with Hungarian Assign Tag */}
                      <rect
                        x={b.x1}
                        y={b.y1}
                        width={b.width}
                        height={b.height}
                        fill={`${color}18`}
                        stroke={color}
                        strokeWidth="0.35"
                        strokeDasharray="1.5, 0.8"
                        rx="0.5"
                      />
                      {/* Bounding Box Corner Reticle */}
                      <path
                        d={`M ${b.x1},${b.y1 + 1} L ${b.x1},${b.y1} L ${b.x1 + 1},${b.y1}`}
                        stroke={color}
                        strokeWidth="0.6"
                        fill="none"
                      />
                      <path
                        d={`M ${b.x2 - 1},${b.y2} L ${b.x2},${b.y2} L ${b.x2},${b.y2 - 1}`}
                        stroke={color}
                        strokeWidth="0.6"
                        fill="none"
                      />
                      {/* Top Label Tag */}
                      <rect
                        x={b.x1}
                        y={Math.max(1, b.y1 - 2.8)}
                        width={Math.max(12, det.lesion_type.length * 1.05 + 6)}
                        height="2.4"
                        fill="#090d16"
                        stroke={color}
                        strokeWidth="0.25"
                        rx="0.4"
                      />
                      <text
                        x={b.x1 + 0.6}
                        y={Math.max(1, b.y1 - 2.8) + 1.7}
                        fill={color}
                        fontSize="1.3"
                        fontFamily="monospace"
                        fontWeight="bold"
                      >
                        {det.lesion_type.slice(0, 8)} {Math.round(det.confidence * 100)}%
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          )}

          {/* HOVER CROSSHAIR PROBE */}
          {hoverCoord && (
            <div
              className="absolute pointer-events-none w-4 h-4 -translate-x-1/2 -translate-y-1/2 border border-cyan-400/80 rounded-full"
              style={{ left: `${hoverCoord.x}%`, top: `${hoverCoord.y}%` }}
            >
              <div className="w-full h-full flex items-center justify-center">
                <div className="w-1 h-1 bg-cyan-300 rounded-full"></div>
              </div>
            </div>
          )}
        </div>

        {/* BOTTOM LEFT OVERLAY: XAI THERMAL SPECTRUM LEGEND */}
        <div className="absolute bottom-3 left-3 bg-slate-950/85 backdrop-blur-md border border-slate-800 p-2 rounded-lg text-[10px] text-slate-300 pointer-events-auto">
          <div className="flex items-center justify-between gap-4 mb-1">
            <span className="font-semibold text-slate-200">Grad-CAM Salience</span>
            <span className="font-mono text-cyan-400">Jet Colormap</span>
          </div>
          <div className="w-36 h-2 rounded bg-gradient-to-r from-blue-600 via-cyan-400 via-yellow-400 to-rose-600 border border-slate-700"></div>
          <div className="flex justify-between text-[9px] font-mono text-slate-400 mt-0.5">
            <span>0.0 (Vessel/Stroma)</span>
            <span>0.5</span>
            <span className="text-rose-400 font-bold">1.0 (Lesion)</span>
          </div>
        </div>

        {/* BOTTOM RIGHT OVERLAY: LIVE PROBE TELEMETRY HUD */}
        <div className="absolute bottom-3 right-3 bg-slate-950/85 backdrop-blur-md border border-slate-800 p-2.5 rounded-lg text-xs font-mono pointer-events-auto min-w-[200px]">
          <div className="flex items-center gap-1.5 text-slate-400 text-[10px] border-b border-slate-800 pb-1 mb-1.5">
            <Crosshair className="w-3 h-3 text-cyan-400" />
            <span>Retinal Surface Probe</span>
          </div>
          <div className="space-y-1 text-[11px]">
            <div className="flex justify-between">
              <span className="text-slate-400">Pixel Vector (X, Y):</span>
              <span className="text-cyan-300">
                {hoverCoord ? `${hoverCoord.x}%, ${hoverCoord.y}%` : `${fovea.x}%, ${fovea.y}%`}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Target Feature:</span>
              <span className="text-slate-200 font-sans">
                {activeHotspot
                  ? activeHotspot.lesionType
                  : hoverCoord && Math.hypot(hoverCoord.x - fovea.x, hoverCoord.y - fovea.y) < 8
                  ? 'Foveal Avascular Zone'
                  : hoverCoord && Math.hypot(hoverCoord.x - disc.x, hoverCoord.y - disc.y) < 10
                  ? 'Optic Nerve Head'
                  : 'Arteriolar Microvasculature'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Activation Energy:</span>
              <span className={`font-bold ${activeHotspot ? 'text-rose-400' : 'text-slate-400'}`}>
                {activeHotspot ? `${(activeHotspot.intensity * 100).toFixed(1)}%` : '0.12%'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* FOOTER STRIP: LESION SUMMARY METRICS */}
      <div className="mt-3 pt-2.5 border-t border-slate-800 grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
        <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800">
          <span className="text-[10px] text-slate-400 block">Microaneurysms</span>
          <span className="font-mono text-sm font-bold text-rose-400">
            {currentCase.lesions.countMA} <span className="text-[10px] text-slate-500 font-normal">pts</span>
          </span>
        </div>
        <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800">
          <span className="text-[10px] text-slate-400 block">Hemorrhages</span>
          <span className="font-mono text-sm font-bold text-amber-400">
            {currentCase.lesions.countHE} <span className="text-[10px] text-slate-500 font-normal">blots</span>
          </span>
        </div>
        <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800">
          <span className="text-[10px] text-slate-400 block">Hard Exudates</span>
          <span className="font-mono text-sm font-bold text-yellow-300">
            {currentCase.lesions.countEX} <span className="text-[10px] text-slate-500 font-normal">clusters</span>
          </span>
        </div>
        <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800">
          <span className="text-[10px] text-slate-400 block">Neovascularization</span>
          <span className={`font-mono text-sm font-bold ${currentCase.lesions.neovascularization ? 'text-rose-500 animate-pulse' : 'text-emerald-400'}`}>
            {currentCase.lesions.neovascularization ? 'DETECTED' : 'ABSENT'}
          </span>
        </div>
      </div>
    </div>
  );
};
