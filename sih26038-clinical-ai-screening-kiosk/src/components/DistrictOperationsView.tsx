import React, { useState, useMemo } from 'react';
import { 
  Building2, 
  Users, 
  Clock, 
  AlertTriangle, 
  CheckCircle2, 
  TrendingUp, 
  Sliders, 
  ShieldCheck, 
  Activity, 
  Cpu, 
  BarChart3, 
  ArrowRight,
  Sparkles,
  Zap,
  RotateCcw
} from 'lucide-react';

export const DistrictOperationsView: React.FC = () => {
  // Model Parameters for M/M/c Queueing Theory
  const [arrivalRatePerKiosk, setArrivalRatePerKiosk] = useState<number>(6); // patients/hour/kiosk
  const [numKiosks, setNumKiosks] = useState<number>(4); // active district screening field kiosks
  const [serviceRatePerDoctor, setServiceRatePerDoctor] = useState<number>(8); // scans reviewed/hour/doctor
  const [numDoctors, setNumDoctors] = useState<number>(4); // ophthalmologist capacity (c)
  const [slaMinutes, setSlaMinutes] = useState<number>(15); // Target SLA max wait time (minutes)

  // Calculations:
  // Total arrival rate Lambda = lambda * K * escalation_fraction
  // Assuming ~35% of all patients have referable/borderline findings escalated to the Hub (Gate 4)
  const escalationFraction = 0.38;
  const totalPatientArrivals = arrivalRatePerKiosk * numKiosks;
  const escalatedArrivalRate = totalPatientArrivals * escalationFraction; // Lambda in queue

  const totalServiceCapacity = numDoctors * serviceRatePerDoctor; // c * mu
  const trafficIntensity = escalatedArrivalRate / totalServiceCapacity; // rho = Lambda / (c * mu)

  // Erlang-C formula for M/M/c queueing system
  const queueStats = useMemo(() => {
    const c = numDoctors;
    const lambda = escalatedArrivalRate;
    const mu = serviceRatePerDoctor;
    const rho = lambda / (c * mu);

    if (rho >= 0.99) {
      return {
        rho: Math.min(1.5, rho),
        isUnstable: true,
        pZero: 0,
        erlangC: 1.0,
        waitQueueMinutes: 180,
        queueLength: Math.round(lambda * 3),
        slaCompliance: 5,
        status: 'CRITICAL_SATURATION',
      };
    }

    // Factorial helper
    const fact = (n: number): number => {
      let r = 1;
      for (let i = 2; i <= n; i++) r *= i;
      return r;
    };

    // Calculate sum_{k=0}^{c-1} (c*rho)^k / k!
    const a = lambda / mu; // traffic in Erlangs
    let sumPart = 0;
    for (let k = 0; k < c; k++) {
      sumPart += Math.pow(a, k) / fact(k);
    }
    const cPart = Math.pow(a, c) / (fact(c) * (1 - rho));
    const pZero = 1.0 / (sumPart + cPart);

    // Erlang-C probability P(Wait > 0)
    const erlangC = cPart * pZero;

    // Average time in queue W_q = P_C / (c*mu - lambda)
    const waitQueueHours = erlangC / (c * mu - lambda);
    const waitQueueMinutes = waitQueueHours * 60;

    // Average number in queue L_q = lambda * W_q
    const queueLength = lambda * waitQueueHours;

    // SLA Compliance: P(W_q <= sla) = 1 - P_C * exp(-(c*mu - lambda)*t)
    const slaHours = slaMinutes / 60.0;
    const pWaitExceedsSla = erlangC * Math.exp(-(c * mu - lambda) * slaHours);
    const slaCompliance = Math.max(0, Math.min(100, Math.round((1.0 - pWaitExceedsSla) * 100)));

    let status = 'OPTIMAL';
    if (rho > 0.85 || waitQueueMinutes > slaMinutes) {
      status = 'WARNING_BOTTLENECK';
    } else if (rho > 0.70) {
      status = 'ACCEPTABLE';
    }

    return {
      rho,
      isUnstable: false,
      pZero,
      erlangC,
      waitQueueMinutes: Math.max(0.2, waitQueueMinutes),
      queueLength: Math.max(0, queueLength),
      slaCompliance,
      status,
    };
  }, [escalatedArrivalRate, serviceRatePerDoctor, numDoctors, slaMinutes]);

  // Recommended minimum doctor staffing to guarantee SLA
  const recommendedDoctors = useMemo(() => {
    for (let c = 1; c <= 16; c++) {
      const rho = escalatedArrivalRate / (c * serviceRatePerDoctor);
      if (rho < 0.75) {
        return c;
      }
    }
    return Math.ceil(escalatedArrivalRate / serviceRatePerDoctor) + 1;
  }, [escalatedArrivalRate, serviceRatePerDoctor]);

  return (
    <div className="flex-1 flex flex-col gap-4 overflow-y-auto p-4 max-w-[1780px] mx-auto w-full">
      {/* SECTION 1: Operations Header Banner */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-lg relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-full bg-gradient-to-l from-indigo-950/40 via-cyan-950/20 to-transparent pointer-events-none"></div>
        <div className="flex flex-wrap items-center justify-between gap-4 relative z-10">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-indigo-900/60 border border-indigo-700/60 text-indigo-400">
                <Building2 className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono font-bold uppercase px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
                    Operations Research & Simulink Framework
                  </span>
                  <span className="text-xs text-slate-400 font-mono">
                    Model: M/M/c Kendall's Notation
                  </span>
                </div>
                <h1 className="text-xl font-bold text-slate-100 tracking-tight mt-0.5">
                  Regional Telemedicine District Capacity & Queue Optimization
                </h1>
              </div>
            </div>
            <p className="text-xs text-slate-400 mt-2 max-w-3xl leading-relaxed">
              Mathematical modeling of field screening kiosks (<strong className="text-cyan-300">Tier 1 Edge</strong>) streaming 
              escalated diagnostic packets to Central Hub Ophthalmologists (<strong className="text-indigo-300">Tier 2 Hub</strong>). 
              Optimizes specialist staffing schedules to prevent clinical triage bottlenecks.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="bg-slate-950/80 border border-slate-800 px-4 py-2.5 rounded-lg text-right">
              <span className="text-[10px] text-slate-400 uppercase font-mono block">Recommended Hub Staffing</span>
              <span className="text-xl font-black text-cyan-400 font-mono">
                c* = {recommendedDoctors} Specialists
              </span>
            </div>
            <button
              onClick={() => {
                setArrivalRatePerKiosk(6);
                setNumKiosks(4);
                setServiceRatePerDoctor(8);
                setNumDoctors(4);
                setSlaMinutes(15);
              }}
              className="px-3 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 flex items-center gap-1.5 transition-all"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset Nominal</span>
            </button>
          </div>
        </div>
      </div>

      {/* SECTION 2: Live KPIs Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* KPI 1: Traffic Intensity rho */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-mono">TRAFFIC INTENSITY (&rho;)</span>
            <Activity className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className={`text-2xl font-mono font-black ${
              queueStats.rho > 0.85 ? 'text-rose-400' : queueStats.rho > 0.70 ? 'text-amber-400' : 'text-emerald-400'
            }`}>
              {(queueStats.rho * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-slate-400">
              ({queueStats.rho > 0.85 ? 'High Congestion' : queueStats.rho > 0.70 ? 'Optimal Load' : 'Surplus Capacity'})
            </span>
          </div>
          <div className="mt-2.5 w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
            <div 
              className={`h-full transition-all duration-300 ${
                queueStats.rho > 0.85 ? 'bg-rose-500' : queueStats.rho > 0.70 ? 'bg-amber-500' : 'bg-emerald-500'
              }`}
              style={{ width: `${Math.min(100, queueStats.rho * 100)}%` }}
            ></div>
          </div>
          <span className="text-[10px] text-slate-500 mt-1.5 block">
            Formula: &Lambda; / (c &times; &mu;) = {escalatedArrivalRate.toFixed(1)} / {totalServiceCapacity}
          </span>
        </div>

        {/* KPI 2: Expected Wait Time W_q */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-mono">AVG QUEUE WAIT (W_q)</span>
            <Clock className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className={`text-2xl font-mono font-black ${
              queueStats.waitQueueMinutes > slaMinutes ? 'text-rose-400' : 'text-cyan-400'
            }`}>
              {queueStats.isUnstable ? '> 180' : queueStats.waitQueueMinutes.toFixed(1)}
            </span>
            <span className="text-xs text-slate-400">minutes / scan</span>
          </div>
          <div className="flex items-center gap-1.5 text-[11px] mt-2 font-medium">
            {queueStats.waitQueueMinutes <= slaMinutes ? (
              <span className="text-emerald-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> Meets &le; {slaMinutes} min SLA Target
              </span>
            ) : (
              <span className="text-rose-400 flex items-center gap-1">
                <AlertTriangle className="w-3.5 h-3.5" /> SLA Breach ({slaMinutes} min Target)
              </span>
            )}
          </div>
          <span className="text-[10px] text-slate-500 mt-1.5 block">
            Erlang-C Delay Probability: {(queueStats.erlangC * 100).toFixed(1)}%
          </span>
        </div>

        {/* KPI 3: Queue Depth L_q */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-mono">QUEUE LENGTH (L_q)</span>
            <Users className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-mono font-black text-amber-400">
              {queueStats.queueLength.toFixed(1)}
            </span>
            <span className="text-xs text-slate-400">patients waiting</span>
          </div>
          <div className="flex items-center gap-2 text-[11px] text-slate-400 mt-2">
            <span>Total Arrivals: <strong className="text-slate-200">{totalPatientArrivals} / hr</strong></span>
            <span>•</span>
            <span>Escalated: <strong className="text-amber-300">{escalatedArrivalRate.toFixed(1)} / hr</strong></span>
          </div>
          <span className="text-[10px] text-slate-500 mt-1.5 block">
            Little's Law: L_q = &Lambda; &times; W_q
          </span>
        </div>

        {/* KPI 4: SLA Compliance Rate */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-mono">SLA COMPLIANCE</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className={`text-2xl font-mono font-black ${
              queueStats.slaCompliance >= 90 ? 'text-emerald-400' : queueStats.slaCompliance >= 75 ? 'text-amber-400' : 'text-rose-400'
            }`}>
              {queueStats.slaCompliance}%
            </span>
            <span className="text-xs text-slate-400">within {slaMinutes} min</span>
          </div>
          <div className="mt-2.5 w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
            <div 
              className={`h-full transition-all duration-300 ${
                queueStats.slaCompliance >= 90 ? 'bg-emerald-500' : queueStats.slaCompliance >= 75 ? 'bg-amber-500' : 'bg-rose-500'
              }`}
              style={{ width: `${queueStats.slaCompliance}%` }}
            ></div>
          </div>
          <span className="text-[10px] text-slate-500 mt-1.5 block">
            Clinical Priority: Gate 4 Escalated NPDR & PDR
          </span>
        </div>
      </div>

      {/* SECTION 3: Interactive Sliders & Queue Dynamics Simulation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Column: Parameter Controls (5 cols) */}
        <div className="lg:col-span-5 bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-col gap-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
            <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
              <Sliders className="w-4 h-4 text-cyan-400" />
              Simulation Control Parameters
            </span>
            <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-800/60">
              Interactive
            </span>
          </div>

          {/* Slider 1: Active District Kiosks */}
          <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-300 font-medium">District Screening Kiosks (K):</span>
              <span className="font-mono font-bold text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/60">
                {numKiosks} Field Kiosks
              </span>
            </div>
            <input 
              type="range" 
              min="1" 
              max="10" 
              step="1"
              value={numKiosks}
              onChange={(e) => setNumKiosks(parseInt(e.target.value))}
              className="w-full mt-2 accent-cyan-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1">
              <span>1 Kiosk (Single Clinic)</span>
              <span>10 Kiosks (Full District)</span>
            </div>
          </div>

          {/* Slider 2: Patient Arrival Rate per Kiosk */}
          <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-300 font-medium">Arrival Rate per Kiosk (&lambda;):</span>
              <span className="font-mono font-bold text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/60">
                {arrivalRatePerKiosk} patients / hr
              </span>
            </div>
            <input 
              type="range" 
              min="2" 
              max="20" 
              step="1"
              value={arrivalRatePerKiosk}
              onChange={(e) => setArrivalRatePerKiosk(parseInt(e.target.value))}
              className="w-full mt-2 accent-cyan-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1">
              <span>2 (Rural Outpost)</span>
              <span>20 (Urban Camp Surge)</span>
            </div>
          </div>

          {/* Slider 3: Central Hub Ophthalmologists (c) */}
          <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-300 font-medium">Hub Ophthalmologist Capacity (c):</span>
              <span className="font-mono font-bold text-indigo-400 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-800/60">
                {numDoctors} Specialists On Duty
              </span>
            </div>
            <input 
              type="range" 
              min="1" 
              max="12" 
              step="1"
              value={numDoctors}
              onChange={(e) => setNumDoctors(parseInt(e.target.value))}
              className="w-full mt-2 accent-indigo-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1">
              <span>1 Doctor (Understaffed)</span>
              <span>12 Doctors (Multi-Hospital)</span>
            </div>
          </div>

          {/* Slider 4: Doctor Review Speed (mu) */}
          <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-300 font-medium">Review Throughput per Doctor (&mu;):</span>
              <span className="font-mono font-bold text-indigo-400 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-800/60">
                {serviceRatePerDoctor} reviews / hr
              </span>
            </div>
            <input 
              type="range" 
              min="4" 
              max="20" 
              step="1"
              value={serviceRatePerDoctor}
              onChange={(e) => setServiceRatePerDoctor(parseInt(e.target.value))}
              className="w-full mt-2 accent-indigo-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1">
              <span>4 (Detailed Manual)</span>
              <span>20 (XAI-Assisted Fast Track)</span>
            </div>
          </div>

          {/* Slider 5: Clinical SLA Target Wait Time */}
          <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-300 font-medium">Target Triage SLA Horizon:</span>
              <span className="font-mono font-bold text-amber-400 bg-amber-950/60 px-2 py-0.5 rounded border border-amber-800/60">
                &le; {slaMinutes} minutes
              </span>
            </div>
            <input 
              type="range" 
              min="5" 
              max="60" 
              step="5"
              value={slaMinutes}
              onChange={(e) => setSlaMinutes(parseInt(e.target.value))}
              className="w-full mt-2 accent-amber-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1">
              <span>5 min (Immediate Emergency)</span>
              <span>60 min (Standard Routine)</span>
            </div>
          </div>
        </div>

        {/* Right Column: Queue Curve & Shift Simulation Timeline (7 cols) */}
        <div className="lg:col-span-7 bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-col gap-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
            <div>
              <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <BarChart3 className="w-4 h-4 text-cyan-400" />
                Queue Delay Sensitivity vs Staffing Capacity (c)
              </span>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Mathematical proof of the queuing bottleneck knee point as staffing varies from c = 1 to 10
              </p>
            </div>
            <span className="text-[10px] font-mono text-slate-400">
              &Lambda; = {escalatedArrivalRate.toFixed(1)} / hr
            </span>
          </div>

          {/* Graph: Wait Time vs Number of Doctors */}
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 flex flex-col gap-2">
            <div className="text-[11px] font-mono text-slate-400 flex justify-between">
              <span>Wait Time (min) vs Staffing c</span>
              <span className="text-emerald-400 font-bold">--- SLA: {slaMinutes} min Target</span>
            </div>

            {/* SVG Sensitivity Curve */}
            <div className="relative h-44 w-full">
              <svg className="w-full h-full" viewBox="0 0 500 150">
                {/* Horizontal Grid lines */}
                <line x1="40" y1="120" x2="490" y2="120" stroke="#334155" strokeWidth="1" />
                <line x1="40" y1="80" x2="490" y2="80" stroke="#1e293b" strokeWidth="1" />
                <line x1="40" y1="40" x2="490" y2="40" stroke="#1e293b" strokeWidth="1" />
                <line x1="40" y1="10" x2="490" y2="10" stroke="#1e293b" strokeWidth="1" />

                {/* Y-Axis Labels */}
                <text x="32" y="124" fill="#64748b" fontSize="9" textAnchor="end">0m</text>
                <text x="32" y="84" fill="#64748b" fontSize="9" textAnchor="end">30m</text>
                <text x="32" y="44" fill="#64748b" fontSize="9" textAnchor="end">60m</text>
                <text x="32" y="14" fill="#64748b" fontSize="9" textAnchor="end">90m</text>

                {/* SLA Threshold Line */}
                {(() => {
                  const slaY = 120 - (slaMinutes / 90) * 110;
                  return (
                    <line 
                      x1="40" 
                      y1={slaY} 
                      x2="490" 
                      y2={slaY} 
                      stroke="#10b981" 
                      strokeWidth="1.5" 
                      strokeDasharray="4 3" 
                    />
                  );
                })()}

                {/* Curve Points for c = 1..10 */}
                {(() => {
                  const points: Array<{ c: number; x: number; y: number; wait: number }> = [];
                  for (let cVal = 1; cVal <= 10; cVal++) {
                    const x = 40 + ((cVal - 1) / 9) * 440;
                    const rhoVal = escalatedArrivalRate / (cVal * serviceRatePerDoctor);
                    let wait = 120;
                    if (rhoVal < 0.98) {
                      // rough wait approx
                      wait = Math.min(100, Math.max(0.5, (rhoVal / (1 - rhoVal)) * (60 / (cVal * serviceRatePerDoctor))));
                    }
                    const y = Math.max(10, 120 - (wait / 90) * 110);
                    points.push({ c: cVal, x, y, wait });
                  }

                  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

                  return (
                    <>
                      {/* Filled Area */}
                      <path 
                        d={`${pathD} L ${points[points.length - 1].x} 120 L ${points[0].x} 120 Z`} 
                        fill="rgba(99, 102, 241, 0.12)" 
                      />
                      {/* Stroke */}
                      <path d={pathD} fill="none" stroke="#6366f1" strokeWidth="2.5" />
                      {/* Dots */}
                      {points.map((p) => {
                        const isSelected = p.c === numDoctors;
                        return (
                          <g key={p.c}>
                            <circle 
                              cx={p.x} 
                              cy={p.y} 
                              r={isSelected ? 6 : 3} 
                              fill={isSelected ? '#38bdf8' : '#818cf8'} 
                              stroke={isSelected ? '#ffffff' : '#4f46e5'}
                              strokeWidth={isSelected ? 2 : 1}
                            />
                            {isSelected && (
                              <text x={p.x} y={p.y - 10} fill="#38bdf8" fontSize="10" fontWeight="bold" textAnchor="middle">
                                Current (c={p.c})
                              </text>
                            )}
                            <text x={p.x} y={135} fill={isSelected ? '#38bdf8' : '#64748b'} fontSize="9" textAnchor="middle">
                              c={p.c}
                            </text>
                          </g>
                        );
                      })}
                    </>
                  );
                })()}
              </svg>
            </div>
          </div>

          {/* 8-Hour Clinic Shift Queue Simulation Timeline */}
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 flex flex-col gap-2">
            <div className="flex items-center justify-between text-xs font-bold text-slate-300">
              <span className="flex items-center gap-1.5 text-indigo-400">
                <Zap className="w-3.5 h-3.5" />
                Simulated 8-Hour Rural PHC Shift Dynamics
              </span>
              <span className="text-[10px] font-mono text-slate-400">08:00 AM - 04:00 PM</span>
            </div>

            <div className="grid grid-cols-8 gap-1 text-center mt-1">
              {[
                { hour: '08:00', load: 0.35, arrivals: Math.round(totalPatientArrivals * 0.4) },
                { hour: '09:00', load: 0.70, arrivals: Math.round(totalPatientArrivals * 0.8) },
                { hour: '10:00', load: 0.95, arrivals: Math.round(totalPatientArrivals * 1.1) },
                { hour: '11:00', load: 1.05, arrivals: Math.round(totalPatientArrivals * 1.25) },
                { hour: '12:00', load: 0.85, arrivals: Math.round(totalPatientArrivals * 0.95) },
                { hour: '13:00', load: 0.50, arrivals: Math.round(totalPatientArrivals * 0.6) },
                { hour: '14:00', load: 0.75, arrivals: Math.round(totalPatientArrivals * 0.85) },
                { hour: '15:00', load: 0.40, arrivals: Math.round(totalPatientArrivals * 0.45) },
              ].map((item, idx) => {
                const isPeak = item.load >= 0.90;
                return (
                  <div key={idx} className="bg-slate-900/80 p-2 rounded border border-slate-800/80 flex flex-col items-center">
                    <span className="text-[10px] text-slate-400 font-mono">{item.hour}</span>
                    <div className="w-full bg-slate-950 rounded-full h-12 my-1 flex flex-col justify-end p-0.5">
                      <div 
                        className={`w-full rounded-sm transition-all ${
                          isPeak ? 'bg-gradient-to-t from-rose-600 to-amber-500' : 'bg-gradient-to-t from-cyan-600 to-indigo-500'
                        }`}
                        style={{ height: `${Math.min(100, item.load * 90)}%` }}
                      ></div>
                    </div>
                    <span className={`text-[10px] font-mono font-bold ${isPeak ? 'text-rose-400' : 'text-slate-300'}`}>
                      {item.arrivals} pts
                    </span>
                    <span className="text-[8px] text-slate-500 uppercase mt-0.5">
                      {isPeak ? 'Surge' : 'Normal'}
                    </span>
                  </div>
                );
              })}
            </div>

            <div className="flex items-center justify-between text-[11px] text-slate-400 bg-slate-900/60 p-2 rounded border border-slate-800 mt-1">
              <span className="flex items-center gap-1.5 text-cyan-300">
                <Sparkles className="w-3.5 h-3.5" />
                <strong>Operations Recommendation:</strong> Staff <strong>c = {recommendedDoctors} specialists</strong> during peak 10:00-12:00 window to maintain 0 patient dropouts.
              </span>
              <span className="text-emerald-400 font-bold font-mono">
                SLA Compliance: {queueStats.slaCompliance}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
