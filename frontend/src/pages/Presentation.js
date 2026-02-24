import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, Download, Home, Maximize2, Minimize2 } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const slides = [
  {
    id: 'title',
    content: (
      <div className="flex flex-col justify-center items-start h-full px-16 md:px-24">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <div className="text-sm font-mono tracking-[0.3em] text-cyan-500/60 mb-6">NETWORK INTRUSION DETECTION SYSTEM</div>
          <h1 className="text-7xl md:text-9xl font-bold font-mono leading-none mb-2 text-white">CYBER</h1>
          <h1 className="text-7xl md:text-9xl font-bold font-mono leading-none mb-8 shimmer-text">SENTINELLE</h1>
          <div className="w-32 h-0.5 bg-gradient-to-r from-cyan-500 to-transparent mb-8" />
          <p className="text-lg text-gray-400 mb-2">Zakarya Oukil</p>
          <p className="text-sm text-gray-600">Master 1 Cybersecurity &bull; HIS 2025/2026</p>
          <p className="text-sm text-gray-600">NSL-KDD Dataset &bull; Decision Tree Classifier</p>
        </motion.div>
      </div>
    )
  },
  {
    id: 'agenda',
    content: (
      <div className="px-16 md:px-24 py-12">
        <h2 className="text-4xl font-mono font-bold text-cyan-400 mb-2">AGENDA</h2>
        <div className="w-20 h-0.5 bg-cyan-500 mb-10" />
        <div className="grid md:grid-cols-2 gap-x-12 gap-y-4">
          {['Problem Statement & Context','Objectives','Dataset: NSL-KDD','System Architecture','Exploratory Data Analysis','Decision Tree Model','Model Performance & Metrics','K-Means Clustering','Real-Time Detection','DoS Attack Simulation','Live Demo','Conclusion & Future Work'].map((item, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 * i }} className="flex items-center gap-4 py-2 border-b border-white/5">
              <span className="text-cyan-500/60 font-mono text-sm w-8">{String(i + 1).padStart(2, '0')}</span>
              <span className="text-gray-300 text-base">{item}</span>
            </motion.div>
          ))}
        </div>
      </div>
    )
  },
  {
    id: 'problem',
    content: (
      <div className="px-16 md:px-24 py-12">
        <h2 className="text-4xl font-mono font-bold text-cyan-400 mb-2">PROBLEM STATEMENT</h2>
        <div className="w-20 h-0.5 bg-cyan-500 mb-10" />
        <div className="grid md:grid-cols-2 gap-12">
          <div>
            <h3 className="text-xl font-bold text-white mb-4">The Challenge</h3>
            {['Cyberattacks are increasing in frequency and sophistication','Traditional firewalls and signature-based systems are insufficient','Need for intelligent, automated threat detection'].map((p, i) => (
              <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 + i * 0.15 }} className="flex gap-3 mb-3">
                <span className="text-red-500 font-mono mt-0.5">&gt;</span>
                <span className="text-gray-400 text-sm leading-relaxed">{p}</span>
              </motion.div>
            ))}
          </div>
          <div>
            <h3 className="text-xl font-bold text-white mb-4">Our Solution</h3>
            {['ML-based Network Intrusion Detection System (NIDS)','Classifies network traffic as Normal or Attack in real-time','Identifies attack categories: DoS, Probe, R2L, U2R'].map((p, i) => (
              <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 + i * 0.15 }} className="flex gap-3 mb-3">
                <span className="text-green-500 font-mono mt-0.5">&gt;</span>
                <span className="text-gray-400 text-sm leading-relaxed">{p}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    )
  },
  {
    id: 'objectives',
    content: (
      <div className="px-16 md:px-24 py-12">
        <h2 className="text-4xl font-mono font-bold text-cyan-400 mb-2">OBJECTIVES</h2>
        <div className="w-20 h-0.5 bg-cyan-500 mb-10" />
        {[
          { n: '01', t: 'Data Analysis', d: 'Comprehensive EDA on NSL-KDD to understand traffic patterns' },
          { n: '02', t: 'Model Training', d: 'Train a Decision Tree classifier for accurate traffic classification' },
          { n: '03', t: 'Real-Time Detection', d: 'Build a monitoring system that detects intrusions as they happen' },
          { n: '04', t: 'Attack Simulation', d: 'Controlled DoS simulation to validate detection capabilities' },
          { n: '05', t: 'Visualization', d: 'Interactive web dashboard with 3D visualizations' },
        ].map((obj, i) => (
          <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 + i * 0.1 }} className="flex gap-5 mb-5 items-start">
            <span className="text-3xl font-mono font-bold text-cyan-500/30">{obj.n}</span>
            <div>
              <h4 className="text-base font-bold text-cyan-400">{obj.t}</h4>
              <p className="text-sm text-gray-400">{obj.d}</p>
            </div>
          </motion.div>
        ))}
      </div>
    )
  },
  {
    id: 'dataset',
    content: (
      <div className="px-16 md:px-24 py-12">
        <h2 className="text-4xl font-mono font-bold text-cyan-400 mb-2">DATASET: NSL-KDD</h2>
        <div className="w-20 h-0.5 bg-cyan-500 mb-8" />
        <div className="grid grid-cols-4 gap-4 mb-10">
          {[{ v: '125,973', l: 'Training Samples' },{ v: '41', l: 'Features' },{ v: '5', l: 'Traffic Classes' },{ v: '2', l: 'Categories' }].map((s, i) => (
            <motion.div key={i} initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: 0.2 + i * 0.1 }} className="text-center p-5 border border-cyan-500/20 bg-cyan-500/5">
              <div className="text-3xl font-mono font-bold text-cyan-400">{s.v}</div>
              <div className="text-[10px] text-gray-500 tracking-widest mt-1">{s.l}</div>
            </motion.div>
          ))}
        </div>
        <div className="grid md:grid-cols-2 gap-6">
          {[
            { t: 'Basic Features', d: 'duration, protocol_type, service, flag, src_bytes, dst_bytes' },
            { t: 'Content Features', d: 'hot, num_failed_logins, logged_in, root_shell, num_file_creations' },
            { t: 'Traffic Features', d: 'count, srv_count, serror_rate, same_srv_rate, dst_host_count' },
            { t: 'Attack Types', d: 'Normal, DoS (neptune, smurf), Probe (portsweep), R2L, U2R' },
          ].map((f, i) => (
            <div key={i} className="border-l-2 border-cyan-500/30 pl-4">
              <h4 className="text-sm font-bold text-white mb-1">{f.t}</h4>
              <p className="text-xs text-gray-500">{f.d}</p>
            </div>
          ))}
        </div>
      </div>
    )
  },
  {
    id: 'architecture',
    content: (
      <div className="px-16 md:px-24 py-12">
        <h2 className="text-4xl font-mono font-bold text-cyan-400 mb-2">SYSTEM ARCHITECTURE</h2>
        <div className="w-20 h-0.5 bg-cyan-500 mb-10" />
        <div className="space-y-4">
          {[
            { name: 'FRONTEND', desc: 'React 19 | 3D Canvas Visualizations | Recharts', color: '#00F0FF', border: 'border-cyan-500/40' },
            { name: 'BACKEND', desc: 'Python FastAPI | RESTful API | Real-Time Monitor', color: '#00FF41', border: 'border-green-500/40' },
            { name: 'ML ENGINE', desc: 'Scikit-learn | Decision Tree | K-Means Clustering', color: '#BD00FF', border: 'border-purple-500/40' },
            { name: 'DATABASE', desc: 'MongoDB Atlas | Model Storage | Traffic Logs', color: '#FAFF00', border: 'border-yellow-500/40' },
          ].map((layer, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 + i * 0.15 }} className={`flex items-center gap-5 p-5 border ${layer.border} bg-white/[0.02]`}>
              <div className="w-1 h-10 rounded-full" style={{ backgroundColor: layer.color }} />
              <div>
                <h4 className="font-mono font-bold text-sm tracking-wider" style={{ color: layer.color }}>{layer.name}</h4>
                <p className="text-xs text-gray-500 mt-1">{layer.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    )
  },
  {
    id: 'decision-tree',
    content: (
      <div className="px-16 md:px-24 py-12">
        <h2 className="text-4xl font-mono font-bold text-cyan-400 mb-2">DECISION TREE MODEL</h2>
        <div className="w-20 h-0.5 bg-cyan-500 mb-8" />
        <div className="grid md:grid-cols-2 gap-10">
          <div>
            <h3 className="text-lg font-bold text-white mb-4">Why Decision Tree?</h3>
            {['Highly interpretable - visualize decisions','Fast training and prediction time','Handles numerical & categorical features','No feature scaling required','Works well with structured data'].map((p, i) => (
              <div key={i} className="flex gap-3 mb-2"><span className="text-cyan-500">+</span><span className="text-sm text-gray-400">{p}</span></div>
            ))}
          </div>
          <div>
            <h3 className="text-lg font-bold text-white mb-4">Training Process</h3>
            {['Label encoding for categorical features','StandardScaler normalization','80/20 train-test split','Gini criterion optimization'].map((p, i) => (
              <div key={i} className="flex gap-3 mb-2"><span className="text-purple-500">&gt;</span><span className="text-sm text-gray-400">{p}</span></div>
            ))}
          </div>
        </div>
      </div>
    )
  },
  {
    id: 'performance',
    content: (
      <div className="px-16 md:px-24 py-12">
        <h2 className="text-4xl font-mono font-bold text-cyan-400 mb-2">MODEL PERFORMANCE</h2>
        <div className="w-20 h-0.5 bg-cyan-500 mb-10" />
        <div className="space-y-4">
          {[
            { name: 'ACCURACY', val: '94.30%', desc: 'Correct predictions out of total' },
            { name: 'PRECISION', val: '94.36%', desc: 'True positives among predicted positives' },
            { name: 'RECALL', val: '94.30%', desc: 'True positives among actual positives' },
            { name: 'F1-SCORE', val: '94.30%', desc: 'Harmonic mean of precision and recall' },
            { name: 'AUC', val: '0.9463', desc: 'Area Under the ROC Curve' },
          ].map((m, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 + i * 0.1 }} className="flex items-center justify-between p-4 border border-cyan-500/15 bg-white/[0.02]">
              <div className="flex items-center gap-6">
                <span className="font-mono font-bold text-cyan-400 text-sm w-24">{m.name}</span>
                <span className="font-mono font-bold text-green-400 text-2xl">{m.val}</span>
              </div>
              <span className="text-xs text-gray-600">{m.desc}</span>
            </motion.div>
          ))}
        </div>
      </div>
    )
  },
  {
    id: 'clustering',
    content: (
      <div className="px-16 md:px-24 py-12">
        <h2 className="text-4xl font-mono font-bold text-purple-400 mb-2">K-MEANS CLUSTERING</h2>
        <div className="w-20 h-0.5 bg-purple-500 mb-10" />
        {[
          { t: 'Purpose', d: 'Unsupervised learning to discover hidden patterns without labels' },
          { t: 'Method', d: 'K-Means with PCA dimensionality reduction for 2D visualization' },
          { t: 'Results', d: 'Clusters naturally separate normal from malicious traffic' },
          { t: 'Application', d: 'Detect novel/zero-day attacks the supervised model has never seen' },
        ].map((item, i) => (
          <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 + i * 0.15 }} className="mb-6">
            <h4 className="font-mono text-purple-400 text-sm tracking-wider mb-1">{item.t}</h4>
            <p className="text-gray-400 text-sm pl-4 border-l border-purple-500/20">{item.d}</p>
          </motion.div>
        ))}
      </div>
    )
  },
  {
    id: 'realtime',
    content: (
      <div className="px-16 md:px-24 py-12">
        <h2 className="text-4xl font-mono font-bold text-cyan-400 mb-2">REAL-TIME DETECTION</h2>
        <div className="w-20 h-0.5 bg-cyan-500 mb-10" />
        {[
          { t: 'Live Traffic Monitor', d: 'Tracks requests/second with visual charts', c: 'text-cyan-400', b: 'bg-cyan-500' },
          { t: 'Anomaly Detection', d: 'Threshold-based detection for traffic spikes', c: 'text-red-400', b: 'bg-red-500' },
          { t: 'Visual Alerts', d: 'Immediate alerts with attack classification', c: 'text-yellow-400', b: 'bg-yellow-500' },
          { t: 'Security Logs', d: 'Timestamped event log for forensics', c: 'text-green-400', b: 'bg-green-500' },
          { t: 'Radar Scanner', d: 'Interactive radar showing network activity', c: 'text-purple-400', b: 'bg-purple-500' },
        ].map((f, i) => (
          <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 + i * 0.1 }} className="flex items-center gap-4 mb-4">
            <div className={`w-1 h-8 rounded-full ${f.b}`} />
            <div>
              <h4 className={`font-mono text-sm font-bold ${f.c}`}>{f.t}</h4>
              <p className="text-xs text-gray-500">{f.d}</p>
            </div>
          </motion.div>
        ))}
      </div>
    )
  },
  {
    id: 'dos-sim',
    content: (
      <div className="px-16 md:px-24 py-12">
        <h2 className="text-4xl font-mono font-bold text-red-400 mb-2">DoS ATTACK SIMULATION</h2>
        <div className="w-20 h-0.5 bg-red-500 mb-8" />
        <p className="text-sm text-gray-500 mb-8">Controlled simulation to validate our detection system in real conditions</p>
        {[
          { s: 'Step 1', t: 'Baseline', d: 'Normal traffic at 0-2 req/s, green status' },
          { s: 'Step 2', t: 'Launch Attack', d: '100+ requests/second flood to the server' },
          { s: 'Step 3', t: 'Detection', d: 'System detects anomaly, triggers red alert' },
          { s: 'Step 4', t: 'Classification', d: 'ML model classifies traffic as DoS' },
          { s: 'Step 5', t: 'Response', d: 'Security logs record event with severity' },
        ].map((step, i) => (
          <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 + i * 0.12 }} className="flex items-center gap-5 mb-3">
            <span className="font-mono text-xs text-red-500/50 w-14">{step.s}</span>
            <span className="font-bold text-white text-sm w-32">{step.t}</span>
            <span className="text-sm text-gray-500">{step.d}</span>
          </motion.div>
        ))}
      </div>
    )
  },
  {
    id: 'demo',
    content: (
      <div className="px-16 md:px-24 py-12">
        <h2 className="text-4xl font-mono font-bold text-green-400 mb-2">LIVE DEMO PLAN</h2>
        <div className="w-20 h-0.5 bg-green-500 mb-10" />
        {[
          { page: 'Homepage', time: '~1 min', what: '3D globe, animated stats, system overview' },
          { page: 'Dashboard EDA', time: '~2 min', what: 'Charts, 3D network topology, attack distribution' },
          { page: 'Model Performance', time: '~2 min', what: 'DT metrics, ROC curve, 3D threat visualization' },
          { page: 'Live Prediction', time: '~2 min', what: 'Demo Normal -> NORMAL, Demo Attack -> INTRUSION' },
          { page: 'Clustering', time: '~1 min', what: 'K-Means scatter plot, cluster separation' },
          { page: 'Live Monitor', time: '~2 min', what: 'Launch DoS simulation, watch real-time detection' },
        ].map((d, i) => (
          <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 + i * 0.1 }} className="flex items-center justify-between py-3 border-b border-white/5">
            <div className="flex items-center gap-4">
              <span className="text-green-400 font-mono text-sm font-bold">{i + 1}.</span>
              <span className="text-white text-sm font-bold">{d.page}</span>
            </div>
            <span className="text-xs text-gray-500 flex-1 mx-6">{d.what}</span>
            <span className="text-green-400 font-mono text-xs">{d.time}</span>
          </motion.div>
        ))}
        <p className="text-green-400 font-mono text-sm mt-8">Total: ~10 minutes</p>
      </div>
    )
  },
  {
    id: 'conclusion',
    content: (
      <div className="px-16 md:px-24 py-12">
        <h2 className="text-4xl font-mono font-bold text-cyan-400 mb-2">CONCLUSION & FUTURE WORK</h2>
        <div className="w-20 h-0.5 bg-cyan-500 mb-8" />
        <div className="grid md:grid-cols-2 gap-10">
          <div>
            <h3 className="text-lg font-bold text-white mb-4">Key Achievements</h3>
            {['Full-stack IDS with real-time monitoring','94.30% accuracy with Decision Tree','Interactive 3D security visualizations','Live DoS simulation validation'].map((a, i) => (
              <div key={i} className="flex gap-3 mb-2"><span className="text-green-500">+</span><span className="text-sm text-gray-400">{a}</span></div>
            ))}
          </div>
          <div>
            <h3 className="text-lg font-bold text-white mb-4">Future Improvements</h3>
            {['Deep learning models (LSTM, CNN)','More attack categories & datasets','Network appliance deployment','SIEM integration'].map((f, i) => (
              <div key={i} className="flex gap-3 mb-2"><span className="text-purple-500">&gt;</span><span className="text-sm text-gray-400">{f}</span></div>
            ))}
          </div>
        </div>
      </div>
    )
  },
  {
    id: 'qa',
    content: (
      <div className="flex flex-col justify-center items-center h-full text-center">
        <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.5 }}>
          <h1 className="text-7xl md:text-8xl font-bold font-mono text-white mb-6">QUESTIONS?</h1>
          <p className="text-xl text-cyan-400 mb-8">Thank you for your attention</p>
          <div className="w-20 h-0.5 bg-cyan-500 mx-auto mb-8" />
          <p className="text-sm text-gray-500">Zakarya Oukil &bull; Master 1 Cybersecurity &bull; HIS 2025/2026</p>
        </motion.div>
      </div>
    )
  },
];

export default function PresentationPage() {
  const [current, setCurrent] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const next = useCallback(() => setCurrent(p => Math.min(p + 1, slides.length - 1)), []);
  const prev = useCallback(() => setCurrent(p => Math.max(p - 1, 0)), []);

  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); next(); }
      if (e.key === 'ArrowLeft') prev();
      if (e.key === 'f' || e.key === 'F') toggleFullscreen();
      if (e.key === 'Escape') exitFullscreen();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [next, prev]);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const exitFullscreen = () => {
    if (document.fullscreenElement) {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-[#08080C] overflow-hidden select-none" data-testid="presentation-page">
      {/* Slide */}
      <div className="flex-1 relative overflow-hidden">
        {/* Corner brackets */}
        <div className="absolute top-4 left-4 w-6 h-6 border-l-2 border-t-2 border-cyan-500/20 z-10" />
        <div className="absolute top-4 right-4 w-6 h-6 border-r-2 border-t-2 border-cyan-500/20 z-10" />
        <div className="absolute bottom-4 left-4 w-6 h-6 border-l-2 border-b-2 border-cyan-500/20 z-10" />
        <div className="absolute bottom-4 right-4 w-6 h-6 border-r-2 border-b-2 border-cyan-500/20 z-10" />

        {/* Top accent */}
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-50" />

        <AnimatePresence mode="wait">
          <motion.div
            key={current}
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -50 }}
            transition={{ duration: 0.3 }}
            className="h-full"
          >
            {slides[current].content}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Bottom bar */}
      <div className="h-14 border-t border-white/5 flex items-center justify-between px-6 bg-black/50">
        <div className="flex items-center gap-3">
          <button onClick={() => window.location.href = '/'} className="text-gray-600 hover:text-cyan-400 transition-colors" title="Home">
            <Home className="w-4 h-4" />
          </button>
          <a href={`${BACKEND_URL}/api/presentation/download`} className="text-gray-600 hover:text-cyan-400 transition-colors" title="Download PDF">
            <Download className="w-4 h-4" />
          </a>
          <button onClick={toggleFullscreen} className="text-gray-600 hover:text-cyan-400 transition-colors" title="Fullscreen (F)">
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button onClick={prev} disabled={current === 0} className="p-2 text-gray-500 hover:text-white disabled:opacity-20 transition-colors">
            <ChevronLeft className="w-5 h-5" />
          </button>
          <span className="font-mono text-xs text-gray-500 w-16 text-center">{current + 1} / {slides.length}</span>
          <button onClick={next} disabled={current === slides.length - 1} className="p-2 text-gray-500 hover:text-white disabled:opacity-20 transition-colors">
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>

        {/* Progress bar */}
        <div className="w-32 h-0.5 bg-white/5 rounded-full overflow-hidden">
          <motion.div className="h-full bg-cyan-500" animate={{ width: `${((current + 1) / slides.length) * 100}%` }} />
        </div>
      </div>
    </div>
  );
}
