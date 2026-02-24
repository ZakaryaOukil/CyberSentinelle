import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, Download, Home, Maximize2, Minimize2, Sun, Moon } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const getSlides = (light) => {
  const t = {
    bg: light ? '#F5F5F0' : '#08080C',
    text: light ? '#0A0A0A' : '#FFFFFF',
    muted: light ? '#1A1A1A' : '#E5E7EB',
    dim: light ? '#333333' : '#D1D5DB',
    faint: light ? '#555555' : '#9CA3AF',
    accent: light ? '#0891B2' : '#00F0FF',
    accentSoft: light ? 'rgba(8,145,178,0.08)' : 'rgba(0,240,255,0.05)',
    green: light ? '#059669' : '#00FF41',
    red: light ? '#DC2626' : '#FF003C',
    purple: light ? '#7C3AED' : '#BD00FF',
    yellow: light ? '#CA8A04' : '#FAFF00',
    border: light ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.05)',
    borderAccent: light ? 'rgba(8,145,178,0.25)' : 'rgba(0,240,255,0.2)',
    cardBg: light ? 'rgba(0,0,0,0.03)' : 'rgba(255,255,255,0.02)',
    shimmer: light ? 'shimmer-text-light' : 'shimmer-text',
  };

  return [
    {
      id: 'title',
      content: (
        <div className="flex flex-col justify-center items-start h-full px-16 md:px-24">
          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <div className="text-lg font-mono tracking-[0.3em] mb-8" style={{ color: `${t.accent}99` }}>NETWORK INTRUSION DETECTION SYSTEM</div>
            <h1 className="text-8xl md:text-[10rem] font-bold font-mono leading-none mb-2" style={{ color: t.text }}>CYBER</h1>
            <h1 className={`text-8xl md:text-[10rem] font-bold font-mono leading-none mb-10 ${t.shimmer}`}>SENTINELLE</h1>
            <div className="w-40 h-1 mb-10" style={{ background: `linear-gradient(to right, ${t.accent}, transparent)` }} />
            <p className="text-2xl mb-3" style={{ color: t.muted }}>Zakarya Oukil</p>
            <p className="text-xl" style={{ color: t.dim }}>Master 1 Cybersecurity &bull; HIS 2025/2026</p>
            <p className="text-xl" style={{ color: t.dim }}>NSL-KDD Dataset &bull; Decision Tree Classifier</p>
          </motion.div>
        </div>
      )
    },
    {
      id: 'agenda',
      content: (
        <div className="px-16 md:px-24 py-10">
          <h2 className="text-5xl font-mono font-bold mb-3" style={{ color: t.accent }}>AGENDA</h2>
          <div className="w-24 h-1 mb-10" style={{ backgroundColor: t.accent }} />
          <div className="grid md:grid-cols-2 gap-x-16 gap-y-3">
            {['Problem Statement & Context','Objectives','Dataset: NSL-KDD','System Architecture','Exploratory Data Analysis','Decision Tree Model','Model Performance & Metrics','K-Means Clustering','Real-Time Detection','DoS Attack Simulation','Live Demo','Conclusion & Future Work'].map((item, i) => (
              <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.08 * i }}
                className="flex items-center gap-5 py-3" style={{ borderBottom: `1px solid ${t.border}` }}>
                <span className="font-mono text-lg w-10" style={{ color: `${t.accent}80` }}>{String(i + 1).padStart(2, '0')}</span>
                <span className="text-xl" style={{ color: t.muted }}>{item}</span>
              </motion.div>
            ))}
          </div>
        </div>
      )
    },
    {
      id: 'problem',
      content: (
        <div className="px-16 md:px-24 py-10">
          <h2 className="text-5xl font-mono font-bold mb-3" style={{ color: t.accent }}>PROBLEM STATEMENT</h2>
          <div className="w-24 h-1 mb-10" style={{ backgroundColor: t.accent }} />
          <div className="grid md:grid-cols-2 gap-16">
            <div>
              <h3 className="text-2xl font-bold mb-6" style={{ color: t.text }}>The Challenge</h3>
              {['Cyberattacks are increasing in frequency and sophistication','Traditional firewalls and signature-based systems are insufficient','Need for intelligent, automated threat detection'].map((p, i) => (
                <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 + i * 0.15 }} className="flex gap-4 mb-4">
                  <span className="text-xl mt-0.5" style={{ color: t.red }}>&gt;</span>
                  <span className="text-xl leading-relaxed" style={{ color: t.muted }}>{p}</span>
                </motion.div>
              ))}
            </div>
            <div>
              <h3 className="text-2xl font-bold mb-6" style={{ color: t.text }}>Our Solution</h3>
              {['ML-based Network Intrusion Detection System (NIDS)','Classifies network traffic as Normal or Attack in real-time','Identifies attack categories: DoS, Probe, R2L, U2R'].map((p, i) => (
                <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 + i * 0.15 }} className="flex gap-4 mb-4">
                  <span className="text-xl mt-0.5" style={{ color: t.green }}>&gt;</span>
                  <span className="text-xl leading-relaxed" style={{ color: t.muted }}>{p}</span>
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
        <div className="px-16 md:px-24 py-10">
          <h2 className="text-5xl font-mono font-bold mb-3" style={{ color: t.accent }}>OBJECTIVES</h2>
          <div className="w-24 h-1 mb-10" style={{ backgroundColor: t.accent }} />
          {[
            { n: '01', ti: 'Data Analysis', d: 'Comprehensive EDA on NSL-KDD to understand traffic patterns' },
            { n: '02', ti: 'Model Training', d: 'Train a Decision Tree classifier for accurate traffic classification' },
            { n: '03', ti: 'Real-Time Detection', d: 'Build a monitoring system that detects intrusions as they happen' },
            { n: '04', ti: 'Attack Simulation', d: 'Controlled DoS simulation to validate detection capabilities' },
            { n: '05', ti: 'Visualization', d: 'Interactive web dashboard with 3D visualizations' },
          ].map((obj, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 + i * 0.1 }} className="flex gap-6 mb-6 items-start">
              <span className="text-4xl font-mono font-bold" style={{ color: `${t.accent}40` }}>{obj.n}</span>
              <div>
                <h4 className="text-xl font-bold" style={{ color: t.accent }}>{obj.ti}</h4>
                <p className="text-lg" style={{ color: t.muted }}>{obj.d}</p>
              </div>
            </motion.div>
          ))}
        </div>
      )
    },
    {
      id: 'dataset',
      content: (
        <div className="px-16 md:px-24 py-10">
          <h2 className="text-5xl font-mono font-bold mb-3" style={{ color: t.accent }}>DATASET: NSL-KDD</h2>
          <div className="w-24 h-1 mb-8" style={{ backgroundColor: t.accent }} />
          <div className="grid grid-cols-4 gap-5 mb-10">
            {[{ v: '125,973', l: 'Training Samples' },{ v: '41', l: 'Features' },{ v: '5', l: 'Traffic Classes' },{ v: '2', l: 'Categories' }].map((s, i) => (
              <motion.div key={i} initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: 0.2 + i * 0.1 }}
                className="text-center p-6" style={{ border: `1px solid ${t.borderAccent}`, backgroundColor: t.accentSoft }}>
                <div className="text-4xl font-mono font-bold" style={{ color: t.accent }}>{s.v}</div>
                <div className="text-sm tracking-widest mt-2" style={{ color: t.dim }}>{s.l}</div>
              </motion.div>
            ))}
          </div>
          <div className="grid md:grid-cols-2 gap-8">
            {[
              { ti: 'Basic Features', d: 'duration, protocol_type, service, flag, src_bytes, dst_bytes' },
              { ti: 'Content Features', d: 'hot, num_failed_logins, logged_in, root_shell, num_file_creations' },
              { ti: 'Traffic Features', d: 'count, srv_count, serror_rate, same_srv_rate, dst_host_count' },
              { ti: 'Attack Types', d: 'Normal, DoS (neptune, smurf), Probe (portsweep), R2L, U2R' },
            ].map((f, i) => (
              <div key={i} className="pl-5" style={{ borderLeft: `3px solid ${t.borderAccent}` }}>
                <h4 className="text-lg font-bold mb-1" style={{ color: t.text }}>{f.ti}</h4>
                <p className="text-base" style={{ color: t.dim }}>{f.d}</p>
              </div>
            ))}
          </div>
        </div>
      )
    },
    {
      id: 'architecture',
      content: (
        <div className="px-16 md:px-24 py-10">
          <h2 className="text-5xl font-mono font-bold mb-3" style={{ color: t.accent }}>SYSTEM ARCHITECTURE</h2>
          <div className="w-24 h-1 mb-10" style={{ backgroundColor: t.accent }} />
          <div className="space-y-5">
            {[
              { name: 'FRONTEND', desc: 'React 19 | 3D Canvas Visualizations | Recharts', color: t.accent },
              { name: 'BACKEND', desc: 'Python FastAPI | RESTful API | Real-Time Monitor', color: t.green },
              { name: 'ML ENGINE', desc: 'Scikit-learn | Decision Tree | K-Means Clustering', color: t.purple },
              { name: 'DATABASE', desc: 'MongoDB Atlas | Model Storage | Traffic Logs', color: t.yellow },
            ].map((layer, i) => (
              <motion.div key={i} initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 + i * 0.15 }}
                className="flex items-center gap-6 p-6" style={{ border: `1px solid ${layer.color}30`, backgroundColor: t.cardBg }}>
                <div className="w-1.5 h-12 rounded-full" style={{ backgroundColor: layer.color }} />
                <div>
                  <h4 className="font-mono font-bold text-lg tracking-wider" style={{ color: layer.color }}>{layer.name}</h4>
                  <p className="text-base mt-1" style={{ color: t.dim }}>{layer.desc}</p>
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
        <div className="px-16 md:px-24 py-10">
          <h2 className="text-5xl font-mono font-bold mb-3" style={{ color: t.accent }}>DECISION TREE MODEL</h2>
          <div className="w-24 h-1 mb-10" style={{ backgroundColor: t.accent }} />
          <div className="grid md:grid-cols-2 gap-14">
            <div>
              <h3 className="text-2xl font-bold mb-6" style={{ color: t.text }}>Why Decision Tree?</h3>
              {['Highly interpretable - visualize decisions','Fast training and prediction time','Handles numerical & categorical features','No feature scaling required','Works well with structured data'].map((p, i) => (
                <div key={i} className="flex gap-4 mb-3"><span className="text-xl" style={{ color: t.accent }}>+</span><span className="text-xl" style={{ color: t.muted }}>{p}</span></div>
              ))}
            </div>
            <div>
              <h3 className="text-2xl font-bold mb-6" style={{ color: t.text }}>Training Process</h3>
              {['Label encoding for categorical features','StandardScaler normalization','80/20 train-test split','Gini criterion optimization'].map((p, i) => (
                <div key={i} className="flex gap-4 mb-3"><span className="text-xl" style={{ color: t.purple }}>&gt;</span><span className="text-xl" style={{ color: t.muted }}>{p}</span></div>
              ))}
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'performance',
      content: (
        <div className="px-16 md:px-24 py-10">
          <h2 className="text-5xl font-mono font-bold mb-3" style={{ color: t.accent }}>MODEL PERFORMANCE</h2>
          <div className="w-24 h-1 mb-10" style={{ backgroundColor: t.accent }} />
          <div className="space-y-5">
            {[
              { name: 'ACCURACY', val: '94.30%', desc: 'Correct predictions out of total' },
              { name: 'PRECISION', val: '94.36%', desc: 'True positives among predicted positives' },
              { name: 'RECALL', val: '94.30%', desc: 'True positives among actual positives' },
              { name: 'F1-SCORE', val: '94.30%', desc: 'Harmonic mean of precision and recall' },
              { name: 'AUC', val: '0.9463', desc: 'Area Under the ROC Curve' },
            ].map((m, i) => (
              <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 + i * 0.1 }}
                className="flex items-center justify-between p-5" style={{ border: `1px solid ${t.borderAccent}`, backgroundColor: t.cardBg }}>
                <div className="flex items-center gap-8">
                  <span className="font-mono font-bold text-lg w-28" style={{ color: t.accent }}>{m.name}</span>
                  <span className="font-mono font-bold text-3xl" style={{ color: t.green }}>{m.val}</span>
                </div>
                <span className="text-base" style={{ color: t.dim }}>{m.desc}</span>
              </motion.div>
            ))}
          </div>
        </div>
      )
    },
    {
      id: 'clustering',
      content: (
        <div className="px-16 md:px-24 py-10">
          <h2 className="text-5xl font-mono font-bold mb-3" style={{ color: t.purple }}>K-MEANS CLUSTERING</h2>
          <div className="w-24 h-1 mb-10" style={{ backgroundColor: t.purple }} />
          {[
            { ti: 'Purpose', d: 'Unsupervised learning to discover hidden patterns without labels' },
            { ti: 'Method', d: 'K-Means with PCA dimensionality reduction for 2D visualization' },
            { ti: 'Results', d: 'Clusters naturally separate normal from malicious traffic' },
            { ti: 'Application', d: 'Detect novel/zero-day attacks the supervised model has never seen' },
          ].map((item, i) => (
            <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 + i * 0.15 }} className="mb-8">
              <h4 className="font-mono text-lg tracking-wider mb-2" style={{ color: t.purple }}>{item.ti}</h4>
              <p className="text-xl pl-5" style={{ color: t.muted, borderLeft: `3px solid ${t.purple}30` }}>{item.d}</p>
            </motion.div>
          ))}
        </div>
      )
    },
    {
      id: 'realtime',
      content: (
        <div className="px-16 md:px-24 py-10">
          <h2 className="text-5xl font-mono font-bold mb-3" style={{ color: t.accent }}>REAL-TIME DETECTION</h2>
          <div className="w-24 h-1 mb-10" style={{ backgroundColor: t.accent }} />
          {[
            { ti: 'Live Traffic Monitor', d: 'Tracks requests/second with visual charts', c: t.accent },
            { ti: 'Anomaly Detection', d: 'Threshold-based detection for traffic spikes', c: t.red },
            { ti: 'Visual Alerts', d: 'Immediate alerts with attack classification', c: t.yellow },
            { ti: 'Security Logs', d: 'Timestamped event log for forensics', c: t.green },
            { ti: 'Radar Scanner', d: 'Interactive radar showing network activity', c: t.purple },
          ].map((f, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 + i * 0.1 }} className="flex items-center gap-5 mb-5">
              <div className="w-1.5 h-10 rounded-full" style={{ backgroundColor: f.c }} />
              <div>
                <h4 className="font-mono text-lg font-bold" style={{ color: f.c }}>{f.ti}</h4>
                <p className="text-base" style={{ color: t.dim }}>{f.d}</p>
              </div>
            </motion.div>
          ))}
        </div>
      )
    },
    {
      id: 'dos-sim',
      content: (
        <div className="px-16 md:px-24 py-10">
          <h2 className="text-5xl font-mono font-bold mb-3" style={{ color: t.red }}>DoS ATTACK SIMULATION</h2>
          <div className="w-24 h-1 mb-8" style={{ backgroundColor: t.red }} />
          <p className="text-lg mb-10" style={{ color: t.dim }}>Controlled simulation to validate our detection system in real conditions</p>
          {[
            { s: 'Step 1', ti: 'Baseline', d: 'Normal traffic at 0-2 req/s, green status' },
            { s: 'Step 2', ti: 'Launch Attack', d: '100+ requests/second flood to the server' },
            { s: 'Step 3', ti: 'Detection', d: 'System detects anomaly, triggers red alert' },
            { s: 'Step 4', ti: 'Classification', d: 'ML model classifies traffic as DoS' },
            { s: 'Step 5', ti: 'Response', d: 'Security logs record event with severity' },
          ].map((step, i) => (
            <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 + i * 0.12 }} className="flex items-center gap-6 mb-4">
              <span className="font-mono text-base w-16" style={{ color: `${t.red}80` }}>{step.s}</span>
              <span className="font-bold text-lg w-40" style={{ color: t.text }}>{step.ti}</span>
              <span className="text-lg" style={{ color: t.dim }}>{step.d}</span>
            </motion.div>
          ))}
        </div>
      )
    },
    {
      id: 'demo',
      content: (
        <div className="px-16 md:px-24 py-10">
          <h2 className="text-5xl font-mono font-bold mb-3" style={{ color: t.green }}>LIVE DEMO PLAN</h2>
          <div className="w-24 h-1 mb-10" style={{ backgroundColor: t.green }} />
          {[
            { page: 'Homepage', time: '~1 min', what: '3D globe, animated stats, system overview' },
            { page: 'Dashboard EDA', time: '~2 min', what: 'Charts, 3D network topology, attack distribution' },
            { page: 'Model Performance', time: '~2 min', what: 'DT metrics, ROC curve, 3D threat visualization' },
            { page: 'Live Prediction', time: '~2 min', what: 'Demo Normal -> NORMAL, Demo Attack -> INTRUSION' },
            { page: 'Clustering', time: '~1 min', what: 'K-Means scatter plot, cluster separation' },
            { page: 'Live Monitor', time: '~2 min', what: 'Launch DoS simulation, watch real-time detection' },
          ].map((d, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 + i * 0.1 }}
              className="flex items-center justify-between py-4" style={{ borderBottom: `1px solid ${t.border}` }}>
              <div className="flex items-center gap-5">
                <span className="font-mono text-lg font-bold" style={{ color: t.green }}>{i + 1}.</span>
                <span className="text-lg font-bold" style={{ color: t.text }}>{d.page}</span>
              </div>
              <span className="text-base flex-1 mx-8" style={{ color: t.dim }}>{d.what}</span>
              <span className="font-mono text-base" style={{ color: t.green }}>{d.time}</span>
            </motion.div>
          ))}
          <p className="font-mono text-lg mt-10" style={{ color: t.green }}>Total: ~10 minutes</p>
        </div>
      )
    },
    {
      id: 'conclusion',
      content: (
        <div className="px-16 md:px-24 py-10">
          <h2 className="text-5xl font-mono font-bold mb-3" style={{ color: t.accent }}>CONCLUSION & FUTURE WORK</h2>
          <div className="w-24 h-1 mb-10" style={{ backgroundColor: t.accent }} />
          <div className="grid md:grid-cols-2 gap-14">
            <div>
              <h3 className="text-2xl font-bold mb-6" style={{ color: t.text }}>Key Achievements</h3>
              {['Full-stack IDS with real-time monitoring','94.30% accuracy with Decision Tree','Interactive 3D security visualizations','Live DoS simulation validation'].map((a, i) => (
                <div key={i} className="flex gap-4 mb-3"><span className="text-xl" style={{ color: t.green }}>+</span><span className="text-xl" style={{ color: t.muted }}>{a}</span></div>
              ))}
            </div>
            <div>
              <h3 className="text-2xl font-bold mb-6" style={{ color: t.text }}>Future Improvements</h3>
              {['Deep learning models (LSTM, CNN)','More attack categories & datasets','Network appliance deployment','SIEM integration'].map((f, i) => (
                <div key={i} className="flex gap-4 mb-3"><span className="text-xl" style={{ color: t.purple }}>&gt;</span><span className="text-xl" style={{ color: t.muted }}>{f}</span></div>
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
            <h1 className="text-8xl md:text-9xl font-bold font-mono mb-8" style={{ color: t.text }}>QUESTIONS?</h1>
            <p className="text-3xl mb-10" style={{ color: t.accent }}>Thank you for your attention</p>
            <div className="w-24 h-1 mx-auto mb-10" style={{ backgroundColor: t.accent }} />
            <p className="text-xl" style={{ color: t.dim }}>Zakarya Oukil &bull; Master 1 Cybersecurity &bull; HIS 2025/2026</p>
          </motion.div>
        </div>
      )
    },
  ];
};

export default function PresentationPage() {
  const [current, setCurrent] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isLight, setIsLight] = useState(false);

  const slides = getSlides(isLight);

  const next = useCallback(() => setCurrent(p => Math.min(p + 1, 13)), []);
  const prev = useCallback(() => setCurrent(p => Math.max(p - 1, 0)), []);

  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); next(); }
      if (e.key === 'ArrowLeft') prev();
      if (e.key === 'f' || e.key === 'F') toggleFullscreen();
      if (e.key === 'Escape') exitFullscreen();
      if (e.key === 't' || e.key === 'T') setIsLight(v => !v);
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

  const bgColor = isLight ? '#F5F5F0' : '#08080C';
  const barBg = isLight ? 'rgba(0,0,0,0.04)' : 'rgba(0,0,0,0.5)';
  const barBorder = isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.05)';
  const iconColor = isLight ? '#555' : '#6B7280';
  const iconHover = isLight ? '#0891B2' : '#00F0FF';
  const progressBg = isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.05)';
  const progressFill = isLight ? '#0891B2' : '#00F0FF';
  const topAccent = isLight ? '#0891B2' : '#00F0FF';
  const cornerBorder = isLight ? 'rgba(8,145,178,0.2)' : 'rgba(0,240,255,0.2)';

  return (
    <div className="h-screen flex flex-col overflow-hidden select-none transition-colors duration-300" style={{ backgroundColor: bgColor }} data-testid="presentation-page">
      {/* Slide area */}
      <div className="flex-1 relative overflow-hidden">
        {/* Corner brackets */}
        <div className="absolute top-4 left-4 w-6 h-6 z-10" style={{ borderLeft: `2px solid ${cornerBorder}`, borderTop: `2px solid ${cornerBorder}` }} />
        <div className="absolute top-4 right-4 w-6 h-6 z-10" style={{ borderRight: `2px solid ${cornerBorder}`, borderTop: `2px solid ${cornerBorder}` }} />
        <div className="absolute bottom-4 left-4 w-6 h-6 z-10" style={{ borderLeft: `2px solid ${cornerBorder}`, borderBottom: `2px solid ${cornerBorder}` }} />
        <div className="absolute bottom-4 right-4 w-6 h-6 z-10" style={{ borderRight: `2px solid ${cornerBorder}`, borderBottom: `2px solid ${cornerBorder}` }} />

        {/* Top accent line */}
        <div className="absolute top-0 left-0 right-0 h-[2px] opacity-50" style={{ background: `linear-gradient(to right, transparent, ${topAccent}, transparent)` }} />

        <AnimatePresence mode="wait">
          <motion.div
            key={`${current}-${isLight}`}
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
      <div className="h-16 flex items-center justify-between px-6 transition-colors duration-300" style={{ backgroundColor: barBg, borderTop: `1px solid ${barBorder}` }}>
        <div className="flex items-center gap-4">
          <button onClick={() => window.location.href = '/'} className="transition-colors p-1" style={{ color: iconColor }}
            onMouseEnter={e => e.currentTarget.style.color = iconHover} onMouseLeave={e => e.currentTarget.style.color = iconColor}
            title="Home" data-testid="pres-home-btn">
            <Home className="w-5 h-5" />
          </button>
          <a href={`${BACKEND_URL}/api/presentation/download`} className="transition-colors p-1" style={{ color: iconColor }}
            onMouseEnter={e => e.currentTarget.style.color = iconHover} onMouseLeave={e => e.currentTarget.style.color = iconColor}
            title="Download PDF" data-testid="pres-download-btn">
            <Download className="w-5 h-5" />
          </a>
          <button onClick={toggleFullscreen} className="transition-colors p-1" style={{ color: iconColor }}
            onMouseEnter={e => e.currentTarget.style.color = iconHover} onMouseLeave={e => e.currentTarget.style.color = iconColor}
            title="Fullscreen (F)" data-testid="pres-fullscreen-btn">
            {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
          </button>
          {/* Theme toggle */}
          <button onClick={() => setIsLight(v => !v)} className="flex items-center gap-2 px-3 py-1.5 rounded-full transition-all duration-300"
            style={{
              backgroundColor: isLight ? 'rgba(8,145,178,0.1)' : 'rgba(255,255,255,0.06)',
              border: `1px solid ${isLight ? 'rgba(8,145,178,0.3)' : 'rgba(255,255,255,0.1)'}`,
              color: isLight ? '#0891B2' : '#FAFF00'
            }}
            title="Toggle theme (T)" data-testid="pres-theme-toggle">
            {isLight ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
            <span className="font-mono text-xs tracking-wider">{isLight ? 'DARK' : 'LIGHT'}</span>
          </button>
        </div>

        <div className="flex items-center gap-3">
          <button onClick={prev} disabled={current === 0} className="p-2 transition-colors disabled:opacity-20" style={{ color: iconColor }}
            data-testid="pres-prev-btn">
            <ChevronLeft className="w-6 h-6" />
          </button>
          <span className="font-mono text-sm w-20 text-center" style={{ color: iconColor }}>{current + 1} / {slides.length}</span>
          <button onClick={next} disabled={current === slides.length - 1} className="p-2 transition-colors disabled:opacity-20" style={{ color: iconColor }}
            data-testid="pres-next-btn">
            <ChevronRight className="w-6 h-6" />
          </button>
        </div>

        {/* Progress bar */}
        <div className="w-40 h-1 rounded-full overflow-hidden" style={{ backgroundColor: progressBg }}>
          <motion.div className="h-full rounded-full" style={{ backgroundColor: progressFill }} animate={{ width: `${((current + 1) / slides.length) * 100}%` }} />
        </div>
      </div>
    </div>
  );
}
