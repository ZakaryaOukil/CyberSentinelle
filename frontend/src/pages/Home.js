import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Shield, ChevronRight, Terminal, Cpu, Activity, Download, Zap, Globe, Lock, Wifi, Database, Server } from 'lucide-react';
import { Button } from '../components/ui/button';
import CyberGlobe from '../components/CyberGlobe';
import HexGrid from '../components/HexGrid';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

function TypeWriter({ text, delay = 0, speed = 40 }) {
  const [displayed, setDisplayed] = useState('');
  useEffect(() => {
    const timeout = setTimeout(() => {
      let i = 0;
      const interval = setInterval(() => {
        setDisplayed(text.slice(0, i + 1));
        i++;
        if (i >= text.length) clearInterval(interval);
      }, speed);
      return () => clearInterval(interval);
    }, delay);
    return () => clearTimeout(timeout);
  }, [text, delay, speed]);
  return <span>{displayed}<span className="animate-pulse text-cyan-400">|</span></span>;
}

function AnimatedCounter({ end, duration = 2000, suffix = '' }) {
  const [count, setCount] = useState(0);
  useEffect(() => {
    const start = 0;
    const increment = end / (duration / 16);
    let current = start;
    const timer = setInterval(() => {
      current += increment;
      if (current >= end) {
        setCount(end);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current * 10) / 10);
      }
    }, 16);
    return () => clearInterval(timer);
  }, [end, duration]);
  return <>{count}{suffix}</>;
}

export default function HomePage() {
  const handleDownloadNotebook = async () => {
    try { window.open(`${BACKEND_URL}/api/notebook/download`, '_blank'); }
    catch (error) { console.error('Erreur:', error); }
  };

  const features = [
    { icon: Activity, title: "MONITORING TEMPS RÉEL", desc: "Surveillance du trafic réseau avec détection automatique des anomalies et alertes en temps réel", color: "#00F0FF", link: "/monitor" },
    { icon: Cpu, title: "MACHINE LEARNING", desc: "Modèles Random Forest & Decision Tree entraînés sur le dataset NSL-KDD avec 97.7% de précision", color: "#BD00FF", link: "/model" },
    { icon: Shield, title: "DÉTECTION D'INTRUSIONS", desc: "Classification des attaques DoS, Probe, R2L, U2R avec analyse des patterns en temps réel", color: "#00FF41", link: "/prediction" },
    { icon: Database, title: "ANALYSE EXPLORATOIRE", desc: "Visualisation 3D du réseau, distribution des attaques, clustering K-Means avancé", color: "#FAFF00", link: "/dashboard" },
    { icon: Wifi, title: "SIMULATION D'ATTAQUES", desc: "Testez le système avec des simulations DoS/DDoS contrôlées et observez la détection", color: "#FF003C", link: "/monitor" },
    { icon: Lock, title: "SÉCURITÉ AVANCÉE", desc: "Analyse approfondie des protocoles réseau TCP, UDP, ICMP avec 41 features extraites", color: "#00F0FF", link: "/dashboard" },
  ];

  const stats = [
    { value: 97.7, label: "PRÉCISION", suffix: "%", color: "#00F0FF" },
    { value: 125, label: "ÉCHANTILLONS", suffix: "K+", color: "#BD00FF" },
    { value: 41, label: "FEATURES", suffix: "", color: "#00FF41" },
    { value: 5, label: "CATÉGORIES", suffix: "", color: "#FAFF00" },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Hero Section */}
      <section className="relative min-h-[85vh] flex items-center">
        <div className="absolute inset-0 opacity-50">
          <CyberGlobe showAlerts={true} />
        </div>
        <div className="absolute inset-0">
          <HexGrid color="#00F0FF" opacity={0.025} />
        </div>
        
        <div className="relative z-10 w-full px-6 md:px-12">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
            className="max-w-3xl"
          >
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3, duration: 0.8 }}
              className="inline-flex items-center gap-3 px-5 py-2.5 mb-8 border border-cyan-500/30 bg-cyan-500/5 backdrop-blur-sm"
            >
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-400" />
              </span>
              <span className="text-cyan-400 font-mono text-sm tracking-[0.2em] uppercase">
                Système de Détection d'Intrusions v4.2
              </span>
            </motion.div>
            
            <h1 className="text-6xl md:text-8xl font-bold font-mono mb-6 leading-none">
              <motion.span 
                className="text-white block"
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4, duration: 0.8 }}
              >
                CYBER
              </motion.span>
              <motion.span 
                className="shimmer-text block"
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6, duration: 0.8 }}
              >
                SENTINELLE
              </motion.span>
            </h1>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
              className="text-lg md:text-xl text-gray-400 mb-10 font-light leading-relaxed max-w-2xl font-mono"
            >
              <TypeWriter 
                text="Plateforme avancée de détection d'intrusions réseau utilisant l'IA et le Machine Learning" 
                delay={1200}
                speed={25}
              />
            </motion.div>
            
            <motion.div 
              className="flex flex-wrap gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.5 }}
            >
              <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>
                <Button 
                  className="cyber-btn h-14 px-8 text-base group"
                  onClick={() => window.location.href = '/monitor'}
                  data-testid="cta-monitor"
                >
                  <Zap className="w-5 h-5 mr-2 group-hover:animate-pulse" />
                  ACCÉDER AU MONITEUR
                  <ChevronRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                </Button>
              </motion.div>
              
              <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>
                <Button 
                  variant="outline"
                  className="h-14 px-8 text-base border-gray-700 text-gray-300 hover:bg-white/5 hover:text-white hover:border-cyan-500/50 font-mono tracking-wider transition-all duration-300"
                  onClick={handleDownloadNotebook}
                  data-testid="cta-download"
                >
                  <Download className="w-5 h-5 mr-2" />
                  NOTEBOOK .IPYNB
                </Button>
              </motion.div>
            </motion.div>
          </motion.div>
        </div>

        <motion.div 
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          animate={{ y: [0, 12, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        >
          <div className="w-6 h-10 border-2 border-cyan-500/30 rounded-full flex justify-center pt-2">
            <div className="w-1 h-3 bg-cyan-400 rounded-full animate-pulse" />
          </div>
        </motion.div>
      </section>

      {/* Stats with counter animation */}
      <section className="py-16 border-t border-white/5 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-cyan-500/[0.02] to-transparent" />
        <motion.div 
          className="grid grid-cols-2 md:grid-cols-4 gap-4 px-6 md:px-12"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              variants={itemVariants}
              className="text-center p-8 border border-white/5 bg-white/[0.02] hover:bg-white/[0.05] transition-all duration-500 holo-card group"
              whileHover={{ borderColor: stat.color + '40' }}
            >
              <div 
                className="text-4xl md:text-6xl font-mono font-bold mb-3 transition-all duration-500"
                style={{ color: stat.color, textShadow: `0 0 20px ${stat.color}40` }}
              >
                <AnimatedCounter end={stat.value} suffix={stat.suffix} />
              </div>
              <div className="text-[10px] text-gray-500 tracking-[0.3em] uppercase">{stat.label}</div>
              <div className="glow-divider mt-4 group-hover:opacity-60 transition-opacity" style={{ background: `linear-gradient(90deg, transparent, ${stat.color}, transparent)` }} />
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* Features */}
      <section className="py-20 px-6 md:px-12">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <h2 className="text-3xl md:text-5xl font-mono font-bold mb-4">
            <span className="text-gray-600">//</span> FONCTIONNALITÉS
          </h2>
          <div className="w-24 h-0.5 bg-gradient-to-r from-cyan-500 via-purple-500 to-red-500" />
        </motion.div>

        <motion.div 
          className="grid md:grid-cols-2 lg:grid-cols-3 gap-5"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              variants={itemVariants}
              onClick={() => window.location.href = feature.link}
              className="cursor-pointer"
            >
              <div className="holo-card p-6 h-full group relative overflow-hidden">
                {/* Top accent line */}
                <div className="absolute top-0 left-0 right-0 h-[1px] opacity-0 group-hover:opacity-100 transition-opacity duration-500" 
                  style={{ background: `linear-gradient(90deg, transparent, ${feature.color}, transparent)` }} />
                
                <div className="flex items-start gap-4">
                  <div 
                    className="p-3 border transition-all duration-500 group-hover:shadow-lg"
                    style={{ 
                      borderColor: feature.color + '30',
                      backgroundColor: feature.color + '08',
                      boxShadow: `0 0 0px ${feature.color}00`
                    }}
                  >
                    <feature.icon className="w-6 h-6 transition-transform duration-500 group-hover:scale-110" style={{ color: feature.color }} />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-mono text-sm mb-2 group-hover:text-cyan-400 transition-colors tracking-wider">
                      {feature.title}
                    </h3>
                    <p className="text-gray-600 text-xs leading-relaxed font-light">
                      {feature.desc}
                    </p>
                  </div>
                </div>
                
                <div className="mt-4 flex items-center text-xs font-mono opacity-0 group-hover:opacity-100 transition-all duration-500 translate-y-2 group-hover:translate-y-0" style={{ color: feature.color }}>
                  <span className="tracking-wider">EXPLORER</span>
                  <ChevronRight className="w-3 h-3 ml-1 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* Terminal */}
      <section className="py-20 px-6 md:px-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto"
        >
          <div className="border border-white/10 bg-black/60 overflow-hidden backdrop-blur-sm holo-card">
            <div className="flex items-center gap-2 px-4 py-3 bg-white/5 border-b border-white/10">
              <div className="flex gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500/80 hover:bg-red-400 transition-colors cursor-pointer" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/80 hover:bg-yellow-400 transition-colors cursor-pointer" />
                <div className="w-3 h-3 rounded-full bg-green-500/80 hover:bg-green-400 transition-colors cursor-pointer" />
              </div>
              <span className="ml-4 text-xs text-gray-500 font-mono">cybersentinelle@ids:~$</span>
            </div>
            
            <div className="p-6 font-mono text-sm space-y-3">
              <div className="flex">
                <span className="text-cyan-600 mr-2">$</span>
                <span className="text-gray-400">./cybersentinelle --status</span>
              </div>
              <div className="text-cyan-400 flex items-center gap-2">
                <span className="text-green-400">&#10003;</span> Système initialisé avec succès
              </div>
              
              <div className="my-2 glow-divider" />
              
              <div className="flex">
                <span className="text-cyan-600 mr-2">$</span>
                <span className="text-gray-400">cat /var/log/ids/summary.log</span>
              </div>
              <div className="space-y-1 text-gray-400 pl-4 border-l border-cyan-500/20">
                <div><span className="text-purple-400">Modèle</span>: Random Forest Classifier</div>
                <div><span className="text-purple-400">Dataset</span>: NSL-KDD (<span className="text-cyan-400">125,973</span> échantillons)</div>
                <div><span className="text-purple-400">Accuracy</span>: <span className="text-green-400">97.70%</span></div>
                <div><span className="text-purple-400">AUC</span>: <span className="text-green-400">0.9971</span></div>
                <div><span className="text-purple-400">Classes</span>: Normal, DoS, Probe, R2L, U2R</div>
              </div>
              
              <div className="my-2 glow-divider" />
              
              <div className="flex">
                <span className="text-cyan-600 mr-2">$</span>
                <span className="text-gray-400">./monitor --realtime</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-cyan-400 text-xs tracking-wider">[LIVE]</span>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-400" />
                </span>
                <span className="text-green-400 text-xs">Monitoring actif</span>
                <span className="text-gray-700">|</span>
                <span className="text-gray-500 text-xs">0 alertes</span>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-6 md:px-12 border-t border-white/5">
        <div className="flex flex-col md:flex-row justify-between items-center gap-4 text-sm text-gray-600">
          <div className="font-mono">
            <span className="text-cyan-500">&lt;</span>CyberSentinelle<span className="text-cyan-500">/&gt;</span>
            <span className="mx-2 text-gray-800">|</span>
            Master 1 Cybersécurité
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span>Zakarya Oukil</span>
            <span className="text-gray-800">•</span>
            <span>HIS 2025/2026</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
