import React from 'react';
import { motion } from 'framer-motion';
import { Shield, ChevronRight, Terminal, Cpu, Activity, Download, Zap, Globe } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent } from '../components/ui/card';
import CyberGlobe from '../components/CyberGlobe';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

export default function HomePage() {
  const handleDownloadNotebook = async () => {
    try {
      window.open(`${BACKEND_URL}/api/notebook/download`, '_blank');
    } catch (error) {
      console.error('Erreur téléchargement:', error);
    }
  };

  const features = [
    {
      icon: Activity,
      title: "MONITORING TEMPS RÉEL",
      description: "Surveillance du trafic réseau avec détection automatique des anomalies",
      color: "#00F0FF"
    },
    {
      icon: Cpu,
      title: "MACHINE LEARNING",
      description: "Modèles Random Forest & Decision Tree entraînés sur NSL-KDD",
      color: "#BD00FF"
    },
    {
      icon: Shield,
      title: "DÉTECTION D'INTRUSIONS",
      description: "Classification des attaques DoS, Probe, R2L, U2R en temps réel",
      color: "#00FF41"
    },
    {
      icon: Globe,
      title: "ANALYSE GLOBALE",
      description: "Visualisation des patterns d'attaque et clustering K-Means",
      color: "#FAFF00"
    }
  ];

  const stats = [
    { value: "97.7%", label: "PRÉCISION", color: "#00F0FF" },
    { value: "125K+", label: "ÉCHANTILLONS", color: "#BD00FF" },
    { value: "41", label: "FEATURES", color: "#00FF41" },
    { value: "5", label: "MODÈLES", color: "#FAFF00" }
  ];

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Hero Section */}
      <section className="relative min-h-[80vh] flex items-center">
        {/* 3D Globe Background */}
        <div className="absolute inset-0 opacity-60">
          <Suspense fallback={<div className="w-full h-full bg-gradient-to-b from-transparent to-black/50" />}>
            <CyberGlobe showAlerts={true} />
          </Suspense>
        </div>
        
        {/* Content */}
        <div className="relative z-10 w-full px-6 md:px-12">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="max-w-3xl"
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-4 py-2 mb-6 border border-cyan-500/30 bg-cyan-500/10"
            >
              <span className="w-2 h-2 bg-cyan-400 animate-pulse" />
              <span className="text-cyan-400 font-mono text-sm tracking-widest uppercase">
                Système de Détection d'Intrusions
              </span>
            </motion.div>
            
            {/* Title */}
            <h1 className="text-5xl md:text-7xl font-bold font-mono mb-6 leading-tight">
              <span className="text-white">CYBER</span>
              <span className="neon-cyan">SENTINELLE</span>
            </h1>
            
            {/* Description */}
            <p className="text-xl md:text-2xl text-gray-400 mb-8 font-light leading-relaxed max-w-2xl">
              Plateforme avancée de détection d'intrusions réseau utilisant 
              <span className="text-cyan-400"> l'intelligence artificielle </span>
              et le <span className="text-purple-400">machine learning</span>.
            </p>
            
            {/* CTA Buttons */}
            <div className="flex flex-wrap gap-4">
              <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                <Button 
                  className="cyber-btn h-14 px-8 text-base"
                  onClick={() => window.location.href = '/monitor'}
                  data-testid="cta-monitor"
                >
                  <Zap className="w-5 h-5 mr-2" />
                  ACCÉDER AU MONITEUR
                  <ChevronRight className="w-5 h-5 ml-2" />
                </Button>
              </motion.div>
              
              <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                <Button 
                  variant="outline"
                  className="h-14 px-8 text-base border-gray-700 text-gray-300 hover:bg-white/5 hover:text-white font-mono tracking-wider"
                  onClick={handleDownloadNotebook}
                  data-testid="cta-download"
                >
                  <Download className="w-5 h-5 mr-2" />
                  TÉLÉCHARGER NOTEBOOK
                </Button>
              </motion.div>
            </div>
          </motion.div>
        </div>

        {/* Scroll indicator */}
        <motion.div 
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <div className="w-6 h-10 border-2 border-cyan-500/30 rounded-full flex justify-center pt-2">
            <div className="w-1 h-3 bg-cyan-400 rounded-full animate-pulse" />
          </div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="py-12 border-t border-white/5">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 px-6 md:px-12">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 + 0.5 }}
              className="text-center p-6 border border-white/5 bg-white/[0.02] hover:bg-white/[0.05] transition-colors"
            >
              <div 
                className="text-4xl md:text-5xl font-mono font-bold mb-2"
                style={{ color: stat.color }}
              >
                {stat.value}
              </div>
              <div className="text-xs text-gray-500 tracking-widest uppercase">
                {stat.label}
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-6 md:px-12">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="mb-12"
        >
          <h2 className="text-3xl md:text-4xl font-mono font-bold mb-4">
            <span className="text-gray-500">//</span> FONCTIONNALITÉS
          </h2>
          <div className="w-20 h-1 bg-gradient-to-r from-cyan-500 to-purple-500" />
        </motion.div>

        <div className="grid md:grid-cols-2 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <Card className="cyber-card h-full group">
                <CardContent className="p-6">
                  <div className="flex items-start gap-4">
                    <div 
                      className="p-3 border border-white/10"
                      style={{ 
                        backgroundColor: `${feature.color}10`,
                        boxShadow: `0 0 20px ${feature.color}20`
                      }}
                    >
                      <feature.icon className="w-6 h-6" style={{ color: feature.color }} />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-mono text-lg mb-2 group-hover:text-cyan-400 transition-colors">
                        {feature.title}
                      </h3>
                      <p className="text-gray-500 text-sm leading-relaxed">
                        {feature.description}
                      </p>
                    </div>
                    <ChevronRight 
                      className="w-5 h-5 text-gray-600 group-hover:text-cyan-400 group-hover:translate-x-1 transition-all" 
                    />
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Terminal Section */}
      <section className="py-20 px-6 md:px-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto"
        >
          <div className="border border-white/10 bg-black/50 overflow-hidden">
            {/* Terminal Header */}
            <div className="flex items-center gap-2 px-4 py-3 bg-white/5 border-b border-white/10">
              <div className="flex gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500/80" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                <div className="w-3 h-3 rounded-full bg-green-500/80" />
              </div>
              <span className="ml-4 text-xs text-gray-500 font-mono">cybersentinelle@ids ~ </span>
            </div>
            
            {/* Terminal Content */}
            <div className="p-6 font-mono text-sm">
              <div className="text-gray-500 mb-2">$ ./cybersentinelle --status</div>
              <div className="text-cyan-400 mb-4">
                [<span className="text-green-400">✓</span>] Système initialisé avec succès
              </div>
              
              <div className="text-gray-500 mb-2">$ cat /var/log/ids/summary.log</div>
              <div className="space-y-1 text-gray-400 mb-4">
                <div><span className="text-purple-400">Modèle:</span> Random Forest Classifier</div>
                <div><span className="text-purple-400">Dataset:</span> NSL-KDD (125,973 échantillons)</div>
                <div><span className="text-purple-400">Accuracy:</span> <span className="text-green-400">97.70%</span></div>
                <div><span className="text-purple-400">Classes:</span> Normal, DoS, Probe, R2L, U2R</div>
              </div>
              
              <div className="text-gray-500 mb-2">$ ./monitor --realtime</div>
              <div className="flex items-center gap-2">
                <span className="text-cyan-400">[LIVE]</span>
                <span className="text-green-400 animate-pulse">● Monitoring actif</span>
                <span className="text-gray-600">|</span>
                <span className="text-gray-400">0 alertes</span>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-6 md:px-12 border-t border-white/5">
        <div className="flex flex-col md:flex-row justify-between items-center gap-4 text-sm text-gray-600">
          <div className="font-mono">
            <span className="text-cyan-500">&lt;</span>
            CyberSentinelle
            <span className="text-cyan-500">/&gt;</span>
            <span className="mx-2">|</span>
            Master 1 Cybersécurité
          </div>
          <div className="flex items-center gap-2">
            <span>Zakarya Oukil</span>
            <span className="text-gray-700">•</span>
            <span>HIS 2025/2026</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
