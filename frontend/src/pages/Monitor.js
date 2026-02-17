import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  Radio, AlertTriangle, CheckCircle, Brain, Globe, Zap,
  Target, X, FileText, Volume2, VolumeX, AlertCircle, Activity,
  Wifi, Shield, Skull
} from "lucide-react";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { toast } from "sonner";
import {
  AreaChart, Area, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const COLORS = {
  primary: "#00F0FF",
  danger: "#FF003C",
  warning: "#FAFF00",
  success: "#00FF41"
};

// Log Entry Component
function LogEntry({ log }) {
  const getColor = () => {
    if (log.severity === 'CRITICAL') return 'text-red-400';
    if (log.severity === 'WARNING') return 'text-yellow-400';
    return 'text-green-400';
  };
  
  const getBgColor = () => {
    if (log.severity === 'CRITICAL') return 'bg-red-500/10 border-red-500/20';
    if (log.severity === 'WARNING') return 'bg-yellow-500/10 border-yellow-500/20';
    return 'bg-green-500/10 border-green-500/20';
  };
  
  const timestamp = new Date(log.timestamp).toLocaleTimeString();
  
  return (
    <motion.div 
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className={`py-2 px-3 border-l-2 mb-2 ${getBgColor()}`}
    >
      <div className="flex items-center gap-2 text-xs text-gray-500 mb-1">
        <span className="font-mono">{timestamp}</span>
        <Badge variant="outline" className={`text-[10px] ${getColor()} border-current`}>
          {log.severity}
        </Badge>
      </div>
      <div className={`text-sm ${getColor()}`}>{log.message}</div>
    </motion.div>
  );
}

// Stat Card Component
function StatCard({ title, value, icon: Icon, color, pulse }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ scale: 1.02 }}
      className={`cyber-card p-4 ${pulse ? 'animate-pulse-glow' : ''}`}
      style={{ 
        borderColor: pulse ? color : 'rgba(255,255,255,0.08)',
        boxShadow: pulse ? `0 0 20px ${color}40` : 'none'
      }}
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[10px] text-gray-500 tracking-widest uppercase mb-1">{title}</p>
          <p className="text-2xl font-mono font-bold" style={{ color }}>{value}</p>
        </div>
        <div 
          className="p-2"
          style={{ 
            backgroundColor: `${color}15`,
            border: `1px solid ${color}30`
          }}
        >
          <Icon className="w-5 h-5" style={{ color }} />
        </div>
      </div>
    </motion.div>
  );
}

export default function LiveMonitorPage() {
  const [trafficData, setTrafficData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isAttacking, setIsAttacking] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [attackIntensity, setAttackIntensity] = useState(100);
  const [logs, setLogs] = useState([]);
  const attackIntervalRef = useRef(null);

  useEffect(() => {
    const fetchTraffic = async () => {
      try {
        const response = await axios.get(`${API}/monitor/traffic`);
        const data = response.data;
        setTrafficData(data);
        
        const alerts = data.alerts || [];
        if (alerts.length > 0) {
          const latestAlert = alerts[alerts.length - 1];
          setLogs(prev => {
            const exists = prev.some(l => l.id === latestAlert.id);
            if (!exists) {
              if (soundEnabled && latestAlert.severity === "CRITICAL") {
                playAlertSound();
              }
              return [...prev.slice(-50), latestAlert];
            }
            return prev;
          });
        }
      } catch (error) {
        console.error("Erreur:", error);
      }
      setLoading(false);
    };
    
    fetchTraffic();
    const interval = setInterval(fetchTraffic, 1000);
    return () => clearInterval(interval);
  }, [soundEnabled]);

  const playAlertSound = () => {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContext();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      oscillator.frequency.value = 800;
      oscillator.type = "square";
      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.5);
    } catch (e) {
      console.log("Audio not supported");
    }
  };

  const startAttackSimulation = async () => {
    setIsAttacking(true);
    toast.warning("Simulation d'attaque DoS lancée!");
    
    setLogs(prev => [...prev, {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      type: "SIMULATION_START",
      severity: "WARNING",
      message: `Simulation d'attaque initiée - Intensité: ${attackIntensity} req/s`
    }]);

    const sendPings = async () => {
      const promises = [];
      for (let i = 0; i < attackIntensity; i++) {
        promises.push(axios.post(`${API}/monitor/ping`).catch(() => {}));
      }
      await Promise.all(promises);
    };

    sendPings();
    attackIntervalRef.current = setInterval(sendPings, 1000);
  };

  const stopAttackSimulation = () => {
    setIsAttacking(false);
    if (attackIntervalRef.current) {
      clearInterval(attackIntervalRef.current);
      attackIntervalRef.current = null;
    }
    toast.success("Simulation arrêtée");
    setLogs(prev => [...prev, {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      type: "SIMULATION_STOP",
      severity: "INFO",
      message: "Simulation d'attaque terminée"
    }]);
  };

  const resetMonitor = async () => {
    try {
      await axios.post(`${API}/monitor/reset`);
      setLogs([]);
      toast.info("Moniteur réinitialisé");
    } catch (error) {
      toast.error("Erreur lors de la réinitialisation");
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="absolute inset-0 border-2 border-cyan-500/30 animate-ping" />
            <div className="absolute inset-2 border-2 border-cyan-500 animate-pulse" />
            <Shield className="absolute inset-4 w-8 h-8 text-cyan-500" />
          </div>
          <p className="text-gray-500 font-mono text-sm tracking-wider">INITIALISATION...</p>
        </div>
      </div>
    );
  }

  const status = trafficData ? trafficData.status : "NORMAL";
  const isUnderAttack = status === "ATTACK_DETECTED";
  const rps = trafficData ? trafficData.requests_per_second : 0;
  const threshold = trafficData ? trafficData.threshold : 50;
  const trafficHistory = trafficData ? trafficData.traffic_history : [];
  const totalRequests = trafficData ? trafficData.total_requests : 0;
  const attackIndicators = trafficData ? trafficData.attack_indicators : null;
  const attackConfidence = attackIndicators ? attackIndicators.confidence : 0;
  const uniqueSources = attackIndicators ? attackIndicators.unique_sources : 0;
  const patterns = attackIndicators ? attackIndicators.patterns : [];

  const reversedLogs = [...logs].reverse();

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
      data-testid="monitor-page"
    >
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className={`p-2 ${isUnderAttack ? 'bg-red-500/20 border-red-500' : 'bg-cyan-500/20 border-cyan-500'} border`}>
              <Radio className={`w-5 h-5 ${isUnderAttack ? 'text-red-500 animate-pulse' : 'text-cyan-500'}`} />
            </div>
            <h1 className="text-2xl font-mono font-bold tracking-wider">
              SURVEILLANCE <span className="neon-cyan">TEMPS RÉEL</span>
            </h1>
          </div>
          <p className="text-gray-500 text-sm">Monitoring du trafic réseau et détection d'intrusions</p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={() => setSoundEnabled(!soundEnabled)}
            className="border-gray-700 hover:border-cyan-500/50"
          >
            {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </Button>
          <Button 
            variant="outline" 
            onClick={resetMonitor}
            className="border-gray-700 hover:border-cyan-500/50 font-mono text-xs tracking-wider"
          >
            RESET
          </Button>
        </div>
      </div>

      {/* Status Banner */}
      <AnimatePresence mode="wait">
        <motion.div
          key={isUnderAttack ? 'attack' : 'normal'}
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          className={`p-6 border-2 relative overflow-hidden ${
            isUnderAttack 
              ? 'bg-red-500/10 border-red-500 glow-box-red' 
              : 'bg-green-500/10 border-green-500'
          }`}
        >
          {/* Animated background */}
          {isUnderAttack && (
            <div className="absolute inset-0 bg-gradient-to-r from-red-500/5 via-red-500/10 to-red-500/5 animate-pulse" />
          )}
          
          <div className="relative flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-4">
              {isUnderAttack ? (
                <motion.div
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 0.5, repeat: Infinity }}
                >
                  <Skull className="w-12 h-12 text-red-500" />
                </motion.div>
              ) : (
                <CheckCircle className="w-12 h-12 text-green-500" />
              )}
              <div>
                <h2 className={`text-2xl font-mono font-bold tracking-wider ${isUnderAttack ? 'neon-red' : 'neon-green'}`}>
                  {isUnderAttack ? 'ATTAQUE DÉTECTÉE' : 'SYSTÈME OPÉRATIONNEL'}
                </h2>
                <p className="text-gray-400 text-sm font-mono">
                  {isUnderAttack 
                    ? `Intrusion DoS en cours | ${rps} req/s (seuil: ${threshold})` 
                    : `Trafic normal | ${rps} req/s`}
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-[10px] text-gray-500 tracking-widest uppercase">Total Requêtes</p>
              <p className="text-3xl font-mono font-bold text-white">{totalRequests.toLocaleString()}</p>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard 
          title="Requêtes/sec" 
          value={rps} 
          icon={Zap} 
          color={rps > threshold ? COLORS.danger : COLORS.success}
          pulse={rps > threshold}
        />
        <StatCard 
          title="Seuil Alerte" 
          value={threshold} 
          icon={AlertTriangle} 
          color={COLORS.warning}
        />
        <StatCard 
          title="Confiance" 
          value={`${attackConfidence}%`} 
          icon={Brain} 
          color="#BD00FF"
        />
        <StatCard 
          title="Sources" 
          value={uniqueSources} 
          icon={Globe} 
          color={COLORS.primary}
        />
      </div>

      {/* Attack Simulation Panel */}
      <Card className="cyber-card border-yellow-500/30">
        <CardHeader className="pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-yellow-500/20 border border-yellow-500/30">
              <Target className="w-5 h-5 text-yellow-500" />
            </div>
            <div>
              <CardTitle className="font-mono text-lg tracking-wider">SIMULATION D'ATTAQUE</CardTitle>
              <CardDescription className="text-xs">Testez le système de détection</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-end gap-4 flex-wrap">
            <div className="flex-1 min-w-[200px]">
              <Label className="text-[10px] text-gray-500 tracking-widest uppercase mb-2 block">
                Intensité (req/s)
              </Label>
              <Input 
                type="number" 
                min="10" 
                max="500" 
                value={attackIntensity}
                onChange={(e) => setAttackIntensity(parseInt(e.target.value) || 100)}
                className="terminal-input w-32"
                disabled={isAttacking}
              />
            </div>
            <div className="flex gap-2">
              {!isAttacking ? (
                <Button 
                  onClick={startAttackSimulation} 
                  className="cyber-btn-danger cyber-btn"
                  data-testid="start-attack-btn"
                >
                  <Zap className="w-4 h-4 mr-2" />
                  LANCER ATTAQUE
                </Button>
              ) : (
                <Button 
                  onClick={stopAttackSimulation} 
                  className="cyber-btn"
                  data-testid="stop-attack-btn"
                >
                  <X className="w-4 h-4 mr-2" />
                  ARRÊTER
                </Button>
              )}
            </div>
          </div>
          
          {isAttacking && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="p-3 bg-red-500/10 border border-red-500/30"
            >
              <div className="flex items-center gap-2 text-red-400 text-sm font-mono">
                <Radio className="w-4 h-4 animate-pulse" />
                ATTAQUE EN COURS... {attackIntensity} req/s
              </div>
            </motion.div>
          )}
        </CardContent>
      </Card>

      {/* Traffic Chart */}
      <Card className="cyber-card">
        <CardHeader className="pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-500/20 border border-cyan-500/30">
              <Activity className="w-5 h-5 text-cyan-500" />
            </div>
            <div>
              <CardTitle className="font-mono text-lg tracking-wider">TRAFIC RÉSEAU</CardTitle>
              <CardDescription className="text-xs">Dernières 60 secondes</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trafficHistory}>
                <defs>
                  <linearGradient id="colorTraffic" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={isUnderAttack ? COLORS.danger : COLORS.primary} stopOpacity={0.3}/>
                    <stop offset="95%" stopColor={isUnderAttack ? COLORS.danger : COLORS.primary} stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis 
                  dataKey="time" 
                  stroke="#505050" 
                  tick={{ fill: '#505050', fontSize: 10, fontFamily: 'Share Tech Mono' }}
                />
                <YAxis 
                  stroke="#505050"
                  tick={{ fill: '#505050', fontSize: 10, fontFamily: 'Share Tech Mono' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#0a0a0a', 
                    border: '1px solid rgba(0,240,255,0.3)', 
                    borderRadius: 0,
                    fontFamily: 'Share Tech Mono'
                  }}
                  labelStyle={{ color: '#00F0FF' }}
                />
                <Area 
                  type="monotone" 
                  dataKey="requests" 
                  name="Requêtes" 
                  stroke={isUnderAttack ? COLORS.danger : COLORS.primary}
                  strokeWidth={2}
                  fill="url(#colorTraffic)"
                />
                <Line 
                  type="monotone" 
                  dataKey="threshold" 
                  name="Seuil" 
                  stroke={COLORS.warning}
                  strokeWidth={1}
                  strokeDasharray="5 5"
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Security Logs */}
      <Card className="cyber-card">
        <CardHeader className="pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-500/20 border border-purple-500/30">
              <FileText className="w-5 h-5 text-purple-500" />
            </div>
            <div>
              <CardTitle className="font-mono text-lg tracking-wider">LOGS SÉCURITÉ</CardTitle>
              <CardDescription className="text-xs">Événements en temps réel</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="bg-black/50 border border-white/5 p-4 max-h-64 overflow-y-auto font-mono text-sm">
            {logs.length === 0 ? (
              <div className="text-gray-600 text-center py-8">
                <Wifi className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-xs tracking-wider">EN ATTENTE D'ÉVÉNEMENTS...</p>
              </div>
            ) : (
              reversedLogs.map((log, idx) => (
                <LogEntry key={log.id || idx} log={log} />
              ))
            )}
          </div>
        </CardContent>
      </Card>

      {/* Attack Patterns */}
      <AnimatePresence>
        {patterns && patterns.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card className="cyber-card border-red-500/30 glow-box-red">
              <CardHeader className="pb-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-red-500/20 border border-red-500/30">
                    <AlertCircle className="w-5 h-5 text-red-500" />
                  </div>
                  <CardTitle className="font-mono text-lg tracking-wider text-red-500">
                    PATTERNS DÉTECTÉS
                  </CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {patterns.map((pattern, idx) => (
                    <Badge 
                      key={idx} 
                      className="bg-red-500/20 text-red-400 border-red-500/30 font-mono text-xs tracking-wider"
                    >
                      {pattern}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
