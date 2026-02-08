import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
  Radio, AlertTriangle, CheckCircle, Brain, Globe, Zap,
  Target, X, FileText, Volume2, VolumeX, AlertCircle
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
  primary: "#06b6d4",
  danger: "#ef4444",
  warning: "#f59e0b"
};

// Component for logs display
function LogEntry({ log }) {
  const getColor = () => {
    if (log.severity === 'CRITICAL') return 'text-red-400';
    if (log.severity === 'WARNING') return 'text-amber-400';
    return 'text-emerald-400';
  };
  
  const getSeverityColor = () => {
    if (log.severity === 'CRITICAL') return 'text-red-500';
    if (log.severity === 'WARNING') return 'text-amber-500';
    return 'text-emerald-500';
  };
  
  const timestamp = new Date(log.timestamp).toLocaleTimeString();
  
  return (
    <div className={`py-1 border-b border-slate-800 last:border-0 ${getColor()}`}>
      <span className="text-muted-foreground">[{timestamp}]</span>{' '}
      <span className={`font-bold ${getSeverityColor()}`}>
        [{log.severity}]
      </span>{' '}
      {log.message}
    </div>
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
      message: `Simulation d'attaque lancée (${attackIntensity} req/s)`
    }]);

    const sendPings = async () => {
      const promises = [];
      for (let i = 0; i < attackIntensity; i++) {
        promises.push(axios.post(`${API}/monitor/ping`).catch(() => {}));
      }
      await Promise.all(promises);
    };

    // Send first batch immediately
    sendPings();
    
    // Then continue sending every second
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
      message: "Simulation d'attaque arrêtée"
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
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary mx-auto mb-4" />
          <p className="text-muted-foreground">Initialisation du moniteur...</p>
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
    <div className="space-y-6 fade-in" data-testid="monitor-page">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold font-mono flex items-center gap-3">
            <Radio className={isUnderAttack ? "w-6 h-6 text-red-500 animate-pulse" : "w-6 h-6 text-emerald-500"} />
            Surveillance en Temps Réel
          </h1>
          <p className="text-muted-foreground">Monitoring du trafic réseau et détection d'intrusions</p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={() => setSoundEnabled(!soundEnabled)}
            title={soundEnabled ? "Désactiver le son" : "Activer le son"}
          >
            {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </Button>
          <Button variant="outline" onClick={resetMonitor}>
            Réinitialiser
          </Button>
        </div>
      </div>

      {/* Status Banner */}
      <div className={isUnderAttack ? "p-6 rounded-xl border-2 bg-red-500/20 border-red-500 animate-pulse" : "p-6 rounded-xl border-2 bg-emerald-500/20 border-emerald-500"}>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4">
            {isUnderAttack ? (
              <AlertCircle className="w-12 h-12 text-red-500" />
            ) : (
              <CheckCircle className="w-12 h-12 text-emerald-500" />
            )}
            <div>
              <h2 className={isUnderAttack ? "text-2xl font-bold font-mono text-red-500" : "text-2xl font-bold font-mono text-emerald-500"}>
                {isUnderAttack ? 'ATTAQUE EN COURS' : 'SYSTÈME NORMAL'}
              </h2>
              <p className="text-muted-foreground">
                {isUnderAttack 
                  ? `Attaque DoS détectée! ${rps} req/s (seuil: ${threshold})` 
                  : `Trafic normal - ${rps} req/s`}
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm text-muted-foreground">Total requêtes</p>
            <p className="text-3xl font-bold font-mono">{totalRequests.toLocaleString()}</p>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className={isUnderAttack ? "cyber-card border-red-500/50" : "cyber-card"}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground">Requêtes/sec</p>
                <p className={rps > threshold ? "text-2xl font-bold font-mono text-red-500" : "text-2xl font-bold font-mono text-emerald-500"}>
                  {rps}
                </p>
              </div>
              <Zap className={rps > threshold ? "w-8 h-8 text-red-500" : "w-8 h-8 text-emerald-500"} />
            </div>
          </CardContent>
        </Card>
        
        <Card className="cyber-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground">Seuil d'alerte</p>
                <p className="text-2xl font-bold font-mono text-amber-500">{threshold}</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-amber-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card className="cyber-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground">Confiance</p>
                <p className="text-2xl font-bold font-mono text-purple-500">{attackConfidence}%</p>
              </div>
              <Brain className="w-8 h-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card className="cyber-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground">Sources uniques</p>
                <p className="text-2xl font-bold font-mono text-cyan-500">{uniqueSources}</p>
              </div>
              <Globe className="w-8 h-8 text-cyan-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Attack Simulation Panel */}
      <Card className="cyber-card border-amber-500/50">
        <CardHeader>
          <CardTitle className="font-mono flex items-center gap-2">
            <Target className="w-5 h-5 text-amber-500" />
            Simulation d'Attaque DoS
          </CardTitle>
          <CardDescription>
            Lancez une simulation pour tester le système de détection
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex-1 min-w-[200px]">
              <Label className="text-xs">Intensité (requêtes/seconde)</Label>
              <div className="flex items-center gap-2 mt-1">
                <Input 
                  type="number" 
                  min="10" 
                  max="500" 
                  value={attackIntensity}
                  onChange={(e) => setAttackIntensity(parseInt(e.target.value) || 100)}
                  className="terminal-input w-24"
                  disabled={isAttacking}
                />
                <span className="text-sm text-muted-foreground">req/s</span>
              </div>
            </div>
            <div className="flex gap-2">
              {!isAttacking ? (
                <Button 
                  onClick={startAttackSimulation} 
                  className="bg-red-600 hover:bg-red-700"
                  data-testid="start-attack-btn"
                >
                  <Zap className="w-4 h-4 mr-2" />
                  Lancer l'attaque
                </Button>
              ) : (
                <Button 
                  onClick={stopAttackSimulation} 
                  variant="outline"
                  className="border-emerald-500 text-emerald-500 hover:bg-emerald-500/20"
                  data-testid="stop-attack-btn"
                >
                  <X className="w-4 h-4 mr-2" />
                  Arrêter l'attaque
                </Button>
              )}
            </div>
          </div>
          {isAttacking && (
            <div className="p-3 bg-red-500/20 border border-red-500/50 rounded-lg">
              <p className="text-sm text-red-400 flex items-center gap-2">
                <Radio className="w-4 h-4 animate-pulse" />
                Simulation en cours... {attackIntensity} requêtes/seconde envoyées
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Traffic Chart */}
      <Card className="cyber-card">
        <CardHeader>
          <CardTitle className="font-mono">Trafic en temps réel</CardTitle>
          <CardDescription>Dernières 60 secondes</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trafficHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="time" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                <Legend />
                <Area 
                  type="monotone" 
                  dataKey="requests" 
                  name="Requêtes" 
                  stroke={isUnderAttack ? COLORS.danger : COLORS.primary} 
                  fill={isUnderAttack ? COLORS.danger : COLORS.primary} 
                  fillOpacity={0.3} 
                />
                <Line 
                  type="monotone" 
                  dataKey="threshold" 
                  name="Seuil" 
                  stroke={COLORS.warning} 
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
        <CardHeader>
          <CardTitle className="font-mono flex items-center gap-2">
            <FileText className="w-5 h-5" />
            Logs de Sécurité
          </CardTitle>
          <CardDescription>Alertes et événements en temps réel</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-slate-950 rounded-lg p-4 font-mono text-sm max-h-64 overflow-y-auto">
            {logs.length === 0 ? (
              <p className="text-muted-foreground">En attente d'événements...</p>
            ) : (
              reversedLogs.map((log, idx) => (
                <LogEntry key={log.id || idx} log={log} />
              ))
            )}
          </div>
        </CardContent>
      </Card>

      {/* Attack Indicators */}
      {patterns && patterns.length > 0 && (
        <Card className="cyber-card border-red-500/50">
          <CardHeader>
            <CardTitle className="font-mono text-red-500">Patterns d'attaque détectés</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {patterns.map((pattern, idx) => (
                <Badge key={idx} variant="destructive" className="font-mono">
                  {pattern}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
