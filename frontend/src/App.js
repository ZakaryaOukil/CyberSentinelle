import React, { useState, useEffect, useCallback, Suspense, lazy } from "react";
import { createPortal } from "react-dom";
import { BrowserRouter, Routes, Route, NavLink, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import {
  Shield, Activity, Brain, Target, Menu, X,
  AlertTriangle, CheckCircle, Globe, Zap,
  BarChart3, PieChart, FileText, Maximize2,
  Radio, Home as HomeIcon, Terminal, ChevronRight
} from "lucide-react";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./components/ui/card";
import { Progress } from "./components/ui/progress";
import { Badge } from "./components/ui/badge";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import { Toaster, toast } from "sonner";
import {
  LineChart, Line, BarChart, Bar, PieChart as RechartsPie, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, AreaChart, Area
} from "recharts";
import "@/App.css";

// Lazy load pages
const HomePage = lazy(() => import("./pages/Home"));
const LiveMonitorPage = lazy(() => import("./pages/Monitor"));

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const COLORS = {
  primary: "#00F0FF",
  secondary: "#BD00FF",
  success: "#00FF41",
  danger: "#FF003C",
  warning: "#FAFF00",
  muted: "#505050"
};

const CHART_COLORS = ["#00F0FF", "#00FF41", "#FF003C", "#FAFF00", "#BD00FF", "#3b82f6", "#ec4899"];

// Chart Modal with Portal
const ChartModal = ({ isOpen, onClose, title, children }) => {
  if (!isOpen) return null;
  
  return createPortal(
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0"
        style={{ zIndex: 99999, backgroundColor: 'rgba(0, 0, 0, 0.95)' }}
        onClick={onClose}
        data-testid="chart-modal-overlay"
      />
      <motion.div 
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="fixed inset-0 flex items-center justify-center p-4 pointer-events-none"
        style={{ zIndex: 100000 }}
      >
        <div 
          className="cyber-card w-full max-w-5xl max-h-[90vh] overflow-auto pointer-events-auto border-cyan-500/30"
          onClick={e => e.stopPropagation()}
          data-testid="chart-modal-content"
        >
          <div className="flex items-center justify-between p-4 border-b border-white/10 sticky top-0 bg-black/90 backdrop-blur-sm">
            <h3 className="text-lg font-mono tracking-wider">{title}</h3>
            <Button variant="ghost" size="icon" onClick={onClose} className="hover:bg-white/10">
              <X className="w-5 h-5" />
            </Button>
          </div>
          <div className="p-6">
            <div className="h-[500px]">{children}</div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>,
    document.body
  );
};

// Clickable Chart Wrapper
const ClickableChart = ({ title, children, chartContent }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  return (
    <>
      <div 
        className="cursor-pointer relative group" 
        onClick={() => setIsModalOpen(true)}
        data-testid={`chart-${title.toLowerCase().replace(/\s+/g, '-')}`}
      >
        {children}
        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="p-1 bg-black/50 border border-cyan-500/30">
            <Maximize2 className="w-4 h-4 text-cyan-400" />
          </div>
        </div>
      </div>
      <ChartModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} title={title}>
        {chartContent}
      </ChartModal>
    </>
  );
};

// Sidebar Component
const Sidebar = ({ isOpen, setIsOpen }) => {
  const location = useLocation();
  
  const navItems = [
    { path: "/", icon: HomeIcon, label: "ACCUEIL" },
    { path: "/monitor", icon: Radio, label: "LIVE MONITOR", highlight: true },
    { path: "/dashboard", icon: Activity, label: "DASHBOARD EDA" },
    { path: "/model", icon: Brain, label: "MODÈLE ML" },
    { path: "/prediction", icon: Target, label: "PRÉDICTION" },
    { path: "/clustering", icon: PieChart, label: "CLUSTERING" }
  ];

  return (
    <>
      {/* Mobile overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 z-40 lg:hidden"
            onClick={() => setIsOpen(false)}
          />
        )}
      </AnimatePresence>
      
      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{ x: isOpen ? 0 : -280 }}
        className="fixed left-0 top-0 bottom-0 w-[280px] sidebar-cyber z-50 flex flex-col lg:translate-x-0"
        style={{ transform: 'none' }}
      >
        {/* Header */}
        <div className="p-6 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 border border-cyan-500/50 flex items-center justify-center">
                <Shield className="w-5 h-5 text-cyan-400" />
              </div>
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 animate-pulse" />
            </div>
            <div>
              <h1 className="font-mono text-lg font-bold tracking-wider">
                <span className="text-white">CYBER</span>
                <span className="text-cyan-400">SENTINELLE</span>
              </h1>
              <p className="text-[10px] text-gray-500 tracking-widest">DÉTECTION D'INTRUSIONS</p>
            </div>
          </div>
        </div>
        
        {/* Navigation */}
        <nav className="flex-1 py-6 overflow-y-auto">
          <div className="px-4 mb-2">
            <span className="text-[10px] text-gray-600 tracking-widest uppercase">Navigation</span>
          </div>
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={() => setIsOpen(false)}
                className={`nav-item flex items-center gap-3 px-6 py-3 mx-2 mb-1 transition-all ${
                  isActive 
                    ? 'active bg-cyan-500/10 text-cyan-400' 
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <item.icon className={`w-4 h-4 ${item.highlight && !isActive ? 'text-red-500' : ''}`} />
                <span className="font-mono text-sm tracking-wider">{item.label}</span>
                {item.highlight && (
                  <span className="ml-auto w-2 h-2 bg-red-500 animate-pulse" />
                )}
              </NavLink>
            );
          })}
        </nav>
        
        {/* Footer */}
        <div className="p-4 border-t border-white/5">
          <div className="text-[10px] text-gray-600 tracking-wider">
            <div>Master 1 Cybersécurité</div>
            <div>HIS - 2025/2026</div>
            <div className="text-cyan-500 mt-1">Zakarya Oukil</div>
          </div>
        </div>
      </motion.aside>
    </>
  );
};

// Loading Spinner
const LoadingSpinner = () => (
  <div className="flex items-center justify-center h-96">
    <div className="relative w-16 h-16">
      <div className="absolute inset-0 border-2 border-cyan-500/30 animate-ping" />
      <div className="absolute inset-2 border-2 border-cyan-500 animate-pulse" />
      <Shield className="absolute inset-4 w-8 h-8 text-cyan-500" />
    </div>
  </div>
);

// Dashboard EDA Page
const DashboardPage = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`${API}/dataset/eda`);
        setData(response.data);
      } catch (error) {
        console.error("Erreur:", error);
        toast.error("Erreur lors du chargement des données");
      }
      setLoading(false);
    };
    fetchData();
  }, []);

  if (loading) return <LoadingSpinner />;
  if (!data) return <div className="text-center text-gray-500">Aucune donnée disponible</div>;

  const attackDistribution = data.attack_distribution || [];
  const protocolDistribution = data.protocol_distribution || [];
  const topFeatures = data.top_features || [];

  const attackChartContent = (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={attackDistribution} layout="vertical" margin={{ left: 80, bottom: 30 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
        <XAxis type="number" stroke="#505050" tick={{ fill: '#505050', fontSize: 11, fontFamily: 'Share Tech Mono' }} label={{ value: "Nombre d'échantillons", position: 'bottom', fill: '#606060' }} />
        <YAxis type="category" dataKey="type" stroke="#505050" tick={{ fill: '#a0a0a0', fontSize: 11, fontFamily: 'Share Tech Mono' }} label={{ value: 'Type', angle: -90, position: 'insideLeft', fill: '#606060' }} />
        <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(0,240,255,0.3)', borderRadius: 0, fontFamily: 'Share Tech Mono' }} />
        <Legend />
        <Bar dataKey="count" name="Nombre d'échantillons" fill={COLORS.primary} />
      </BarChart>
    </ResponsiveContainer>
  );

  const featuresChartContent = (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={topFeatures} margin={{ bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="feature" stroke="#505050" tick={{ fill: '#a0a0a0', fontSize: 10, fontFamily: 'Share Tech Mono', angle: -45, textAnchor: 'end' }} label={{ value: 'Feature', position: 'bottom', fill: '#606060', dy: 45 }} />
        <YAxis stroke="#505050" scale="log" domain={['auto', 'auto']} tick={{ fill: '#505050', fontSize: 11, fontFamily: 'Share Tech Mono' }} label={{ value: 'Variance (log)', angle: -90, position: 'insideLeft', fill: '#606060' }} />
        <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(0,240,255,0.3)', borderRadius: 0, fontFamily: 'Share Tech Mono' }} />
        <Bar dataKey="variance" name="Variance" fill={COLORS.success} />
      </BarChart>
    </ResponsiveContainer>
  );

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-cyan-500/20 border border-cyan-500/30">
          <BarChart3 className="w-5 h-5 text-cyan-500" />
        </div>
        <div>
          <h1 className="text-2xl font-mono font-bold tracking-wider">ANALYSE <span className="neon-cyan">EXPLORATOIRE</span></h1>
          <p className="text-gray-500 text-sm">Dataset NSL-KDD • Cliquer pour agrandir</p>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card className="cyber-card lg:col-span-2">
          <CardHeader>
            <CardTitle className="font-mono tracking-wider text-lg">DISTRIBUTION DES ATTAQUES</CardTitle>
          </CardHeader>
          <CardContent>
            <ClickableChart title="Distribution des types d'attaques" chartContent={attackChartContent}>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={attackDistribution} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis type="number" stroke="#505050" tick={{ fill: '#505050', fontSize: 10 }} />
                    <YAxis type="category" dataKey="type" stroke="#505050" tick={{ fill: '#a0a0a0', fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(0,240,255,0.3)', borderRadius: 0 }} />
                    <Bar dataKey="count" fill={COLORS.primary} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ClickableChart>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="font-mono tracking-wider text-lg">PROTOCOLES RÉSEAU</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsPie>
                  <Pie
                    data={protocolDistribution}
                    dataKey="count"
                    nameKey="protocol"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label={({ protocol, percent }) => `${protocol} ${(percent * 100).toFixed(0)}%`}
                    labelLine={{ stroke: '#505050' }}
                  >
                    {protocolDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(0,240,255,0.3)', borderRadius: 0 }} />
                </RechartsPie>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="font-mono tracking-wider text-lg">TOP FEATURES</CardTitle>
            <CardDescription className="text-xs">Échelle logarithmique</CardDescription>
          </CardHeader>
          <CardContent>
            <ClickableChart title="Top Features (par variance)" chartContent={featuresChartContent}>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={topFeatures}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="feature" stroke="#505050" tick={{ fill: '#505050', fontSize: 8, angle: -45, textAnchor: 'end' }} />
                    <YAxis stroke="#505050" scale="log" domain={['auto', 'auto']} tick={{ fill: '#505050', fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(0,240,255,0.3)', borderRadius: 0 }} />
                    <Bar dataKey="variance" fill={COLORS.success} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ClickableChart>
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
};

// Model Page
const ModelPage = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);

  const fetchMetrics = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/model/metrics`);
      setMetrics(response.data);
    } catch (error) {
      console.error("Erreur:", error);
    }
    setLoading(false);
  }, []);

  useEffect(() => { fetchMetrics(); }, [fetchMetrics]);

  const handleTrain = async () => {
    setTraining(true);
    try {
      await axios.post(`${API}/model/train`);
      toast.success("Modèles entraînés avec succès!");
      await fetchMetrics();
    } catch (error) {
      toast.error("Erreur lors de l'entraînement");
    }
    setTraining(false);
  };

  if (loading) return <LoadingSpinner />;

  const rfMetrics = metrics?.results?.random_forest || {};
  const dtMetrics = metrics?.results?.decision_tree || {};

  const comparisonData = [
    { name: 'Accuracy', RF: (rfMetrics.accuracy || 0) * 100, DT: (dtMetrics.accuracy || 0) * 100 },
    { name: 'Precision', RF: (rfMetrics.precision || 0) * 100, DT: (dtMetrics.precision || 0) * 100 },
    { name: 'Recall', RF: (rfMetrics.recall || 0) * 100, DT: (dtMetrics.recall || 0) * 100 },
    { name: 'F1-Score', RF: (rfMetrics.f1_score || 0) * 100, DT: (dtMetrics.f1_score || 0) * 100 }
  ];

  const rocData = rfMetrics.roc_curve ? rfMetrics.roc_curve.fpr.map((fpr, i) => ({
    fpr, tpr: rfMetrics.roc_curve.tpr[i]
  })) : [];

  const featureImportance = rfMetrics.feature_importance ? 
    Object.entries(rfMetrics.feature_importance).map(([feature, importance]) => ({ feature, importance })).slice(0, 10) : [];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/20 border border-purple-500/30">
            <Brain className="w-5 h-5 text-purple-500" />
          </div>
          <div>
            <h1 className="text-2xl font-mono font-bold tracking-wider">MODÈLE <span className="text-purple-400">ML</span></h1>
            <p className="text-gray-500 text-sm">Random Forest & Decision Tree</p>
          </div>
        </div>
        <Button onClick={handleTrain} disabled={training} className="cyber-btn">
          {training ? "ENTRAÎNEMENT..." : "ENTRAÎNER"}
        </Button>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'ACCURACY', value: ((rfMetrics.accuracy || 0) * 100).toFixed(2) + '%', color: COLORS.primary },
          { label: 'PRECISION', value: ((rfMetrics.precision || 0) * 100).toFixed(2) + '%', color: COLORS.secondary },
          { label: 'RECALL', value: ((rfMetrics.recall || 0) * 100).toFixed(2) + '%', color: COLORS.warning },
          { label: 'AUC', value: (rfMetrics.auc || 0).toFixed(4), color: COLORS.success }
        ].map((metric) => (
          <Card key={metric.label} className="cyber-card">
            <CardContent className="p-4">
              <p className="text-[10px] text-gray-500 tracking-widest mb-1">{metric.label} (RF)</p>
              <p className="text-2xl font-mono font-bold" style={{ color: metric.color }}>{metric.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="font-mono tracking-wider">COMPARAISON</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="name" stroke="#505050" tick={{ fill: '#a0a0a0', fontSize: 10 }} />
                  <YAxis stroke="#505050" tick={{ fill: '#505050', fontSize: 10 }} domain={[0, 100]} />
                  <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(0,240,255,0.3)', borderRadius: 0 }} />
                  <Legend />
                  <Bar dataKey="RF" name="Random Forest" fill={COLORS.primary} />
                  <Bar dataKey="DT" name="Decision Tree" fill={COLORS.secondary} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="font-mono tracking-wider">COURBE ROC</CardTitle>
            <CardDescription>AUC = {(rfMetrics.auc || 0).toFixed(4)}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={rocData}>
                  <defs>
                    <linearGradient id="rocGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={COLORS.success} stopOpacity={0.3}/>
                      <stop offset="95%" stopColor={COLORS.success} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="fpr" stroke="#505050" tick={{ fill: '#505050', fontSize: 10 }} />
                  <YAxis stroke="#505050" tick={{ fill: '#505050', fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(0,240,255,0.3)', borderRadius: 0 }} />
                  <Area type="monotone" dataKey="tpr" stroke={COLORS.success} fill="url(#rocGradient)" />
                  <Line type="monotone" dataKey="fpr" stroke="#505050" strokeDasharray="5 5" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="cyber-card lg:col-span-2">
          <CardHeader>
            <CardTitle className="font-mono tracking-wider">IMPORTANCE DES FEATURES</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={featureImportance} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis type="number" stroke="#505050" tick={{ fill: '#505050', fontSize: 10 }} />
                  <YAxis type="category" dataKey="feature" stroke="#505050" tick={{ fill: '#a0a0a0', fontSize: 10 }} width={120} />
                  <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(0,240,255,0.3)', borderRadius: 0 }} />
                  <Bar dataKey="importance" fill={COLORS.warning} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
};

// Prediction Page
const PredictionPage = () => {
  const [formData, setFormData] = useState({
    duration: 0, protocol_type: 'tcp', service: 'http', flag: 'SF',
    src_bytes: 0, dst_bytes: 0, land: 0, wrong_fragment: 0, urgent: 0,
    hot: 0, num_failed_logins: 0, logged_in: 1, num_compromised: 0,
    root_shell: 0, su_attempted: 0, num_root: 0, num_file_creations: 0,
    num_shells: 0, num_access_files: 0, num_outbound_cmds: 0, is_host_login: 0,
    is_guest_login: 0, count: 1, srv_count: 1, serror_rate: 0, srv_serror_rate: 0,
    rerror_rate: 0, srv_rerror_rate: 0, same_srv_rate: 1, diff_srv_rate: 0,
    srv_diff_host_rate: 0, dst_host_count: 0, dst_host_srv_count: 0,
    dst_host_same_srv_rate: 0, dst_host_diff_srv_rate: 0, dst_host_same_src_port_rate: 0,
    dst_host_srv_diff_host_rate: 0, dst_host_serror_rate: 0, dst_host_srv_serror_rate: 0,
    dst_host_rerror_rate: 0, dst_host_srv_rerror_rate: 0
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post(`${API}/model/predict`, formData);
      setResult(response.data);
      const isAttack = response.data.prediction === 'Attack';
      toast[isAttack ? 'error' : 'success'](isAttack ? 'Intrusion détectée!' : 'Trafic normal');
    } catch (error) {
      toast.error("Erreur de prédiction");
    }
    setLoading(false);
  };

  const loadDemo = (type) => {
    if (type === 'normal') {
      setFormData(prev => ({ ...prev, duration: 0, src_bytes: 200, dst_bytes: 1000, count: 2, srv_count: 2, same_srv_rate: 1, logged_in: 1 }));
    } else {
      setFormData(prev => ({ ...prev, duration: 0, src_bytes: 0, dst_bytes: 0, count: 500, srv_count: 500, same_srv_rate: 1, serror_rate: 1, srv_serror_rate: 1 }));
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-green-500/20 border border-green-500/30">
          <Target className="w-5 h-5 text-green-500" />
        </div>
        <div>
          <h1 className="text-2xl font-mono font-bold tracking-wider">PRÉDICTION <span className="neon-green">LIVE</span></h1>
          <p className="text-gray-500 text-sm">Classification du trafic réseau</p>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <Card className="cyber-card lg:col-span-2">
          <CardHeader>
            <CardTitle className="font-mono tracking-wider">PARAMÈTRES</CardTitle>
            <div className="flex gap-2 mt-2">
              <Button variant="outline" size="sm" onClick={() => loadDemo('normal')} className="text-xs">DÉMO NORMAL</Button>
              <Button variant="outline" size="sm" onClick={() => loadDemo('attack')} className="text-xs">DÉMO ATTAQUE</Button>
            </div>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 'serror_rate', 'same_srv_rate', 'dst_host_count'].map((field) => (
                  <div key={field}>
                    <Label className="text-[10px] text-gray-500 tracking-widest uppercase">{field}</Label>
                    <Input
                      type="number"
                      step="any"
                      value={formData[field]}
                      onChange={(e) => setFormData(prev => ({ ...prev, [field]: parseFloat(e.target.value) || 0 }))}
                      className="terminal-input mt-1"
                    />
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <Label className="text-[10px] text-gray-500 tracking-widest uppercase">Protocol</Label>
                  <Select value={formData.protocol_type} onValueChange={(v) => setFormData(prev => ({ ...prev, protocol_type: v }))}>
                    <SelectTrigger className="terminal-input mt-1"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="tcp">TCP</SelectItem>
                      <SelectItem value="udp">UDP</SelectItem>
                      <SelectItem value="icmp">ICMP</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-[10px] text-gray-500 tracking-widest uppercase">Service</Label>
                  <Select value={formData.service} onValueChange={(v) => setFormData(prev => ({ ...prev, service: v }))}>
                    <SelectTrigger className="terminal-input mt-1"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {['http', 'ftp', 'smtp', 'ssh', 'dns', 'telnet', 'other'].map(s => <SelectItem key={s} value={s}>{s.toUpperCase()}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-[10px] text-gray-500 tracking-widest uppercase">Flag</Label>
                  <Select value={formData.flag} onValueChange={(v) => setFormData(prev => ({ ...prev, flag: v }))}>
                    <SelectTrigger className="terminal-input mt-1"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'RSTOS0', 'S3', 'S2', 'OTH'].map(f => <SelectItem key={f} value={f}>{f}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <Button type="submit" disabled={loading} className="cyber-btn w-full">
                {loading ? "ANALYSE..." : "ANALYSER LE TRAFIC"}
              </Button>
            </form>
          </CardContent>
        </Card>

        <Card className={`cyber-card ${result ? (result.prediction === 'Attack' ? 'border-red-500/50 glow-box-red' : 'border-green-500/50') : ''}`}>
          <CardHeader>
            <CardTitle className="font-mono tracking-wider">RÉSULTAT</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col items-center justify-center min-h-[200px]">
            {result ? (
              <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} className="text-center">
                {result.prediction === 'Attack' ? (
                  <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                ) : (
                  <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
                )}
                <p className={`text-2xl font-mono font-bold mb-2 ${result.prediction === 'Attack' ? 'neon-red' : 'neon-green'}`}>
                  {result.prediction === 'Attack' ? 'INTRUSION' : 'NORMAL'}
                </p>
                <p className="text-gray-500 text-sm">Confiance: {(result.confidence * 100).toFixed(1)}%</p>
                {result.attack_type && <Badge className="mt-2">{result.attack_type}</Badge>}
              </motion.div>
            ) : (
              <div className="text-gray-600 text-center">
                <Target className="w-12 h-12 mx-auto mb-2 opacity-30" />
                <p className="text-xs tracking-wider">EN ATTENTE D'ANALYSE</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
};

// Clustering Page
const ClusteringPage = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [numClusters, setNumClusters] = useState(5);

  const runClustering = async () => {
    setLoading(true);
    try {
      await axios.post(`${API}/clustering/run`, { n_clusters: numClusters });
      const response = await axios.get(`${API}/clustering/results`);
      setResults(response.data);
      toast.success("Clustering terminé!");
    } catch (error) {
      toast.error("Erreur de clustering");
    }
    setLoading(false);
  };

  const scatterData = results?.visualization_data || [];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-yellow-500/20 border border-yellow-500/30">
            <PieChart className="w-5 h-5 text-yellow-500" />
          </div>
          <div>
            <h1 className="text-2xl font-mono font-bold tracking-wider">CLUSTERING <span className="neon-yellow">K-MEANS</span></h1>
            <p className="text-gray-500 text-sm">Analyse non supervisée</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Label className="text-xs text-gray-500">K =</Label>
            <Input type="number" min="2" max="10" value={numClusters} onChange={(e) => setNumClusters(parseInt(e.target.value) || 5)} className="terminal-input w-16" />
          </div>
          <Button onClick={runClustering} disabled={loading} className="cyber-btn">
            {loading ? "CALCUL..." : "EXÉCUTER"}
          </Button>
        </div>
      </div>

      {results && (
        <div className="grid lg:grid-cols-2 gap-6">
          <Card className="cyber-card lg:col-span-2">
            <CardHeader>
              <CardTitle className="font-mono tracking-wider">VISUALISATION 2D</CardTitle>
              <CardDescription>Projection PCA des clusters</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ bottom: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis type="number" dataKey="x" name="PC1" stroke="#505050" tick={{ fill: '#505050', fontSize: 10 }} />
                    <YAxis type="number" dataKey="y" name="PC2" stroke="#505050" tick={{ fill: '#505050', fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(0,240,255,0.3)', borderRadius: 0 }} />
                    <Legend />
                    {[...new Set(scatterData.map(d => d.cluster))].map((cluster, idx) => (
                      <Scatter
                        key={cluster}
                        name={`Cluster ${cluster}`}
                        data={scatterData.filter(d => d.cluster === cluster)}
                        fill={CHART_COLORS[idx % CHART_COLORS.length]}
                      />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card className="cyber-card">
            <CardHeader>
              <CardTitle className="font-mono tracking-wider">MÉTRIQUES</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 border border-white/10">
                  <p className="text-[10px] text-gray-500 tracking-widest">SILHOUETTE</p>
                  <p className="text-2xl font-mono font-bold text-cyan-400">{(results.silhouette_score || 0).toFixed(4)}</p>
                </div>
                <div className="p-4 border border-white/10">
                  <p className="text-[10px] text-gray-500 tracking-widest">INERTIE</p>
                  <p className="text-2xl font-mono font-bold text-purple-400">{(results.inertia || 0).toFixed(0)}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="cyber-card">
            <CardHeader>
              <CardTitle className="font-mono tracking-wider">TAILLE DES CLUSTERS</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={Object.entries(results.cluster_sizes || {}).map(([k, v]) => ({ cluster: `C${k}`, size: v }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="cluster" stroke="#505050" tick={{ fill: '#a0a0a0', fontSize: 10 }} />
                    <YAxis stroke="#505050" tick={{ fill: '#505050', fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(0,240,255,0.3)', borderRadius: 0 }} />
                    <Bar dataKey="size" fill={COLORS.warning} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </motion.div>
  );
};

// Main App Component
function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-[#050505] text-white cyber-grid-bg">
        {/* Scanlines effect */}
        <div className="scanlines" />
        
        <Toaster 
          position="top-right" 
          toastOptions={{
            style: { background: '#0a0a0a', border: '1px solid rgba(0,240,255,0.3)', color: '#fff', fontFamily: 'Share Tech Mono' }
          }}
        />
        
        <Sidebar isOpen={sidebarOpen} setIsOpen={setSidebarOpen} />
        
        {/* Mobile header */}
        <div className="lg:hidden fixed top-0 left-0 right-0 z-30 bg-black/90 backdrop-blur-sm border-b border-white/5 px-4 py-3">
          <div className="flex items-center justify-between">
            <Button variant="ghost" size="icon" onClick={() => setSidebarOpen(true)}>
              <Menu className="w-5 h-5" />
            </Button>
            <span className="font-mono text-sm tracking-wider">
              <span className="text-white">CYBER</span>
              <span className="text-cyan-400">SENTINELLE</span>
            </span>
            <div className="w-10" />
          </div>
        </div>
        
        {/* Main content */}
        <main className="lg:ml-[280px] min-h-screen pt-16 lg:pt-0">
          <div className="p-6 md:p-8 lg:p-12">
            <Suspense fallback={<LoadingSpinner />}>
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/monitor" element={<LiveMonitorPage />} />
                <Route path="/dashboard" element={<DashboardPage />} />
                <Route path="/model" element={<ModelPage />} />
                <Route path="/prediction" element={<PredictionPage />} />
                <Route path="/clustering" element={<ClusteringPage />} />
              </Routes>
            </Suspense>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
