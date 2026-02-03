import React, { useState, useEffect, useCallback } from "react";
import { BrowserRouter, Routes, Route, NavLink, useLocation } from "react-router-dom";
import axios from "axios";
import {
  Shield, Activity, Database, Brain, Target, Download, Menu,
  AlertTriangle, CheckCircle, Server, Cpu, Globe, Lock, Zap,
  BarChart3, PieChart, TrendingUp, FileText, Play
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

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const COLORS = {
  primary: "#06b6d4",
  secondary: "#3b82f6", 
  success: "#10b981",
  danger: "#ef4444",
  warning: "#f59e0b",
  info: "#8b5cf6",
  muted: "#64748b"
};

const CHART_COLORS = ["#06b6d4", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6", "#3b82f6", "#ec4899"];

// Sidebar - Sans page Télécharger
const Sidebar = ({ isOpen, setIsOpen }) => {
  const location = useLocation();
  
  const navItems = [
    { path: "/", icon: Shield, label: "Accueil", testId: "nav-home" },
    { path: "/dashboard", icon: Activity, label: "Dashboard EDA", testId: "nav-dashboard" },
    { path: "/model", icon: Brain, label: "Modèle ML", testId: "nav-model" },
    { path: "/prediction", icon: Target, label: "Prédiction", testId: "nav-prediction" },
    { path: "/clustering", icon: PieChart, label: "Clustering", testId: "nav-clustering" }
  ];

  return (
    <>
      {isOpen && (
        <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={() => setIsOpen(false)} />
      )}
      
      <aside className={`
        fixed top-0 left-0 z-50 h-full w-64 bg-card border-r border-border
        transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        lg:translate-x-0 lg:static
      `}>
        <div className="flex flex-col h-full">
          <div className="p-6 border-b border-border">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/20 rounded-lg">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-foreground font-mono">IDS Project</h1>
                <p className="text-xs text-muted-foreground">Détection d'intrusions</p>
              </div>
            </div>
          </div>

          <nav className="flex-1 p-4 space-y-2">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                data-testid={item.testId}
                onClick={() => setIsOpen(false)}
                className={({ isActive }) => `
                  flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200
                  ${isActive 
                    ? 'bg-primary/20 text-primary border border-primary/50' 
                    : 'text-muted-foreground hover:bg-secondary/50 hover:text-foreground border border-transparent'
                  }
                `}
              >
                <item.icon className="w-5 h-5" />
                <span className="font-medium">{item.label}</span>
              </NavLink>
            ))}
          </nav>

          <div className="p-4 border-t border-border">
            <div className="text-xs text-muted-foreground space-y-1">
              <p className="font-mono">Master 1 Cybersécurité</p>
              <p>HIS - 2025/2026</p>
              <p className="text-primary font-medium">Zakarya Oukil</p>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};

const MobileHeader = ({ setIsOpen }) => (
  <header className="lg:hidden fixed top-0 left-0 right-0 z-30 bg-card border-b border-border">
    <div className="flex items-center justify-between p-4">
      <div className="flex items-center gap-2">
        <Shield className="w-6 h-6 text-primary" />
        <span className="font-mono font-bold">IDS Project</span>
      </div>
      <Button variant="ghost" size="icon" onClick={() => setIsOpen(true)} data-testid="mobile-menu-btn">
        <Menu className="w-6 h-6" />
      </Button>
    </div>
  </header>
);

const StatCard = ({ title, value, icon: Icon, color = "primary", testId }) => (
  <Card className="cyber-card" data-testid={testId}>
    <CardContent className="p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-muted-foreground mb-1">{title}</p>
          <p className="text-2xl font-bold font-mono" style={{ color: COLORS[color] }}>{value}</p>
        </div>
        <div className="p-3 rounded-lg" style={{ backgroundColor: `${COLORS[color]}20` }}>
          <Icon className="w-5 h-5" style={{ color: COLORS[color] }} />
        </div>
      </div>
    </CardContent>
  </Card>
);

// Page d'accueil avec bouton télécharger
const HomePage = () => {
  const [stats, setStats] = useState(null);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes, metricsRes] = await Promise.all([
          axios.get(`${API}/dataset/info`),
          axios.get(`${API}/model/metrics`)
        ]);
        setStats(statsRes.data);
        setModelMetrics(metricsRes.data);
      } catch (error) {
        console.error("Erreur:", error);
      }
      setLoading(false);
    };
    fetchData();
  }, []);

  const handleDownloadNotebook = async () => {
    setDownloading(true);
    toast.info("Génération du notebook...");
    try {
      await axios.post(`${API}/notebook/generate`);
      const response = await axios.get(`${API}/notebook/download`, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'Mini_Projet_Detection_Intrusion_Zakarya_Oukil.ipynb');
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast.success("Notebook téléchargé !");
    } catch (error) {
      toast.error("Erreur lors du téléchargement");
    }
    setDownloading(false);
  };

  const rfAccuracy = modelMetrics?.results?.random_forest?.accuracy;
  const attackData = stats?.attack_categories 
    ? Object.entries(stats.attack_categories).map(([name, value]) => ({ name, value }))
    : [];

  return (
    <div className="space-y-8 fade-in" data-testid="home-page">
      {/* Hero */}
      <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-slate-900 to-slate-800 border border-border p-8">
        <div className="absolute inset-0 grid-pattern opacity-20" />
        <div className="relative z-10">
          <Badge variant="outline" className="border-primary/50 text-primary mb-4">
            Mini-Projet Data Science
          </Badge>
          <h1 className="text-3xl md:text-4xl font-bold mb-3 font-mono">
            Détection d'Intrusions Réseau
          </h1>
          <p className="text-muted-foreground max-w-2xl mb-6">
            Système de détection d'attaques DoS/DDoS basé sur le Machine Learning.
            Ce projet utilise le dataset NSL-KDD et implémente des algorithmes de 
            classification (Decision Tree, Random Forest) et de clustering (K-Means).
          </p>
          <div className="flex flex-wrap gap-3">
            <NavLink to="/prediction">
              <Button className="bg-primary hover:bg-primary/90" data-testid="start-btn">
                <Play className="w-4 h-4 mr-2" />
                Tester le modèle
              </Button>
            </NavLink>
            <Button 
              variant="outline" 
              className="border-primary/50 text-primary hover:bg-primary/20"
              onClick={handleDownloadNotebook}
              disabled={downloading}
              data-testid="download-notebook-btn"
            >
              <Download className="w-4 h-4 mr-2" />
              {downloading ? "Génération..." : "Télécharger le Notebook"}
            </Button>
          </div>
        </div>
      </div>

      {/* Stats - Données dynamiques */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard 
          title="Échantillons" 
          value={stats?.total_samples?.toLocaleString() || "-"} 
          icon={Database} 
          color="primary"
          testId="stat-samples"
        />
        <StatCard 
          title="Features" 
          value={stats?.num_features ? stats.num_features - 1 : "-"} 
          icon={Cpu} 
          color="secondary"
          testId="stat-features"
        />
        <StatCard 
          title="Types d'attaques" 
          value={Object.keys(stats?.attack_categories || {}).length || "-"} 
          icon={AlertTriangle} 
          color="danger"
          testId="stat-categories"
        />
        <StatCard 
          title="Accuracy (RF)" 
          value={rfAccuracy ? `${(rfAccuracy * 100).toFixed(1)}%` : "-"} 
          icon={CheckCircle} 
          color="success"
          testId="stat-accuracy"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="font-mono text-lg">Distribution des classes</CardTitle>
            <CardDescription>Normal vs Attaques dans le dataset</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsPie>
                  <Pie
                    data={attackData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={90}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {attackData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                </RechartsPie>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="font-mono text-lg">Informations du projet</CardTitle>
            <CardDescription>Master 1 Cybersécurité - HIS 2025/2026</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-3 p-3 bg-secondary/30 rounded-lg">
              <Database className="w-5 h-5 text-primary" />
              <div>
                <p className="text-sm font-medium">Dataset</p>
                <p className="text-xs text-muted-foreground">NSL-KDD (version améliorée de KDD Cup 99)</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-secondary/30 rounded-lg">
              <Brain className="w-5 h-5 text-primary" />
              <div>
                <p className="text-sm font-medium">Algorithmes</p>
                <p className="text-xs text-muted-foreground">Decision Tree, Random Forest, K-Means</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-secondary/30 rounded-lg">
              <Target className="w-5 h-5 text-danger" />
              <div>
                <p className="text-sm font-medium">Objectif</p>
                <p className="text-xs text-muted-foreground">Classification binaire : Normal vs Attaque</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-secondary/30 rounded-lg">
              <Lock className="w-5 h-5 text-success" />
              <div>
                <p className="text-sm font-medium">Focus</p>
                <p className="text-xs text-muted-foreground">Détection des attaques DoS/DDoS</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// Dashboard EDA - Corrigé
const DashboardPage = () => {
  const [edaData, setEdaData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchEDA = async () => {
      try {
        const response = await axios.get(`${API}/dataset/eda`);
        setEdaData(response.data);
      } catch (error) {
        console.error("Erreur:", error);
        toast.error("Erreur lors du chargement des données");
      }
      setLoading(false);
    };
    fetchEDA();
  }, []);

  // Préparer les données pour le graphique des features (corrigé - utiliser log scale)
  const topFeaturesData = edaData?.top_features?.slice(0, 10).map(f => ({
    name: f.feature.length > 12 ? f.feature.substring(0, 12) + '...' : f.feature,
    fullName: f.feature,
    variance: f.variance,
    // Utiliser échelle logarithmique pour meilleure visualisation
    logVariance: f.variance > 0 ? Math.log10(f.variance) : 0
  })) || [];

  const labelDistribution = edaData?.feature_distributions?.label 
    ? Object.entries(edaData.feature_distributions.label).slice(0, 10).map(([name, value]) => ({ name, value }))
    : [];

  const protocolData = edaData?.feature_distributions?.protocol_type
    ? Object.entries(edaData.feature_distributions.protocol_type).map(([name, value]) => ({ name, value }))
    : [];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary mx-auto mb-4" />
          <p className="text-muted-foreground">Chargement...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 fade-in" data-testid="dashboard-page">
      <div>
        <h1 className="text-2xl font-bold font-mono">Analyse Exploratoire (EDA)</h1>
        <p className="text-muted-foreground">Exploration du dataset NSL-KDD</p>
      </div>

      {/* Distribution des labels */}
      <Card className="cyber-card">
        <CardHeader>
          <CardTitle className="font-mono">Distribution des types d'attaques</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={labelDistribution} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" stroke="#94a3b8" />
                <YAxis dataKey="name" type="category" stroke="#94a3b8" width={80} fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                <Bar dataKey="value" fill={COLORS.primary} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Protocoles */}
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="font-mono">Protocoles réseau</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsPie>
                  <Pie
                    data={protocolData}
                    cx="50%"
                    cy="50%"
                    outerRadius={70}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {protocolData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                </RechartsPie>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Top Features - avec échelle log */}
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="font-mono">Top Features (par variance)</CardTitle>
            <CardDescription>Échelle logarithmique pour meilleure visualisation</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topFeaturesData} margin={{ bottom: 70 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis 
                    dataKey="name" 
                    stroke="#94a3b8" 
                    angle={-45} 
                    textAnchor="end" 
                    height={70} 
                    fontSize={9}
                    interval={0}
                  />
                  <YAxis 
                    stroke="#94a3b8"
                    domain={[0, 'auto']}
                    tickFormatter={(v) => `10^${v.toFixed(0)}`}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                    formatter={(value, name, props) => {
                      const realVariance = props.payload.variance;
                      return [realVariance >= 1000000 ? `${(realVariance/1000000).toFixed(1)}M` : realVariance >= 1000 ? `${(realVariance/1000).toFixed(1)}K` : realVariance.toFixed(1), 'Variance'];
                    }}
                    labelFormatter={(label, payload) => payload[0]?.payload?.fullName || label}
                  />
                  <Bar dataKey="logVariance" fill={COLORS.success} radius={[4, 4, 0, 0]} name="Variance (log)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Aperçu des données */}
      <Card className="cyber-card">
        <CardHeader>
          <CardTitle className="font-mono">Aperçu des données</CardTitle>
          <CardDescription>Premiers échantillons du dataset</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-2 font-mono text-muted-foreground">duration</th>
                  <th className="text-left p-2 font-mono text-muted-foreground">protocol</th>
                  <th className="text-left p-2 font-mono text-muted-foreground">service</th>
                  <th className="text-left p-2 font-mono text-muted-foreground">src_bytes</th>
                  <th className="text-left p-2 font-mono text-muted-foreground">dst_bytes</th>
                  <th className="text-left p-2 font-mono text-muted-foreground">label</th>
                </tr>
              </thead>
              <tbody>
                {edaData?.sample_data?.slice(0, 8).map((row, idx) => (
                  <tr key={idx} className="border-b border-border/50 hover:bg-secondary/20">
                    <td className="p-2 font-mono">{row.duration}</td>
                    <td className="p-2">{row.protocol_type}</td>
                    <td className="p-2">{row.service}</td>
                    <td className="p-2 font-mono">{row.src_bytes}</td>
                    <td className="p-2 font-mono">{row.dst_bytes}</td>
                    <td className="p-2">
                      <Badge variant={row.label === 'normal' ? 'outline' : 'destructive'} className="text-xs">
                        {row.label}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Page Modèle ML
const ModelPage = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);

  const fetchMetrics = useCallback(async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/model/metrics`);
      setMetrics(response.data);
    } catch (error) {
      console.error("Erreur:", error);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchMetrics();
  }, [fetchMetrics]);

  const handleTrain = async () => {
    setTraining(true);
    toast.info("Entraînement en cours...");
    try {
      const response = await axios.post(`${API}/model/train`);
      setMetrics(response.data);
      toast.success("Entraînement terminé !");
    } catch (error) {
      toast.error("Erreur lors de l'entraînement");
    }
    setTraining(false);
  };

  const dtMetrics = metrics?.results?.decision_tree;
  const rfMetrics = metrics?.results?.random_forest;

  const comparisonData = dtMetrics && rfMetrics ? [
    { name: 'Accuracy', dt: (dtMetrics.accuracy * 100).toFixed(1), rf: (rfMetrics.accuracy * 100).toFixed(1) },
    { name: 'Precision', dt: (dtMetrics.precision * 100).toFixed(1), rf: (rfMetrics.precision * 100).toFixed(1) },
    { name: 'Recall', dt: (dtMetrics.recall * 100).toFixed(1), rf: (rfMetrics.recall * 100).toFixed(1) },
    { name: 'F1-Score', dt: (dtMetrics.f1_score * 100).toFixed(1), rf: (rfMetrics.f1_score * 100).toFixed(1) },
  ] : [];

  const featureImportance = rfMetrics?.feature_importance 
    ? Object.entries(rfMetrics.feature_importance).slice(0, 10).map(([name, value]) => ({ 
        name: name.length > 15 ? name.substring(0, 15) + '...' : name, 
        value: (value * 100).toFixed(1) 
      }))
    : [];

  return (
    <div className="space-y-6 fade-in" data-testid="model-page">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold font-mono">Classification Supervisée</h1>
          <p className="text-muted-foreground">Decision Tree & Random Forest</p>
        </div>
        <Button onClick={handleTrain} disabled={training} className="bg-primary hover:bg-primary/90" data-testid="train-btn">
          {training ? "Entraînement..." : "Entraîner les modèles"}
        </Button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary" />
        </div>
      ) : metrics?.results ? (
        <>
          {/* Métriques */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard title="Accuracy (RF)" value={`${(rfMetrics?.accuracy * 100).toFixed(1)}%`} icon={CheckCircle} color="success" testId="m-acc" />
            <StatCard title="Precision (RF)" value={`${(rfMetrics?.precision * 100).toFixed(1)}%`} icon={Target} color="primary" testId="m-prec" />
            <StatCard title="Recall (RF)" value={`${(rfMetrics?.recall * 100).toFixed(1)}%`} icon={Zap} color="warning" testId="m-rec" />
            <StatCard title="AUC (RF)" value={rfMetrics?.roc_data?.auc?.toFixed(3)} icon={TrendingUp} color="info" testId="m-auc" />
          </div>

          {/* Comparaison */}
          <Card className="cyber-card">
            <CardHeader>
              <CardTitle className="font-mono">Comparaison des modèles</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" domain={[0, 100]} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                    <Legend />
                    <Bar dataKey="dt" name="Decision Tree" fill={COLORS.secondary} radius={[4, 4, 0, 0]} />
                    <Bar dataKey="rf" name="Random Forest" fill={COLORS.success} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* ROC */}
            <Card className="cyber-card">
              <CardHeader>
                <CardTitle className="font-mono">Courbe ROC</CardTitle>
                <CardDescription>AUC = {rfMetrics?.roc_data?.auc?.toFixed(4)}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={rfMetrics?.roc_data?.fpr?.map((fpr, i) => ({
                      fpr: fpr.toFixed(2),
                      tpr: rfMetrics.roc_data.tpr[i].toFixed(2)
                    })).filter((_, i) => i % 10 === 0)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="fpr" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                      <Area type="monotone" dataKey="tpr" stroke={COLORS.primary} fill={COLORS.primary} fillOpacity={0.3} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Feature Importance */}
            <Card className="cyber-card">
              <CardHeader>
                <CardTitle className="font-mono">Importance des features</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={featureImportance} layout="vertical" margin={{ left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" stroke="#94a3b8" />
                      <YAxis dataKey="name" type="category" stroke="#94a3b8" width={90} fontSize={10} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                      <Bar dataKey="value" fill={COLORS.warning} radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Matrice de confusion */}
          <Card className="cyber-card">
            <CardHeader>
              <CardTitle className="font-mono">Matrice de confusion (Random Forest)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4 max-w-md mx-auto">
                <div className="p-4 bg-emerald-500/20 border border-emerald-500/50 rounded-lg text-center">
                  <p className="text-xs text-muted-foreground mb-1">Vrais Négatifs (TN)</p>
                  <p className="text-2xl font-bold font-mono text-emerald-400">{rfMetrics?.confusion_matrix?.[0]?.[0] || 0}</p>
                </div>
                <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-center">
                  <p className="text-xs text-muted-foreground mb-1">Faux Positifs (FP)</p>
                  <p className="text-2xl font-bold font-mono text-red-400">{rfMetrics?.confusion_matrix?.[0]?.[1] || 0}</p>
                </div>
                <div className="p-4 bg-amber-500/20 border border-amber-500/50 rounded-lg text-center">
                  <p className="text-xs text-muted-foreground mb-1">Faux Négatifs (FN)</p>
                  <p className="text-2xl font-bold font-mono text-amber-400">{rfMetrics?.confusion_matrix?.[1]?.[0] || 0}</p>
                </div>
                <div className="p-4 bg-cyan-500/20 border border-cyan-500/50 rounded-lg text-center">
                  <p className="text-xs text-muted-foreground mb-1">Vrais Positifs (TP)</p>
                  <p className="text-2xl font-bold font-mono text-cyan-400">{rfMetrics?.confusion_matrix?.[1]?.[1] || 0}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      ) : (
        <Card className="cyber-card p-8 text-center">
          <Brain className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground">Cliquez sur "Entraîner les modèles"</p>
        </Card>
      )}
    </div>
  );
};

// Page Prédiction
const PredictionPage = () => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [features, setFeatures] = useState({
    duration: 0, protocol_type: 'tcp', service: 'http', flag: 'SF',
    src_bytes: 0, dst_bytes: 0, count: 1, srv_count: 1, serror_rate: 0,
    same_srv_rate: 1, dst_host_count: 255, dst_host_srv_count: 255
  });

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/model/predict`, { features });
      setPrediction(response.data);
      toast[response.data.prediction === 'Attack' ? 'error' : 'success'](
        response.data.prediction === 'Attack' ? 'Intrusion détectée !' : 'Trafic normal'
      );
    } catch (error) {
      toast.error("Erreur - Entraînez d'abord le modèle");
    }
    setLoading(false);
  };

  const setDemoNormal = () => {
    // These values are typical of legitimate HTTP traffic
    setFeatures({
      duration: 8, protocol_type: 'tcp', service: 'http', flag: 'SF',
      src_bytes: 320, dst_bytes: 12500, count: 2, srv_count: 2, serror_rate: 0.0,
      same_srv_rate: 1, dst_host_count: 180, dst_host_srv_count: 175, logged_in: 1
    });
    setPrediction(null);
  };

  const setDemoAttack = () => {
    // These values are typical of DoS SYN flood attack (high count, S0 flag, no response)
    setFeatures({
      duration: 0, protocol_type: 'tcp', service: 'private', flag: 'S0',
      src_bytes: 0, dst_bytes: 0, count: 450, srv_count: 400, serror_rate: 0.95,
      same_srv_rate: 1, dst_host_count: 255, dst_host_srv_count: 5, logged_in: 0
    });
    setPrediction(null);
  };

  return (
    <div className="space-y-6 fade-in" data-testid="prediction-page">
      <div>
        <h1 className="text-2xl font-bold font-mono">Test de prédiction</h1>
        <p className="text-muted-foreground">Testez le modèle avec vos propres données</p>
      </div>

      <div className="flex flex-wrap gap-3">
        <Button variant="outline" onClick={setDemoNormal} data-testid="demo-normal-btn">
          <CheckCircle className="w-4 h-4 mr-2 text-emerald-400" />
          Exemple: Trafic Normal
        </Button>
        <Button variant="outline" onClick={setDemoAttack} data-testid="demo-attack-btn">
          <AlertTriangle className="w-4 h-4 mr-2 text-red-400" />
          Exemple: Attaque DoS
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="font-mono">Caractéristiques du trafic</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label className="text-xs">Duration</Label>
                <Input type="number" value={features.duration} onChange={(e) => setFeatures({...features, duration: parseInt(e.target.value) || 0})} className="terminal-input" />
              </div>
              <div>
                <Label className="text-xs">Protocol</Label>
                <Select value={features.protocol_type} onValueChange={(v) => setFeatures({...features, protocol_type: v})}>
                  <SelectTrigger className="terminal-input"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="tcp">TCP</SelectItem>
                    <SelectItem value="udp">UDP</SelectItem>
                    <SelectItem value="icmp">ICMP</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-xs">Service</Label>
                <Select value={features.service} onValueChange={(v) => setFeatures({...features, service: v})}>
                  <SelectTrigger className="terminal-input"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="http">HTTP</SelectItem>
                    <SelectItem value="ftp">FTP</SelectItem>
                    <SelectItem value="smtp">SMTP</SelectItem>
                    <SelectItem value="ssh">SSH</SelectItem>
                    <SelectItem value="private">Private</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-xs">Flag</Label>
                <Select value={features.flag} onValueChange={(v) => setFeatures({...features, flag: v})}>
                  <SelectTrigger className="terminal-input"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="SF">SF</SelectItem>
                    <SelectItem value="S0">S0</SelectItem>
                    <SelectItem value="REJ">REJ</SelectItem>
                    <SelectItem value="RSTR">RSTR</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-xs">Source Bytes</Label>
                <Input type="number" value={features.src_bytes} onChange={(e) => setFeatures({...features, src_bytes: parseInt(e.target.value) || 0})} className="terminal-input" />
              </div>
              <div>
                <Label className="text-xs">Dest Bytes</Label>
                <Input type="number" value={features.dst_bytes} onChange={(e) => setFeatures({...features, dst_bytes: parseInt(e.target.value) || 0})} className="terminal-input" />
              </div>
              <div>
                <Label className="text-xs">Count</Label>
                <Input type="number" value={features.count} onChange={(e) => setFeatures({...features, count: parseInt(e.target.value) || 0})} className="terminal-input" />
              </div>
              <div>
                <Label className="text-xs">Serror Rate (0-1)</Label>
                <Input type="number" step="0.1" min="0" max="1" value={features.serror_rate} onChange={(e) => setFeatures({...features, serror_rate: parseFloat(e.target.value) || 0})} className="terminal-input" />
              </div>
            </div>
            <Button onClick={handlePredict} disabled={loading} className="w-full bg-primary hover:bg-primary/90" data-testid="predict-btn">
              {loading ? "Analyse..." : "Analyser le trafic"}
            </Button>
          </CardContent>
        </Card>

        <Card className={`cyber-card ${prediction ? (prediction.prediction === 'Attack' ? 'border-red-500/50' : 'border-emerald-500/50') : ''}`}>
          <CardHeader>
            <CardTitle className="font-mono">Résultat</CardTitle>
          </CardHeader>
          <CardContent>
            {prediction ? (
              <div className="space-y-4" data-testid="prediction-result">
                <div className={`p-6 rounded-lg text-center ${prediction.prediction === 'Attack' ? 'bg-red-500/20 border border-red-500/50' : 'bg-emerald-500/20 border border-emerald-500/50'}`}>
                  {prediction.prediction === 'Attack' ? (
                    <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-3" />
                  ) : (
                    <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
                  )}
                  <h2 className={`text-2xl font-bold font-mono ${prediction.prediction === 'Attack' ? 'text-red-400' : 'text-emerald-400'}`}>
                    {prediction.prediction === 'Attack' ? 'INTRUSION DÉTECTÉE' : 'TRAFIC NORMAL'}
                  </h2>
                  <p className="text-sm text-muted-foreground mt-2">
                    Risque: <span className={`font-bold ${prediction.risk_level === 'CRITICAL' ? 'text-red-400' : prediction.risk_level === 'HIGH' ? 'text-amber-400' : prediction.risk_level === 'LOW' ? 'text-emerald-400' : 'text-yellow-400'}`}>
                      {prediction.risk_level}
                    </span>
                  </p>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Probabilité Normal</span>
                    <span className="font-mono text-emerald-400">{(prediction.probability.normal * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={prediction.probability.normal * 100} className="h-2" />
                  <div className="flex justify-between text-sm">
                    <span>Probabilité Attaque</span>
                    <span className="font-mono text-red-400">{(prediction.probability.attack * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={prediction.probability.attack * 100} className="h-2" />
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <Target className="w-12 h-12 text-muted-foreground mx-auto mb-3" />
                <p className="text-muted-foreground">Entrez les données et cliquez sur Analyser</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// Page Clustering
const ClusteringPage = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/clustering/results`);
      setData(response.data);
    } catch (error) {
      console.error("Erreur:", error);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleRun = async () => {
    setLoading(true);
    toast.info("Exécution du K-Means...");
    try {
      const response = await axios.post(`${API}/clustering/run`);
      setData(response.data);
      toast.success("Clustering terminé !");
    } catch (error) {
      toast.error("Erreur");
    }
    setLoading(false);
  };

  const elbowData = data?.elbow_data || [];
  const pcaData = data?.pca_data?.slice(0, 500) || [];

  return (
    <div className="space-y-6 fade-in" data-testid="clustering-page">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold font-mono">Clustering K-Means</h1>
          <p className="text-muted-foreground">Apprentissage non-supervisé</p>
        </div>
        <Button onClick={handleRun} disabled={loading} className="bg-primary hover:bg-primary/90">
          {loading ? "Exécution..." : "Exécuter K-Means"}
        </Button>
      </div>

      {data ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatCard title="Clusters" value={data.n_clusters} icon={PieChart} color="primary" />
            <StatCard title="Silhouette Score" value={data.silhouette_score?.toFixed(3)} icon={TrendingUp} color="success" />
            <StatCard title="Variance PCA" value={data.pca_explained_variance ? `${(data.pca_explained_variance.reduce((a, b) => a + b, 0) * 100).toFixed(0)}%` : '-'} icon={BarChart3} color="info" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="cyber-card">
              <CardHeader>
                <CardTitle className="font-mono">Méthode du coude</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={elbowData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="k" stroke="#94a3b8" />
                      <YAxis yAxisId="left" stroke="#94a3b8" />
                      <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                      <Legend />
                      <Line yAxisId="left" type="monotone" dataKey="inertia" name="Inertie" stroke={COLORS.primary} strokeWidth={2} />
                      <Line yAxisId="right" type="monotone" dataKey="silhouette" name="Silhouette" stroke={COLORS.success} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card className="cyber-card">
              <CardHeader>
                <CardTitle className="font-mono">Distribution des clusters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsPie>
                      <Pie
                        data={Object.entries(data.cluster_distribution || {}).map(([name, value]) => ({ name, value }))}
                        cx="50%"
                        cy="50%"
                        outerRadius={70}
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {Object.entries(data.cluster_distribution || {}).map((_, index) => (
                          <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                    </RechartsPie>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="cyber-card">
            <CardHeader>
              <CardTitle className="font-mono">Visualisation PCA</CardTitle>
              <CardDescription>Projection 2D des clusters</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="x" name="PC1" stroke="#94a3b8" />
                    <YAxis dataKey="y" name="PC2" stroke="#94a3b8" />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                    <Legend />
                    {[...new Set(pcaData.map(d => d.cluster))].map((cluster, idx) => (
                      <Scatter key={cluster} name={`Cluster ${cluster}`} data={pcaData.filter(d => d.cluster === cluster)} fill={CHART_COLORS[idx % CHART_COLORS.length]} />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </>
      ) : (
        <Card className="cyber-card p-8 text-center">
          <PieChart className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground">Cliquez sur "Exécuter K-Means"</p>
        </Card>
      )}
    </div>
  );
};

// App principal - Sans route Download
function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-background">
      <BrowserRouter>
        <Toaster position="top-right" toastOptions={{ style: { background: '#0f172a', border: '1px solid #334155', color: '#f8fafc' } }} />
        <div className="flex">
          <Sidebar isOpen={sidebarOpen} setIsOpen={setSidebarOpen} />
          <MobileHeader setIsOpen={setSidebarOpen} />
          
          <main className="flex-1 min-h-screen lg:ml-0 pt-16 lg:pt-0">
            <div className="p-4 md:p-6 lg:p-8 max-w-6xl mx-auto">
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/dashboard" element={<DashboardPage />} />
                <Route path="/model" element={<ModelPage />} />
                <Route path="/prediction" element={<PredictionPage />} />
                <Route path="/clustering" element={<ClusteringPage />} />
              </Routes>
            </div>
          </main>
        </div>
      </BrowserRouter>
    </div>
  );
}

export default App;
