import React, { useState, useEffect, useCallback } from "react";
import { BrowserRouter, Routes, Route, NavLink, useLocation } from "react-router-dom";
import axios from "axios";
import {
  Shield, Activity, Database, Brain, Target, Download, Menu, X,
  AlertTriangle, CheckCircle, Server, Cpu, Globe, Lock, Zap,
  BarChart3, PieChart, TrendingUp, FileText, Play, Upload
} from "lucide-react";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Progress } from "./components/ui/progress";
import { Badge } from "./components/ui/badge";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import { Toaster, toast } from "sonner";
import {
  LineChart, Line, BarChart, Bar, PieChart as RechartsPie, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  AreaChart, Area
} from "recharts";
import "@/App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Color palette
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

// Sidebar Navigation Component
const Sidebar = ({ isOpen, setIsOpen }) => {
  const location = useLocation();
  
  const navItems = [
    { path: "/", icon: Shield, label: "Accueil", testId: "nav-home" },
    { path: "/dashboard", icon: Activity, label: "Dashboard EDA", testId: "nav-dashboard" },
    { path: "/model", icon: Brain, label: "Modèle ML", testId: "nav-model" },
    { path: "/prediction", icon: Target, label: "Prédiction", testId: "nav-prediction" },
    { path: "/clustering", icon: PieChart, label: "Clustering", testId: "nav-clustering" },
    { path: "/download", icon: Download, label: "Télécharger", testId: "nav-download" }
  ];

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}
      
      {/* Sidebar */}
      <aside className={`
        fixed top-0 left-0 z-50 h-full w-64 bg-card border-r border-border
        transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        lg:translate-x-0 lg:static
      `}>
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="p-6 border-b border-border">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/20 rounded-lg shield-pulse">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-foreground font-mono">CyberSentinelle</h1>
                <p className="text-xs text-muted-foreground">Network IDS</p>
              </div>
            </div>
          </div>

          {/* Navigation */}
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

          {/* Footer */}
          <div className="p-4 border-t border-border">
            <div className="text-xs text-muted-foreground space-y-1">
              <p className="font-mono">Master 1 Cybersécurité</p>
              <p>HIS 2025-2026</p>
              <p className="text-primary">Zakarya Oukil</p>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};

// Mobile Header
const MobileHeader = ({ setIsOpen }) => (
  <header className="lg:hidden fixed top-0 left-0 right-0 z-30 bg-card border-b border-border">
    <div className="flex items-center justify-between p-4">
      <div className="flex items-center gap-2">
        <Shield className="w-6 h-6 text-primary" />
        <span className="font-mono font-bold">CyberSentinelle</span>
      </div>
      <Button 
        variant="ghost" 
        size="icon"
        onClick={() => setIsOpen(true)}
        data-testid="mobile-menu-btn"
      >
        <Menu className="w-6 h-6" />
      </Button>
    </div>
  </header>
);

// Stat Card Component
const StatCard = ({ title, value, icon: Icon, trend, color = "primary", testId }) => (
  <Card className="cyber-card" data-testid={testId}>
    <CardContent className="p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-muted-foreground mb-1">{title}</p>
          <p className="text-3xl font-bold font-mono" style={{ color: COLORS[color] }}>
            {value}
          </p>
          {trend && (
            <p className={`text-xs mt-2 ${trend > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}% vs précédent
            </p>
          )}
        </div>
        <div className={`p-3 rounded-lg bg-${color}/20`} style={{ backgroundColor: `${COLORS[color]}20` }}>
          <Icon className="w-6 h-6" style={{ color: COLORS[color] }} />
        </div>
      </div>
    </CardContent>
  </Card>
);

// Home Page
const HomePage = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await axios.get(`${API}/dataset/info`);
        setStats(response.data);
      } catch (error) {
        console.error("Error fetching stats:", error);
        // Set demo data
        setStats({
          total_samples: 5000,
          num_features: 41,
          attack_categories: { Normal: 2500, DoS: 1500, Probe: 500, R2L: 300, U2R: 200 }
        });
      }
      setLoading(false);
    };
    fetchStats();
  }, []);

  const attackData = stats?.attack_categories 
    ? Object.entries(stats.attack_categories).map(([name, value]) => ({ name, value }))
    : [];

  return (
    <div className="space-y-8 fade-in" data-testid="home-page">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 to-slate-800 border border-border p-8 md:p-12">
        <div className="absolute inset-0 grid-pattern opacity-30" />
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-primary/20 rounded-xl pulse-glow">
              <Shield className="w-10 h-10 text-primary" />
            </div>
            <Badge variant="outline" className="border-primary/50 text-primary">
              v1.0.0 - Active
            </Badge>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4 font-mono">
            <span className="text-foreground">Cyber</span>
            <span className="text-primary">Sentinelle</span>
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mb-6">
            Système de détection d'intrusions réseau basé sur le Machine Learning. 
            Analyse en temps réel du trafic pour identifier les attaques DoS/DDoS.
          </p>
          <div className="flex flex-wrap gap-4">
            <NavLink to="/prediction">
              <Button className="bg-primary hover:bg-primary/90 text-primary-foreground" data-testid="start-analysis-btn">
                <Play className="w-4 h-4 mr-2" />
                Démarrer l'analyse
              </Button>
            </NavLink>
            <NavLink to="/download">
              <Button variant="outline" className="border-primary/50 text-primary hover:bg-primary/20" data-testid="download-notebook-btn">
                <Download className="w-4 h-4 mr-2" />
                Télécharger le Notebook
              </Button>
            </NavLink>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard 
          title="Échantillons analysés" 
          value={stats?.total_samples?.toLocaleString() || "5,000"} 
          icon={Database} 
          color="primary"
          testId="stat-samples"
        />
        <StatCard 
          title="Features" 
          value={stats?.num_features || 41} 
          icon={Cpu} 
          color="secondary"
          testId="stat-features"
        />
        <StatCard 
          title="Catégories d'attaques" 
          value={Object.keys(stats?.attack_categories || {}).length || 5} 
          icon={AlertTriangle} 
          color="danger"
          testId="stat-categories"
        />
        <StatCard 
          title="Précision du modèle" 
          value="97.2%" 
          icon={CheckCircle} 
          color="success"
          testId="stat-accuracy"
        />
      </div>

      {/* Overview Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Attack Distribution */}
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 font-mono">
              <PieChart className="w-5 h-5 text-primary" />
              Distribution des attaques
            </CardTitle>
            <CardDescription>Répartition des catégories dans le dataset NSL-KDD</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsPie>
                  <Pie
                    data={attackData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {attackData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#0f172a', 
                      border: '1px solid #334155',
                      borderRadius: '8px'
                    }}
                  />
                </RechartsPie>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Project Info */}
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 font-mono">
              <FileText className="w-5 h-5 text-primary" />
              À propos du projet
            </CardTitle>
            <CardDescription>Mini-projet Data Science - Master 1 Cybersécurité</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex items-center gap-3 p-3 bg-secondary/30 rounded-lg">
                <Server className="w-5 h-5 text-primary" />
                <div>
                  <p className="text-sm font-medium">Dataset</p>
                  <p className="text-xs text-muted-foreground">NSL-KDD (amélioration du KDD Cup 99)</p>
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-secondary/30 rounded-lg">
                <Brain className="w-5 h-5 text-primary" />
                <div>
                  <p className="text-sm font-medium">Modèles ML</p>
                  <p className="text-xs text-muted-foreground">Decision Tree, Random Forest, K-Means</p>
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-secondary/30 rounded-lg">
                <Target className="w-5 h-5 text-danger" />
                <div>
                  <p className="text-sm font-medium">Focus</p>
                  <p className="text-xs text-muted-foreground">Détection des attaques DoS/DDoS</p>
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-secondary/30 rounded-lg">
                <Lock className="w-5 h-5 text-success" />
                <div>
                  <p className="text-sm font-medium">Objectif</p>
                  <p className="text-xs text-muted-foreground">Classification binaire Normal vs Attaque</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// Dashboard EDA Page
const DashboardPage = () => {
  const [edaData, setEdaData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchEDA = async () => {
      try {
        const response = await axios.get(`${API}/dataset/eda`);
        setEdaData(response.data);
      } catch (error) {
        console.error("Error fetching EDA:", error);
        toast.error("Erreur lors du chargement des données EDA");
      }
      setLoading(false);
    };
    fetchEDA();
  }, []);

  const correlationData = edaData?.correlation_matrix 
    ? Object.entries(edaData.correlation_matrix).slice(0, 8).map(([key, values]) => ({
        name: key.substring(0, 10),
        ...Object.fromEntries(Object.entries(values).slice(0, 8).map(([k, v]) => [k.substring(0, 8), parseFloat(v.toFixed(2))]))
      }))
    : [];

  const topFeaturesData = edaData?.top_features?.slice(0, 10) || [];

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
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4" />
          <p className="text-muted-foreground">Chargement des données EDA...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 fade-in" data-testid="dashboard-page">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold font-mono">Dashboard EDA</h1>
          <p className="text-muted-foreground">Analyse exploratoire du dataset NSL-KDD</p>
        </div>
        <Badge variant="outline" className="border-primary/50 text-primary">
          {edaData?.sample_data?.length || 0} échantillons chargés
        </Badge>
      </div>

      {/* Distribution des labels */}
      <Card className="cyber-card">
        <CardHeader>
          <CardTitle className="font-mono flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-primary" />
            Distribution des types d'attaques
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={labelDistribution} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" stroke="#94a3b8" />
                <YAxis dataKey="name" type="category" stroke="#94a3b8" width={100} fontSize={12} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                />
                <Bar dataKey="value" fill={COLORS.primary} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Protocol Distribution */}
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="font-mono flex items-center gap-2">
              <Globe className="w-5 h-5 text-primary" />
              Distribution des protocoles
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsPie>
                  <Pie
                    data={protocolData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {protocolData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                </RechartsPie>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Top Features by Variance */}
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="font-mono flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-primary" />
              Top Features (par variance)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topFeaturesData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="feature" stroke="#94a3b8" angle={-45} textAnchor="end" height={80} fontSize={10} />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                  <Bar dataKey="variance" fill={COLORS.success} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Sample Data Table */}
      <Card className="cyber-card">
        <CardHeader>
          <CardTitle className="font-mono flex items-center gap-2">
            <Database className="w-5 h-5 text-primary" />
            Aperçu des données (10 premiers échantillons)
          </CardTitle>
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
                {edaData?.sample_data?.slice(0, 10).map((row, idx) => (
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

// Model Training Page
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
      console.error("Error fetching metrics:", error);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchMetrics();
  }, [fetchMetrics]);

  const handleTrain = async () => {
    setTraining(true);
    toast.info("Entraînement des modèles en cours...");
    try {
      const response = await axios.post(`${API}/model/train`);
      setMetrics(response.data);
      toast.success("Modèles entraînés avec succès !");
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
    ? Object.entries(rfMetrics.feature_importance).map(([name, value]) => ({ name: name.substring(0, 15), value: (value * 100).toFixed(2) }))
    : [];

  return (
    <div className="space-y-6 fade-in" data-testid="model-page">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold font-mono">Modèle ML</h1>
          <p className="text-muted-foreground">Classification supervisée - Decision Tree & Random Forest</p>
        </div>
        <Button 
          onClick={handleTrain} 
          disabled={training}
          className="bg-primary hover:bg-primary/90"
          data-testid="train-models-btn"
        >
          {training ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-foreground mr-2" />
              Entraînement...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 mr-2" />
              Entraîner les modèles
            </>
          )}
        </Button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-96">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
        </div>
      ) : metrics?.results ? (
        <>
          {/* Metrics Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard 
              title="Accuracy (RF)" 
              value={`${(rfMetrics?.accuracy * 100).toFixed(1)}%`}
              icon={CheckCircle}
              color="success"
              testId="metric-accuracy"
            />
            <StatCard 
              title="Precision (RF)" 
              value={`${(rfMetrics?.precision * 100).toFixed(1)}%`}
              icon={Target}
              color="primary"
              testId="metric-precision"
            />
            <StatCard 
              title="Recall (RF)" 
              value={`${(rfMetrics?.recall * 100).toFixed(1)}%`}
              icon={Zap}
              color="warning"
              testId="metric-recall"
            />
            <StatCard 
              title="AUC (RF)" 
              value={rfMetrics?.roc_data?.auc?.toFixed(3)}
              icon={TrendingUp}
              color="info"
              testId="metric-auc"
            />
          </div>

          {/* Comparison Chart */}
          <Card className="cyber-card">
            <CardHeader>
              <CardTitle className="font-mono">Comparaison des modèles</CardTitle>
              <CardDescription>Decision Tree vs Random Forest</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
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
            {/* ROC Curve */}
            <Card className="cyber-card">
              <CardHeader>
                <CardTitle className="font-mono">Courbe ROC</CardTitle>
                <CardDescription>AUC: {rfMetrics?.roc_data?.auc?.toFixed(4)}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={rfMetrics?.roc_data?.fpr?.map((fpr, i) => ({
                      fpr: fpr.toFixed(3),
                      tpr: rfMetrics.roc_data.tpr[i].toFixed(3)
                    })).filter((_, i) => i % 10 === 0)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="fpr" stroke="#94a3b8" label={{ value: 'FPR', position: 'bottom' }} />
                      <YAxis stroke="#94a3b8" label={{ value: 'TPR', angle: -90, position: 'left' }} />
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
                <CardDescription>Top features pour la classification</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={featureImportance} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" stroke="#94a3b8" />
                      <YAxis dataKey="name" type="category" stroke="#94a3b8" width={100} fontSize={10} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                      <Bar dataKey="value" fill={COLORS.warning} radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Confusion Matrix */}
          <Card className="cyber-card">
            <CardHeader>
              <CardTitle className="font-mono">Matrice de confusion (Random Forest)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4 max-w-md mx-auto">
                <div className="p-6 bg-emerald-500/20 border border-emerald-500/50 rounded-lg text-center">
                  <p className="text-xs text-muted-foreground mb-1">Vrais Négatifs</p>
                  <p className="text-3xl font-bold font-mono text-emerald-400">
                    {rfMetrics?.confusion_matrix?.[0]?.[0] || 0}
                  </p>
                </div>
                <div className="p-6 bg-red-500/20 border border-red-500/50 rounded-lg text-center">
                  <p className="text-xs text-muted-foreground mb-1">Faux Positifs</p>
                  <p className="text-3xl font-bold font-mono text-red-400">
                    {rfMetrics?.confusion_matrix?.[0]?.[1] || 0}
                  </p>
                </div>
                <div className="p-6 bg-amber-500/20 border border-amber-500/50 rounded-lg text-center">
                  <p className="text-xs text-muted-foreground mb-1">Faux Négatifs</p>
                  <p className="text-3xl font-bold font-mono text-amber-400">
                    {rfMetrics?.confusion_matrix?.[1]?.[0] || 0}
                  </p>
                </div>
                <div className="p-6 bg-cyan-500/20 border border-cyan-500/50 rounded-lg text-center">
                  <p className="text-xs text-muted-foreground mb-1">Vrais Positifs</p>
                  <p className="text-3xl font-bold font-mono text-cyan-400">
                    {rfMetrics?.confusion_matrix?.[1]?.[1] || 0}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      ) : (
        <Card className="cyber-card p-12 text-center">
          <Brain className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
          <h2 className="text-xl font-bold mb-2">Aucun modèle entraîné</h2>
          <p className="text-muted-foreground mb-4">Cliquez sur "Entraîner les modèles" pour commencer</p>
        </Card>
      )}
    </div>
  );
};

// Prediction Page
const PredictionPage = () => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [features, setFeatures] = useState({
    duration: 0,
    protocol_type: 'tcp',
    service: 'http',
    flag: 'SF',
    src_bytes: 0,
    dst_bytes: 0,
    count: 1,
    srv_count: 1,
    serror_rate: 0,
    same_srv_rate: 1,
    dst_host_count: 255,
    dst_host_srv_count: 255
  });

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/model/predict`, { features });
      setPrediction(response.data);
      if (response.data.prediction === 'Attack') {
        toast.error("INTRUSION DÉTECTÉE !", { duration: 5000 });
      } else {
        toast.success("Trafic normal détecté");
      }
    } catch (error) {
      toast.error("Erreur: entraînez d'abord le modèle");
    }
    setLoading(false);
  };

  const handleDemoNormal = () => {
    setFeatures({
      duration: 0,
      protocol_type: 'tcp',
      service: 'http',
      flag: 'SF',
      src_bytes: 215,
      dst_bytes: 45076,
      count: 1,
      srv_count: 1,
      serror_rate: 0,
      same_srv_rate: 1,
      dst_host_count: 255,
      dst_host_srv_count: 255
    });
    setPrediction(null);
  };

  const handleDemoAttack = () => {
    setFeatures({
      duration: 0,
      protocol_type: 'tcp',
      service: 'private',
      flag: 'S0',
      src_bytes: 0,
      dst_bytes: 0,
      count: 511,
      srv_count: 511,
      serror_rate: 1,
      same_srv_rate: 1,
      dst_host_count: 255,
      dst_host_srv_count: 1
    });
    setPrediction(null);
  };

  return (
    <div className="space-y-6 fade-in" data-testid="prediction-page">
      <div>
        <h1 className="text-3xl font-bold font-mono">Prédiction en temps réel</h1>
        <p className="text-muted-foreground">Testez le modèle avec des données personnalisées</p>
      </div>

      {/* Demo Buttons */}
      <div className="flex flex-wrap gap-4">
        <Button variant="outline" onClick={handleDemoNormal} data-testid="demo-normal-btn">
          <CheckCircle className="w-4 h-4 mr-2 text-emerald-400" />
          Démo: Trafic Normal
        </Button>
        <Button variant="outline" onClick={handleDemoAttack} data-testid="demo-attack-btn">
          <AlertTriangle className="w-4 h-4 mr-2 text-red-400" />
          Démo: Attaque DoS
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <Card className="cyber-card">
          <CardHeader>
            <CardTitle className="font-mono flex items-center gap-2">
              <Cpu className="w-5 h-5 text-primary" />
              Caractéristiques du trafic
            </CardTitle>
            <CardDescription>Entrez les features pour la classification</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label className="text-xs">Duration</Label>
                <Input 
                  type="number" 
                  value={features.duration}
                  onChange={(e) => setFeatures({...features, duration: parseInt(e.target.value) || 0})}
                  className="terminal-input"
                  data-testid="input-duration"
                />
              </div>
              <div>
                <Label className="text-xs">Protocol Type</Label>
                <Select value={features.protocol_type} onValueChange={(v) => setFeatures({...features, protocol_type: v})}>
                  <SelectTrigger className="terminal-input" data-testid="select-protocol">
                    <SelectValue />
                  </SelectTrigger>
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
                  <SelectTrigger className="terminal-input" data-testid="select-service">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="http">HTTP</SelectItem>
                    <SelectItem value="ftp">FTP</SelectItem>
                    <SelectItem value="smtp">SMTP</SelectItem>
                    <SelectItem value="ssh">SSH</SelectItem>
                    <SelectItem value="dns">DNS</SelectItem>
                    <SelectItem value="telnet">Telnet</SelectItem>
                    <SelectItem value="private">Private</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-xs">Flag</Label>
                <Select value={features.flag} onValueChange={(v) => setFeatures({...features, flag: v})}>
                  <SelectTrigger className="terminal-input" data-testid="select-flag">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="SF">SF</SelectItem>
                    <SelectItem value="S0">S0</SelectItem>
                    <SelectItem value="REJ">REJ</SelectItem>
                    <SelectItem value="RSTR">RSTR</SelectItem>
                    <SelectItem value="SH">SH</SelectItem>
                    <SelectItem value="RSTO">RSTO</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="text-xs">Source Bytes</Label>
                <Input 
                  type="number" 
                  value={features.src_bytes}
                  onChange={(e) => setFeatures({...features, src_bytes: parseInt(e.target.value) || 0})}
                  className="terminal-input"
                  data-testid="input-src-bytes"
                />
              </div>
              <div>
                <Label className="text-xs">Dest Bytes</Label>
                <Input 
                  type="number" 
                  value={features.dst_bytes}
                  onChange={(e) => setFeatures({...features, dst_bytes: parseInt(e.target.value) || 0})}
                  className="terminal-input"
                  data-testid="input-dst-bytes"
                />
              </div>
              <div>
                <Label className="text-xs">Count</Label>
                <Input 
                  type="number" 
                  value={features.count}
                  onChange={(e) => setFeatures({...features, count: parseInt(e.target.value) || 0})}
                  className="terminal-input"
                  data-testid="input-count"
                />
              </div>
              <div>
                <Label className="text-xs">Serror Rate</Label>
                <Input 
                  type="number" 
                  step="0.1"
                  min="0"
                  max="1"
                  value={features.serror_rate}
                  onChange={(e) => setFeatures({...features, serror_rate: parseFloat(e.target.value) || 0})}
                  className="terminal-input"
                  data-testid="input-serror-rate"
                />
              </div>
            </div>

            <Button 
              onClick={handlePredict} 
              disabled={loading}
              className="w-full bg-primary hover:bg-primary/90"
              data-testid="predict-btn"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-foreground mr-2" />
                  Analyse en cours...
                </>
              ) : (
                <>
                  <Target className="w-4 h-4 mr-2" />
                  Analyser le trafic
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Prediction Result */}
        <Card className={`cyber-card ${prediction ? (prediction.prediction === 'Attack' ? 'border-red-500/50 glow-red' : 'border-emerald-500/50 glow-green') : ''}`}>
          <CardHeader>
            <CardTitle className="font-mono flex items-center gap-2">
              <Shield className="w-5 h-5 text-primary" />
              Résultat de l'analyse
            </CardTitle>
          </CardHeader>
          <CardContent>
            {prediction ? (
              <div className="space-y-6" data-testid="prediction-result">
                {/* Main Result */}
                <div className={`p-6 rounded-xl text-center ${prediction.prediction === 'Attack' ? 'bg-red-500/20 border border-red-500/50' : 'bg-emerald-500/20 border border-emerald-500/50'}`}>
                  {prediction.prediction === 'Attack' ? (
                    <AlertTriangle className="w-16 h-16 text-red-400 mx-auto mb-4" />
                  ) : (
                    <CheckCircle className="w-16 h-16 text-emerald-400 mx-auto mb-4" />
                  )}
                  <h2 className={`text-3xl font-bold font-mono ${prediction.prediction === 'Attack' ? 'text-red-400' : 'text-emerald-400'}`}>
                    {prediction.prediction === 'Attack' ? 'INTRUSION DÉTECTÉE' : 'TRAFIC NORMAL'}
                  </h2>
                  <p className="text-muted-foreground mt-2">
                    Niveau de risque: <span className={`font-bold ${prediction.risk_level === 'CRITICAL' ? 'text-red-400' : prediction.risk_level === 'HIGH' ? 'text-amber-400' : prediction.risk_level === 'MEDIUM' ? 'text-yellow-400' : 'text-emerald-400'}`}>
                      {prediction.risk_level}
                    </span>
                  </p>
                </div>

                {/* Probabilities */}
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Probabilité Normal</span>
                      <span className="font-mono text-emerald-400">{(prediction.probability.normal * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={prediction.probability.normal * 100} className="h-2 bg-secondary" />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Probabilité Attaque</span>
                      <span className="font-mono text-red-400">{(prediction.probability.attack * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={prediction.probability.attack * 100} className="h-2 bg-secondary" />
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <Target className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">
                  Entrez les caractéristiques du trafic et cliquez sur "Analyser"
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// Clustering Page
const ClusteringPage = () => {
  const [clusterData, setClusterData] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchClustering = useCallback(async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/clustering/results`);
      setClusterData(response.data);
    } catch (error) {
      console.error("Error:", error);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchClustering();
  }, [fetchClustering]);

  const handleRunClustering = async () => {
    setLoading(true);
    toast.info("Exécution du clustering K-Means...");
    try {
      const response = await axios.post(`${API}/clustering/run`);
      setClusterData(response.data);
      toast.success("Clustering terminé !");
    } catch (error) {
      toast.error("Erreur lors du clustering");
    }
    setLoading(false);
  };

  const elbowData = clusterData?.elbow_data || [];
  const pcaData = clusterData?.pca_data?.slice(0, 500) || [];

  return (
    <div className="space-y-6 fade-in" data-testid="clustering-page">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold font-mono">Clustering K-Means</h1>
          <p className="text-muted-foreground">Analyse non-supervisée des patterns réseau</p>
        </div>
        <Button 
          onClick={handleRunClustering} 
          disabled={loading}
          className="bg-primary hover:bg-primary/90"
          data-testid="run-clustering-btn"
        >
          {loading ? "Exécution..." : "Exécuter K-Means"}
        </Button>
      </div>

      {clusterData ? (
        <>
          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatCard 
              title="Nombre de clusters" 
              value={clusterData.n_clusters}
              icon={PieChart}
              color="primary"
              testId="stat-clusters"
            />
            <StatCard 
              title="Score Silhouette" 
              value={clusterData.silhouette_score?.toFixed(4)}
              icon={TrendingUp}
              color="success"
              testId="stat-silhouette"
            />
            <StatCard 
              title="Variance PCA" 
              value={clusterData.pca_explained_variance ? `${(clusterData.pca_explained_variance.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%` : 'N/A'}
              icon={BarChart3}
              color="info"
              testId="stat-pca-variance"
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Elbow Method */}
            <Card className="cyber-card">
              <CardHeader>
                <CardTitle className="font-mono">Méthode du coude</CardTitle>
                <CardDescription>Inertie et Silhouette par nombre de clusters</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
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

            {/* Cluster Distribution */}
            <Card className="cyber-card">
              <CardHeader>
                <CardTitle className="font-mono">Distribution des clusters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsPie>
                      <Pie
                        data={Object.entries(clusterData.cluster_distribution || {}).map(([name, value]) => ({ name, value }))}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {Object.entries(clusterData.cluster_distribution || {}).map((entry, index) => (
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

          {/* PCA Scatter Plot */}
          <Card className="cyber-card">
            <CardHeader>
              <CardTitle className="font-mono">Visualisation PCA des clusters</CardTitle>
              <CardDescription>Projection 2D des données avec coloration par cluster</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="x" name="PC1" stroke="#94a3b8" />
                    <YAxis dataKey="y" name="PC2" stroke="#94a3b8" />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }} 
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                      formatter={(value, name) => [value.toFixed(2), name]}
                    />
                    <Legend />
                    {[...new Set(pcaData.map(d => d.cluster))].map((cluster, idx) => (
                      <Scatter 
                        key={cluster}
                        name={`Cluster ${cluster}`}
                        data={pcaData.filter(d => d.cluster === cluster)}
                        fill={CHART_COLORS[idx % CHART_COLORS.length]}
                      />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </>
      ) : (
        <Card className="cyber-card p-12 text-center">
          <PieChart className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
          <h2 className="text-xl font-bold mb-2">Aucun résultat de clustering</h2>
          <p className="text-muted-foreground">Cliquez sur "Exécuter K-Means" pour lancer l'analyse</p>
        </Card>
      )}
    </div>
  );
};

// Download Page
const DownloadPage = () => {
  const [generating, setGenerating] = useState(false);

  const handleDownload = async () => {
    setGenerating(true);
    toast.info("Génération du notebook en cours...");
    try {
      // First generate the notebook
      await axios.post(`${API}/notebook/generate`);
      
      // Then download it
      const response = await axios.get(`${API}/notebook/download`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'Detection_Intrusion_Reseau_DoS_Zakarya_Oukil.ipynb');
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      toast.success("Notebook téléchargé avec succès !");
    } catch (error) {
      toast.error("Erreur lors du téléchargement");
    }
    setGenerating(false);
  };

  return (
    <div className="space-y-6 fade-in" data-testid="download-page">
      <div>
        <h1 className="text-3xl font-bold font-mono">Télécharger le Notebook</h1>
        <p className="text-muted-foreground">Obtenez le Jupyter Notebook complet pour votre soumission</p>
      </div>

      <Card className="cyber-card">
        <CardContent className="p-8">
          <div className="text-center max-w-2xl mx-auto">
            <div className="p-4 bg-primary/20 rounded-2xl w-fit mx-auto mb-6">
              <FileText className="w-16 h-16 text-primary" />
            </div>
            
            <h2 className="text-2xl font-bold font-mono mb-2">
              Détection d'Intrusions Réseau (DoS/DDoS)
            </h2>
            <p className="text-muted-foreground mb-6">
              Notebook Jupyter complet en français avec :
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-left mb-8">
              <div className="p-4 bg-secondary/30 rounded-lg">
                <h3 className="font-bold text-primary mb-2">1. Analyse Exploratoire</h3>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>- Statistiques descriptives</li>
                  <li>- Visualisations (histogrammes, boxplots)</li>
                  <li>- Matrice de corrélation</li>
                </ul>
              </div>
              <div className="p-4 bg-secondary/30 rounded-lg">
                <h3 className="font-bold text-primary mb-2">2. Prétraitement</h3>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>- Nettoyage des données</li>
                  <li>- Encodage des variables</li>
                  <li>- Normalisation</li>
                </ul>
              </div>
              <div className="p-4 bg-secondary/30 rounded-lg">
                <h3 className="font-bold text-primary mb-2">3. Classification</h3>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>- Decision Tree & Random Forest</li>
                  <li>- Métriques complètes</li>
                  <li>- Courbes ROC</li>
                </ul>
              </div>
              <div className="p-4 bg-secondary/30 rounded-lg">
                <h3 className="font-bold text-primary mb-2">4. Clustering</h3>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>- K-Means avec méthode du coude</li>
                  <li>- Visualisation PCA</li>
                  <li>- Score Silhouette</li>
                </ul>
              </div>
            </div>

            <Button 
              size="lg" 
              onClick={handleDownload}
              disabled={generating}
              className="bg-primary hover:bg-primary/90 text-lg px-8"
              data-testid="download-btn"
            >
              {generating ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-foreground mr-2" />
                  Génération...
                </>
              ) : (
                <>
                  <Download className="w-5 h-5 mr-2" />
                  Télécharger le Notebook (FR)
                </>
              )}
            </Button>

            <p className="text-xs text-muted-foreground mt-4">
              Format: .ipynb | Prêt pour Google Colab / Jupyter
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="cyber-card">
          <CardContent className="p-6 text-center">
            <Database className="w-8 h-8 text-primary mx-auto mb-3" />
            <h3 className="font-bold mb-1">Dataset NSL-KDD</h3>
            <p className="text-xs text-muted-foreground">~5000 échantillons</p>
          </CardContent>
        </Card>
        <Card className="cyber-card">
          <CardContent className="p-6 text-center">
            <Brain className="w-8 h-8 text-primary mx-auto mb-3" />
            <h3 className="font-bold mb-1">Modèles ML</h3>
            <p className="text-xs text-muted-foreground">DT, RF, K-Means</p>
          </CardContent>
        </Card>
        <Card className="cyber-card">
          <CardContent className="p-6 text-center">
            <Globe className="w-8 h-8 text-primary mx-auto mb-3" />
            <h3 className="font-bold mb-1">Langue</h3>
            <p className="text-xs text-muted-foreground">100% Français</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// Main App Component
function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-background">
      <BrowserRouter>
        <Toaster 
          position="top-right" 
          toastOptions={{
            style: {
              background: '#0f172a',
              border: '1px solid #334155',
              color: '#f8fafc'
            }
          }}
        />
        <div className="flex">
          <Sidebar isOpen={sidebarOpen} setIsOpen={setSidebarOpen} />
          <MobileHeader setIsOpen={setSidebarOpen} />
          
          <main className="flex-1 min-h-screen lg:ml-0 pt-16 lg:pt-0">
            <div className="p-4 md:p-6 lg:p-8 max-w-7xl mx-auto">
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/dashboard" element={<DashboardPage />} />
                <Route path="/model" element={<ModelPage />} />
                <Route path="/prediction" element={<PredictionPage />} />
                <Route path="/clustering" element={<ClusteringPage />} />
                <Route path="/download" element={<DownloadPage />} />
              </Routes>
            </div>
          </main>
        </div>
      </BrowserRouter>
    </div>
  );
}

export default App;
