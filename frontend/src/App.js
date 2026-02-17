import React, { useState, useEffect, useCallback, Suspense } from "react";
import { createPortal } from "react-dom";
import { BrowserRouter, Routes, Route, NavLink, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import {
  Shield, Activity, Brain, Target, Menu, X,
  AlertTriangle, CheckCircle, Globe, Zap,
  BarChart3, PieChart, FileText, Maximize2,
  Radio, Home as HomeIcon, Terminal, ChevronRight,
  Cpu, Lock, Eye, Wifi, Database, Server
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

// Import pages
import HomePage from "./pages/Home";
import LiveMonitorPage from "./pages/Monitor";

// Import 3D components
const NetworkTopology3D = React.lazy(() => import("./components/NetworkTopology3D"));
const ThreatVisualization3D = React.lazy(() => import("./components/ThreatVisualization3D"));

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

// Animated Background Particles
const ParticleBackground = () => {
  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
      {[...Array(30)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 bg-cyan-500/30 rounded-full"
          initial={{
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
          }}
          animate={{
            x: [null, Math.random() * window.innerWidth],
            y: [null, Math.random() * window.innerHeight],
            opacity: [0.2, 0.5, 0.2],
          }}
          transition={{
            duration: 10 + Math.random() * 20,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      ))}
    </div>
  );
};

// Glitch Text Effect
const GlitchText = ({ children, className = "" }) => {
  return (
    <span className={`relative inline-block ${className}`}>
      <span className="relative z-10">{children}</span>
      <span className="absolute top-0 left-0 -ml-[2px] text-red-500/50 animate-glitch-1 z-0" aria-hidden="true">{children}</span>
      <span className="absolute top-0 left-0 ml-[2px] text-cyan-500/50 animate-glitch-2 z-0" aria-hidden="true">{children}</span>
    </span>
  );
};

// Cyber Border Animation
const CyberBorder = ({ children, className = "", glowColor = "cyan" }) => {
  const colorMap = {
    cyan: "border-cyan-500/50 hover:border-cyan-400 hover:shadow-[0_0_30px_rgba(0,240,255,0.3)]",
    red: "border-red-500/50 hover:border-red-400 hover:shadow-[0_0_30px_rgba(255,0,60,0.3)]",
    green: "border-green-500/50 hover:border-green-400 hover:shadow-[0_0_30px_rgba(0,255,65,0.3)]",
    purple: "border-purple-500/50 hover:border-purple-400 hover:shadow-[0_0_30px_rgba(189,0,255,0.3)]",
    yellow: "border-yellow-500/50 hover:border-yellow-400 hover:shadow-[0_0_30px_rgba(250,255,0,0.3)]",
  };
  
  return (
    <div className={`relative border transition-all duration-500 ${colorMap[glowColor]} ${className}`}>
      <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-current -translate-x-px -translate-y-px" />
      <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-current translate-x-px -translate-y-px" />
      <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-current -translate-x-px translate-y-px" />
      <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-current translate-x-px translate-y-px" />
      {children}
    </div>
  );
};

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
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 20 }}
        transition={{ type: "spring", damping: 25 }}
        className="fixed inset-0 flex items-center justify-center p-4 pointer-events-none"
        style={{ zIndex: 100000 }}
      >
        <CyberBorder glowColor="cyan" className="bg-black/95 w-full max-w-5xl max-h-[90vh] overflow-auto pointer-events-auto">
          <div className="flex items-center justify-between p-4 border-b border-cyan-500/20 sticky top-0 bg-black/95 backdrop-blur-sm">
            <h3 className="text-lg font-mono tracking-wider text-cyan-400">{title}</h3>
            <Button variant="ghost" size="icon" onClick={onClose} className="hover:bg-cyan-500/10 hover:text-cyan-400">
              <X className="w-5 h-5" />
            </Button>
          </div>
          <div className="p-6">
            <div className="h-[500px]">{children}</div>
          </div>
        </CyberBorder>
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
      <motion.div 
        className="cursor-pointer relative group" 
        onClick={() => setIsModalOpen(true)}
        whileHover={{ scale: 1.01 }}
        transition={{ type: "spring", stiffness: 400 }}
        data-testid={`chart-${title.toLowerCase().replace(/\s+/g, '-')}`}
      >
        {children}
        <motion.div 
          className="absolute top-2 right-2 opacity-0 group-hover:opacity-100"
          initial={{ scale: 0 }}
          whileHover={{ scale: 1.1 }}
          animate={{ scale: 1 }}
        >
          <div className="p-2 bg-cyan-500/20 border border-cyan-500/50 backdrop-blur-sm">
            <Maximize2 className="w-4 h-4 text-cyan-400" />
          </div>
        </motion.div>
      </motion.div>
      <ChartModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} title={title}>
        {chartContent}
      </ChartModal>
    </>
  );
};

// Animated Stat Card
const StatCard = ({ icon: Icon, label, value, color, delay = 0 }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay, type: "spring", stiffness: 100 }}
  >
    <CyberBorder glowColor={color} className="bg-black/50 p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[10px] text-gray-500 tracking-widest uppercase mb-1">{label}</p>
          <motion.p 
            className="text-2xl font-mono font-bold"
            style={{ color: COLORS[color] || color }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: delay + 0.2 }}
          >
            {value}
          </motion.p>
        </div>
        <motion.div 
          className="p-3"
          style={{ backgroundColor: `${COLORS[color] || color}15`, border: `1px solid ${COLORS[color] || color}30` }}
          whileHover={{ scale: 1.1, rotate: 5 }}
        >
          <Icon className="w-5 h-5" style={{ color: COLORS[color] || color }} />
        </motion.div>
      </div>
    </CyberBorder>
  </motion.div>
);

// Sidebar Component - FIXED
const Sidebar = ({ isOpen, setIsOpen }) => {
  const location = useLocation();
  const [hoveredItem, setHoveredItem] = useState(null);
  
  const navItems = [
    { path: "/", icon: HomeIcon, label: "ACCUEIL", color: "cyan" },
    { path: "/monitor", icon: Radio, label: "LIVE MONITOR", highlight: true, color: "red" },
    { path: "/dashboard", icon: Activity, label: "DASHBOARD EDA", color: "green" },
    { path: "/model", icon: Brain, label: "MODÈLE ML", color: "purple" },
    { path: "/prediction", icon: Target, label: "PRÉDICTION", color: "yellow" },
    { path: "/clustering", icon: PieChart, label: "CLUSTERING", color: "cyan" }
  ];

  // Only close sidebar on mobile
  const handleNavClick = () => {
    if (window.innerWidth < 1024) {
      setIsOpen(false);
    }
  };

  return (
    <>
      {/* Mobile overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/90 backdrop-blur-sm z-40 lg:hidden"
            onClick={() => setIsOpen(false)}
          />
        )}
      </AnimatePresence>
      
      {/* Sidebar */}
      <aside className={`fixed left-0 top-0 bottom-0 w-[280px] z-50 flex flex-col transition-transform duration-300 lg:translate-x-0 ${isOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        {/* Animated border effect */}
        <div className="absolute inset-0 bg-gradient-to-b from-black via-gray-950 to-black" />
        <div className="absolute top-0 right-0 bottom-0 w-px bg-gradient-to-b from-transparent via-cyan-500/50 to-transparent" />
        
        {/* Scanning line effect */}
        <motion.div
          className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan-400 to-transparent"
          animate={{ top: ["0%", "100%", "0%"] }}
          transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
        />
        
        {/* Content */}
        <div className="relative z-10 flex flex-col h-full">
          {/* Header */}
          <div className="p-6 border-b border-cyan-500/10">
            <motion.div 
              className="flex items-center gap-3"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="relative">
                <motion.div 
                  className="w-12 h-12 border-2 border-cyan-500/50 flex items-center justify-center bg-cyan-500/5"
                  animate={{ 
                    boxShadow: ["0 0 20px rgba(0,240,255,0.2)", "0 0 40px rgba(0,240,255,0.4)", "0 0 20px rgba(0,240,255,0.2)"]
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  <Shield className="w-6 h-6 text-cyan-400" />
                </motion.div>
                <motion.div 
                  className="absolute -top-1 -right-1 w-3 h-3 bg-green-500"
                  animate={{ scale: [1, 1.2, 1], opacity: [1, 0.7, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
              </div>
              <div>
                <h1 className="font-mono text-xl font-bold tracking-wider">
                  <span className="text-white">CYBER</span>
                  <span className="text-cyan-400">SENTINELLE</span>
                </h1>
                <p className="text-[10px] text-cyan-500/50 tracking-[0.2em]">SYSTÈME IDS v2.0</p>
              </div>
            </motion.div>
          </div>
          
          {/* System Status */}
          <div className="px-6 py-4 border-b border-cyan-500/10">
            <div className="flex items-center gap-2 text-xs">
              <motion.div 
                className="w-2 h-2 bg-green-500 rounded-full"
                animate={{ opacity: [1, 0.5, 1] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
              <span className="text-green-400 font-mono">SYSTÈME ACTIF</span>
            </div>
          </div>
          
          {/* Navigation */}
          <nav className="flex-1 py-4 overflow-y-auto">
            <div className="px-6 mb-3">
              <span className="text-[10px] text-gray-600 tracking-[0.2em] uppercase">Navigation</span>
            </div>
            {navItems.map((item, index) => {
              const isActive = location.pathname === item.path;
              const isHovered = hoveredItem === item.path;
              
              return (
                <motion.div
                  key={item.path}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <NavLink
                    to={item.path}
                    onClick={handleNavClick}
                    onMouseEnter={() => setHoveredItem(item.path)}
                    onMouseLeave={() => setHoveredItem(null)}
                    className="relative block mx-3 mb-1"
                  >
                    <motion.div
                      className={`flex items-center gap-3 px-4 py-3 transition-colors ${
                        isActive 
                          ? 'bg-cyan-500/10 text-cyan-400' 
                          : 'text-gray-400 hover:text-white'
                      }`}
                      whileHover={{ x: 5 }}
                    >
                      {/* Active indicator */}
                      {isActive && (
                        <motion.div
                          className="absolute left-0 top-0 bottom-0 w-1 bg-cyan-400"
                          layoutId="activeIndicator"
                          transition={{ type: "spring", stiffness: 500, damping: 30 }}
                        />
                      )}
                      
                      {/* Hover glow */}
                      {(isHovered || isActive) && (
                        <motion.div
                          className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 to-transparent"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                        />
                      )}
                      
                      <item.icon className={`w-4 h-4 relative z-10 ${item.highlight && !isActive ? 'text-red-500' : ''}`} />
                      <span className="font-mono text-sm tracking-wider relative z-10">{item.label}</span>
                      
                      {item.highlight && (
                        <motion.span 
                          className="ml-auto w-2 h-2 bg-red-500 rounded-full relative z-10"
                          animate={{ scale: [1, 1.3, 1], opacity: [1, 0.7, 1] }}
                          transition={{ duration: 1, repeat: Infinity }}
                        />
                      )}
                    </motion.div>
                  </NavLink>
                </motion.div>
              );
            })}
          </nav>
          
          {/* Footer */}
          <div className="p-4 border-t border-cyan-500/10">
            <div className="flex items-center gap-2 mb-2">
              <Database className="w-3 h-3 text-cyan-500/50" />
              <span className="text-[10px] text-gray-600 font-mono">NSL-KDD Dataset</span>
            </div>
            <div className="text-[10px] text-gray-600 tracking-wider font-mono">
              <div className="flex items-center gap-2">
                <span className="text-cyan-500">◆</span>
                <span>Master 1 Cybersécurité</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-cyan-500">◆</span>
                <span>HIS - 2025/2026</span>
              </div>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-cyan-400">◆</span>
                <span className="text-cyan-400">Zakarya Oukil</span>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};

// Loading Spinner with 3D effect
const LoadingSpinner = () => (
  <div className="flex items-center justify-center h-96">
    <div className="relative">
      {/* Outer ring */}
      <motion.div
        className="w-24 h-24 border-2 border-cyan-500/30 rounded-full"
        animate={{ rotate: 360 }}
        transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
      />
      {/* Middle ring */}
      <motion.div
        className="absolute inset-2 border-2 border-purple-500/30 rounded-full"
        animate={{ rotate: -360 }}
        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
      />
      {/* Inner ring */}
      <motion.div
        className="absolute inset-4 border-2 border-cyan-400/50 rounded-full"
        animate={{ rotate: 360 }}
        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
      />
      {/* Center icon */}
      <motion.div
        className="absolute inset-0 flex items-center justify-center"
        animate={{ scale: [1, 1.1, 1] }}
        transition={{ duration: 1, repeat: Infinity }}
      >
        <Shield className="w-8 h-8 text-cyan-400" />
      </motion.div>
      
      {/* Orbiting dots */}
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="absolute w-2 h-2 bg-cyan-400 rounded-full"
          style={{ top: '50%', left: '50%' }}
          animate={{
            x: [0, Math.cos((i * 120 * Math.PI) / 180) * 40],
            y: [0, Math.sin((i * 120 * Math.PI) / 180) * 40],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            repeatType: "reverse",
            delay: i * 0.2,
          }}
        />
      ))}
    </div>
    <motion.p
      className="absolute mt-40 text-cyan-500/50 font-mono text-sm tracking-widest"
      animate={{ opacity: [0.5, 1, 0.5] }}
      transition={{ duration: 1.5, repeat: Infinity }}
    >
      CHARGEMENT...
    </motion.p>
  </div>
);

// Dashboard EDA Page with animations
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

  // Transform API data into chart-ready format
  const labelDist = data.feature_distributions?.label || {};
  const attackDistribution = Object.entries(labelDist).map(([type, count]) => ({ type, count }));
  
  const protocolDist = data.feature_distributions?.protocol_type || {};
  const protocolDistribution = Object.entries(protocolDist).map(([protocol, count]) => ({ protocol, count }));
  
  const topFeatures = data.top_features || [];

  const attackChartContent = (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={attackDistribution} layout="vertical" margin={{ left: 80, bottom: 30 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,240,255,0.1)" />
        <XAxis type="number" stroke="#00F0FF" tick={{ fill: '#00F0FF', fontSize: 11, fontFamily: 'Share Tech Mono' }} />
        <YAxis type="category" dataKey="type" stroke="#00F0FF" tick={{ fill: '#fff', fontSize: 11, fontFamily: 'Share Tech Mono' }} />
        <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #00F0FF', fontFamily: 'Share Tech Mono' }} />
        <Legend />
        <Bar dataKey="count" name="Échantillons" fill="#00F0FF" />
      </BarChart>
    </ResponsiveContainer>
  );

  const featuresChartContent = (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={topFeatures} margin={{ bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,255,65,0.1)" />
        <XAxis dataKey="feature" stroke="#00FF41" tick={{ fill: '#00FF41', fontSize: 10, fontFamily: 'Share Tech Mono', angle: -45, textAnchor: 'end' }} />
        <YAxis stroke="#00FF41" scale="log" domain={['auto', 'auto']} tick={{ fill: '#00FF41', fontSize: 11, fontFamily: 'Share Tech Mono' }} />
        <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #00FF41', fontFamily: 'Share Tech Mono' }} />
        <Bar dataKey="variance" name="Variance" fill="#00FF41" />
      </BarChart>
    </ResponsiveContainer>
  );

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      {/* Header */}
      <motion.div 
        className="flex items-center gap-4"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <CyberBorder glowColor="cyan" className="p-3 bg-cyan-500/10">
          <BarChart3 className="w-6 h-6 text-cyan-400" />
        </CyberBorder>
        <div>
          <h1 className="text-2xl font-mono font-bold tracking-wider">
            ANALYSE <GlitchText className="text-cyan-400">EXPLORATOIRE</GlitchText>
          </h1>
          <p className="text-gray-500 text-sm font-mono">Dataset NSL-KDD • Cliquer pour agrandir</p>
        </div>
      </motion.div>

      {/* 3D Network Topology */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <CyberBorder glowColor="cyan" className="bg-black/50 p-6">
          <h3 className="font-mono text-lg tracking-wider mb-1 text-cyan-400">TOPOLOGIE RÉSEAU 3D</h3>
          <p className="text-xs text-gray-500 mb-2">Modèle interactif du réseau • Faites glisser pour tourner</p>
          <div className="h-[400px]">
            <Suspense fallback={<div className="flex items-center justify-center h-full"><p className="text-cyan-500/50 font-mono text-sm animate-pulse">CHARGEMENT 3D...</p></div>}>
              <NetworkTopology3D />
            </Suspense>
          </div>
          <div className="flex items-center gap-6 mt-4 text-xs font-mono">
            <div className="flex items-center gap-2"><div className="w-3 h-3 bg-cyan-400" /><span className="text-gray-400">Serveur</span></div>
            <div className="flex items-center gap-2"><div className="w-3 h-3 bg-yellow-400 rotate-45" /><span className="text-gray-400">Pare-feu / IDS</span></div>
            <div className="flex items-center gap-2"><div className="w-3 h-3 bg-green-400 rounded-full" /><span className="text-gray-400">Client</span></div>
            <div className="flex items-center gap-2"><div className="w-3 h-3 bg-red-400" style={{ clipPath: 'polygon(50% 0%, 0% 100%, 100% 100%)' }} /><span className="text-gray-400">Attaquant</span></div>
          </div>
        </CyberBorder>
      </motion.div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Attack Distribution */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2"
        >
          <CyberBorder glowColor="cyan" className="bg-black/50 p-6">
            <h3 className="font-mono text-lg tracking-wider mb-4 text-cyan-400">DISTRIBUTION DES ATTAQUES</h3>
            <ClickableChart title="Distribution des types d'attaques" chartContent={attackChartContent}>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={attackDistribution} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,240,255,0.1)" />
                    <XAxis type="number" stroke="#505050" tick={{ fill: '#00F0FF', fontSize: 10 }} />
                    <YAxis type="category" dataKey="type" stroke="#505050" tick={{ fill: '#fff', fontSize: 10 }} width={80} />
                    <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #00F0FF' }} />
                    <Bar dataKey="count" fill="#00F0FF">
                      {attackDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ClickableChart>
          </CyberBorder>
        </motion.div>

        {/* Protocol Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <CyberBorder glowColor="purple" className="bg-black/50 p-6 h-full">
            <h3 className="font-mono text-lg tracking-wider mb-4 text-purple-400">PROTOCOLES RÉSEAU</h3>
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
                    innerRadius={40}
                    label={({ protocol, percent }) => `${protocol} ${(percent * 100).toFixed(0)}%`}
                    labelLine={{ stroke: '#BD00FF' }}
                  >
                    {protocolDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #BD00FF' }} />
                </RechartsPie>
              </ResponsiveContainer>
            </div>
          </CyberBorder>
        </motion.div>

        {/* Top Features */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <CyberBorder glowColor="green" className="bg-black/50 p-6 h-full">
            <h3 className="font-mono text-lg tracking-wider mb-1 text-green-400">TOP FEATURES</h3>
            <p className="text-xs text-gray-500 mb-4">Échelle logarithmique</p>
            <ClickableChart title="Top Features (par variance)" chartContent={featuresChartContent}>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={topFeatures}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,255,65,0.1)" />
                    <XAxis dataKey="feature" stroke="#505050" tick={{ fill: '#00FF41', fontSize: 8, angle: -45, textAnchor: 'end' }} />
                    <YAxis stroke="#505050" scale="log" domain={['auto', 'auto']} tick={{ fill: '#00FF41', fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #00FF41' }} />
                    <Bar dataKey="variance" fill="#00FF41" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ClickableChart>
          </CyberBorder>
        </motion.div>
      </div>
    </motion.div>
  );
};

// Model Page with animations
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

  const rocData = rfMetrics.roc_data ? rfMetrics.roc_data.fpr.map((fpr, i) => ({
    fpr, tpr: rfMetrics.roc_data.tpr[i]
  })) : [];

  const featureImportance = rfMetrics.feature_importance ? 
    Object.entries(rfMetrics.feature_importance).map(([feature, importance]) => ({ feature, importance })).slice(0, 10) : [];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      {/* Header */}
      <motion.div 
        className="flex items-center justify-between flex-wrap gap-4"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-4">
          <CyberBorder glowColor="purple" className="p-3 bg-purple-500/10">
            <Brain className="w-6 h-6 text-purple-400" />
          </CyberBorder>
          <div>
            <h1 className="text-2xl font-mono font-bold tracking-wider">
              MODÈLE <GlitchText className="text-purple-400">ML</GlitchText>
            </h1>
            <p className="text-gray-500 text-sm font-mono">Random Forest & Decision Tree</p>
          </div>
        </div>
        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
          <Button onClick={handleTrain} disabled={training} className="cyber-btn">
            <Cpu className="w-4 h-4 mr-2" />
            {training ? "ENTRAÎNEMENT..." : "ENTRAÎNER"}
          </Button>
        </motion.div>
      </motion.div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon={CheckCircle} label="ACCURACY (RF)" value={`${((rfMetrics.accuracy || 0) * 100).toFixed(2)}%`} color="primary" delay={0.1} />
        <StatCard icon={Target} label="PRECISION (RF)" value={`${((rfMetrics.precision || 0) * 100).toFixed(2)}%`} color="secondary" delay={0.2} />
        <StatCard icon={Eye} label="RECALL (RF)" value={`${((rfMetrics.recall || 0) * 100).toFixed(2)}%`} color="warning" delay={0.3} />
        <StatCard icon={Activity} label="AUC (RF)" value={(rfMetrics.roc_data?.auc || 0).toFixed(4)} color="success" delay={0.4} />
      </div>

      {/* 3D Threat Visualization */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <CyberBorder glowColor="red" className="bg-black/50 p-6">
          <h3 className="font-mono text-lg tracking-wider mb-1 text-red-400">VISUALISATION 3D DES MENACES</h3>
          <p className="text-xs text-gray-500 mb-2">Bouclier IDS • Trafic normal (vert) vs Attaques (rouge)</p>
          <div className="h-[400px]">
            <Suspense fallback={<div className="flex items-center justify-center h-full"><p className="text-red-500/50 font-mono text-sm animate-pulse">CHARGEMENT 3D...</p></div>}>
              <ThreatVisualization3D />
            </Suspense>
          </div>
          <div className="flex items-center gap-6 mt-3 text-xs font-mono">
            <div className="flex items-center gap-2"><div className="w-3 h-3 bg-cyan-400 rounded-full" /><span className="text-gray-400">Bouclier IDS</span></div>
            <div className="flex items-center gap-2"><div className="w-3 h-3 bg-green-400 rounded-full" /><span className="text-gray-400">Trafic Normal</span></div>
            <div className="flex items-center gap-2"><div className="w-3 h-3 bg-red-400 rounded-full" /><span className="text-gray-400">Attaques Détectées</span></div>
          </div>
        </CyberBorder>
      </motion.div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Comparison Chart */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <CyberBorder glowColor="cyan" className="bg-black/50 p-6">
            <h3 className="font-mono text-lg tracking-wider mb-4 text-cyan-400">COMPARAISON DES MODÈLES</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,240,255,0.1)" />
                  <XAxis dataKey="name" stroke="#505050" tick={{ fill: '#fff', fontSize: 10 }} />
                  <YAxis stroke="#505050" tick={{ fill: '#505050', fontSize: 10 }} domain={[0, 100]} />
                  <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #00F0FF' }} />
                  <Legend />
                  <Bar dataKey="RF" name="Random Forest" fill={COLORS.primary} />
                  <Bar dataKey="DT" name="Decision Tree" fill={COLORS.secondary} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CyberBorder>
        </motion.div>

        {/* ROC Curve */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
        >
          <CyberBorder glowColor="green" className="bg-black/50 p-6">
            <h3 className="font-mono text-lg tracking-wider mb-1 text-green-400">COURBE ROC</h3>
            <p className="text-xs text-gray-500 mb-4">AUC = {(rfMetrics.roc_data?.auc || 0).toFixed(4)}</p>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={rocData}>
                  <defs>
                    <linearGradient id="rocGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={COLORS.success} stopOpacity={0.4}/>
                      <stop offset="95%" stopColor={COLORS.success} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,255,65,0.1)" />
                  <XAxis dataKey="fpr" stroke="#505050" tick={{ fill: '#00FF41', fontSize: 10 }} />
                  <YAxis stroke="#505050" tick={{ fill: '#00FF41', fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #00FF41' }} />
                  <Area type="monotone" dataKey="tpr" stroke={COLORS.success} strokeWidth={2} fill="url(#rocGradient)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CyberBorder>
        </motion.div>

        {/* Feature Importance */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="lg:col-span-2"
        >
          <CyberBorder glowColor="yellow" className="bg-black/50 p-6">
            <h3 className="font-mono text-lg tracking-wider mb-4 text-yellow-400">IMPORTANCE DES FEATURES</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={featureImportance} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(250,255,0,0.1)" />
                  <XAxis type="number" stroke="#505050" tick={{ fill: '#FAFF00', fontSize: 10 }} />
                  <YAxis type="category" dataKey="feature" stroke="#505050" tick={{ fill: '#fff', fontSize: 10 }} width={140} />
                  <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #FAFF00' }} />
                  <Bar dataKey="importance" fill={COLORS.warning} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CyberBorder>
        </motion.div>
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
      const response = await axios.post(`${API}/model/predict`, { features: formData });
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
      setFormData(prev => ({ ...prev, duration: 0, src_bytes: 200, dst_bytes: 1000, count: 2, srv_count: 2, same_srv_rate: 1, logged_in: 1, serror_rate: 0, srv_serror_rate: 0 }));
    } else {
      setFormData(prev => ({ ...prev, duration: 0, src_bytes: 0, dst_bytes: 0, count: 500, srv_count: 500, same_srv_rate: 1, serror_rate: 1, srv_serror_rate: 1 }));
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      {/* Header */}
      <motion.div 
        className="flex items-center gap-4"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <CyberBorder glowColor="green" className="p-3 bg-green-500/10">
          <Target className="w-6 h-6 text-green-400" />
        </CyberBorder>
        <div>
          <h1 className="text-2xl font-mono font-bold tracking-wider">
            PRÉDICTION <GlitchText className="text-green-400">LIVE</GlitchText>
          </h1>
          <p className="text-gray-500 text-sm font-mono">Classification du trafic réseau</p>
        </div>
      </motion.div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Form */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2"
        >
          <CyberBorder glowColor="cyan" className="bg-black/50 p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="font-mono text-lg tracking-wider text-cyan-400">PARAMÈTRES</h3>
              <div className="flex gap-2">
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Button variant="outline" size="sm" onClick={() => loadDemo('normal')} className="text-xs border-green-500/50 text-green-400 hover:bg-green-500/10">
                    DÉMO NORMAL
                  </Button>
                </motion.div>
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Button variant="outline" size="sm" onClick={() => loadDemo('attack')} className="text-xs border-red-500/50 text-red-400 hover:bg-red-500/10">
                    DÉMO ATTAQUE
                  </Button>
                </motion.div>
              </div>
            </div>
            
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 'serror_rate', 'same_srv_rate', 'dst_host_count'].map((field, i) => (
                  <motion.div 
                    key={field}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 + i * 0.05 }}
                  >
                    <Label className="text-[10px] text-gray-500 tracking-widest uppercase">{field}</Label>
                    <Input
                      type="number"
                      step="any"
                      value={formData[field]}
                      onChange={(e) => setFormData(prev => ({ ...prev, [field]: parseFloat(e.target.value) || 0 }))}
                      className="terminal-input mt-1"
                    />
                  </motion.div>
                ))}
              </div>
              
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <Label className="text-[10px] text-gray-500 tracking-widest uppercase">Protocol</Label>
                  <Select value={formData.protocol_type} onValueChange={(v) => setFormData(prev => ({ ...prev, protocol_type: v }))}>
                    <SelectTrigger className="terminal-input mt-1"><SelectValue /></SelectTrigger>
                    <SelectContent className="bg-black border-cyan-500/30">
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
                    <SelectContent className="bg-black border-cyan-500/30">
                      {['http', 'ftp', 'smtp', 'ssh', 'dns', 'telnet', 'other'].map(s => <SelectItem key={s} value={s}>{s.toUpperCase()}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-[10px] text-gray-500 tracking-widest uppercase">Flag</Label>
                  <Select value={formData.flag} onValueChange={(v) => setFormData(prev => ({ ...prev, flag: v }))}>
                    <SelectTrigger className="terminal-input mt-1"><SelectValue /></SelectTrigger>
                    <SelectContent className="bg-black border-cyan-500/30">
                      {['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'RSTOS0', 'S3', 'S2', 'OTH'].map(f => <SelectItem key={f} value={f}>{f}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <motion.div whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}>
                <Button type="submit" disabled={loading} className="cyber-btn w-full h-12">
                  <Lock className="w-4 h-4 mr-2" />
                  {loading ? "ANALYSE EN COURS..." : "ANALYSER LE TRAFIC"}
                </Button>
              </motion.div>
            </form>
          </CyberBorder>
        </motion.div>

        {/* Result */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <CyberBorder 
            glowColor={result ? (result.prediction === 'Attack' ? 'red' : 'green') : 'cyan'} 
            className="bg-black/50 p-6 h-full"
          >
            <h3 className="font-mono text-lg tracking-wider mb-6 text-center">RÉSULTAT</h3>
            <div className="flex flex-col items-center justify-center min-h-[200px]">
              <AnimatePresence mode="wait">
                {result ? (
                  <motion.div 
                    key="result"
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0, opacity: 0 }}
                    className="text-center"
                  >
                    <motion.div
                      animate={{ 
                        scale: [1, 1.1, 1],
                        rotate: result.prediction === 'Attack' ? [0, -5, 5, 0] : 0
                      }}
                      transition={{ duration: 0.5 }}
                    >
                      {result.prediction === 'Attack' ? (
                        <AlertTriangle className="w-20 h-20 text-red-500 mx-auto mb-4" />
                      ) : (
                        <CheckCircle className="w-20 h-20 text-green-500 mx-auto mb-4" />
                      )}
                    </motion.div>
                    <p className={`text-3xl font-mono font-bold mb-2 ${result.prediction === 'Attack' ? 'neon-red' : 'neon-green'}`}>
                      {result.prediction === 'Attack' ? 'INTRUSION' : 'NORMAL'}
                    </p>
                    <p className="text-gray-500 text-sm font-mono">Confiance: {((result.prediction === 'Attack' ? result.probability?.attack : result.probability?.normal) * 100).toFixed(1)}%</p>
                    {result.attack_category && result.attack_category !== 'Normal' && (
                      <Badge className="mt-3 bg-red-500/20 text-red-400 border-red-500/30">{result.attack_category}</Badge>
                    )}
                  </motion.div>
                ) : (
                  <motion.div 
                    key="waiting"
                    className="text-center"
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    <Wifi className="w-16 h-16 mx-auto mb-4 text-cyan-500/30" />
                    <p className="text-xs tracking-widest text-gray-600 font-mono">EN ATTENTE D'ANALYSE</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </CyberBorder>
        </motion.div>
      </div>
    </motion.div>
  );
};

// Clustering Page
const ClusteringPage = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [numClusters, setNumClusters] = useState(5);

  // Auto-load clustering results on mount
  useEffect(() => {
    const fetchResults = async () => {
      try {
        const response = await axios.get(`${API}/clustering/results`);
        setResults(response.data);
      } catch (error) {
        console.error("Erreur:", error);
      }
      setLoading(false);
    };
    fetchResults();
  }, []);

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

  const scatterData = results?.pca_data || [];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      {/* Header */}
      <motion.div 
        className="flex items-center justify-between flex-wrap gap-4"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-4">
          <CyberBorder glowColor="yellow" className="p-3 bg-yellow-500/10">
            <PieChart className="w-6 h-6 text-yellow-400" />
          </CyberBorder>
          <div>
            <h1 className="text-2xl font-mono font-bold tracking-wider">
              CLUSTERING <GlitchText className="text-yellow-400">K-MEANS</GlitchText>
            </h1>
            <p className="text-gray-500 text-sm font-mono">Analyse non supervisée</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Label className="text-xs text-gray-500 font-mono">K =</Label>
            <Input 
              type="number" 
              min="2" 
              max="10" 
              value={numClusters} 
              onChange={(e) => setNumClusters(parseInt(e.target.value) || 5)} 
              className="terminal-input w-16" 
            />
          </div>
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button onClick={runClustering} disabled={loading} className="cyber-btn">
              <Server className="w-4 h-4 mr-2" />
              {loading ? "CALCUL..." : "EXÉCUTER"}
            </Button>
          </motion.div>
        </div>
      </motion.div>

      {loading && <LoadingSpinner />}

      {!loading && results && (
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Scatter Plot */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2"
          >
            <CyberBorder glowColor="cyan" className="bg-black/50 p-6">
              <h3 className="font-mono text-lg tracking-wider mb-1 text-cyan-400">VISUALISATION 2D</h3>
              <p className="text-xs text-gray-500 mb-4">Projection PCA</p>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ bottom: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,240,255,0.1)" />
                    <XAxis type="number" dataKey="x" name="PC1" stroke="#505050" tick={{ fill: '#00F0FF', fontSize: 10 }} />
                    <YAxis type="number" dataKey="y" name="PC2" stroke="#505050" tick={{ fill: '#00F0FF', fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #00F0FF' }} cursor={{ strokeDasharray: '3 3' }} />
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
            </CyberBorder>
          </motion.div>

          {/* Metrics */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <CyberBorder glowColor="purple" className="bg-black/50 p-6 h-full">
              <h3 className="font-mono text-lg tracking-wider mb-4 text-purple-400">MÉTRIQUES</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 border border-purple-500/30 bg-purple-500/5">
                  <p className="text-[10px] text-gray-500 tracking-widest mb-1">SILHOUETTE</p>
                  <p className="text-2xl font-mono font-bold text-purple-400">{(results.silhouette_score || 0).toFixed(4)}</p>
                </div>
                <div className="p-4 border border-cyan-500/30 bg-cyan-500/5">
                  <p className="text-[10px] text-gray-500 tracking-widest mb-1">CLUSTERS</p>
                  <p className="text-2xl font-mono font-bold text-cyan-400">{results.n_clusters || 0}</p>
                </div>
              </div>
            </CyberBorder>
          </motion.div>

          {/* Cluster Sizes */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <CyberBorder glowColor="green" className="bg-black/50 p-6 h-full">
              <h3 className="font-mono text-lg tracking-wider mb-4 text-green-400">TAILLE DES CLUSTERS</h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={Object.entries(results.cluster_distribution || {}).map(([k, v]) => ({ cluster: k, size: v }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,255,65,0.1)" />
                    <XAxis dataKey="cluster" stroke="#505050" tick={{ fill: '#00FF41', fontSize: 10 }} />
                    <YAxis stroke="#505050" tick={{ fill: '#00FF41', fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #00FF41' }} />
                    <Bar dataKey="size" fill={COLORS.success}>
                      {Object.keys(results.cluster_distribution || {}).map((_, idx) => (
                        <Cell key={idx} fill={CHART_COLORS[idx % CHART_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CyberBorder>
          </motion.div>
        </div>
      )}

      {!loading && !results && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center justify-center h-64 text-center"
        >
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          >
            <PieChart className="w-16 h-16 text-yellow-500/30" />
          </motion.div>
          <p className="mt-4 text-gray-600 font-mono text-sm tracking-wider">
            Cliquez sur EXÉCUTER pour lancer le clustering
          </p>
        </motion.div>
      )}
    </motion.div>
  );
};

// Main App Component
function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-[#030303] text-white overflow-x-hidden">
        {/* Animated background */}
        <ParticleBackground />
        
        {/* Grid overlay */}
        <div className="fixed inset-0 cyber-grid-bg pointer-events-none z-0" />
        
        {/* Scanlines */}
        <div className="scanlines" />
        
        <Toaster 
          position="top-right" 
          toastOptions={{
            style: { 
              background: '#0a0a0a', 
              border: '1px solid rgba(0,240,255,0.3)', 
              color: '#fff', 
              fontFamily: 'Share Tech Mono',
              borderRadius: 0
            }
          }}
        />
        
        <Sidebar isOpen={sidebarOpen} setIsOpen={setSidebarOpen} />
        
        {/* Mobile header */}
        <motion.div 
          className="lg:hidden fixed top-0 left-0 right-0 z-30 bg-black/95 backdrop-blur-sm border-b border-cyan-500/20 px-4 py-3"
          initial={{ y: -100 }}
          animate={{ y: 0 }}
        >
          <div className="flex items-center justify-between">
            <motion.div whileTap={{ scale: 0.9 }}>
              <Button variant="ghost" size="icon" onClick={() => setSidebarOpen(true)} className="hover:bg-cyan-500/10">
                <Menu className="w-5 h-5 text-cyan-400" />
              </Button>
            </motion.div>
            <span className="font-mono text-sm tracking-wider">
              <span className="text-white">CYBER</span>
              <span className="text-cyan-400">SENTINELLE</span>
            </span>
            <div className="w-10 flex justify-end">
              <motion.div 
                className="w-2 h-2 bg-green-500 rounded-full"
                animate={{ opacity: [1, 0.5, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
              />
            </div>
          </div>
        </motion.div>
        
        {/* Main content */}
        <main className="lg:ml-[280px] min-h-screen pt-16 lg:pt-0 relative z-10">
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
