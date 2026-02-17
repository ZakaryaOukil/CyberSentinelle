# CyberSentinelle - Network Intrusion Detection System
## Product Requirements Document (PRD)

### Project Overview
- **Project Name**: CyberSentinelle
- **Author**: Zakarya Oukil
- **Formation**: Master 1 Cybersecurite - HIS 2025/2026
- **Technology Stack**: React 19 (Frontend), FastAPI (Backend), MongoDB, Scikit-learn (ML)
- **Dataset**: NSL-KDD

---

## Current Features (All Working - Tested 100%)

### Pages
1. **Homepage** - Shimmer title, typewriter animation, CyberGlobe 3D, HexGrid overlay, animated counters, 6 holographic feature cards, terminal section
2. **Live Monitor** - Real-time traffic monitoring, DoS simulation, attack detection
3. **Dashboard EDA** - 3D Network Topology (17 nodes, hex grid floor, HUD), attack distribution, protocol analysis, top features
4. **Model ML** - 3D Threat Visualization (icosahedron shield, orbiting rings, data streams), RF/DT metrics, comparison, ROC, feature importance
5. **Prediction** - Radar Scanner (sweep beam, blips), Demo Normal/Attack modes, live classification
6. **Clustering** - K-Means with PCA scatter plot, auto-loads

### 3D Visualizations (Canvas-based)
- **3D Network Topology**: 17 nodes (server hexagons, firewall diamonds, client circles, attacker triangles), animated data packets, hex grid floor, HUD overlay
- **3D Threat Visualization**: Icosahedron wireframe shield, 4 orbiting rings, 200 attack particles, 120 normal traffic particles, 30 data streams, HUD corners
- **Radar Scanner**: Animated sweep beam, 15 blips (green/red), concentric rings, compass labels, color changes based on result
- **CyberGlobe**: Animated 3D globe with orbital connections
- **Matrix Rain**: Full-screen background code rain (green characters)
- **HexGrid**: Pulsing hexagonal grid overlay

### Visual Effects
- Shimmer text animation, typewriter effect, animated counters
- Holographic card hover effects, glow dividers
- Scanlines overlay, cyber grid background
- Advanced CSS animations (15+ keyframe animations)

---

## Model Metrics
- Accuracy: 97.70%, Precision: 97.70%, Recall: 97.70%, AUC: 0.9971

---

## Bug Fixes (2026-02-17)
- Fixed empty charts (wrong property names in frontend)
- Fixed prediction errors (wrong request format)
- Fixed clustering display (wrong data mapping)

---

## Remaining Tasks

### P1
1. Jupyter Notebook refinement for student presentation

### P2
2. Final QA sweep

---

## Architecture
```
/app
├── backend/server.py
├── frontend/src/
│   ├── App.js (main + all page components)
│   ├── pages/ (Home.js, Monitor.js)
│   ├── components/
│   │   ├── CyberGlobe.js, MatrixRain.js, HexGrid.js
│   │   ├── NetworkTopology3D.js, ThreatVisualization3D.js
│   │   ├── RadarScanner.js, ChartModal.js
│   │   └── ui/ (shadcn components)
│   └── index.css (cyber-tactical theme)
└── memory/PRD.md
```
