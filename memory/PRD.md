# CyberSentinelle - Network Intrusion Detection System
## Product Requirements Document (PRD)

### Project Overview
- **Project Name**: CyberSentinelle
- **Author**: Zakarya Oukil
- **Formation**: Master 1 Cybersecurite - HIS 2025/2026
- **Technology Stack**: React (Frontend), FastAPI (Backend), MongoDB, Scikit-learn (ML)
- **Dataset**: NSL-KDD

---

## Current Features (All Working)

1. **Homepage** - 3D animated globe, stats, feature cards, terminal section
2. **Live Monitor** - Real-time traffic monitoring, DoS simulation, attack detection
3. **Dashboard EDA** - Attack distribution, protocol analysis, top features, **3D Network Topology**
4. **Model ML** - RF/DT metrics, comparison charts, ROC curve, feature importance, **3D Threat Visualization**
5. **Prediction** - Live traffic classification with demo modes (Normal/Attack)
6. **Clustering** - K-Means visualization with PCA scatter plot, auto-loads on page visit

---

## Bug Fixes (2026-02-17)

### Fixed: Empty Charts on Dashboard EDA
- **Root Cause**: Frontend used wrong property names (`attack_distribution` instead of `feature_distributions.label`, etc.)
- **Fix**: Transformed API response data into chart-ready format

### Fixed: Empty Charts on Model ML
- **Root Cause**: Frontend used `roc_curve` instead of `roc_data`, `auc` at top-level instead of `roc_data.auc`
- **Fix**: Updated all property references

### Fixed: Prediction "Erreur de prediction"
- **Root Cause**: Frontend sent `formData` directly instead of `{features: formData}`. Also used `result.confidence` / `result.attack_type` instead of `result.probability` / `result.attack_category`
- **Fix**: Wrapped data correctly and fixed response field mapping

### Fixed: Empty Clustering Charts
- **Root Cause**: Frontend used `visualization_data` / `cluster_sizes` / `inertia` instead of `pca_data` / `cluster_distribution` / `n_clusters`
- **Fix**: Updated all property references and added auto-loading on page mount

---

## 3D Enhancements (2026-02-17)

### 3D Network Topology (Dashboard EDA)
- Interactive canvas-based 3D visualization
- Shows Server, Firewalls/IDS, Clients, and Attackers as distinct shapes
- Animated data packets flowing along connections
- Grid floor and background particles
- Auto-rotating perspective with depth-based rendering

### 3D Threat Visualization (Model ML)
- Icosahedron wireframe shield representing IDS
- Green normal traffic ring orbiting the shield
- Red attack particles surrounding in outer orbits
- Three colored orbiting rings (threat levels)
- Pulsing core with scan line effect

---

## Model Metrics
- **Accuracy**: 97.70%
- **Precision**: 97.70%
- **Recall**: 97.70%
- **AUC**: 0.9971

---

## Remaining Tasks

### P1 - High Priority
1. **Jupyter Notebook Refinement** - Make it look more "student-made" for presentation

### P2 - Medium Priority
2. **Final QA** - Review for typos and UI consistency

---

## Architecture

```
/app
├── backend/
│   ├── server.py
│   ├── requirements.txt
│   ├── data/
│   ├── models/
│   └── tests/
├── frontend/
│   └── src/
│       ├── App.js          # Main app with all page components
│       ├── pages/
│       │   ├── Home.js     # Homepage with 3D globe
│       │   └── Monitor.js  # Live monitoring page
│       ├── components/
│       │   ├── CyberGlobe.js          # Canvas globe
│       │   ├── NetworkTopology3D.js    # Canvas 3D network
│       │   ├── ThreatVisualization3D.js # Canvas 3D threats
│       │   └── ui/                     # Shadcn components
│       ├── index.css       # Cyber-tactical styles
│       └── App.css
├── design_guidelines.json
└── memory/PRD.md
```

---

## Notes
- User language: French
- 3D visualizations use HTML5 Canvas (not React Three Fiber) for React 19 compatibility
- All pages have consistent cyber-tactical theme
- No hardcoded data - all metrics come from ML models
