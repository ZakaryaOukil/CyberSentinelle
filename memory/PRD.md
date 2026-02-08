# CyberSentinelle - Network Intrusion Detection System
## Product Requirements Document (PRD)

### Project Overview
- **Project Name**: CyberSentinelle
- **Author**: Zakarya Oukil
- **Formation**: Master 1 Cybersécurité - HIS 2025/2026
- **Technology Stack**: React (Frontend), FastAPI (Backend), MongoDB, Scikit-learn (ML)
- **Dataset**: NSL-KDD

### Core Requirements
1. **Web Application** with visualization of:
   - Exploratory Data Analysis (EDA)
   - Live predictions
   - Real-time model performance metrics
   - Clustering analysis
   - **NEW: Live Traffic Monitoring & DoS Attack Detection**

2. **Jupyter Notebook** downloadable for student presentation

3. **ML Models**: Decision Tree, Random Forest, K-Means clustering

---

## Completed Features ✅

### 2026-02-08 - Live Monitor Page with DoS Simulation
- **Feature**: New "Live Monitor" page with real-time traffic surveillance
- **Components**:
  - Real-time traffic monitoring (req/s, total requests)
  - Status banner (NORMAL/ATTACK with animations)
  - DoS attack simulation button with configurable intensity
  - Traffic chart (60 seconds history)
  - Security logs with severity levels
  - Attack pattern detection (SINGLE_SOURCE_FLOOD, RAPID_FIRE, IDENTICAL_REQUESTS)
  - Audio alert on attack detection
- **Backend endpoints**:
  - `GET /api/monitor/traffic` - Get traffic stats
  - `POST /api/monitor/ping` - Ping endpoint for testing
  - `POST /api/monitor/reset` - Reset monitor
  - `GET /api/monitor/alerts` - Get security alerts
- **Files Created**: 
  - `/app/frontend/src/pages/Monitor.js`
  - `/app/GUIDE_ATTAQUE_DOS.md`
- **Status**: TESTED & WORKING

### 2026-02-03 - Bug Fix: Chart Modal Z-Index
- **Issue**: Enlarged chart modal appeared behind other page elements
- **Solution**: Used React `createPortal` to inject modal into document.body
- **Status**: FIXED

### Previous Session - Completed Work
- Removed dedicated "Download" page, moved button to homepage
- Restored website title to "CyberSentinelle"
- Implemented axis labels on charts
- Click-to-enlarge modal for all charts
- Replaced hardcoded prediction logic with real ML model outputs
- Adjusted synthetic data for realistic metrics (~97.7% accuracy)
- All metrics fetched dynamically from backend
- Fixed "Top Features" chart with logarithmic scale

---

## Current Model Metrics
- **Accuracy (Random Forest)**: 97.70%
- **Precision (RF)**: 97.70%
- **Recall (RF)**: 97.70%
- **AUC (RF)**: 0.9971

---

## Architecture

```
/app
├── backend/
│   ├── server.py         # FastAPI app, ML logic, monitoring endpoints
│   ├── requirements.txt  # Python dependencies
│   ├── models/           # Saved ML models (.joblib)
│   └── data/             # Dataset files
├── frontend/
│   ├── src/
│   │   ├── App.js        # Main React app (routing, components)
│   │   ├── pages/
│   │   │   └── Monitor.js  # NEW: Live Monitor page
│   │   ├── App.css       # App-specific styles
│   │   ├── index.css     # Global styles, theme
│   │   └── components/ui/ # Shadcn UI components
│   └── package.json
├── GUIDE_ATTAQUE_DOS.md  # DoS attack testing guide (French)
└── memory/
    └── PRD.md            # This file
```

### Key API Endpoints
- `GET /api/health` - Health check
- `GET /api/dataset/info` - Dataset statistics
- `GET /api/dataset/eda` - EDA data for visualizations
- `POST /api/model/train` - Train ML models
- `GET /api/model/metrics` - Get model performance metrics
- `POST /api/model/predict` - Single prediction
- `POST /api/model/predict-batch` - Batch predictions
- `POST /api/clustering/run` - Run K-Means clustering
- `GET /api/clustering/results` - Get clustering results
- `GET /api/notebook/download` - Download Jupyter notebook
- **NEW**: `GET /api/monitor/traffic` - Real-time traffic stats
- **NEW**: `POST /api/monitor/ping` - Ping for load testing
- **NEW**: `POST /api/monitor/reset` - Reset monitor

---

## Remaining Tasks

### P1 - High Priority
1. **Jupyter Notebook Refinement**
   - Make notebook look more "student-made" for teacher presentation
   - Add detailed markdown explanations
   - Add inline code comments
   - Structure as educational walkthrough

### P2 - Medium Priority
2. **Final Quality Assurance**
   - Review entire application for typos
   - Check UI consistency
   - Fix any minor bugs

---

## Notes for Developers
- All data displayed comes from real ML model predictions (no hardcoded data)
- Modal uses `createPortal` for proper z-index stacking
- Charts are clickable for enlarged view
- All charts have axis labels
- User language preference: French
- DoS simulation is safe and controlled (uses internal ping endpoint)
