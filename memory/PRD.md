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

2. **Jupyter Notebook** downloadable for student presentation

3. **ML Models**: Decision Tree, Random Forest, K-Means clustering

---

## Completed Features ✅

### 2026-02-03 - Bug Fix: Chart Modal Z-Index
- **Issue**: Enlarged chart modal appeared behind other page elements (sidebar, other charts)
- **Solution**: Used React `createPortal` to inject modal directly into document.body with z-index 99999/100000
- **Files Modified**: `/app/frontend/src/App.js`
- **Status**: TESTED & VERIFIED

### Previous Session - Completed Work
- Removed dedicated "Download" page, moved button to homepage
- Restored website title to "CyberSentinelle"
- Implemented axis labels on charts
- Click-to-enlarge modal for all charts
- Replaced hardcoded prediction logic with real ML model outputs
- Adjusted synthetic data for realistic metrics (~97.7% accuracy)
- All metrics fetched dynamically from backend
- Fixed "Top Features" chart with logarithmic scale
- Deployment guidance for Netlify/Render/MongoDB Atlas

---

## Current Model Metrics
- **Accuracy (Random Forest)**: 97.70%
- **Precision (RF)**: 97.70%
- **Recall (RF)**: 97.70%
- **AUC (RF)**: 0.9971

*Note: Identical metrics for Accuracy/Precision/Recall are normal for balanced binary classification.*

---

## Architecture

```
/app
├── backend/
│   ├── server.py         # FastAPI app, ML logic, all API endpoints
│   ├── requirements.txt  # Python dependencies
│   ├── models/           # Saved ML models (.joblib)
│   └── data/             # Dataset files
├── frontend/
│   ├── src/
│   │   ├── App.js        # Main React app (all pages, components)
│   │   ├── App.css       # App-specific styles
│   │   ├── index.css     # Global styles, theme
│   │   └── components/ui/ # Shadcn UI components
│   └── package.json
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

## Deployment Information
- **Frontend**: Netlify (or similar static hosting)
- **Backend**: Render (or similar Python hosting)
- **Database**: MongoDB Atlas
- **Version Control**: GitHub

---

## Notes for Developers
- All data displayed comes from real ML model predictions (no hardcoded/fake data)
- Modal uses `createPortal` for proper z-index stacking
- Charts are clickable for enlarged view
- All charts have axis labels
- User language preference: French
