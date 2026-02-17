# CyberSentinelle - Network Intrusion Detection System
## Product Requirements Document (PRD)

### Project Overview
- **Project Name**: CyberSentinelle
- **Author**: Zakarya Oukil
- **Formation**: Master 1 Cybersécurité - HIS 2025/2026
- **Technology Stack**: React (Frontend), FastAPI (Backend), MongoDB, Scikit-learn (ML)
- **Dataset**: NSL-KDD

---

## MAJOR UPDATE: Cyber-Tactical UI Redesign ✅

### 2026-02-17 - Complete UI Overhaul
The website has been completely redesigned with a professional "Cyber-Tactical" aesthetic:

**Design System:**
- **Colors**: Deep black (#050505) with neon cyan (#00F0FF), alert red (#FF003C), success green (#00FF41), warning yellow (#FAFF00)
- **Typography**: 
  - Rajdhani (body text)
  - Share Tech Mono (headings, data)
  - JetBrains Mono (code)
- **Effects**: Scanlines overlay, grid background, glassmorphism, neon glows

**New Features:**
- Animated 3D globe on homepage (canvas-based particles and rings)
- Cyber-styled sidebar with navigation indicators
- Motion animations (Framer Motion) throughout
- Sharp-edged cards with hover glow effects
- Terminal-style inputs with cyan glow
- Neon text effects for status indicators

**Files Created/Modified:**
- `/app/frontend/src/index.css` - Complete CSS redesign
- `/app/frontend/src/App.js` - Refactored with new components
- `/app/frontend/src/pages/Home.js` - New homepage with 3D globe
- `/app/frontend/src/pages/Monitor.js` - Redesigned monitoring page
- `/app/frontend/src/components/CyberGlobe.js` - Canvas-based animated globe
- `/app/design_guidelines.json` - Design system documentation

**Packages Added:**
- framer-motion (animations)

---

## Current Features ✅

1. **Homepage** - 3D animated globe, stats, feature cards, terminal section
2. **Live Monitor** - Real-time traffic monitoring, DoS simulation, attack detection
3. **Dashboard EDA** - Attack distribution, protocol analysis, top features
4. **Model ML** - RF/DT metrics, comparison charts, ROC curve, feature importance
5. **Prediction** - Live traffic classification with demo modes
6. **Clustering** - K-Means visualization with scatter plot

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
│   └── server.py         # FastAPI, ML, monitoring endpoints
├── frontend/
│   └── src/
│       ├── App.js        # Main app with routing
│       ├── pages/
│       │   ├── Home.js   # Homepage with 3D globe
│       │   └── Monitor.js # Live monitoring page
│       ├── components/
│       │   ├── CyberGlobe.js # Animated globe
│       │   └── ui/       # Shadcn components
│       ├── index.css     # Cyber-tactical styles
│       └── App.css       # Additional styles
├── design_guidelines.json # Design system
└── memory/PRD.md
```

---

## Notes
- User language: French
- Design inspired by cybersecurity/hacking aesthetics
- No AI-generated look (sharp edges, unique colors, custom animations)
- All pages have consistent cyber-tactical theme
