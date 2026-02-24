# CyberSentinelle - Network Intrusion Detection System
## Product Requirements Document (PRD)

### Project Overview
- **Author**: Zakarya Oukil | Master 1 Cybersecurite - HIS 2025/2026
- **Stack**: React 19 + FastAPI + MongoDB + Scikit-learn
- **Dataset**: NSL-KDD
- **Primary Model**: Decision Tree (Arbre de Decision)

---

## Features (All Working)
1. Homepage - Logo, shimmer title, typewriter, globe 3D, HexGrid, animated counters, feature cards, terminal
2. Dashboard EDA - 3D Network Topology (17 nodes), charts (attacks, protocols, features)
3. Model ML - **Decision Tree** as primary model, 3D Threat Visualization, comparison with RF, ROC, feature importance
4. Prediction - Radar Scanner, Demo Normal/Attack, live classification via Decision Tree
5. Clustering - K-Means PCA scatter, auto-loads
6. Live Monitor - Real-time traffic, DoS simulation

## Changes (2026-02-24)
- Primary model switched from Random Forest to **Decision Tree** (backend + all frontend labels)
- User logo added to sidebar and homepage hero
- Decision Tree metrics: Accuracy 94.30%, Precision 94.36%, AUC 0.9463

## Remaining: Jupyter Notebook refinement (P1), Final QA (P2)
