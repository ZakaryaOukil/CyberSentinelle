# CyberSentinelle - PRD

## Stack: React 19 + FastAPI + MongoDB + Scikit-learn | Primary Model: Decision Tree | Dataset: NSL-KDD

## All Features (Working)
1. Homepage, Dashboard EDA (3D Network), Model ML (3D Threats), Prediction (Radar), Clustering, Live Monitor
2. Presentation page (`/presentation`) - 14 slides, keyboard navigation, fullscreen, light/dark mode, large text for datashow
3. PDF download (`/api/presentation/download`) - 15 slides, downloadable PDF with matching theme
4. **Light/Dark mode toggle** - In sidebar for main app, in bottom bar for presentation page
5. **Sidebar presentation link** - Direct navigation to `/presentation` from sidebar

## Changes (2026-02-24)
- Switched model: Random Forest -> Decision Tree (all references updated)
- Added user logo (sidebar only, removed from hero)
- Created web presentation (14 slides, arrow keys, fullscreen with F)
- Created PDF presentation endpoint (15 slides, cyber dark theme)
- Both presentations in English, include live demo plan (~10 min)
- **NEW**: Light/dark mode toggle in sidebar (ThemeContext shared across app)
- **NEW**: Presentation page light/dark mode with large text for datashow projector
- **NEW**: Sidebar link to presentation page
- **NEW**: Theme-aware components: CyberBorder, StatCard, GlitchText, ChartModal
- **NEW**: Theme-aware pages: Home, Dashboard EDA, Model ML, Prediction, Clustering

## Architecture
- ThemeContext.js - Shared theme context (createContext/useContext)
- App.js - ThemeContext.Provider wraps all routes, sidebar passes isLight/setIsLight
- Pages use `useTheme()` hook for conditional styling

## Remaining Tasks
- Jupyter Notebook refinement for student audience (P1)
- Final Quality Assurance sweep (P2)
- Potential Next.js migration (P2, future)
