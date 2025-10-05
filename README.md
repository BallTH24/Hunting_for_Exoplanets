# 🪐 Hunting for Exoplanets with AI  
*A NASA Space Apps Challenge 2025 Project*

![NASA Space Apps](https://www.spaceappschallenge.org/_next/static/media/spaceapps-logo.0c88c7a5.svg)

## 🌌 Overview
**Hunting for Exoplanets with AI** is a machine learning project that automatically analyzes NASA’s open-source **Kepler, K2, and TESS** datasets to identify potential exoplanets.  
By analyzing stellar brightness variations (light curves), the model detects periodic dimming patterns that indicate exoplanet transits.

Developed for the [NASA International Space Apps Challenge 2025](https://www.spaceappschallenge.org/):  
👉 *A World Away: Hunting for Exoplanets with AI*

---

## 🚀 Features
- 🌠 Retrieve exoplanet metadata from **NASA Exoplanet Archive**
- 📡 Download & preprocess light curves from **Kepler/K2/TESS missions**
- 🧠 Extract advanced light curve features (skewness, kurtosis, percentiles, flux ratio, autocorrelation, etc.)
- 🤖 Classify exoplanet signals using multiple ML models (RandomForest, Logistic Regression, XGBoost)
- 📊 Visualize transit events and feature importance
- 🪄 Interactive **Streamlit Web Interface** for users to upload & analyze data
- 🐳 Containerized deployment with **Docker** and **Railway**

---

## 🔭 Exoplanet Light Curve Analysis
Light curve analysis is the scientific foundation of this project.  
When a planet passes in front of its host star (transit), the observed brightness dips periodically. Detecting these dips requires filtering, normalization, and precise feature extraction.

### 🧩 Analysis Steps
1. **Light Curve Acquisition** — Download raw light curves from NASA missions (Kepler, K2, TESS)  
2. **Preprocessing** — Remove noise, normalize flux, and fold data by orbital period  
3. **Feature Extraction** — Compute 15+ statistical and transit-based metrics:  
   - Mean, Standard Deviation, Median  
   - Percentiles (P10, P25, P75, P90)  
   - Skewness, Kurtosis  
   - Flux Amplitude, Flux Ratio  
   - Depth Proxy (transit depth approximation)  
   - Duration Proxy (number of consecutive low flux points)  
   - Autocorrelation lag-1 (signal periodicity)
4. **Labeling** — Combine known exoplanet signals and artificial noise for supervised training  

### 🪐 Visualization Example
![Light Curve Example](https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Kepler_Lightcurve.png/800px-Kepler_Lightcurve.png)
*Kepler light curve showing periodic dimming from an exoplanet transit.*

---

## 🧠 Model Pipeline
1. **Data Acquisition** → Fetch planet metadata & light curves from NASA archives  
2. **Feature Engineering** → Extract advanced numerical descriptors  
3. **Training** → Fit RandomForest / XGBoost / Logistic Regression  
4. **Evaluation** → Classification report + ROC AUC + confusion matrix  
5. **Web Interface** → Streamlit-based analysis tool  
6. **Deployment** → Docker container (Railway / local / ngrok)

---

## 🌍 Tech Stack
| Category | Tools |
|-----------|-------|
| **Languages** | Python 3.13.7 |
| **Data Sources** | NASA Exoplanet Archive (NExScI), MAST (Kepler, K2, TESS) |
| **Libraries** | Lightkurve, Astroquery, Scikit-learn, XGBoost, Pandas, NumPy, SciPy |
| **Web App** | Streamlit |
| **Deployment** | Docker, Railway |
| **Visualization** | Matplotlib, Plotly |

---