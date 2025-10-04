import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

try:
    from lightkurve import search_lightcurvefile
    HAS_LIGHTKURVE = True
except Exception:
    HAS_LIGHTKURVE = False

st.set_page_config(page_title="Exoplanet Light Curve Analysis", layout="wide")

def extract_features_from_flux(flux):
    """Return dict of advanced features given 1D flux array-like."""
    flux = np.array(flux, dtype=float)
    flux = flux[~np.isnan(flux)]
    if flux.size < 5:
        return None
    feats = {}
    feats['min'] = float(np.min(flux))
    feats['max'] = float(np.max(flux))
    feats['mean'] = float(np.mean(flux))
    feats['std'] = float(np.std(flux))
    feats['median'] = float(np.median(flux))
    feats['p10'] = float(np.percentile(flux, 10))
    feats['p25'] = float(np.percentile(flux, 25))
    feats['p75'] = float(np.percentile(flux, 75))
    feats['p90'] = float(np.percentile(flux, 90))
    feats['skew'] = float(skew(flux))
    feats['kurtosis'] = float(kurtosis(flux))
    feats['amplitude'] = float(feats['max'] - feats['min'])
    feats['flux_ratio'] = float(feats['max'] / feats['min']) if feats['min'] != 0 else 0.0
    feats['depth_proxy'] = float(feats['median'] - feats['min'])
    feats['duration_proxy'] = int(np.sum(flux < (feats['mean'] - feats['std'])))
    feats['autocorr_lag1'] = float(np.corrcoef(flux[:-1], flux[1:])[0,1]) if flux.size > 1 else 0.0
    feats['var'] = float(np.var(flux))
    return feats

def guess_flux_column(df):
    """Try to find a column representing flux in common names."""
    candidates = ['flux', 'flux_pdcsap', 'PDCSAP_FLUX', 'pdcsap_flux', 'sap_flux', 'SAP_FLUX', 'PDCSAP']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: numeric column except time-like
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # prefer any numeric column not named time
    for c in numeric_cols:
        if 'time' not in c.lower():
            return c
    return None

ARTIFACT_DIR = "artifacts"
models = {}
scaler = None

def load_artifacts():
    global models, scaler
    models = {}
    scaler = None
    if not os.path.exists(ARTIFACT_DIR):
        st.warning(f"Artifact folder not found: {ARTIFACT_DIR}. Models will be unavailable.")
        return
    # try common names used in notebook
    candidates = {
        "rf": os.path.join(ARTIFACT_DIR, "model_rf.joblib"),
        "logreg": os.path.join(ARTIFACT_DIR, "model_logreg.joblib"),
        "xgb": os.path.join(ARTIFACT_DIR, "model_xgb.joblib"),
    }
    for name, path in candidates.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                st.error(f"Failed to load model {name} from {path}: {e}")
    scaler_path = os.path.join(ARTIFACT_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.error(f"Failed to load scaler from {scaler_path}: {e}")

load_artifacts()

st.title("Exoplanet Light Curve Analysis")
st.write("Explore the light curves of exoplanets and their properties.")
st.write("Upload your own light curve data or fetch from the Kepler/TESS database.")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Model artifacts")
    if models:
        for k in models:
            st.write(f"- Model: `{k}` loaded")
    else:
        st.write("- No models loaded.")
    if scaler is not None:
        st.write("- Scaler loaded")
    st.write("---")
    st.markdown("**Instructions**")
    st.markdown("""
    - CSV should contain a time column (e.g., `time`, `bjd`, `t`) and a flux column (e.g., `flux`, `PDCSAP_FLUX`).  
    - Alternatively, use 'Fetch by name' (requires `lightkurve` and internet).  
    - If no models exist, app will still compute features from uploaded data.
    """)

with col1:
    uploaded = st.file_uploader("Upload light curve CSV", type=["csv"])
    fetch_host = st.text_input("Fetch by target name (optional, e.g., TIC 25155310 or a star name)", value="")
    use_fetch = st.button("Fetch light curve (requires internet & lightkurve)") if HAS_LIGHTKURVE else None

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success("CSV loaded")
    except Exception as e:
        st.error(f"Cannot read CSV: {e}")
elif fetch_host and HAS_LIGHTKURVE and use_fetch:
    try:
        st.info(f"Searching for {fetch_host} ...")
        search = search_lightcurvefile(fetch_host, mission="TESS")
        if len(search) == 0:
            st.warning("No lightcurve found for that target via lightkurve.")
        else:
            lcfile = search.download()
            # try common flux attributes
            if hasattr(lcfile, "PDCSAP_FLUX"):
                lc = lcfile.PDCSAP_FLUX
            elif hasattr(lcfile, "SAP_FLUX"):
                lc = lcfile.SAP_FLUX
            else:
                try:
                    lc = lcfile.get_lightcurve()
                except Exception:
                    lc = None
            if lc is None:
                st.error("Could not extract LC from the downloaded file.")
            else:
                lc = lc.remove_nans().normalize()
                df = pd.DataFrame({'time': lc.time.value, 'flux': lc.flux.value})
                st.success("Light curve fetched via lightkurve")
    except Exception as e:
        st.error(f"lightkurve fetch failed: {e}")
elif not uploaded and not fetch_host:
    # Offer a synthetic example for demo
    if st.checkbox("Use synthetic demo light curve (example)"):
        t = np.linspace(0, 30, 3000)
        flux = np.ones_like(t) + np.random.normal(0, 0.0008, size=t.shape)
        # inject a fake transit at t~10 with depth 0.01
        mask = (t > 9.95) & (t < 10.05)
        flux[mask] -= 0.01
        df = pd.DataFrame({'time': t, 'flux': flux})
        st.success("Using synthetic demo light curve")

if df is not None:
    st.subheader("Preview light curve data")
    st.dataframe(df.head(10))
    flux_col = guess_flux_column(df)
    flux_col_user = st.text_input("Flux column", value=flux_col if flux_col else "")
    time_col_guess = None
    if 'time' in df.columns:
        time_col_guess = 'time'
    else:
        # try to find any time-like column
        for c in df.columns:
            if 'time' in c.lower() or 'bjd' in c.lower() or 't' == c.lower():
                time_col_guess = c
                break
    time_col_user = st.text_input("Time column (optional)", value=time_col_guess if time_col_guess else "")

    # plot raw light curve
    try:
        flux_series = df[flux_col_user].astype(float).values
        time_series = df[time_col_user].astype(float).values if time_col_user and time_col_user in df.columns else np.arange(len(flux_series))
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(time_series, flux_series, lw=0.6)
        ax.set_xlabel("time")
        ax.set_ylabel("flux")
        ax.set_title("Light curve")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not plot light curve preview: {e}")

    # Allow user to select a segment to analyze (start/end idx)
    st.markdown("### Select a segment to analyze")
    n_points = len(df)
    c1, c2 = st.columns(3)
    with c1:
        seg_start = st.number_input("Segment start index", min_value=0, max_value=max(0,n_points-1), value=0, step=1)
    with c2:
        seg_end = st.number_input("Segment end index", min_value=0, max_value=max(0,n_points-1), value=min(199, n_points-1), step=1)
    seg_end = max(seg_end, seg_start+1)

    if st.button("Extract features & Predict"):
        # extract selected segment
        try:
            seg_flux = df[flux_col_user].astype(float).values[seg_start:seg_end+1]
        except Exception as e:
            st.error(f"Cannot extract segment: {e}")
            seg_flux = None

        if seg_flux is not None:
            feats = extract_features_from_flux(seg_flux)
            if feats is None:
                st.error("Segment too short or invalid for feature extraction.")
            else:
                st.subheader("Extracted features")
                feats_df = pd.DataFrame([feats])
                st.table(feats_df.T.rename(columns={0:'value'}))

                # Prepare features for models
                X = feats_df.copy()
                # ensure column order consistent with scaler/model training: try to load column order from scaler if possible
                # We'll use X.columns order; scaler expects same feature order used in training.
                # If scaler exists, transform; else use raw.
                if scaler is not None:
                    try:
                        X_scaled = scaler.transform(X)
                    except Exception as e:
                        st.warning(f"Scaler transform failed: {e}. Using raw features.")
                        X_scaled = None
                else:
                    X_scaled = None

                # Predict with available models
                if not models:
                    st.warning("No models found in artifacts. Only features are shown.")
                else:
                    for name, mdl in models.items():
                        try:
                            if X_scaled is not None and name in ('logreg','xgb'):
                                inp = X_scaled
                            elif X_scaled is not None and name not in ('logreg','xgb'):
                                # random forest was trained on raw features in notebook; try raw X
                                inp = X.values
                            else:
                                inp = X.values
                            pred = mdl.predict(inp)
                            proba = mdl.predict_proba(inp)[:,1] if hasattr(mdl, "predict_proba") else None
                            st.write(f"### Model: {name}")
                            st.write(f"- Prediction (label): {int(pred[0])}")
                            if proba is not None:
                                st.write(f"- Probability (positive): {float(proba[0]):.3f}")
                        except Exception as e:
                            st.error(f"Prediction failed for model {name}: {e}")

                # plot the selected segment and highlight
                fig2, ax2 = plt.subplots(figsize=(10,3))
                seg_time = df[time_col_user].astype(float).values[seg_start:seg_end+1] if time_col_user and time_col_user in df.columns else np.arange(len(seg_flux))
                ax2.plot(seg_time, seg_flux, lw=0.8)
                ax2.set_title("Selected segment")
                ax2.set_xlabel("time")
                ax2.set_ylabel("flux")
                st.pyplot(fig2)

st.markdown("---")
st.markdown("If you want, place `main_ready.ipynb` and artifacts (models + scaler) together with this app. If models are missing you can still use the app to compute features.")
st.markdown("Developed by NOVA-5")
