"""
Phase 3 inference helper.

Loads the trained classifier + regressor saved by train_flood_risk_model.py
and exposes a simple predict_flood_risk() function the Streamlit dashboard
(Phase 5) can call directly — no retraining needed at dashboard runtime.
"""

import os

import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "..", "..", "models")


def _load(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


_clf_bundle = _load("best_classifier.joblib")
_reg_bundle = _load("best_regressor.joblib")


def models_available() -> bool:
    return _clf_bundle is not None and _reg_bundle is not None


def _build_features(precipitation_mm, elevation_m, drainage_dist_km,
                     building_count, latitude, longitude, month,
                     rain_mm=None, rain_3day_sum=None, rain_7day_sum=None,
                     rain_lag1=0.0):
    """
    Builds a single-row feature vector matching FEATURE_COLS used at
    training time. For a one-off dashboard prediction (no history of
    previous days for that point), rolling rainfall features default to
    the current day's rainfall unless supplied.
    """
    rain_mm = precipitation_mm if rain_mm is None else rain_mm
    rain_3day_sum = precipitation_mm if rain_3day_sum is None else rain_3day_sum
    rain_7day_sum = precipitation_mm if rain_7day_sum is None else rain_7day_sum
    rain_change = precipitation_mm - rain_lag1
    is_monsoon = int(month in (6, 7, 8, 9))
    rain_elev_interaction = precipitation_mm / (elevation_m + 1.0)

    row = {
        "precipitation_mm": precipitation_mm,
        "rain_mm": rain_mm,
        "elevation_m": elevation_m,
        "building_count": building_count,
        "drainage_dist_km": drainage_dist_km,
        "latitude": latitude,
        "longitude": longitude,
        "rain_3day_sum": rain_3day_sum,
        "rain_7day_sum": rain_7day_sum,
        "rain_lag1": rain_lag1,
        "rain_change": rain_change,
        "is_monsoon": is_monsoon,
        "rain_elev_interaction": rain_elev_interaction,
    }
    return pd.DataFrame([row])


def predict_flood_risk(precipitation_mm, elevation_m, drainage_dist_km,
                        building_count, latitude, longitude, month, **kwargs):
    """
    Returns dict: {"risk_label": "Low"/"Medium"/"High", "risk_score": float}
    or None if models aren't loaded.
    """
    if not models_available():
        return None

    X = _build_features(precipitation_mm, elevation_m, drainage_dist_km,
                         building_count, latitude, longitude, month, **kwargs)

    clf_model = _clf_bundle["model"]
    clf_scaler = _clf_bundle["scaler"]
    clf_features = _clf_bundle["features"]
    labels = _clf_bundle["labels"]

    reg_model = _reg_bundle["model"]
    reg_scaler = _reg_bundle["scaler"]

    X_clf = clf_scaler.transform(X[clf_features])
    X_reg = reg_scaler.transform(X[clf_features])

    # XGBoost classifier was trained on numeric labels 0/1/2
    pred = clf_model.predict(X_clf)[0]
    if isinstance(pred, (int, np.integer)):
        risk_label = labels[int(pred)]
    else:
        risk_label = pred

    risk_score = float(reg_model.predict(X_reg)[0])
    risk_score = max(0.0, min(100.0, risk_score))

    # Confidence: max class probability from predict_proba (if supported)
    try:
        proba = clf_model.predict_proba(X_clf)[0]
        confidence = round(float(np.max(proba)) * 100, 1)
    except Exception:
        confidence = None

    return {
        "risk_label": risk_label,
        "risk_score": round(risk_score, 1),
        "confidence": confidence,
    }
