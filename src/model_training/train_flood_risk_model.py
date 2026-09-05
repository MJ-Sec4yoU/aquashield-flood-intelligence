"""
Urban Flood Intelligence System — Phase 3
Flood Risk Prediction Model Training Pipeline

Trains on flood_risk_dataset.csv (produced by Phase 2) to predict flood risk
in two ways:
    1. Classification -> Low / Medium / High
    2. Regression      -> continuous 0-100 flood_risk_score

Design choices (see project report Phase 0-2 + Phase 3 discussion):
  - The label was originally built with a RULE-BASED formula
    (60% rainfall, 25% elevation, 15% distance-to-drainage). Training on the
    exact same three raw features would let the model simply re-derive the
    formula rather than learn anything new. To avoid that, this script adds
    engineered features the formula does NOT use directly:
        * 3-day / 7-day rolling rainfall sums (accumulation matters, not just
          a single day's rain)
        * day-over-day rainfall change (is it intensifying?)
        * monsoon-season flag (Jun-Sep)
        * a rainfall/elevation interaction term
  - TWO separate train/test splits are run and reported, because a random
    row split would leak information (same 12 locations repeat across 5
    years):
        * Temporal split : train 2019-2022, test 2023 (can it predict the
          future?)
        * Spatial split  : leave a few of the 12 grid locations out
          entirely (can it generalize to new places?)
  - High-risk class is ~1.4% of rows. Accuracy is NOT reported as the
    headline metric. Macro-F1, per-class recall (especially High), and the
    confusion matrix are used instead. class_weight='balanced' is used by
    default; SMOTE is available as a toggle (train fold only, never on
    test data).
  - The verified 18 July 2021 flood event is used purely as a QUALITATIVE
    sanity check after training, not as a training signal.

Usage:
    python train_flood_risk_model.py --data /path/to/flood_risk_dataset.csv
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
RISK_ORDER = ["Low", "Medium", "High"]  # fixed label order everywhere


# --------------------------------------------------------------------------
# 1. LOAD + FEATURE ENGINEERING
# --------------------------------------------------------------------------
def load_and_engineer(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # Stable location id from unique (lat, lon) pairs — needed for grouping
    # rolling features per location and for the spatial split.
    locs = df[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    locs["location_id"] = locs.index
    df = df.merge(locs, on=["latitude", "longitude"], how="left")

    df = df.sort_values(["location_id", "date"]).reset_index(drop=True)

    # --- Engineered features NOT used by the original rule-based formula ---
    g = df.groupby("location_id")["precipitation_mm"]
    df["rain_3day_sum"] = g.transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["rain_7day_sum"] = g.transform(lambda s: s.rolling(7, min_periods=1).sum())
    df["rain_lag1"] = g.transform(lambda s: s.shift(1).fillna(0))
    df["rain_change"] = df["precipitation_mm"] - df["rain_lag1"]

    df["month"] = df["date"].dt.month
    df["is_monsoon"] = df["month"].between(6, 9).astype(int)

    df["rain_elev_interaction"] = df["precipitation_mm"] / (df["elevation_m"] + 1.0)

    return df


FEATURE_COLS = [
    "precipitation_mm",
    "rain_mm",
    "elevation_m",
    "building_count",
    "drainage_dist_km",
    "latitude",
    "longitude",
    "rain_3day_sum",
    "rain_7day_sum",
    "rain_lag1",
    "rain_change",
    "is_monsoon",
    "rain_elev_interaction",
]


# --------------------------------------------------------------------------
# 2. SPLITS
# --------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame, test_year: int = 2023):
    train = df[df["date"].dt.year < test_year]
    test = df[df["date"].dt.year == test_year]
    return train, test


def spatial_split(df: pd.DataFrame, n_holdout_locations: int = 3, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
    all_locs = df["location_id"].unique()
    holdout_locs = rng.choice(all_locs, size=n_holdout_locations, replace=False)
    train = df[~df["location_id"].isin(holdout_locs)]
    test = df[df["location_id"].isin(holdout_locs)]
    return train, test, holdout_locs


# --------------------------------------------------------------------------
# 3. CLASSIFICATION
# --------------------------------------------------------------------------
def run_classification(train_df, test_df, split_name: str, use_smote: bool = False):
    X_train, y_train = train_df[FEATURE_COLS], train_df["flood_risk"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["flood_risk"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if use_smote:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        ),
    }

    results = {}
    for name, model in models.items():
        if name == "XGBoost":
            # XGBoost needs numeric labels + sample_weight for imbalance
            label_map = {l: i for i, l in enumerate(RISK_ORDER)}
            y_train_num = y_train.map(label_map)
            y_test_num = y_test.map(label_map)
            counts = y_train_num.value_counts()
            weight_map = {c: len(y_train_num) / (len(counts) * n) for c, n in counts.items()}
            sample_weight = y_train_num.map(weight_map)
            model.fit(X_train_s, y_train_num, sample_weight=sample_weight)
            y_pred_num = model.predict(X_test_s)
            y_pred = pd.Series(y_pred_num).map({i: l for l, i in label_map.items()})
            y_true = y_test
        else:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_true = y_test

        report = classification_report(
            y_true, y_pred, labels=RISK_ORDER, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=RISK_ORDER)
        macro_f1 = f1_score(y_true, y_pred, labels=RISK_ORDER, average="macro", zero_division=0)
        high_recall = recall_score(
            y_true, y_pred, labels=["High"], average="macro", zero_division=0
        )

        results[name] = {
            "model": model,
            "scaler": scaler,
            "report": report,
            "confusion_matrix": cm,
            "macro_f1": macro_f1,
            "high_recall": high_recall,
            "y_true": y_true,
            "y_pred": y_pred,
        }

        print(f"\n--- [{split_name}] {name} ---")
        print(f"Macro F1: {macro_f1:.3f}   |   High-risk recall: {high_recall:.3f}")
        print(f"Confusion matrix (rows=true, cols=pred) order {RISK_ORDER}:")
        print(cm)

    return results


# --------------------------------------------------------------------------
# 4. REGRESSION (continuous 0-100 score)
# --------------------------------------------------------------------------
def run_regression(train_df, test_df, split_name: str):
    X_train, y_train = train_df[FEATURE_COLS], train_df["flood_risk_score"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["flood_risk_score"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoostRegressor": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "scaler": scaler,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "y_true": y_test,
            "y_pred": y_pred,
        }

        print(f"\n--- [{split_name}] {name} ---")
        print(f"RMSE: {rmse:.2f}   MAE: {mae:.2f}   R2: {r2:.3f}")

    return results


# --------------------------------------------------------------------------
# 5. QUALITATIVE SANITY CHECK — the known 18 July 2021 flood event
# --------------------------------------------------------------------------
def known_event_check(df: pd.DataFrame, clf_results: dict, split_name: str):
    """
    Reports how the trained models score the verified 2021 flood event,
    compared to how the ORIGINAL rule-based formula scored it (which,
    per the Phase 0-2 report, labeled it Medium/Low everywhere — none High).
    This is reported as a finding, not baked into training.
    """
    known = df[df["is_known_flood_event"] == True]  # noqa: E712
    if known.empty:
        return

    best_model_name = max(clf_results, key=lambda k: clf_results[k]["macro_f1"])
    model = clf_results[best_model_name]["model"]
    scaler = clf_results[best_model_name]["scaler"]

    X_known = known[FEATURE_COLS]
    X_known_s = scaler.transform(X_known)

    if best_model_name == "XGBoost":
        label_map = {i: l for i, l in enumerate(RISK_ORDER)}
        preds_num = model.predict(X_known_s)
        preds = pd.Series(preds_num).map(label_map)
    else:
        preds = model.predict(X_known_s)

    print(f"\n=== [{split_name}] Known 18-July-2021 flood event — model: {best_model_name} ===")
    out = known[["latitude", "longitude", "precipitation_mm", "flood_risk"]].copy()
    out["original_formula_label"] = out["flood_risk"]
    out["model_predicted_label"] = list(preds)
    print(out[["latitude", "longitude", "precipitation_mm",
               "original_formula_label", "model_predicted_label"]].to_string(index=False))
    n_high = (out["model_predicted_label"] == "High").sum()
    print(f"\nOriginal formula rated 0/{len(out)} grid points as High on this real flood day.")
    print(f"Trained model rates {n_high}/{len(out)} grid points as High on this real flood day.")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main(data_path: str, out_dir: str, use_smote: bool):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data and engineering features...")
    df = load_and_engineer(data_path)
    print(f"Total rows: {len(df)} | Locations: {df['location_id'].nunique()} "
          f"| Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    all_metrics = {}

    # ---------------- Temporal split ----------------
    print("\n" + "=" * 70)
    print("TEMPORAL SPLIT (train 2019-2022, test 2023)")
    print("=" * 70)
    train_t, test_t = temporal_split(df)
    clf_temporal = run_classification(train_t, test_t, "Temporal", use_smote=use_smote)
    reg_temporal = run_regression(train_t, test_t, "Temporal")
    known_event_check(test_t if not test_t.empty else df, clf_temporal, "Temporal")
    # Known event date is 2021, which falls in TRAIN for the temporal split,
    # not test — check against full df's known rows using the temporal
    # model regardless, purely for the qualitative report.
    known_event_check(df, clf_temporal, "Temporal (model trained on 2019-2022)")

    # ---------------- Spatial split ----------------
    print("\n" + "=" * 70)
    print("SPATIAL SPLIT (3 of 12 locations held out entirely)")
    print("=" * 70)
    train_s, test_s, holdout_locs = spatial_split(df)
    print(f"Held-out location_ids: {list(holdout_locs)}")
    clf_spatial = run_classification(train_s, test_s, "Spatial", use_smote=use_smote)
    reg_spatial = run_regression(train_s, test_s, "Spatial")
    known_event_check(df, clf_spatial, "Spatial")

    # ---------------- Save best models ----------------
    best_clf_name = max(clf_temporal, key=lambda k: clf_temporal[k]["macro_f1"])
    best_reg_name = min(reg_temporal, key=lambda k: reg_temporal[k]["rmse"])

    joblib.dump(
        {"model": clf_temporal[best_clf_name]["model"], "scaler": clf_temporal[best_clf_name]["scaler"],
         "features": FEATURE_COLS, "labels": RISK_ORDER},
        out / "best_classifier.joblib",
    )
    joblib.dump(
        {"model": reg_temporal[best_reg_name]["model"], "scaler": reg_temporal[best_reg_name]["scaler"],
         "features": FEATURE_COLS},
        out / "best_regressor.joblib",
    )

    # ---------------- Summary report ----------------
    summary = {
        "temporal_split": {
            name: {"macro_f1": r["macro_f1"], "high_recall": r["high_recall"]}
            for name, r in clf_temporal.items()
        },
        "spatial_split": {
            name: {"macro_f1": r["macro_f1"], "high_recall": r["high_recall"]}
            for name, r in clf_spatial.items()
        },
        "regression_temporal": {
            name: {"rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"]}
            for name, r in reg_temporal.items()
        },
        "regression_spatial": {
            name: {"rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"]}
            for name, r in reg_spatial.items()
        },
        "best_classifier": best_clf_name,
        "best_regressor": best_reg_name,
    }
    with open(out / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print(f"\nBest classifier ({best_clf_name}) and regressor ({best_reg_name}) saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to flood_risk_dataset.csv")
    parser.add_argument("--out", type=str, default="./phase3_outputs", help="Output directory")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE oversampling on train fold")
    args = parser.parse_args()
    main(args.data, args.out, args.smote)
