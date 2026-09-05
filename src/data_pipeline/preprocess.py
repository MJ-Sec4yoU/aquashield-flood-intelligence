"""
PHASE 2 - Data Cleaning & Preprocessing
--------------------------------------------
Ye script teeno raw datasets (rainfall, elevation) ko:
1. Missing values handle karta hai
2. Ek common location grid pe merge karta hai
3. Ek flood-risk label add karta hai (Low/Medium/High) rainfall thresholds ke basis pe
   -- (ye sirf ek simple starting label hai, Phase 3 mein proper ML model isko replace karega)

Output: data/processed/merged_dataset.csv
        -> Ye file Phase 3 (model training) ko diya jayega
"""

import pandas as pd
import numpy as np
import yaml
import os


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_rainfall(df):
    """Missing rainfall values ko 0 se fill karo (no rain that day) aur duplicates hatao."""
    df = df.drop_duplicates()
    df["precipitation_mm"] = df["precipitation_mm"].fillna(0)
    df["rain_mm"] = df["rain_mm"].fillna(0)
    return df


def clean_elevation(df):
    """Missing elevation ko us column ke average se fill karo."""
    df = df.drop_duplicates()
    df["elevation_m"] = df["elevation_m"].fillna(df["elevation_m"].mean())
    return df


def assign_risk_label(precipitation_mm, thresholds):
    """Simple rule-based risk label - Phase 3 mein ML model isse better karega."""
    if precipitation_mm < thresholds["low"]:
        return "Low"
    elif precipitation_mm < thresholds["medium"]:
        return "Medium"
    else:
        return "High"


def nearest_elevation(lat, lon, elevation_df):
    """Rainfall point ke sabse paas wali elevation grid cell dhoondta hai."""
    distances = np.sqrt(
        (elevation_df["latitude"] - lat) ** 2 + (elevation_df["longitude"] - lon) ** 2
    )
    nearest_idx = distances.idxmin()
    return elevation_df.loc[nearest_idx, "elevation_m"]


def main():
    config = load_config()
    raw_dir = config["paths"]["raw_dir"]
    processed_dir = config["paths"]["processed_dir"]
    thresholds = config["flood_risk_thresholds"]

    # ---- Load raw data ----
    rainfall_df = pd.read_csv(os.path.join(raw_dir, "rainfall_raw.csv"))
    elevation_df = pd.read_csv(os.path.join(raw_dir, "elevation_raw.csv"))

    # ---- Clean ----
    rainfall_df = clean_rainfall(rainfall_df)
    elevation_df = clean_elevation(elevation_df)

    # ---- Merge: har rainfall row ko nearest elevation se jodo ----
    print("Merging rainfall + elevation data...")
    rainfall_df["elevation_m"] = rainfall_df.apply(
        lambda row: nearest_elevation(row["latitude"], row["longitude"], elevation_df),
        axis=1
    )

    # ---- Add risk label ----
    rainfall_df["flood_risk"] = rainfall_df["precipitation_mm"].apply(
        lambda mm: assign_risk_label(mm, thresholds)
    )

    # ---- Save ----
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "merged_dataset.csv")
    rainfall_df.to_csv(output_path, index=False)

    print(f"Saved merged dataset with {len(rainfall_df)} rows to {output_path}")
    print(rainfall_df.head())
    print("\nRisk label distribution:")
    print(rainfall_df["flood_risk"].value_counts())


if __name__ == "__main__":
    main()
