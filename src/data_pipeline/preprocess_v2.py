"""
PHASE 2 - COMPLETE Data Cleaning, Merging & Feature Engineering
--------------------------------------------------------------------
Ye script SAB raw data sources ko jodta hai:
  1. rainfall_raw.csv          (multi-point historical rainfall)
  2. elevation_raw.csv         (grid-wise elevation)
  3. water_features.geojson    (rivers/drains/coastline - for drainage distance)
  4. building_density.csv      (population proxy)
  5. known_flood_events.csv    (real ground-truth reference)
  6. roads.geojson             (for infrastructure risk scoring)

Aur do FINAL outputs banata hai:
  A. data/processed/flood_risk_dataset.csv   -> Phase 3 (ML model) ke liye
  B. data/processed/road_risk_dataset.csv    -> Phase 4 (routing) ke liye

Run karne se PEHLE ye sab scripts chala chuka hona chahiye:
  collect_rainfall_v2.py, collect_dem.py, collect_osm.py,
  collect_drainage.py, collect_population.py, known_flood_events.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import yaml
import os


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------
# STEP 1: Basic cleaning
# ---------------------------------------------------------------

def clean_rainfall(df):
    df = df.drop_duplicates()
    df["precipitation_mm"] = df["precipitation_mm"].fillna(0)
    df["rain_mm"] = df["rain_mm"].fillna(0)
    return df


def clean_elevation(df):
    df = df.drop_duplicates()
    df["elevation_m"] = df["elevation_m"].fillna(df["elevation_m"].mean())
    return df


# ---------------------------------------------------------------
# STEP 2: Nearest-neighbour helpers (grid points alag-alag resolution
# ke ho sakte hain, isliye "nearest match" use karte hain, exact match nahi)
# ---------------------------------------------------------------

def nearest_value(lat, lon, ref_df, lat_col, lon_col, value_col):
    distances = np.sqrt((ref_df[lat_col] - lat) ** 2 + (ref_df[lon_col] - lon) ** 2)
    return ref_df.loc[distances.idxmin(), value_col]


def distance_to_nearest_feature(lat, lon, features_gdf):
    """Point se sabse paas wale water feature ki distance (degrees mein, roughly km * 111)."""
    if features_gdf is None or len(features_gdf) == 0:
        return np.nan
    point = Point(lon, lat)
    distances = features_gdf.geometry.distance(point)
    return distances.min() * 111  # rough degree-to-km conversion


# ---------------------------------------------------------------
# STEP 3: Risk labeling (rainfall + elevation + drainage-distance combine)
# ---------------------------------------------------------------

def assign_flood_risk(precipitation_mm, elevation_m, drainage_dist_km, thresholds):
    """
    Simple weighted rule-based risk score (0-100), phir Low/Medium/High mein convert.
    - Zyada rainfall = zyada risk
    - Kam elevation = zyada risk (paani neeche jama hota hai)
    - Drainage ke paas hona thoda risk kam karta hai (paani nikal sakta hai)
      lekin bahut zyada paas hona (overflow ka risk) thoda badha bhi sakta hai -
      isliye hum sirf "bahut door" ko positive treat karte hain.
    """
    rainfall_score = min(precipitation_mm / thresholds["medium"], 2.0) * 50   # 0-100
    elevation_score = max(0, (20 - elevation_m) / 20) * 100                    # kam elevation = zyada score
    drainage_score = max(0, (2 - min(drainage_dist_km, 2)) / 2) * 100 if not np.isnan(drainage_dist_km) else 0

    total_score = (rainfall_score * 0.6) + (elevation_score * 0.25) + (drainage_score * 0.15)

    if total_score < 30:
        return "Low", round(total_score, 1)
    elif total_score < 60:
        return "Medium", round(total_score, 1)
    else:
        return "High", round(total_score, 1)


def assign_infra_risk(elevation_m, drainage_dist_km):
    """Road segment ke liye infrastructure risk - low elevation + drainage ke paas = zyada risk."""
    elevation_score = max(0, (20 - elevation_m) / 20) * 100
    drainage_score = max(0, (2 - min(drainage_dist_km, 2)) / 2) * 100 if not np.isnan(drainage_dist_km) else 0
    total = (elevation_score * 0.7) + (drainage_score * 0.3)

    if total < 30:
        return "Low", round(total, 1)
    elif total < 60:
        return "Medium", round(total, 1)
    else:
        return "High", round(total, 1)


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    config = load_config()
    raw_dir = config["paths"]["raw_dir"]
    processed_dir = config["paths"]["processed_dir"]
    thresholds = config["flood_risk_thresholds"]

    print("=" * 60)
    print("PHASE 2: Loading all raw data sources...")
    print("=" * 60)

    rainfall_df = clean_rainfall(pd.read_csv(os.path.join(raw_dir, "rainfall_raw.csv")))
    elevation_df = clean_elevation(pd.read_csv(os.path.join(raw_dir, "elevation_raw.csv")))
    building_df = pd.read_csv(os.path.join(raw_dir, "building_density.csv"))

    water_path = os.path.join(raw_dir, "water_features.geojson")
    water_gdf = gpd.read_file(water_path) if os.path.exists(water_path) else None

    print(f"  Rainfall rows: {len(rainfall_df)}")
    print(f"  Elevation points: {len(elevation_df)}")
    print(f"  Building density cells: {len(building_df)}")
    print(f"  Water features: {len(water_gdf) if water_gdf is not None else 0}")

    # -----------------------------------------------------------
    # A. FLOOD RISK DATASET (point-in-time, per location)
    # -----------------------------------------------------------
    print("\nBuilding flood_risk_dataset.csv ...")

    df = rainfall_df.copy()

    # Nearest elevation
    df["elevation_m"] = df.apply(
        lambda r: nearest_value(r["latitude"], r["longitude"], elevation_df,
                                 "latitude", "longitude", "elevation_m"),
        axis=1
    )

    # Nearest building density (population proxy)
    df["building_count"] = df.apply(
        lambda r: nearest_value(r["latitude"], r["longitude"], building_df,
                                 "center_lat", "center_lon", "building_count"),
        axis=1
    )

    # Distance to nearest drainage/water feature (km)
    unique_locations = df[["latitude", "longitude"]].drop_duplicates()
    unique_locations["drainage_dist_km"] = unique_locations.apply(
        lambda r: distance_to_nearest_feature(r["latitude"], r["longitude"], water_gdf),
        axis=1
    )
    df = df.merge(unique_locations, on=["latitude", "longitude"], how="left")

    # Risk label + score
    risk_results = df.apply(
        lambda r: assign_flood_risk(r["precipitation_mm"], r["elevation_m"],
                                     r["drainage_dist_km"], thresholds),
        axis=1
    )
    df["flood_risk"] = [r[0] for r in risk_results]
    df["flood_risk_score"] = [r[1] for r in risk_results]

    # Flag known real flood event dates (ground truth reference)
    known_events_path = os.path.join(raw_dir, "known_flood_events.csv")
    if os.path.exists(known_events_path):
        known_df = pd.read_csv(known_events_path)
        known_dates = set(known_df["date"].unique())
        df["is_known_flood_event"] = df["date"].isin(known_dates)
    else:
        df["is_known_flood_event"] = False

    os.makedirs(processed_dir, exist_ok=True)
    flood_output = os.path.join(processed_dir, "flood_risk_dataset.csv")
    df.to_csv(flood_output, index=False)

    print(f"Saved {len(df)} rows to {flood_output}")
    print("\nFlood risk distribution:")
    print(df["flood_risk"].value_counts())
    print(f"\nKnown flood event rows matched: {df['is_known_flood_event'].sum()}")

    # -----------------------------------------------------------
    # B. ROAD / INFRASTRUCTURE RISK DATASET
    # -----------------------------------------------------------
    roads_path = os.path.join(raw_dir, "roads.geojson")
    if os.path.exists(roads_path):
        print("\nBuilding road_risk_dataset.csv ...")
        roads_gdf = gpd.read_file(roads_path)

        # Har road segment ka midpoint nikal lo
        roads_gdf["midpoint"] = roads_gdf.geometry.centroid
        roads_gdf["mid_lat"] = roads_gdf["midpoint"].y
        roads_gdf["mid_lon"] = roads_gdf["midpoint"].x

        # Nearest elevation + drainage distance for each road
        roads_gdf["elevation_m"] = roads_gdf.apply(
            lambda r: nearest_value(r["mid_lat"], r["mid_lon"], elevation_df,
                                     "latitude", "longitude", "elevation_m"),
            axis=1
        )
        roads_gdf["drainage_dist_km"] = roads_gdf.apply(
            lambda r: distance_to_nearest_feature(r["mid_lat"], r["mid_lon"], water_gdf),
            axis=1
        )

        infra_results = roads_gdf.apply(
            lambda r: assign_infra_risk(r["elevation_m"], r["drainage_dist_km"]),
            axis=1
        )
        roads_gdf["infra_risk"] = [r[0] for r in infra_results]
        roads_gdf["infra_risk_score"] = [r[1] for r in infra_results]

        # Sirf zaroori columns rakhte hain (poora geometry road_network.graphml mein already hai)
        keep_cols = ["name", "highway", "mid_lat", "mid_lon", "elevation_m",
                     "drainage_dist_km", "infra_risk", "infra_risk_score"]
        keep_cols = [c for c in keep_cols if c in roads_gdf.columns]
        road_output_df = pd.DataFrame(roads_gdf[keep_cols])

        road_output = os.path.join(processed_dir, "road_risk_dataset.csv")
        road_output_df.to_csv(road_output, index=False)

        print(f"Saved {len(road_output_df)} road segments to {road_output}")
        print("\nInfrastructure risk distribution:")
        print(road_output_df["infra_risk"].value_counts())
    else:
        print("\nroads.geojson not found - skipping road risk dataset. Run collect_osm.py first.")

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
