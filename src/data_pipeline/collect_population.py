"""
PHASE 1 - Population Density (Proxy via Building Density)
----------------------------------------------------------------
Real population data (census-block level) India ke liye easily free API
mein available nahi hai. Isliye ek reliable PROXY use karte hain:
har grid cell mein kitni buildings hain (OSM se) - jyada buildings =
jyada log rehte hain us area mein = jyada logon par flood ka impact.

Ye ek standard, widely-accepted approach hai jab actual census data
na mile.

Output: data/raw/building_density.csv
"""

import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
import os


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_grid_cells(bbox, resolution):
    """Grid cells banata hai (har cell ek chhota rectangle hai)."""
    lats = np.arange(bbox["south"], bbox["north"], resolution)
    lons = np.arange(bbox["west"], bbox["east"], resolution)
    cells = []
    for lat in lats:
        for lon in lons:
            cells.append({
                "lat_min": lat, "lat_max": lat + resolution,
                "lon_min": lon, "lon_max": lon + resolution,
                "center_lat": round(lat + resolution / 2, 4),
                "center_lon": round(lon + resolution / 2, 4),
            })
    return cells


def main():
    config = load_config()
    bbox = config["city"]["bbox"]
    resolution = config["elevation"]["grid_resolution"]

    bbox_tuple = (bbox["west"], bbox["south"], bbox["east"], bbox["north"])

    print("Downloading building footprints from OSM (thoda time lagega)...")
    try:
        buildings = ox.features_from_bbox(bbox=bbox_tuple, tags={"building": True})
    except TypeError:
        buildings = ox.features_from_bbox(bbox["north"], bbox["south"], bbox["east"], bbox["west"], tags={"building": True})

    print(f"Total buildings found: {len(buildings)}")

    # Har building ka center point nikal lo (density counting easy ho jayega)
    buildings["center"] = buildings.geometry.centroid
    building_lats = buildings["center"].y.values
    building_lons = buildings["center"].x.values

    grid_cells = make_grid_cells(bbox, resolution)

    for cell in grid_cells:
        count = np.sum(
            (building_lats >= cell["lat_min"]) & (building_lats < cell["lat_max"]) &
            (building_lons >= cell["lon_min"]) & (building_lons < cell["lon_max"])
        )
        cell["building_count"] = int(count)

    df = pd.DataFrame(grid_cells)

    # Simple density category - jitne zyada buildings, utna zyada dense
    df["density_category"] = pd.cut(
        df["building_count"],
        bins=[-1, 10, 50, float("inf")],
        labels=["Low", "Medium", "High"]
    )

    os.makedirs(config["paths"]["raw_dir"], exist_ok=True)
    output_path = os.path.join(config["paths"]["raw_dir"], "building_density.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} grid cells with building density to {output_path}")
    print(df[["center_lat", "center_lon", "building_count", "density_category"]].head())


if __name__ == "__main__":
    main()
