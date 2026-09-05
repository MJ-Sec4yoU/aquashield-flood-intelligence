"""
PHASE 1 - Elevation (DEM) Data Collection
---------------------------------------------
NASA SRTM DEM download karna USGS Earth Explorer account maangta hai jo
complicated hai. Isliye hum Open-Meteo Elevation API use kar rahe hain -
same underlying SRTM data hi deta hai, bina kisi signup ke.

Ye script city ke bounding box ke andar ek grid banata hai aur har grid
point ki elevation (height) fetch karta hai.

Output: data/raw/elevation_raw.csv
"""

import requests
import pandas as pd
import numpy as np
import yaml
import os
import time


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_grid(bbox, resolution):
    """Bounding box ke andar lat/lon grid points banata hai."""
    lats = np.arange(bbox["south"], bbox["north"], resolution)
    lons = np.arange(bbox["west"], bbox["east"], resolution)
    grid_points = [(lat, lon) for lat in lats for lon in lons]
    return grid_points


def fetch_elevation_batch(points, max_retries=5):
    """
    Open-Meteo Elevation API - ek call mein multiple points ki elevation
    fetch karta hai. Agar 429 (Too Many Requests) mile to thoda रुक कर retry karta hai.
    """
    lats = ",".join(str(round(p[0], 4)) for p in points)
    lons = ",".join(str(round(p[1], 4)) for p in points)

    url = "https://api.open-meteo.com/v1/elevation"
    params = {"latitude": lats, "longitude": lons}

    for attempt in range(max_retries):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 429:
            wait_time = 5 * (attempt + 1)   # 5s, 10s, 15s... badhta hua wait
            print(f"  Rate limited, waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            continue
        response.raise_for_status()
        return response.json()["elevation"]

    raise RuntimeError("Elevation API repeatedly rate-limited, try again later.")


def main():
    config = load_config()
    bbox = config["city"]["bbox"]
    resolution = config["elevation"]["grid_resolution"]

    grid_points = make_grid(bbox, resolution)
    print(f"Grid mein total {len(grid_points)} points hain. Fetching elevation...")

    batch_size = 50   # chota batch size taaki rate limit na lage
    all_rows = []

    for i in range(0, len(grid_points), batch_size):
        batch = grid_points[i:i + batch_size]
        elevations = fetch_elevation_batch(batch)

        for (lat, lon), elev in zip(batch, elevations):
            all_rows.append({"latitude": lat, "longitude": lon, "elevation_m": elev})

        print(f"  {min(i + batch_size, len(grid_points))}/{len(grid_points)} done")
        time.sleep(2)   # API ko friendly rehne ke liye zyada delay

    df = pd.DataFrame(all_rows)

    os.makedirs(config["paths"]["raw_dir"], exist_ok=True)
    output_path = os.path.join(config["paths"]["raw_dir"], "elevation_raw.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
