"""
PHASE 1 - Rainfall Data Collection (Multi-Point Grid)
------------------------------------------------------
Ye script Open-Meteo Historical Weather API se rainfall data download karta hai -
lekin ab sirf 1 point ke bajaye poore area (VVN) ke andar multiple grid points
ke liye, taaki alag-alag jagah ka alag rainfall pattern capture ho sake.

Output: data/raw/rainfall_raw.csv
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
    return [(round(lat, 4), round(lon, 4)) for lat in lats for lon in lons]


def fetch_rainfall(lat, lon, start_date, end_date, max_retries=5):
    """
    Open-Meteo Archive API se ek point ka daily rainfall data fetch karta hai.
    429 (rate limit) aane par retry karta hai.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum,rain_sum",
        "timezone": "auto"
    }

    for attempt in range(max_retries):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 429:
            wait_time = 5 * (attempt + 1)
            print(f"    Rate limited, waiting {wait_time}s...")
            time.sleep(wait_time)
            continue
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame({
            "date": data["daily"]["time"],
            "precipitation_mm": data["daily"]["precipitation_sum"],
            "rain_mm": data["daily"]["rain_sum"],
        })
        df["latitude"] = lat
        df["longitude"] = lon
        return df

    raise RuntimeError("Rainfall API repeatedly rate-limited, try again later.")


def main():
    config = load_config()
    city = config["city"]
    rainfall_cfg = config["rainfall"]

    # Rainfall ke liye elevation se thoda bada grid resolution use karte hain,
    # kyunki rainfall itni fine detail par change nahi hota - isse API calls kam lagenge
    grid_points = make_grid(city["bbox"], resolution=0.05)
    print(f"Fetching rainfall for {city['name']} ({len(grid_points)} grid points)...")

    all_dfs = []
    for i, (lat, lon) in enumerate(grid_points):
        print(f"  Point {i+1}/{len(grid_points)}: ({lat}, {lon})")
        df = fetch_rainfall(
            lat=lat, lon=lon,
            start_date=rainfall_cfg["start_date"],
            end_date=rainfall_cfg["end_date"],
        )
        all_dfs.append(df)
        time.sleep(1.5)   # API ko friendly rehne ke liye delay

    final_df = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(config["paths"]["raw_dir"], exist_ok=True)
    output_path = os.path.join(config["paths"]["raw_dir"], "rainfall_raw.csv")
    final_df.to_csv(output_path, index=False)

    print(f"\nSaved {len(final_df)} rows ({len(grid_points)} points x 1 year) to {output_path}")
    print(final_df.head())


if __name__ == "__main__":
    main()
