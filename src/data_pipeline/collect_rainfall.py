"""
PHASE 1 - Rainfall Data Collection
------------------------------------
Ye script Open-Meteo Historical Weather API se rainfall data download karta hai.
Open-Meteo FREE hai, koi API key nahi chahiye, koi signup nahi chahiye.
(IMD ka direct data mushkil se milta hai, isliye ye reliable alternative hai)

Output: data/raw/rainfall_raw.csv
"""

import requests
import pandas as pd
import yaml
import os

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fetch_rainfall(lat, lon, start_date, end_date):
    """
    Open-Meteo Archive API se daily rainfall data fetch karta hai.
    Docs: https://open-meteo.com/en/docs/historical-weather-api
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

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()   # agar error aaya to yahin ruk jayega
    data = response.json()

    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "precipitation_mm": data["daily"]["precipitation_sum"],
        "rain_mm": data["daily"]["rain_sum"],
    })
    df["latitude"] = lat
    df["longitude"] = lon
    return df


def main():
    config = load_config()
    city = config["city"]
    rainfall_cfg = config["rainfall"]

    print(f"Fetching rainfall data for {city['name']}...")

    df = fetch_rainfall(
        lat=city["center_lat"],
        lon=city["center_lon"],
        start_date=rainfall_cfg["start_date"],
        end_date=rainfall_cfg["end_date"],
    )

    os.makedirs(config["paths"]["raw_dir"], exist_ok=True)
    output_path = os.path.join(config["paths"]["raw_dir"], "rainfall_raw.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
