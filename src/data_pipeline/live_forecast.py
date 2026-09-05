"""
LIVE RAINFALL FORECAST (agle kuch din ka prediction)
-------------------------------------------------------
Ye historical data se ALAG hai. Ye Open-Meteo ka FORECAST API use karta hai
jo agle 16 din tak ka rainfall forecast deta hai - LIVE / current data.

Isse pata chalega: "kal ya agle 3 din mein kitni barish ho sakti hai aur
kitna risk hai" - Dashboard mein "Live" section ke liye ye use hoga.

Output: data/raw/live_forecast.csv (har baar chalane pe fresh data aayega)
"""

import requests
import pandas as pd
import yaml
import os


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fetch_forecast(lat, lon, days=16):
    """
    Open-Meteo Forecast API - LIVE/current forecast deta hai, historical nahi.
    days: kitne din aage ka forecast chahiye (max 16)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,rain_sum,precipitation_probability_max",
        "forecast_days": days,
        "timezone": "auto"
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "precipitation_mm": data["daily"]["precipitation_sum"],
        "rain_mm": data["daily"]["rain_sum"],
        "rain_probability_pct": data["daily"]["precipitation_probability_max"],
    })
    df["latitude"] = lat
    df["longitude"] = lon
    return df


def assign_risk(precipitation_mm, thresholds):
    if precipitation_mm < thresholds["low"]:
        return "Low"
    elif precipitation_mm < thresholds["medium"]:
        return "Medium"
    else:
        return "High"


def main():
    config = load_config()
    city = config["city"]
    thresholds = config["flood_risk_thresholds"]

    print(f"Fetching LIVE forecast for {city['name']}...")
    df = fetch_forecast(city["center_lat"], city["center_lon"])

    df["flood_risk_forecast"] = df["precipitation_mm"].apply(
        lambda mm: assign_risk(mm, thresholds)
    )

    os.makedirs(config["paths"]["raw_dir"], exist_ok=True)
    output_path = os.path.join(config["paths"]["raw_dir"], "live_forecast.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)}-day forecast to {output_path}\n")
    print(df[["date", "precipitation_mm", "rain_probability_pct", "flood_risk_forecast"]])


if __name__ == "__main__":
    main()
