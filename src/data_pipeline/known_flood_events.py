"""
KNOWN HISTORICAL FLOOD EVENTS - Vasai-Virar-Nalasopara
------------------------------------------------------------
Ye file manually curated hai verified news reports se (Deccan Herald, PTI).
Isका use hoga: model ko ek REAL ground-truth reference dena ki jab itna
rainfall hua tha, tab actually flood/severe waterlogging hua tha.

NOTE: Ye chhota sa reference set hai (comprehensive nahi). Project report
mein likhna: "aage aur historical events add karke labeling accuracy
improve ki ja sakti hai" - ye ek achha 'future scope' point bhi banta hai.

Output: data/raw/known_flood_events.csv
"""

import pandas as pd
import os
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # Verified event: 18 July 2021 - severe waterlogging across Vasai, Nalasopara,
    # Virar. NDRF had to rescue people. Source: Deccan Herald / PTI reports.
    known_events = [
        {
            "date": "2021-07-18",
            "location": "Vasai",
            "rainfall_mm_24hr": 204,
            "severity": "Severe",
            "notes": "NDRF rescue operations, salt pans and ground-floor flats flooded"
        },
        {
            "date": "2021-07-18",
            "location": "Virar",
            "rainfall_mm_24hr": 191,
            "severity": "Severe",
            "notes": "Roads submerged, traffic badly affected"
        },
        {
            "date": "2021-07-18",
            "location": "Pelhar",
            "rainfall_mm_24hr": 225,
            "severity": "Severe",
            "notes": "Highest recorded rainfall in this event"
        },
        {
            "date": "2021-07-18",
            "location": "Mandvi",
            "rainfall_mm_24hr": 175,
            "severity": "Severe",
            "notes": "Part of same widespread flooding event"
        },
    ]

    df = pd.DataFrame(known_events)

    os.makedirs(config["paths"]["raw_dir"], exist_ok=True)
    output_path = os.path.join(config["paths"]["raw_dir"], "known_flood_events.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} known flood event records to {output_path}")
    print(df)


if __name__ == "__main__":
    main()
