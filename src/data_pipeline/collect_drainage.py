"""
PHASE 1 - Drainage / Water-Body Data Collection
----------------------------------------------------
Waterlogging estimate karne ke liye pata hona chahiye ki koi jagah
rivers/nullahs/creeks/lakes ke kitni paas hai - jitna paas, utna zyada
waterlogging risk (paani nikalne ki jagah kam / overflow zyada).

Ye script OSM se in features ko download karta hai:
- waterway (river, stream, drain, canal)
- natural=water (lakes, ponds)
- Vasai creek jaisi coastal water bodies

Output: data/raw/water_features.geojson
"""

import osmnx as ox
import geopandas as gpd
import yaml
import os


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_water_features(bbox):
    """OSM se rivers, drains, lakes, aur coastal water bodies download karta hai."""
    bbox_tuple = (bbox["west"], bbox["south"], bbox["east"], bbox["north"])

    tags = {
        "waterway": True,                  # rivers, streams, drains, canals
        "natural": ["water", "coastline"], # lakes, ponds, aur coastline/creek
        "landuse": "reservoir",
    }

    print("Downloading water/drainage features from OSM...")
    try:
        gdf = ox.features_from_bbox(bbox=bbox_tuple, tags=tags)
    except TypeError:
        # OSMnx ke purane version ke liye alag argument order
        gdf = ox.features_from_bbox(bbox["north"], bbox["south"], bbox["east"], bbox["west"], tags=tags)

    return gdf


def main():
    config = load_config()
    bbox = config["city"]["bbox"]

    gdf = download_water_features(bbox)

    os.makedirs(config["paths"]["raw_dir"], exist_ok=True)
    output_path = os.path.join(config["paths"]["raw_dir"], "water_features.geojson")

    # Geometry column ke alawa kuch columns list-type ho sakte hain jo GeoJSON mein
    # dikkat karte hain - unhe string mein convert kar dete hain
    for col in gdf.columns:
        if col != "geometry" and gdf[col].apply(lambda x: isinstance(x, list)).any():
            gdf[col] = gdf[col].astype(str)

    gdf.to_file(output_path, driver="GeoJSON")
    print(f"Saved {len(gdf)} water/drainage features to {output_path}")


if __name__ == "__main__":
    main()
