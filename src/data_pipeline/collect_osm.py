"""
PHASE 1 - Road / Infrastructure Data Collection (OpenStreetMap)
------------------------------------------------------------------
Ye script OSMnx library use karke city ka road network download karta hai.
OSMnx free hai, background mein OpenStreetMap se data leta hai.

Install: pip install osmnx

Output:
- data/raw/road_network.graphml  (poora road graph, routing ke liye)
- data/raw/roads.geojson         (roads ka geo data, mapping ke liye)
"""

import osmnx as ox
import yaml
import os


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_road_network(bbox):
    """
    Bounding box (south, west, north, east) se drivable road network download karta hai.
    City ke naam se geocode karne ke bajaye seedha coordinates use karte hain -
    ye zyada reliable hai kyunki OSM kabhi kabhi city naam se exact boundary nahi deta.
    """
    print("Downloading road network for the given area... (thoda time lagega)")
    print(f"OSMnx version: {ox.__version__}")

    # Note: OSMnx ke naye versions (2.x) mein bbox order (west, south, east, north) hai.
    # Purane versions (1.x) mein (north, south, east, west) tha.
    # Agar ye fail ho to error message dekh kar order badalna padega.
    try:
        graph = ox.graph_from_bbox(
            bbox=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
            network_type="drive"
        )
    except TypeError:
        graph = ox.graph_from_bbox(
            bbox["north"], bbox["south"], bbox["east"], bbox["west"],
            network_type="drive"
        )
    return graph


def main():
    config = load_config()
    bbox = config["city"]["bbox"]

    graph = download_road_network(bbox)

    os.makedirs(config["paths"]["raw_dir"], exist_ok=True)

    # Graph save karo (Phase 4 mein routing ke liye kaam ayega)
    graphml_path = os.path.join(config["paths"]["raw_dir"], "road_network.graphml")
    ox.save_graphml(graph, graphml_path)
    print(f"Saved road graph to {graphml_path}")

    # GeoJSON bhi save karo (mapping/visualization ke liye)
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    geojson_path = os.path.join(config["paths"]["raw_dir"], "roads.geojson")
    edges.to_file(geojson_path, driver="GeoJSON")
    print(f"Saved roads geojson to {geojson_path}")
    print(f"Total road segments: {len(edges)}")


if __name__ == "__main__":
    main()
