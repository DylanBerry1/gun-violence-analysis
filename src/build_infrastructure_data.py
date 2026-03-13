"""
Finds the lat/lon of several forms of infrastructure in Chicago (e.g., parks, churches, libraries, liquor stores, etc.)
Infrastructure on the OSM map can be found here: https://wiki.openstreetmap.org/wiki/Map_features
"""

import osmnx as ox
import warnings

warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")


def fetch_infrastructure_data():
    place_name = "Chicago, Illinois, USA"
    tags = {
        "amenity": [
            "library",
            "community_centre",
            "social_facility",
            "place_of_worship",
            "school",
            "hospital",
            "clinic",
            "bar",
            "pub",
            "research_institute",
            "atm",
            "fuel",
            "payment_terminal",
            "nursing_home",
            "pharmacy",
            "arts_centre",
            "brothel",
            "casino",
            "gambling",
            "love_hotel",
            "nightclub",
            "stripclub",
            "swingerclub",
            "police",
            "fire_station",
            "courthouse",
            "prison",
            "monastery",
        ],
        "leisure": [
            "adult_gaming_centre",
            "park",
            "recreation_ground",
            "playground",
        ],
        "shop": [
            "alcohol",
            "cannabisconvenience",
            "supermarket",
            "e-cigarette",
            "laundry",
            "tobacco",
            "weapons",
        ],
    }

    gdf = ox.features_from_place(place_name, tags=tags)
    ## OSM returns points, lines, and polygons and we want a single coordinate (centroid) for each feature
    gdf["geometry"] = gdf["geometry"].centroid
    gdf["longitude"] = gdf["geometry"].x
    gdf["latitude"] = gdf["geometry"].y
    ## grabs the main category
    gdf["infrastructure_type"] = (
        gdf["amenity"].combine_first(gdf["leisure"]).combine_first(gdf["shop"])
    )
    ## we really only want these cols
    columns_to_keep = ["name", "infrastructure_type", "latitude", "longitude"]
    existing_columns = [col for col in columns_to_keep if col in gdf.columns]

    df_clean = gdf[existing_columns].copy()
    if "name" in df_clean.columns:
        df_clean["name"] = df_clean["name"].fillna("Unknown")

    ## drop any weird rows
    df_clean = df_clean.dropna(subset=["latitude", "longitude"])

    output_path = "data/raw/infrastructure_locations.csv"
    df_clean.to_csv(output_path, index=False)


if __name__ == "__main__":
    fetch_infrastructure_data()
