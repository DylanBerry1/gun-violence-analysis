import osmnx as ox
import pandas as pd
import warnings

# Suppress geometry warnings for cleaner output
warnings.filterwarnings('ignore', message='Geometry is in a geographic CRS')

def fetch_infrastructure_data():
    place_name = "Chicago, Illinois, USA"
    print(f"Fetching infrastructure data for: {place_name}...")

    # Define the OpenStreetMap tags for the infrastructure you want.
    # You can expand these based on the OSM map features wiki:
    # https://wiki.openstreetmap.org/wiki/Map_features
    tags = {
        'amenity': [
            'library', 
            'community_centre', 
            'social_facility',
            'place_of_worship',
            'school',
            'hospital',
            'clinic',
            'bar',
            'pub',
            'research_institute',
            'atm',
            'fuel',
            'payment_terminal',
            'nursing_home',
            'pharmacy',
            'arts_centre',
            'brothel',
            'casino',
            'gambling',
            'love_hotel',
            'nightclub',
            'stripclub',
            'swingerclub',
            'police',
            'fire_station',
            'courthouse',
            'prison',
            'monastery',
        ],
        'leisure': [
            'adult_gaming_centre',
            'park', 
            'recreation_ground', 
            'playground',
            
        ],
        'shop': [
            'alcohol',   # Liquor stores
            'cannabis'
            'convenience',
            'supermarket',
            'e-cigarette',
            'laundry',
            'tobacco',
            'weapons'
        ]
    }

    # Use osmnx to fetch the features (modern osmnx uses features_from_place)
    try:
        gdf = ox.features_from_place(place_name, tags=tags)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    print(f"Successfully fetched {len(gdf)} points of interest.")

    # OSM data returns points, lines, and polygons. 
    # We want a single coordinate (centroid) for each feature.
    gdf['geometry'] = gdf['geometry'].centroid
    gdf['longitude'] = gdf['geometry'].x
    gdf['latitude'] = gdf['geometry'].y

    # Extract the main category. 
    # Since a feature might have an 'amenity' tag OR a 'shop' tag, we coalesce them.
    gdf['infrastructure_type'] = gdf['amenity'].combine_first(gdf['leisure']).combine_first(gdf['shop'])

    # Keep only the most useful columns for your analysis
    # Note: Not all OSM features have names, so we fill NA with 'Unknown'
    columns_to_keep = ['name', 'infrastructure_type', 'latitude', 'longitude']
    
    # Filter out columns that don't exist in the fetched dataframe to avoid errors
    existing_columns = [col for col in columns_to_keep if col in gdf.columns]
    
    df_clean = gdf[existing_columns].copy()
    if 'name' in df_clean.columns:
        df_clean['name'] = df_clean['name'].fillna('Unknown')

    # Drop any rows that completely failed to process geometries
    df_clean = df_clean.dropna(subset=['latitude', 'longitude'])

    # Save to your processed data folder
    output_path = 'data/raw/infrastructure_locations.csv'
    
    # Ensure directories exist
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_clean.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path}")

if __name__ == "__main__":
    fetch_infrastructure_data()