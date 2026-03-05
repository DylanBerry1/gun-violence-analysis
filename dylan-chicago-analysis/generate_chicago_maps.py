"""
Generate two folium maps from chicago_violence_homicides.csv:
- chicago_homicides_points_map.html (clustered/circle markers)
- chicago_homicides_hex_map.html (hexbin aggregation)

Run: python generate_chicago_maps.py
"""

import os
import sys
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm

CSV_PATH = 'chicago_violence_homicides.csv'
OUT_POINTS = 'chicago_homicides_points_map.html'
OUT_HEX = 'chicago_homicides_hex_map.html'

if not os.path.exists(CSV_PATH):
    print(f"ERROR: CSV not found at {CSV_PATH}. Run from the directory containing the CSV.")
    sys.exit(1)

print('Reading', CSV_PATH)
df = pd.read_csv(CSV_PATH)

# Auto-detect lat/lon columns
lat_col = None
lon_col = None
for c in df.columns:
    lc = c.lower()
    if 'lat' in lc and lat_col is None:
        lat_col = c
    if ('lon' in lc or 'lng' in lc or 'long' in lc) and lon_col is None:
        lon_col = c

if lat_col is None:
    for c in df.columns:
        if c.lower() in ('y', 'latitude'):
            lat_col = c
if lon_col is None:
    for c in df.columns:
        if c.lower() in ('x', 'longitude'):
            lon_col = c

if lat_col is None or lon_col is None:
    print('Could not auto-detect latitude/longitude columns. Columns found:\n', list(df.columns))
    sys.exit(1)

print('Using:', lat_col, lon_col)

# Convert
df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

before = len(df)
df = df.dropna(subset=[lat_col, lon_col])
print(f'Dropped {before - len(df)} rows without valid coords')

# timestamp -> year if present
time_col = None
for c in df.columns:
    if 'date' in c.lower() or 'time' in c.lower() or 'occur' in c.lower():
        time_col = c
        """
        Generate two folium maps from chicago_violence_homicides.csv:
        - chicago_homicides_points_map.html (scatter points; every point visible)
        - chicago_homicides_hex_map.html (hexagon aggregation using GeoPandas)

        Run: python generate_chicago_maps.py
        """

        import os
        import sys
        import pandas as pd
        import numpy as np
        import folium
        import branca.colormap as cm

        CSV_PATH = 'chicago_violence_homicides.csv'
        OUT_POINTS = 'chicago_homicides_points_map.html'
        OUT_HEX = 'chicago_homicides_hex_map.html'

        if not os.path.exists(CSV_PATH):
            print(f"ERROR: CSV not found at {CSV_PATH}. Run from the directory containing the CSV.")
            sys.exit(1)

        print('Reading', CSV_PATH)
        df = pd.read_csv(CSV_PATH)

        # Auto-detect lat/lon columns
        lat_col = None
        lon_col = None
        for c in df.columns:
            lc = c.lower()
            if 'lat' in lc and lat_col is None:
                lat_col = c
            if ('lon' in lc or 'lng' in lc or 'long' in lc) and lon_col is None:
                lon_col = c

        if lat_col is None:
            for c in df.columns:
                if c.lower() in ('y', 'latitude'):
                    lat_col = c
        if lon_col is None:
            for c in df.columns:
                if c.lower() in ('x', 'longitude'):
                    lon_col = c

        if lat_col is None or lon_col is None:
            print('Could not auto-detect latitude/longitude columns. Columns found:\n', list(df.columns))
            sys.exit(1)

        print('Using:', lat_col, lon_col)

        # Convert
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

        before = len(df)
        df = df.dropna(subset=[lat_col, lon_col])
        print(f'Dropped {before - len(df)} rows without valid coords')

        # timestamp -> year if present
        time_col = None
        for c in df.columns:
            if 'date' in c.lower() or 'time' in c.lower() or 'occur' in c.lower():
                time_col = c
                break

        if time_col is not None:
            try:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                df['year'] = df[time_col].dt.year
            except Exception:
                df['year'] = np.nan
        else:
            df['year'] = np.nan

        center_lat = df[lat_col].median()
        center_lon = df[lon_col].median()
        print('Map center:', center_lat, center_lon)

        # --- Points map (UNCLUSTERED scatter) ---
        point_map = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB positron')

        years = df['year'].dropna().unique()
        if len(years) > 0:
            colormap = cm.linear.YlOrRd_09.scale(min(years), max(years))
        else:
            colormap = cm.linear.YlOrRd_09.scale(0, 1)

        # Add every point individually so no grouping occurs
        for _, row in df.iterrows():
            lat = row[lat_col]
            lon = row[lon_col]
            yr = row.get('year', None)
            color = '#3186cc' if (yr is None or (isinstance(yr, float) and np.isnan(yr))) else colormap(yr)
            popup_items = []
            if time_col is not None:
                popup_items.append(f"Date: {row.get(time_col)}")
            for c in ('block', 'description', 'primary_type'):
                if c in df.columns:
                    popup_items.append(f"{c.capitalize()}: {row.get(c)}")
            popup_html = '<br>'.join([str(x) for x in popup_items if x is not None])
            folium.CircleMarker(location=[lat, lon], radius=3, color=color, fill=True, fill_opacity=0.5, popup=popup_html).add_to(point_map)

        folium.LayerControl().add_to(point_map)
        point_map.save(OUT_POINTS)
        print('Saved', OUT_POINTS)

        # --- Hexagon-aggregated map using GeoPandas ---
        hex_map = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB dark_matter')

        try:
            import geopandas as gpd
            from shapely.geometry import Point, Polygon

            # Create GeoDataFrame of points
            gdf = gpd.GeoDataFrame(df.copy(), geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])], crs='EPSG:4326')
            # Project to Web Mercator (meters) for uniform hex sizing
            gdf_m = gdf.to_crs(epsg=3857)

            # Create hex grid covering the points extent
            bounds = gdf_m.total_bounds  # minx, miny, maxx, maxy
            minx, miny, maxx, maxy = bounds
            hex_size = 1500  # meters

            from math import ceil

            dx = hex_size * (3 ** 0.5)
            dy = hex_size * 1.5

            cols = int(ceil((maxx - minx) / dx)) + 3
            rows = int(ceil((maxy - miny) / dy)) + 3

            hexes = []
            for col in range(-1, cols + 1):
                for row in range(-1, rows + 1):
                    x = minx + col * dx
                    y = miny + row * dy
                    if row % 2 == 0:
                        x += dx / 2
                    coords = []
                    for i in range(6):
                        angle = 2 * np.pi * (i + 0.5) / 6.0
                        px = x + hex_size * np.cos(angle)
                        py = y + hex_size * np.sin(angle)
                        coords.append((px, py))
                    hexes.append(Polygon(coords))

            hex_gdf = gpd.GeoDataFrame({'geometry': hexes}, crs='EPSG:3857')

            # Spatial join to count points in each hex
            joined = gpd.sjoin(hex_gdf, gdf_m[['geometry']], how='left', predicate='contains')
            counts = joined.groupby(joined.index).size()
            hex_gdf['count'] = counts.reindex(hex_gdf.index).fillna(0).astype(int)

            # Project hexes back to WGS84 for folium
            hex_gdf_wgs = hex_gdf.to_crs(epsg=4326)

            # Create colormap
            vmax = int(hex_gdf['count'].max())
            vmin = int(hex_gdf['count'].min())
            colormap = cm.linear.YlOrRd_09.scale(vmin if vmin is not None else 0, vmax if vmax > 0 else 1)

            def style_function(feature):
                c = feature['properties'].get('count', 0)
                return {
                    'fillColor': colormap(c) if c > 0 else 'transparent',
                    'color': 'grey',
                    'weight': 0.5,
                    'fillOpacity': 0.6 if c > 0 else 0,
                }

            folium.GeoJson(
                hex_gdf_wgs.to_json(),
                name='hex_grid',
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(fields=['count'], aliases=['Count'])
            ).add_to(hex_map)

            colormap.caption = 'Incidents per hex'
            hex_map.add_child(colormap)
        except Exception as e:
            folium.map.Marker([center_lat, center_lon], popup=f'Hex aggregation failed: {e}').add_to(hex_map)

        folium.LayerControl().add_to(hex_map)
        hex_map.save(OUT_HEX)
        print('Saved', OUT_HEX)

        print('\nDone. Open the generated HTML files in a browser.')
