"""
Generate Chicago homicide visualization and reusable hex IDs.

Outputs:
- chicago_homicides_hex_map.html
- hex_csv_outputs/chicago_hex_counts.csv
- hex_csv_outputs/chicago_homicides_with_hex.csv 
- hex_csv_outputs/chicago_hex_time_season_counts.csv

Run:
    python3 scripts/03_build_maps.py
"""

import sys
import math
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "data" / "raw" / "chicago_violence_homicides.csv"
OUT_HEX_MAP = ROOT_DIR / "reports" / "maps" / "chicago_homicides_hex_map.html"
CSV_OUT_DIR = ROOT_DIR / "data" / "processed" / "hex"
OUT_HEX_COUNTS = CSV_OUT_DIR / "chicago_hex_counts.csv"
OUT_INCIDENTS = CSV_OUT_DIR / "chicago_homicides_with_hex.csv"
OUT_TIME_SEASON = CSV_OUT_DIR / "chicago_hex_time_season_counts.csv"

CHICAGO_BOUNDS = {
    "lat_min": 41.5,
    "lat_max": 42.1,
    "lon_min": -88.0,
    "lon_max": -87.5,
}


def detect_lat_lon_columns(df: pd.DataFrame) -> tuple[str, str]:
    lat_col = None
    lon_col = None
    for c in df.columns:
        lc = c.lower()
        if lat_col is None and "lat" in lc:
            lat_col = c
        if lon_col is None and ("lon" in lc or "lng" in lc or "long" in lc):
            lon_col = c

    if lat_col is None:
        for c in df.columns:
            if c.lower() in ("y", "latitude"):
                lat_col = c
                break
    if lon_col is None:
        for c in df.columns:
            if c.lower() in ("x", "longitude"):
                lon_col = c
                break

    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Could not auto-detect latitude/longitude columns. Columns: {list(df.columns)}"
        )
    return lat_col, lon_col


def cube_round(qf: np.ndarray, rf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xf = qf
    zf = rf
    yf = -xf - zf

    rx = np.round(xf)
    ry = np.round(yf)
    rz = np.round(zf)

    x_diff = np.abs(rx - xf)
    y_diff = np.abs(ry - yf)
    z_diff = np.abs(rz - zf)

    x_largest = (x_diff > y_diff) & (x_diff > z_diff)
    y_largest = (~x_largest) & (y_diff > z_diff)
    z_largest = (~x_largest) & (~y_largest)

    rx[x_largest] = -ry[x_largest] - rz[x_largest]
    ry[y_largest] = -rx[y_largest] - rz[y_largest]
    rz[z_largest] = -rx[z_largest] - ry[z_largest]

    return rx.astype(int), rz.astype(int)


def assign_hex_ids(gdf_m: gpd.GeoDataFrame, hex_size_m: float) -> pd.DataFrame:
    x = gdf_m.geometry.x.to_numpy()
    y = gdf_m.geometry.y.to_numpy()
    sqrt3 = math.sqrt(3.0)

    qf = ((sqrt3 / 3.0) * x - (1.0 / 3.0) * y) / hex_size_m
    rf = ((2.0 / 3.0) * y) / hex_size_m
    q, r = cube_round(qf, rf)

    out = gdf_m.copy()
    out["hex_q"] = q
    out["hex_r"] = r
    out["hex_id"] = out["hex_q"].astype(str) + "_" + out["hex_r"].astype(str)
    return out


def axial_to_center_xy(q: int, r: int, hex_size_m: float) -> tuple[float, float]:
    x = hex_size_m * math.sqrt(3.0) * (q + r / 2.0)
    y = hex_size_m * 1.5 * r
    return x, y


def hex_polygon_from_center(
    center_x: float, center_y: float, hex_size_m: float
) -> Polygon:
    coords = []
    for i in range(6):
        angle = 2 * math.pi * (i + 0.5) / 6.0
        px = center_x + hex_size_m * math.cos(angle)
        py = center_y + hex_size_m * math.sin(angle)
        coords.append((px, py))
    return Polygon(coords)


def season_from_month(month: pd.Series) -> pd.Series:
    mapping = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Fall",
        10: "Fall",
        11: "Fall",
    }
    return month.map(mapping)


def time_bin_from_hour(hour: pd.Series) -> pd.Series:
    bins = [-1, 5, 11, 17, 23]
    labels = [
        "Night (00-05)",
        "Morning (06-11)",
        "Afternoon (12-17)",
        "Evening (18-23)",
    ]
    return pd.cut(hour, bins=bins, labels=labels)


def main() -> int:
    if not CSV_PATH.exists():
        print(
            f"ERROR: CSV not found at {CSV_PATH}. Run from the directory containing the CSV."
        )
        return 1
    CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HEX_MAP.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    lat_col, lon_col = detect_lat_lon_columns(df)
    print(f"Using coordinate columns: {lat_col}, {lon_col}")

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    before_coords = len(df)
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    print(f"Dropped {before_coords - len(df)} rows with null coordinates")

    in_bounds = (
        (df[lat_col] >= CHICAGO_BOUNDS["lat_min"])
        & (df[lat_col] <= CHICAGO_BOUNDS["lat_max"])
        & (df[lon_col] >= CHICAGO_BOUNDS["lon_min"])
        & (df[lon_col] <= CHICAGO_BOUNDS["lon_max"])
    )
    dropped_bounds = int((~in_bounds).sum())
    df = df[in_bounds].copy()
    print(f"Dropped {dropped_bounds} out-of-Chicago coordinate rows")

    if df.empty:
        print("ERROR: No rows left after coordinate filtering.")
        return 1

    date_col = "Date" if "Date" in df.columns else None
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["hour"] = df[date_col].dt.hour
        df["season"] = season_from_month(df["month"])
        df["time_bin"] = time_bin_from_hour(df["hour"])
    else:
        df["year"] = np.nan
        df["month"] = np.nan
        df["hour"] = np.nan
        df["season"] = pd.NA
        df["time_bin"] = pd.NA

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
    gdf_m = gdf.to_crs(epsg=3857)

    # Doesn't create too many bins.
    hex_size_m = 1000.0
    gdf_hex = assign_hex_ids(gdf_m, hex_size_m)

    hex_counts = (
        gdf_hex.groupby(["hex_id", "hex_q", "hex_r"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    print(f"Incidents kept: {len(gdf_hex)}")
    print(f"Occupied hexes: {len(hex_counts)}")
    print(f"Count integrity check (sum hex counts): {int(hex_counts['count'].sum())}")

    # Build geometry for occupied hexes only
    hex_polys = []
    for _, row in hex_counts.iterrows():
        cx, cy = axial_to_center_xy(int(row["hex_q"]), int(row["hex_r"]), hex_size_m)
        hex_polys.append(hex_polygon_from_center(cx, cy, hex_size_m))

    hex_gdf_m = gpd.GeoDataFrame(hex_counts.copy(), geometry=hex_polys, crs="EPSG:3857")
    centroids = hex_gdf_m.copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids_wgs = centroids.to_crs(epsg=4326)
    hex_gdf_wgs = hex_gdf_m.to_crs(epsg=4326)

    hex_gdf_wgs["centroid_lat"] = centroids_wgs.geometry.y
    hex_gdf_wgs["centroid_lon"] = centroids_wgs.geometry.x

    # Save team-facing tables.
    hex_gdf_wgs[
        ["hex_id", "hex_q", "hex_r", "count", "centroid_lat", "centroid_lon"]
    ].to_csv(OUT_HEX_COUNTS, index=False)

    incident_cols = [
        c
        for c in [
            "ID",
            "Case Number",
            "Date",
            "Primary Type",
            "Description",
            "Block",
            lat_col,
            lon_col,
        ]
        if c in gdf_hex.columns
    ]
    
    gdf_hex_out = gdf_hex.copy()
    gdf_hex_out["latitude"] = gdf_hex_out.geometry.y
    gdf_hex_out["longitude"] = gdf_hex_out.geometry.x
    gdf_hex_out[
        incident_cols
        + ["year", "month", "hour", "season", "time_bin", "hex_id", "hex_q", "hex_r"]
    ].to_csv(OUT_INCIDENTS, index=False)

    hex_time_season = (
        gdf_hex.groupby(["hex_id", "time_bin", "season"], dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["hex_id", "time_bin", "season"])
    )
    hex_time_season.to_csv(OUT_TIME_SEASON, index=False)

    # Build folium map
    center_lat = float(df[lat_col].median())
    center_lon = float(df[lon_col].median())
    hex_map = folium.Map(
        location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB dark_matter"
    )

    vmax = int(hex_gdf_wgs["count"].max())
    vmin = int(hex_gdf_wgs["count"].min())
    colormap = cm.linear.YlOrRd_09.scale(vmin, max(vmax, 1))

    render_cols = ["hex_id", "count", "centroid_lat", "centroid_lon", "geometry"]
    render_geojson = hex_gdf_wgs[render_cols].to_json()

    def style_function(feature: dict) -> dict:
        c = feature["properties"].get("count", 0)
        return {
            "fillColor": colormap(c),
            "color": "#8d8d8d",
            "weight": 0.5,
            "fillOpacity": 0.65,
        }

    folium.GeoJson(
        render_geojson,
        name="hex_grid",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["hex_id", "count"],
            aliases=["Hex ID", "Homicides"],
            localize=True,
        ),
    ).add_to(hex_map)

    colormap.caption = "Homicides per occupied hex"
    hex_map.add_child(colormap)
    folium.LayerControl().add_to(hex_map)
    hex_map.save(OUT_HEX_MAP)

    print(f"Saved {OUT_HEX_MAP}")
    print(f"Saved {OUT_HEX_COUNTS}")
    print(f"Saved {OUT_INCIDENTS}")
    print(f"Saved {OUT_TIME_SEASON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
