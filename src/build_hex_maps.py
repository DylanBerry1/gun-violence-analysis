"""
Generate a combined Chicago crime hex map with a crime-type selector.

Outputs:
- reports/maps/crime_hex_maps/chicago_hex_map.html
- data/processed/hex/chicago_<crime>_hex_counts.csv
- data/processed/hex/chicago_<crime>_with_hex.csv
- data/processed/hex/chicago_<crime>_hex_time_season_counts.csv

Run:
    python3 src/build_hex_maps.py
"""

import json
import math
import re
import sys
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.element import MacroElement, Template
from shapely.geometry import Polygon

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
CSV_OUT_DIR = ROOT_DIR / "data" / "processed" / "hex"
OUT_HEX_MAP = ROOT_DIR / "reports" / "maps" / "crime_hex_maps" / "chicago_hex_map.html"
FULL_CRIMES_DATASET_CANDIDATES = (
    "chicago_crimes_2001_to_present.csv",
    "Crimes_-_2001_to_Present.csv",
    "chicago_crimes.csv",
)
COMPATIBILITY_CRIMES: dict[str, dict[str, str]] = {
    "HOMICIDE": {
        "slug": "homicides",
        "label": "Homicides",
        "legend_title": "Homicides Per Occupied Hex",
        "popup_label": "Homicides",
    },
    "NARCOTICS": {
        "slug": "drug",
        "label": "Drug Crimes",
        "legend_title": "Drug Crimes Per Occupied Hex",
        "popup_label": "Drug Crimes",
    },
    "chicago_violence_homicides.csv": {
        "slug": "homicides",
        "label": "Homicides",
        "legend_title": "Homicides Per Occupied Hex",
        "popup_label": "Homicides",
    },
    "chicago_drug_crimes.csv": {
        "slug": "drug",
        "label": "Drug Crimes",
        "legend_title": "Drug Crimes Per Occupied Hex",
        "popup_label": "Drug Crimes",
    },
}

CHICAGO_BOUNDS = {
    "lat_min": 41.5,
    "lat_max": 42.1,
    "lon_min": -88.0,
    "lon_max": -87.5,
}

CHICAGO_VIEW_BOUNDS = [[41.63, -87.94], [42.03, -87.30]]
CHICAGO_NAV_BOUNDS = {
    "min_lat": 41.55,
    "max_lat": 42.10,
    "min_lon": -88.05,
    "max_lon": -87.10,
}
MAP_MIN_ZOOM = 10
MAP_MAX_ZOOM = 14
MAP_ZOOM_DELTA = 0.5
MAP_ZOOM_SNAP = 0.5
MAP_WHEEL_PX_PER_ZOOM_LEVEL = 100
HEX_SIZE_MIN_M = 100
HEX_SIZE_MAX_M = 2000
HEX_SIZE_STEP_M = 100
DEFAULT_HEX_SIZE_M = 500
HEX_SIZE_OPTIONS = tuple(range(HEX_SIZE_MIN_M, HEX_SIZE_MAX_M + 1, HEX_SIZE_STEP_M))
LAYER_CACHE_LIMIT = 6
GEOJSON_COORD_PRECISION = 4
FINE_DETAIL_SIZES = (100, 200)
FINE_DETAIL_ZOOM_THRESHOLD = 12.5
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
TIME_BIN_BINS = [-1, 5, 11, 17, 23]
TIME_BIN_LABELS = [
    "Night (00-05)",
    "Morning (06-11)",
    "Afternoon (12-17)",
    "Evening (18-23)",
]
MAP_TITLE = "Chicago Crime Hex Map"
MAP_DESCRIPTION = (
    "Interactive Chicago crime hex map with adjustable hex size and "
    "a crime-type selector."
)
COLOR_RAMP = tuple(cm.linear.YlOrRd_09.scale(0, 1)(index / 8) for index in range(9))
UNUSED_FOLIUM_ASSETS = (
    '<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>',
    '<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js"></script>',
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js"></script>',
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css"/>',
    '<link rel="stylesheet" href="https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css"/>',
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css"/>',
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css"/>',
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css"/>',
)

CrimeConfig = dict[str, str]


def detect_lat_lon_columns(df: pd.DataFrame) -> tuple[str, str]:
    lat_col = None
    lon_col = None
    for column in df.columns:
        lowered = column.lower()
        if lat_col is None and "lat" in lowered:
            lat_col = column
        if lon_col is None and (
            "lon" in lowered or "lng" in lowered or "long" in lowered
        ):
            lon_col = column

    if lat_col is None:
        for column in df.columns:
            if column.lower() in ("y", "latitude"):
                lat_col = column
                break
    if lon_col is None:
        for column in df.columns:
            if column.lower() in ("x", "longitude"):
                lon_col = column
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


def assign_hex_ids(gdf_m: gpd.GeoDataFrame, hex_size_m: float) -> gpd.GeoDataFrame:
    x_values = gdf_m.geometry.x.to_numpy()
    y_values = gdf_m.geometry.y.to_numpy()
    sqrt3 = math.sqrt(3.0)

    qf = ((sqrt3 / 3.0) * x_values - (1.0 / 3.0) * y_values) / hex_size_m
    rf = ((2.0 / 3.0) * y_values) / hex_size_m
    q_values, r_values = cube_round(qf, rf)

    out = gdf_m.copy()
    out["hex_q"] = q_values
    out["hex_r"] = r_values
    out["hex_id"] = out["hex_q"].astype(str) + "_" + out["hex_r"].astype(str)
    return out


def axial_to_center_xy(
    q_value: int, r_value: int, hex_size_m: float
) -> tuple[float, float]:
    x_value = hex_size_m * math.sqrt(3.0) * (q_value + r_value / 2.0)
    y_value = hex_size_m * 1.5 * r_value
    return x_value, y_value


def hex_polygon_from_center(
    center_x: float, center_y: float, hex_size_m: float
) -> Polygon:
    coords = []
    for index in range(6):
        angle = 2 * math.pi * (index + 0.5) / 6.0
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
    season = month.map(mapping)
    return pd.Series(
        pd.Categorical(season, categories=SEASON_ORDER, ordered=True),
        index=month.index,
    )


def time_bin_from_hour(hour: pd.Series) -> pd.Series:
    return pd.cut(hour, bins=TIME_BIN_BINS, labels=TIME_BIN_LABELS, ordered=True)


def build_hex_layer(
    gdf_m: gpd.GeoDataFrame, hex_size_m: float
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    gdf_hex = assign_hex_ids(gdf_m, hex_size_m)

    hex_counts = (
        gdf_hex.groupby(["hex_id", "hex_q", "hex_r"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )

    hex_polygons = []
    for _, row in hex_counts.iterrows():
        center_x, center_y = axial_to_center_xy(
            int(row["hex_q"]), int(row["hex_r"]), hex_size_m
        )
        hex_polygons.append(hex_polygon_from_center(center_x, center_y, hex_size_m))

    hex_gdf_m = gpd.GeoDataFrame(
        hex_counts.copy(), geometry=hex_polygons, crs="EPSG:3857"
    )
    centroids = hex_gdf_m.copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids_wgs = centroids.to_crs(epsg=4326)
    hex_gdf_wgs = hex_gdf_m.to_crs(epsg=4326)

    hex_gdf_wgs["centroid_lat"] = centroids_wgs.geometry.y
    hex_gdf_wgs["centroid_lon"] = centroids_wgs.geometry.x
    return gdf_hex, hex_gdf_wgs


def round_geojson_coordinates(value, precision: int = GEOJSON_COORD_PRECISION):
    if isinstance(value, float):
        return round(value, precision)
    if isinstance(value, list):
        return [round_geojson_coordinates(item, precision) for item in value]
    if isinstance(value, dict):
        return {
            key: round_geojson_coordinates(item, precision)
            for key, item in value.items()
        }
    return value


def build_layer_payload(hex_gdf_wgs: gpd.GeoDataFrame) -> dict:
    vmin = int(hex_gdf_wgs["count"].min())
    vmax = int(hex_gdf_wgs["count"].max())
    render_frame = hex_gdf_wgs[["hex_id", "count", "geometry"]].copy()
    geojson = json.loads(render_frame.to_json(drop_id=True, to_wgs84=True))
    for feature in geojson["features"]:
        props = feature["properties"]
        feature["properties"] = {"h": props["hex_id"], "c": int(props["count"])}

    return {"g": round_geojson_coordinates(geojson), "n": vmin, "x": vmax}


def finalize_map_html(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")

    if not html.lstrip().lower().startswith("<!doctype html>"):
        html = "<!DOCTYPE html>\n" + html.lstrip()

    html = re.sub(r"<html(?![^>]*\blang=)", '<html lang="en"', html, count=1)

    if '<link rel="icon"' not in html:
        html = html.replace("<head>", '<head>\n    <link rel="icon" href="data:,">', 1)

    if "<title>" not in html:
        html = html.replace(
            '<meta http-equiv="content-type" content="text/html; charset=UTF-8" />',
            (
                '<meta http-equiv="content-type" content="text/html; charset=UTF-8" />\n'
                f"    <title>{MAP_TITLE}</title>"
            ),
            1,
        )

    if 'name="description"' not in html:
        html = html.replace(
            f"<title>{MAP_TITLE}</title>",
            (
                f"<title>{MAP_TITLE}</title>\n"
                f'    <meta name="description" content="{MAP_DESCRIPTION}" />'
            ),
            1,
        )

    html = re.sub(
        r'<meta name="viewport" content="[^"]*" ?/>',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0" />',
        html,
        count=1,
    )

    for asset_tag in UNUSED_FOLIUM_ASSETS:
        html = html.replace(f"    {asset_tag}\n", "")
        html = html.replace(f"{asset_tag}\n", "")

    html_path.write_text(html, encoding="utf-8")


def build_output_paths(slug: str) -> dict[str, Path]:
    return {
        "hex_counts": CSV_OUT_DIR / f"chicago_{slug}_hex_counts.csv",
        "incidents": CSV_OUT_DIR / f"chicago_{slug}_with_hex.csv",
        "time_season": CSV_OUT_DIR / f"chicago_{slug}_hex_time_season_counts.csv",
    }


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "crime"


def titleize_slug(slug: str) -> str:
    return slug.replace("_", " ").title()


def make_crime_config(slug: str, label: str) -> CrimeConfig:
    return {
        "slug": slug,
        "label": label,
        "legend_title": f"{label} Per Occupied Hex",
        "popup_label": label,
    }


def make_crime_config_from_primary_type(primary_type: str) -> CrimeConfig:
    primary_type_key = primary_type.strip().upper()
    if primary_type_key in COMPATIBILITY_CRIMES:
        return COMPATIBILITY_CRIMES[primary_type_key].copy()

    label = primary_type.strip().title()
    return make_crime_config(slugify(primary_type), label)


def make_crime_config_from_csv(csv_path: Path) -> CrimeConfig:
    if csv_path.name in COMPATIBILITY_CRIMES:
        return COMPATIBILITY_CRIMES[csv_path.name].copy()

    stem = csv_path.stem
    if stem.startswith("chicago_"):
        stem = stem.removeprefix("chicago_")
    if stem.endswith("_crimes"):
        stem = stem.removesuffix("_crimes")
    return make_crime_config(stem, titleize_slug(stem))


def find_full_crimes_dataset() -> Path | None:
    for candidate in FULL_CRIMES_DATASET_CANDIDATES:
        csv_path = RAW_DATA_DIR / candidate
        if csv_path.exists():
            return csv_path

    for csv_path in sorted(RAW_DATA_DIR.glob("*.csv")):
        lowered = csv_path.name.lower()
        if "crime" in lowered and "2001" in lowered and "present" in lowered:
            return csv_path

    return None


def discover_filtered_crime_csvs() -> list[tuple[CrimeConfig, Path]]:
    discovered: list[tuple[CrimeConfig, Path]] = []

    for csv_path in sorted(RAW_DATA_DIR.glob("chicago_*_crimes.csv")):
        discovered.append((make_crime_config_from_csv(csv_path), csv_path))

    homicide_csv = RAW_DATA_DIR / "chicago_violence_homicides.csv"
    if homicide_csv.exists():
        discovered.append((make_crime_config_from_csv(homicide_csv), homicide_csv))

    unique: dict[str, tuple[CrimeConfig, Path]] = {}
    for config, csv_path in discovered:
        unique[config["slug"]] = (config, csv_path)

    return sorted(
        unique.values(),
        key=lambda item: (0 if item[0]["slug"] == "homicides" else 1, item[0]["label"]),
    )


def load_and_prepare_incidents_frame(
    df: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, str, str]:
    lat_col, lon_col = detect_lat_lon_columns(df)

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
        raise ValueError("No rows left after coordinate filtering.")

    if "Date" in df.columns:
        parsed_dates = pd.to_datetime(
            df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
        )
        if parsed_dates.isna().any():
            fallback_mask = parsed_dates.isna()
            fallback_dates = pd.to_datetime(
                df.loc[fallback_mask, "Date"], errors="coerce"
            )
            parsed_dates.loc[fallback_mask] = fallback_dates
            print(f"Date parse fallbacks used: {int(fallback_dates.notna().sum())}")
        print(f"Rows with unparsed dates: {int(parsed_dates.isna().sum())}")
        df["Date"] = parsed_dates
        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["hour"] = df["Date"].dt.hour
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
    return gdf.to_crs(epsg=3857), lat_col, lon_col


def load_and_prepare_incidents(csv_path: Path) -> tuple[gpd.GeoDataFrame, str, str]:
    df = pd.read_csv(csv_path, low_memory=False)
    return load_and_prepare_incidents_frame(df)


def write_crime_outputs(
    crime_config: CrimeConfig,
    gdf_hex: gpd.GeoDataFrame,
    hex_gdf_wgs: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
) -> None:
    output_paths = build_output_paths(crime_config["slug"])

    hex_gdf_wgs[
        ["hex_id", "hex_q", "hex_r", "count", "centroid_lat", "centroid_lon"]
    ].to_csv(output_paths["hex_counts"], index=False)

    incident_cols = [
        column
        for column in [
            "ID",
            "Case Number",
            "Date",
            "Primary Type",
            "Description",
            "Block",
            lat_col,
            lon_col,
        ]
        if column in gdf_hex.columns
    ]

    gdf_hex_out = gdf_hex.copy()
    gdf_hex_out["latitude"] = gdf_hex_out.geometry.y
    gdf_hex_out["longitude"] = gdf_hex_out.geometry.x
    gdf_hex_out[
        incident_cols
        + ["year", "month", "hour", "season", "time_bin", "hex_id", "hex_q", "hex_r"]
    ].to_csv(output_paths["incidents"], index=False)

    hex_time_season = (
        gdf_hex.groupby(
            ["hex_id", "time_bin", "season"],
            dropna=False,
            observed=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["hex_id", "time_bin", "season"])
    )
    hex_time_season.to_csv(output_paths["time_season"], index=False)

    print(f"Saved {output_paths['hex_counts']}")
    print(f"Saved {output_paths['incidents']}")
    print(f"Saved {output_paths['time_season']}")


class CrimeHexMapControl(MacroElement):
    _template = Template(
        """
        {% macro header(this, kwargs) %}
        <style>
            .hex-ui-control,
            .hex-legend-control {
                max-width: calc(100vw - 20px);
                padding: 12px 14px;
                border-radius: 10px;
                background: rgba(13, 13, 13, 0.92);
                color: #f4efe2;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
                border: 1px solid rgba(255, 255, 255, 0.12);
                font-family: "Helvetica Neue", Arial, sans-serif;
                backdrop-filter: blur(6px);
            }

            .hex-ui-control {
                width: min(290px, calc(100vw - 20px));
            }

            .hex-legend-control {
                width: min(260px, calc(100vw - 20px));
            }

            .hex-control-block + .hex-control-block {
                margin-top: 12px;
            }

            .hex-control-title,
            .hex-legend-title {
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .hex-control-value {
                margin-top: 6px;
                font-size: 20px;
                font-weight: 700;
                line-height: 1;
            }

            .hex-control-status {
                margin-top: 6px;
                font-size: 11px;
                min-height: 1.2em;
                color: rgba(244, 239, 226, 0.78);
            }

            .hex-control-status[data-state="error"] {
                color: #ff9f80;
            }

            .hex-crime-select {
                width: 100%;
                margin-top: 10px;
                padding: 10px 12px;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.14);
                background: rgba(32, 32, 32, 0.95);
                color: #f4efe2;
                font-size: 14px;
                cursor: pointer;
                outline: none;
            }

            .hex-size-slider {
                width: 100%;
                margin: 12px 0 10px;
                accent-color: #f16913;
                cursor: pointer;
            }

            .hex-size-endpoints {
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                color: rgba(244, 239, 226, 0.72);
            }

            .hex-legend-bar {
                height: 18px;
                margin-top: 10px;
                border-radius: 999px;
            }

            .hex-legend-ticks {
                display: flex;
                justify-content: space-between;
                gap: 8px;
                margin-top: 8px;
                font-size: 11px;
                color: rgba(244, 239, 226, 0.84);
            }

            .hex-legend-note {
                margin-top: 8px;
                font-size: 11px;
                color: rgba(244, 239, 226, 0.72);
            }

            @media (max-width: 640px) {
                .hex-ui-control,
                .hex-legend-control {
                    max-width: calc(100vw - 24px);
                    padding: 9px 10px;
                    border-radius: 8px;
                }

                .hex-ui-control {
                    width: min(250px, calc(100vw - 24px));
                }

                .hex-legend-control {
                    width: min(230px, calc(100vw - 24px));
                }

                .leaflet-top.leaflet-right .hex-legend-control {
                    margin-top: 8px;
                    margin-right: 8px;
                }

                .leaflet-bottom.leaflet-left .hex-ui-control {
                    margin-left: 8px;
                    margin-bottom: 34px;
                }

                .leaflet-control-attribution {
                    margin: 0 4px 4px 0;
                    padding: 2px 6px;
                    font-size: 9px;
                    line-height: 1.25;
                }

                .hex-control-value {
                    margin-top: 4px;
                    font-size: 16px;
                }

                .hex-control-title,
                .hex-legend-title {
                    font-size: 11px;
                }

                .hex-crime-select {
                    margin-top: 8px;
                    padding: 8px 10px;
                    font-size: 12px;
                }

                .hex-legend-bar {
                    height: 12px;
                    margin-top: 8px;
                }

                .hex-legend-ticks,
                .hex-size-endpoints,
                .hex-legend-note,
                .hex-control-status {
                    font-size: 10px;
                }

                .hex-legend-ticks {
                    gap: 4px;
                    margin-top: 6px;
                }

                .hex-legend-note,
                .hex-control-status {
                    margin-top: 5px;
                }

                .hex-size-slider {
                    margin: 10px 0 8px;
                }

                .hex-control-block + .hex-control-block {
                    margin-top: 10px;
                }
            }
        </style>
        {% endmacro %}
        {% macro script(this, kwargs) %}
        (function() {
            const map = {{ this._parent.get_name() }};
            const defaultCrime = {{ this.default_crime_json | safe }};
            const defaultSize = {{ this.default_size }};
            const crimeKeys = {{ this.crime_keys_json | safe }};
            const crimeMeta = {{ this.crime_meta_json | safe }};
            const layerPayloads = {{ this.layer_payloads_json | safe }};
            const sizeOptions = {{ this.size_options_json | safe }};
            const colorRamp = {{ this.color_ramp_json | safe }};
            const legendGradientCss = {{ this.legend_gradient_css_json | safe }};
            const fineDetailSizes = new Set({{ this.fine_detail_sizes_json | safe }});
            const fineDetailZoomThreshold = {{ this.fine_detail_zoom_threshold }};
            const isCoarsePointer = window.matchMedia("(pointer: coarse)").matches || navigator.maxTouchPoints > 0;
            const layerCache = new Map();
            const layerCacheLimit = {{ this.layer_cache_limit }};
            let activeCrime = defaultCrime;
            let activeSize = defaultSize;
            let pendingSize = defaultSize;
            let scheduledFrame = null;
            let activeLayer = null;
            let activePopupLayer = null;
            let crimeSelectEl = null;
            let sliderValueEl = null;
            let legendTitleEl = null;
            let legendTicksEl = null;
            let statusEl = null;

            function buildLegendTicks(vmin, vmax, tickCount = 5) {
                if (vmax <= vmin) {
                    return [vmin];
                }
                const ticks = [];
                for (let index = 0; index < tickCount; index += 1) {
                    const ratio = index / (tickCount - 1);
                    const tick = Math.round(vmin + ((vmax - vmin) * ratio));
                    if (!ticks.length || tick !== ticks[ticks.length - 1]) {
                        ticks.push(tick);
                    }
                }
                ticks[0] = vmin;
                ticks[ticks.length - 1] = vmax;
                return ticks;
            }

            function getCrimeMeta(crimeKey) {
                return crimeMeta[crimeKey] || crimeMeta[defaultCrime];
            }

            function getPayload(crimeKey, size) {
                const crimePayloads = layerPayloads[crimeKey] || null;
                if (!crimePayloads) {
                    return null;
                }
                return crimePayloads[String(size)] || null;
            }

            function getFillColor(count, payload) {
                if (!payload || payload.x <= payload.n) {
                    return colorRamp[colorRamp.length - 1];
                }
                const normalized = (count - payload.n) / (payload.x - payload.n);
                const clamped = Math.min(1, Math.max(0, normalized));
                const rampIndex = Math.min(
                    colorRamp.length - 1,
                    Math.round(clamped * (colorRamp.length - 1))
                );
                return colorRamp[rampIndex];
            }

            function buildDetailHtml(props, crimeKey) {
                return `<strong>Hex ID</strong>: ${props.h}<br><strong>${getCrimeMeta(crimeKey).popup_label}</strong>: ${props.c}`;
            }

            function shouldShowFineDetailHint(size) {
                return fineDetailSizes.has(Number(size)) && map.getZoom() < fineDetailZoomThreshold;
            }

            function buildReadyStatus(crimeKey, size) {
                const base = `${getCrimeMeta(crimeKey).label}, ${size} m hexes loaded`;
                return shouldShowFineDetailHint(size)
                    ? `${base}. Zoom in for neighborhood detail.`
                    : base;
            }

            function styleFeature(feature, payload, size) {
                const fineResolutionWideView = fineDetailSizes.has(Number(size))
                    && map.getZoom() < fineDetailZoomThreshold;
                return {
                    fillColor: getFillColor(feature.properties.c, payload),
                    color: "#fff3d3",
                    weight: fineResolutionWideView ? 0.45 : 0.75,
                    opacity: fineResolutionWideView ? 0.14 : 0.28,
                    fillOpacity: fineResolutionWideView ? 0.52 : 0.72
                };
            }

            function hoverStyle(feature, payload) {
                return {
                    fillColor: getFillColor(feature.properties.c, payload),
                    color: "#fff7e4",
                    weight: 1.45,
                    opacity: 0.68,
                    fillOpacity: 0.9
                };
            }

            function popupStyle(feature, payload) {
                return {
                    fillColor: getFillColor(feature.properties.c, payload),
                    color: "#ffffff",
                    weight: 1.8,
                    opacity: 0.82,
                    fillOpacity: 0.93
                };
            }

            function resetLayerStyle(layer, feature, payload, size) {
                layer.setStyle(styleFeature(feature, payload, size));
            }

            function closeDetailSurfaces() {
                if (activePopupLayer) {
                    activePopupLayer.closePopup();
                    activePopupLayer = null;
                }
                map.closePopup();
            }

            function onEachFeature(feature, layer, payload, size, crimeKey) {
                const detailHtml = buildDetailHtml(feature.properties, crimeKey);
                if (!isCoarsePointer) {
                    layer.bindTooltip(detailHtml, {
                        sticky: true,
                        direction: "auto",
                        className: "hex-detail-tooltip"
                    });
                }
                layer.bindPopup(detailHtml);
                layer.on("mouseover", () => {
                    if (isCoarsePointer || activePopupLayer === layer) {
                        return;
                    }
                    layer.setStyle(hoverStyle(feature, payload));
                    if (layer.getTooltip() && !layer.isPopupOpen()) {
                        layer.openTooltip();
                    }
                });
                layer.on("mouseout", () => {
                    if (isCoarsePointer || activePopupLayer === layer) {
                        return;
                    }
                    if (layer.getTooltip()) {
                        layer.closeTooltip();
                    }
                    resetLayerStyle(layer, feature, payload, size);
                });
                layer.on("click", () => {
                    if (layer.getTooltip()) {
                        layer.closeTooltip();
                    }
                });
                layer.on("popupopen", () => {
                    if (activePopupLayer && activePopupLayer !== layer) {
                        activePopupLayer.closePopup();
                    }
                    if (layer.getTooltip()) {
                        layer.closeTooltip();
                    }
                    activePopupLayer = layer;
                    layer.setStyle(popupStyle(feature, payload));
                });
                layer.on("popupclose", () => {
                    if (activePopupLayer === layer) {
                        activePopupLayer = null;
                    }
                    resetLayerStyle(layer, feature, payload, size);
                });
            }

            function updateStatus(message, state = "idle") {
                if (!statusEl) {
                    return;
                }
                statusEl.textContent = message;
                statusEl.dataset.state = state;
            }

            function refreshLegend(crimeKey, size) {
                const payload = getPayload(crimeKey, size);
                if (!legendTicksEl || !legendTitleEl || !payload) {
                    return;
                }
                legendTitleEl.textContent = getCrimeMeta(crimeKey).legend_title;
                legendTicksEl.innerHTML = buildLegendTicks(payload.n, payload.x)
                    .map((tick) => `<span>${tick}</span>`)
                    .join("");
            }

            function getLayer(crimeKey, size) {
                const cacheKey = `${crimeKey}::${size}`;
                if (layerCache.has(cacheKey)) {
                    const cached = layerCache.get(cacheKey);
                    layerCache.delete(cacheKey);
                    layerCache.set(cacheKey, cached);
                    return cached;
                }

                const payload = getPayload(crimeKey, size);
                if (!payload) {
                    throw new Error(`Missing payload for ${crimeKey} at ${size}m`);
                }

                const layer = L.geoJSON(payload.g, {
                    style: (feature) => styleFeature(feature, payload, size),
                    onEachFeature: (feature, featureLayer) =>
                        onEachFeature(feature, featureLayer, payload, size, crimeKey)
                });
                layerCache.set(cacheKey, layer);

                while (layerCache.size > layerCacheLimit) {
                    const oldestKey = layerCache.keys().next().value;
                    if (oldestKey === `${activeCrime}::${activeSize}`) {
                        const activeEntry = layerCache.get(oldestKey);
                        layerCache.delete(oldestKey);
                        layerCache.set(oldestKey, activeEntry);
                        continue;
                    }
                    layerCache.delete(oldestKey);
                }

                return layer;
            }

            function setActiveView(crimeKey, size) {
                const payload = getPayload(crimeKey, size);
                if (!payload) {
                    updateStatus(`Missing ${getCrimeMeta(crimeKey).label} data for ${size} m`, "error");
                    return;
                }

                refreshLegend(crimeKey, size);
                updateStatus(`Rendering ${getCrimeMeta(crimeKey).label}, ${size} m hexes...`, "loading");

                try {
                    const nextLayer = getLayer(crimeKey, size);
                    closeDetailSurfaces();
                    if (activeLayer) {
                        map.removeLayer(activeLayer);
                    }
                    activeLayer = nextLayer;
                    activeLayer.addTo(map);
                    activeCrime = crimeKey;
                    activeSize = Number(size);

                    if (crimeSelectEl) {
                        crimeSelectEl.value = crimeKey;
                    }
                    if (sliderValueEl) {
                        sliderValueEl.textContent = `${activeSize} m`;
                    }

                    updateStatus(buildReadyStatus(crimeKey, size), "ready");
                } catch (error) {
                    console.error("Failed to load hex layer", crimeKey, size, error);
                    updateStatus(`Unable to render ${getCrimeMeta(crimeKey).label}`, "error");
                }
            }

            function scheduleActiveSize(size) {
                pendingSize = Number(size);
                if (scheduledFrame !== null) {
                    return;
                }
                scheduledFrame = window.requestAnimationFrame(() => {
                    scheduledFrame = null;
                    setActiveView(activeCrime, pendingSize);
                });
            }

            function refreshActiveLayerStyle() {
                if (!activeLayer) {
                    return;
                }
                const payload = getPayload(activeCrime, activeSize);
                activeLayer.setStyle((feature) => styleFeature(feature, payload, activeSize));
                if (activePopupLayer) {
                    activePopupLayer.setStyle(popupStyle(activePopupLayer.feature, payload));
                }
                updateStatus(buildReadyStatus(activeCrime, activeSize), "ready");
            }

            const UiControl = L.Control.extend({
                options: { position: "bottomleft" },
                onAdd() {
                    const container = L.DomUtil.create("div", "hex-ui-control leaflet-control");
                    L.DomEvent.disableClickPropagation(container);
                    L.DomEvent.disableScrollPropagation(container);

                    const crimeBlock = L.DomUtil.create("div", "hex-control-block", container);
                    const crimeTitle = L.DomUtil.create("div", "hex-control-title", crimeBlock);
                    crimeTitle.textContent = "Crime Type";

                    crimeSelectEl = L.DomUtil.create("select", "hex-crime-select", crimeBlock);
                    crimeSelectEl.setAttribute("aria-label", "Crime type");
                    crimeKeys.forEach((crimeKey) => {
                        const option = document.createElement("option");
                        option.value = crimeKey;
                        option.textContent = getCrimeMeta(crimeKey).label;
                        crimeSelectEl.appendChild(option);
                    });
                    crimeSelectEl.value = defaultCrime;
                    crimeSelectEl.addEventListener("change", (event) => {
                        setActiveView(event.target.value, activeSize);
                    });

                    const sizeBlock = L.DomUtil.create("div", "hex-control-block", container);
                    const sizeTitle = L.DomUtil.create("div", "hex-control-title", sizeBlock);
                    sizeTitle.textContent = "Hex Size";

                    sliderValueEl = L.DomUtil.create("div", "hex-control-value", sizeBlock);
                    sliderValueEl.textContent = `${defaultSize} m`;

                    statusEl = L.DomUtil.create("div", "hex-control-status", sizeBlock);
                    statusEl.textContent = "Ready";
                    statusEl.dataset.state = "ready";

                    const slider = L.DomUtil.create("input", "hex-size-slider", sizeBlock);
                    slider.type = "range";
                    slider.min = String(sizeOptions[0]);
                    slider.max = String(sizeOptions[sizeOptions.length - 1]);
                    slider.step = String(sizeOptions.length > 1 ? sizeOptions[1] - sizeOptions[0] : 1);
                    slider.value = String(defaultSize);
                    slider.setAttribute("aria-label", "Hex size");
                    slider.setAttribute("aria-valuetext", `${defaultSize} meters`);

                    slider.addEventListener("input", (event) => {
                        const nextSize = Number(event.target.value);
                        sliderValueEl.textContent = `${nextSize} m`;
                        slider.setAttribute("aria-valuetext", `${nextSize} meters`);
                        scheduleActiveSize(nextSize);
                    });

                    const endpoints = L.DomUtil.create("div", "hex-size-endpoints", sizeBlock);
                    endpoints.innerHTML = `<span>${sizeOptions[0]} m</span><span>${sizeOptions[sizeOptions.length - 1]} m</span>`;

                    return container;
                }
            });

            const LegendControl = L.Control.extend({
                options: { position: "topright" },
                onAdd() {
                    const container = L.DomUtil.create("div", "hex-legend-control leaflet-control");
                    L.DomEvent.disableClickPropagation(container);
                    L.DomEvent.disableScrollPropagation(container);

                    legendTitleEl = L.DomUtil.create("div", "hex-legend-title", container);
                    legendTitleEl.textContent = getCrimeMeta(defaultCrime).legend_title;

                    const bar = L.DomUtil.create("div", "hex-legend-bar", container);
                    bar.style.background = legendGradientCss;

                    legendTicksEl = L.DomUtil.create("div", "hex-legend-ticks", container);
                    const note = L.DomUtil.create("div", "hex-legend-note", container);
                    note.textContent = "Legend rescales to the selected hex size.";
                    refreshLegend(defaultCrime, defaultSize);

                    return container;
                }
            });

            new LegendControl().addTo(map);
            new UiControl().addTo(map);
            map.on("zoomend", refreshActiveLayerStyle);
            setActiveView(defaultCrime, defaultSize);
        })();
        {% endmacro %}
        """
    )

    def __init__(
        self,
        layer_payloads: dict[str, dict[str, dict]],
        crime_meta: dict[str, dict[str, str]],
        crime_keys: list[str],
        default_crime: str,
        size_options: tuple[int, ...],
        default_size: int,
        legend_gradient_css: str,
    ) -> None:
        super().__init__()
        self._name = "CrimeHexMapControl"
        self.layer_payloads_json = json.dumps(layer_payloads, separators=(",", ":"))
        self.crime_meta_json = json.dumps(crime_meta, separators=(",", ":"))
        self.crime_keys_json = json.dumps(crime_keys, separators=(",", ":"))
        self.default_crime_json = json.dumps(default_crime, separators=(",", ":"))
        self.size_options_json = json.dumps(list(size_options), separators=(",", ":"))
        self.default_size = default_size
        self.layer_cache_limit = LAYER_CACHE_LIMIT
        self.color_ramp_json = json.dumps(list(COLOR_RAMP), separators=(",", ":"))
        self.fine_detail_sizes_json = json.dumps(
            list(FINE_DETAIL_SIZES), separators=(",", ":")
        )
        self.fine_detail_zoom_threshold = FINE_DETAIL_ZOOM_THRESHOLD
        self.legend_gradient_css_json = json.dumps(
            legend_gradient_css, separators=(",", ":")
        )


def build_map(
    layer_payloads: dict[str, dict[str, dict]],
    crime_meta: dict[str, dict[str, str]],
    crime_keys: list[str],
    default_crime: str,
    center_lat: float,
    center_lon: float,
) -> None:
    hex_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB dark_matter",
        width="100%",
        height="100%",
        min_zoom=MAP_MIN_ZOOM,
        max_zoom=MAP_MAX_ZOOM,
        zoom_animation=True,
        fade_animation=False,
        marker_zoom_animation=True,
        zoom_delta=MAP_ZOOM_DELTA,
        zoom_snap=MAP_ZOOM_SNAP,
        wheel_px_per_zoom_level=MAP_WHEEL_PX_PER_ZOOM_LEVEL,
        min_lat=CHICAGO_NAV_BOUNDS["min_lat"],
        max_lat=CHICAGO_NAV_BOUNDS["max_lat"],
        min_lon=CHICAGO_NAV_BOUNDS["min_lon"],
        max_lon=CHICAGO_NAV_BOUNDS["max_lon"],
        max_bounds=True,
        max_bounds_viscosity=1.0,
        prefer_canvas=True,
    )

    legend_gradient_css = "linear-gradient(90deg, " + ", ".join(COLOR_RAMP) + ")"
    hex_map.add_child(
        CrimeHexMapControl(
            layer_payloads=layer_payloads,
            crime_meta=crime_meta,
            crime_keys=crime_keys,
            default_crime=default_crime,
            size_options=HEX_SIZE_OPTIONS,
            default_size=DEFAULT_HEX_SIZE_M,
            legend_gradient_css=legend_gradient_css,
        )
    )
    hex_map.fit_bounds(CHICAGO_VIEW_BOUNDS)
    hex_map.save(OUT_HEX_MAP)
    finalize_map_html(OUT_HEX_MAP)


def main() -> int:
    CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HEX_MAP.parent.mkdir(parents=True, exist_ok=True)

    available_crime_keys: list[str] = []
    layer_payloads: dict[str, dict[str, dict]] = {}
    centers: list[tuple[float, float]] = []
    crime_configs: dict[str, CrimeConfig] = {}

    full_dataset_path = find_full_crimes_dataset()
    if full_dataset_path is not None:
        print(f"Reading full crimes dataset {full_dataset_path}")
        full_df = pd.read_csv(full_dataset_path, low_memory=False)
        if "Primary Type" not in full_df.columns:
            print(
                f"ERROR: Full crimes dataset {full_dataset_path} is missing `Primary Type`."
            )
            return 1

        normalized_primary_types = full_df["Primary Type"].astype("string").str.strip()
        discovered_primary_types = sorted(
            {
                primary_type
                for primary_type in normalized_primary_types.dropna()
                if primary_type
            }
        )

        if not discovered_primary_types:
            print(
                f"ERROR: Full crimes dataset {full_dataset_path} did not contain any primary crime types."
            )
            return 1

        for primary_type in discovered_primary_types:
            crime_config = make_crime_config_from_primary_type(primary_type)
            crime_key = crime_config["slug"]
            crime_df = full_df[normalized_primary_types == primary_type]
            print(f"Preparing {primary_type} as {crime_key}")
            try:
                gdf_m, lat_col, lon_col = load_and_prepare_incidents_frame(
                    crime_df.copy()
                )
            except ValueError as error:
                print(f"WARNING: Skipping {crime_key}; {error}")
                continue

            centers_wgs = gdf_m.to_crs(epsg=4326)
            centers.append(
                (
                    float(centers_wgs.geometry.y.median()),
                    float(centers_wgs.geometry.x.median()),
                )
            )

            crime_payloads: dict[str, dict] = {}
            gdf_hex_default: gpd.GeoDataFrame | None = None
            hex_gdf_default: gpd.GeoDataFrame | None = None

            for hex_size_m in HEX_SIZE_OPTIONS:
                layer_gdf_hex, layer_hex_gdf_wgs = build_hex_layer(gdf_m, hex_size_m)
                crime_payloads[str(hex_size_m)] = build_layer_payload(layer_hex_gdf_wgs)

                if hex_size_m == DEFAULT_HEX_SIZE_M:
                    gdf_hex_default = layer_gdf_hex
                    hex_gdf_default = layer_hex_gdf_wgs

            if gdf_hex_default is None or hex_gdf_default is None:
                print(
                    f"WARNING: Skipping {crime_key}; default hex size {DEFAULT_HEX_SIZE_M}m failed"
                )
                continue

            print(f"Incidents kept for {crime_key}: {len(gdf_hex_default)}")
            print(
                f"Occupied hexes for {crime_key} at {DEFAULT_HEX_SIZE_M}m: "
                f"{len(hex_gdf_default)}"
            )
            print(
                f"Count integrity check for {crime_key}: "
                f"{int(hex_gdf_default['count'].sum())}"
            )

            write_crime_outputs(
                crime_config=crime_config,
                gdf_hex=gdf_hex_default,
                hex_gdf_wgs=hex_gdf_default,
                lat_col=lat_col,
                lon_col=lon_col,
            )
            crime_configs[crime_key] = crime_config
            layer_payloads[crime_key] = crime_payloads
            available_crime_keys.append(crime_key)
    else:
        discovered_csvs = discover_filtered_crime_csvs()
        if not discovered_csvs:
            print(
                "ERROR: No crime datasets were available to build the combined hex map."
            )
            return 1

        for crime_config, csv_path in discovered_csvs:
            crime_key = crime_config["slug"]
            print(f"Reading {csv_path}")
            try:
                gdf_m, lat_col, lon_col = load_and_prepare_incidents(csv_path)
            except ValueError as error:
                print(f"WARNING: Skipping {crime_key}; {error}")
                continue

            centers_wgs = gdf_m.to_crs(epsg=4326)
            centers.append(
                (
                    float(centers_wgs.geometry.y.median()),
                    float(centers_wgs.geometry.x.median()),
                )
            )

            crime_payloads: dict[str, dict] = {}
            gdf_hex_default: gpd.GeoDataFrame | None = None
            hex_gdf_default: gpd.GeoDataFrame | None = None

            for hex_size_m in HEX_SIZE_OPTIONS:
                layer_gdf_hex, layer_hex_gdf_wgs = build_hex_layer(gdf_m, hex_size_m)
                crime_payloads[str(hex_size_m)] = build_layer_payload(layer_hex_gdf_wgs)

                if hex_size_m == DEFAULT_HEX_SIZE_M:
                    gdf_hex_default = layer_gdf_hex
                    hex_gdf_default = layer_hex_gdf_wgs

            if gdf_hex_default is None or hex_gdf_default is None:
                print(
                    f"WARNING: Skipping {crime_key}; default hex size {DEFAULT_HEX_SIZE_M}m failed"
                )
                continue

            print(f"Incidents kept for {crime_key}: {len(gdf_hex_default)}")
            print(
                f"Occupied hexes for {crime_key} at {DEFAULT_HEX_SIZE_M}m: "
                f"{len(hex_gdf_default)}"
            )
            print(
                f"Count integrity check for {crime_key}: "
                f"{int(hex_gdf_default['count'].sum())}"
            )

            write_crime_outputs(
                crime_config=crime_config,
                gdf_hex=gdf_hex_default,
                hex_gdf_wgs=hex_gdf_default,
                lat_col=lat_col,
                lon_col=lon_col,
            )
            crime_configs[crime_key] = crime_config
            layer_payloads[crime_key] = crime_payloads
            available_crime_keys.append(crime_key)

    if not available_crime_keys:
        print("ERROR: No crime datasets were available to build the combined hex map.")
        return 1

    default_crime = (
        "homicides" if "homicides" in available_crime_keys else available_crime_keys[0]
    )
    center_lat = float(np.mean([lat for lat, _ in centers]))
    center_lon = float(np.mean([lon for _, lon in centers]))
    crime_meta = {
        crime_key: {
            "label": crime_configs[crime_key]["label"],
            "legend_title": crime_configs[crime_key]["legend_title"],
            "popup_label": crime_configs[crime_key]["popup_label"],
        }
        for crime_key in available_crime_keys
    }

    build_map(
        layer_payloads=layer_payloads,
        crime_meta=crime_meta,
        crime_keys=available_crime_keys,
        default_crime=default_crime,
        center_lat=center_lat,
        center_lon=center_lon,
    )

    print(f"Saved {OUT_HEX_MAP}")
    print(
        f"Built combined slider-controlled map with {len(available_crime_keys)} crime types "
        f"and hex sizes {HEX_SIZE_MIN_M}m to {HEX_SIZE_MAX_M}m"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
