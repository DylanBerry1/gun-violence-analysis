"""
Generate Chicago crime visualization and reusable hex IDs.

Have to give it a specific type of crime (the one in the primary type column) and the data.

Outputs:
- chicago_CRIME_TYPE_hex_map.html
- data/processed/hex/chicago_CRIME_TYPE_hex_counts.csv
- data/processed/hex/chicago_CRIME_TYPE_with_hex.csv
- data/processed/hex/chicago_CRIME_TYPE_hex_time_season_counts.csv

Run:
    python3 src/build_CRIME_TYPE_hex_map.py
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

#--------
CRIME_TYPE = ''
#--------


ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "data" / "raw" / f"chicago_{CRIME_TYPE}_crimes.csv"
OUT_HEX_MAP = ROOT_DIR / "reports" / "maps" / f"chicago_{CRIME_TYPE}_hex_map.html"
CSV_OUT_DIR = ROOT_DIR / "data" / "processed" / "hex"
OUT_HEX_COUNTS = CSV_OUT_DIR / f"chicago_{CRIME_TYPE}_hex_counts.csv"
OUT_INCIDENTS = CSV_OUT_DIR / f"chicago_{CRIME_TYPE}_with_hex.csv"
OUT_TIME_SEASON = CSV_OUT_DIR / f"chicago_{CRIME_TYPE}_hex_time_season_counts.csv"

CHICAGO_BOUNDS = {
    "lat_min": 41.5,
    "lat_max": 42.1,
    "lon_min": -88.0,
    "lon_max": -87.5,
}

# Analysis-focused map viewport controls
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
LAYER_CACHE_LIMIT = 4
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
MAP_TITLE = f"Chicago {CRIME_TYPE} Hex Map"
MAP_DESCRIPTION = (
    f"Interactive Chicago {CRIME_TYPE} hex map with adjustable hex size and "
    f"per-hex {CRIME_TYPE} counts."
)
COLOR_RAMP = tuple(cm.linear.YlOrRd_09.scale(0, 1)(i / 8) for i in range(9))
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
    season = month.map(mapping)
    return pd.Series(
        pd.Categorical(season, categories=SEASON_ORDER, ordered=True), index=month.index
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
        feature["properties"] = {
            "h": props["hex_id"],
            "c": int(props["count"]),
        }

    return {
        "g": round_geojson_coordinates(geojson),
        "n": vmin,
        "x": vmax,
    }


def finalize_map_html(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")

    if not html.lstrip().lower().startswith("<!doctype html>"):
        html = "<!DOCTYPE html>\n" + html.lstrip()

    html = re.sub(r"<html(?![^>]*\blang=)", '<html lang="en"', html, count=1)

    if '<link rel="icon"' not in html:
        html = html.replace(
            "<head>",
            '<head>\n    <link rel="icon" href="data:,">',
            1,
        )

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


class HexSizeSliderControl(MacroElement):
    _template = Template(
        """
        {% macro header(this, kwargs) %}
        <style>
            .hex-size-control {
                width: min(280px, calc(100vw - 20px));
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

            .hex-size-title,
            .hex-legend-title {
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .hex-size-value {
                margin-top: 6px;
                font-size: 20px;
                font-weight: 700;
                line-height: 1;
            }

            .hex-size-status {
                margin-top: 6px;
                font-size: 11px;
                min-height: 1.2em;
                color: rgba(244, 239, 226, 0.78);
            }

            .hex-size-status[data-state="error"] {
                color: #ff9f80;
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

            .hex-legend-control {
                width: min(260px, calc(100vw - 20px));
                max-width: calc(100vw - 20px);
                padding: 12px 14px;
                border-radius: 10px;
                background: rgba(13, 13, 13, 0.9);
                color: #f4efe2;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
                border: 1px solid rgba(255, 255, 255, 0.12);
                font-family: "Helvetica Neue", Arial, sans-serif;
                backdrop-filter: blur(6px);
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
                .hex-size-control,
                .hex-legend-control {
                    padding: 10px 12px;
                    border-radius: 8px;
                }

                .hex-size-value {
                    font-size: 18px;
                }

                .hex-size-title,
                .hex-legend-title {
                    font-size: 11px;
                }

                .hex-legend-bar {
                    height: 14px;
                }

                .hex-legend-ticks,
                .hex-size-endpoints,
                .hex-legend-note,
                .hex-size-status {
                    font-size: 10px;
                }
            }
        </style>
        {% endmacro %}
        {% macro script(this, kwargs) %}
        (function() {
            const map = {{ this._parent.get_name() }};
            const defaultSize = {{ this.default_size }};
            const sizeOptions = {{ this.size_options_json | safe }};
            const layerPayloads = {{ this.layer_payloads_json | safe }};
            const colorRamp = {{ this.color_ramp_json | safe }};
            const legendGradientCss = {{ this.legend_gradient_css_json | safe }};
            const fineDetailSizes = new Set({{ this.fine_detail_sizes_json | safe }});
            const fineDetailZoomThreshold = {{ this.fine_detail_zoom_threshold }};
            const isCoarsePointer = window.matchMedia("(pointer: coarse)").matches || navigator.maxTouchPoints > 0;
            const layerCache = new Map();
            const layerCacheLimit = {{ this.layer_cache_limit }};
            let activeLayer = null;
            let activeSize = defaultSize;
            let pendingSize = defaultSize;
            let scheduledFrame = null;
            let sliderValueEl = null;
            let legendTicksEl = null;
            let sliderStatusEl = null;
            let activePopupLayer = null;

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

            function buildDetailHtml(props) {
                return `<strong>Hex ID</strong>: ${props.h}<br><strong>Instances</strong>: ${props.c}`;
            }

            function shouldShowFineDetailHint(size) {
                return fineDetailSizes.has(Number(size)) && map.getZoom() < fineDetailZoomThreshold;
            }

            function buildReadyStatus(size) {
                const base = `${size} m hexes loaded`;
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

            function onEachFeature(feature, layer, payload, size) {
                const props = feature.properties;
                const detailHtml = buildDetailHtml(props);
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
                if (!sliderStatusEl) {
                    return;
                }
                sliderStatusEl.textContent = message;
                sliderStatusEl.dataset.state = state;
            }

            function getPayload(size) {
                const key = String(size);
                return layerPayloads[key] || null;
            }

            function refreshActiveLayerStyle() {
                if (!activeLayer) {
                    return;
                }
                const payload = getPayload(activeSize);
                activeLayer.setStyle((feature) => styleFeature(feature, payload, activeSize));
                if (activePopupLayer) {
                    activePopupLayer.setStyle(popupStyle(activePopupLayer.feature, payload));
                }
                updateStatus(buildReadyStatus(activeSize), "ready");
            }

            function getLayer(size) {
                const key = String(size);
                if (layerCache.has(key)) {
                    const cached = layerCache.get(key);
                    layerCache.delete(key);
                    layerCache.set(key, cached);
                    return cached;
                }

                const payload = getPayload(key);
                if (!payload) {
                    throw new Error(`Missing payload for hex size ${key}`);
                }
                const layer = L.geoJSON(payload.g, {
                    style: (feature) => styleFeature(feature, payload, key),
                    onEachFeature: (feature, featureLayer) =>
                        onEachFeature(feature, featureLayer, payload, key)
                });
                layerCache.set(key, layer);

                while (layerCache.size > layerCacheLimit) {
                    const oldestKey = layerCache.keys().next().value;
                    if (oldestKey === String(activeSize)) {
                        const activeEntry = layerCache.get(oldestKey);
                        layerCache.delete(oldestKey);
                        layerCache.set(oldestKey, activeEntry);
                        continue;
                    }
                    layerCache.delete(oldestKey);
                }

                return layer;
            }

            function updateLegend(size) {
                const payload = getPayload(size);
                if (!legendTicksEl || !payload) {
                    return;
                }
                legendTicksEl.innerHTML = buildLegendTicks(payload.n, payload.x)
                    .map((tick) => `<span>${tick}</span>`)
                    .join("");
            }

            function setActiveSize(size) {
                const key = String(size);
                if (!getPayload(key)) {
                    return;
                }
                if (Number(size) === activeSize && activeLayer) {
                    return;
                }

                updateLegend(key);
                updateStatus(`Rendering ${key} m hexes...`, "loading");
                try {
                    const nextLayer = getLayer(key);
                    closeDetailSurfaces();
                    if (activeLayer) {
                        map.removeLayer(activeLayer);
                    }
                    activeLayer = nextLayer;
                    activeLayer.addTo(map);
                    activeSize = Number(size);

                    if (sliderValueEl) {
                        sliderValueEl.textContent = `${activeSize} m`;
                    }

                    updateStatus(buildReadyStatus(key), "ready");
                } catch (error) {
                    console.error("Failed to load hex layer", key, error);
                    updateStatus(`Unable to render ${key} m layer`, "error");
                }
            }

            function scheduleActiveSize(size) {
                pendingSize = Number(size);
                if (scheduledFrame !== null) {
                    return;
                }
                scheduledFrame = window.requestAnimationFrame(() => {
                    scheduledFrame = null;
                    setActiveSize(pendingSize);
                });
            }

            const SliderControl = L.Control.extend({
                options: { position: "bottomleft" },
                onAdd() {
                    const container = L.DomUtil.create("div", "hex-size-control leaflet-control");
                    L.DomEvent.disableClickPropagation(container);
                    L.DomEvent.disableScrollPropagation(container);

                    const title = L.DomUtil.create("div", "hex-size-title", container);
                    title.textContent = "Hex Size";

                    sliderValueEl = L.DomUtil.create("div", "hex-size-value", container);
                    sliderValueEl.textContent = `${defaultSize} m`;

                    sliderStatusEl = L.DomUtil.create("div", "hex-size-status", container);
                    sliderStatusEl.textContent = "Ready";
                    sliderStatusEl.dataset.state = "ready";

                    const slider = L.DomUtil.create("input", "hex-size-slider", container);
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

                    const endpoints = L.DomUtil.create("div", "hex-size-endpoints", container);
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
                    const title = L.DomUtil.create("div", "hex-legend-title", container);
                    title.textContent = "Instances Per Occupied Hex";

                    const bar = L.DomUtil.create("div", "hex-legend-bar", container);
                    bar.style.background = legendGradientCss;

                    legendTicksEl = L.DomUtil.create("div", "hex-legend-ticks", container);
                    const note = L.DomUtil.create("div", "hex-legend-note", container);
                    note.textContent = "Legend rescales to the selected hex size.";
                    updateLegend(defaultSize);

                    return container;
                }
            });

            new LegendControl().addTo(map);
            new SliderControl().addTo(map);
            map.on("zoomend", refreshActiveLayerStyle);
            setActiveSize(defaultSize);
        })();
        {% endmacro %}
        """
    )

    def __init__(
        self,
        layer_payloads: dict[str, dict],
        size_options: tuple[int, ...],
        default_size: int,
        legend_gradient_css: str,
    ) -> None:
        super().__init__()
        self._name = "HexSizeSliderControl"
        self.layer_payloads_json = json.dumps(layer_payloads, separators=(",", ":"))
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
        parsed_dates = pd.to_datetime(
            df[date_col], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
        )
        if parsed_dates.isna().any():
            fallback_mask = parsed_dates.isna()
            fallback_dates = pd.to_datetime(
                df.loc[fallback_mask, date_col], errors="coerce"
            )
            parsed_dates.loc[fallback_mask] = fallback_dates
            print(f"Date parse fallbacks used: {int(fallback_dates.notna().sum())}")
        print(f"Rows with unparsed dates: {int(parsed_dates.isna().sum())}")
        df[date_col] = parsed_dates
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

    slider_layer_payloads: dict[str, dict] = {}
    gdf_hex: gpd.GeoDataFrame | None = None
    hex_gdf_wgs: gpd.GeoDataFrame | None = None

    for hex_size_m in HEX_SIZE_OPTIONS:
        layer_gdf_hex, layer_hex_gdf_wgs = build_hex_layer(gdf_m, hex_size_m)
        layer_payload = build_layer_payload(layer_hex_gdf_wgs)
        slider_layer_payloads[str(hex_size_m)] = layer_payload

        if hex_size_m == DEFAULT_HEX_SIZE_M:
            gdf_hex = layer_gdf_hex
            hex_gdf_wgs = layer_hex_gdf_wgs

    if gdf_hex is None or hex_gdf_wgs is None:
        raise RuntimeError(f"Default hex size {DEFAULT_HEX_SIZE_M}m was not generated.")

    print(f"Incidents kept: {len(gdf_hex)}")
    print(f"Occupied hexes at {DEFAULT_HEX_SIZE_M}m: {len(hex_gdf_wgs)}")
    print(
        f"Count integrity check (sum hex counts at {DEFAULT_HEX_SIZE_M}m): "
        f"{int(hex_gdf_wgs['count'].sum())}"
    )

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
    hex_time_season.to_csv(OUT_TIME_SEASON, index=False)

    # Build folium map
    center_lat = float(df[lat_col].median())
    center_lon = float(df[lon_col].median())
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
        HexSizeSliderControl(
            layer_payloads=slider_layer_payloads,
            size_options=HEX_SIZE_OPTIONS,
            default_size=DEFAULT_HEX_SIZE_M,
            legend_gradient_css=legend_gradient_css,
        )
    )
    hex_map.fit_bounds(CHICAGO_VIEW_BOUNDS)
    hex_map.save(OUT_HEX_MAP)
    finalize_map_html(OUT_HEX_MAP)

    print(f"Saved {OUT_HEX_MAP}")
    print(f"Saved {OUT_HEX_COUNTS}")
    print(f"Saved {OUT_INCIDENTS}")
    print(f"Saved {OUT_TIME_SEASON}")
    print(
        f"Built slider-controlled map with sizes {HEX_SIZE_MIN_M}m to {HEX_SIZE_MAX_M}m "
        f"in {HEX_SIZE_STEP_M}m increments"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
