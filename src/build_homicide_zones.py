"""
Spatial homicide zone analysis — Getis-Ord Gi* hotspot detection.

Identifies statistically significant homicide hotspot clusters on the
existing 500 m hex grid, groups adjacent hot hexes into named zones,
labels them by Chicago community area, and produces static figures plus
an interactive Folium map.

Outputs:
  data/processed/hex/homicide_gi_star_hex.csv
  reports/figures/homicide_zones/zone_summary.csv
  reports/figures/homicide_zones/gi_star_hotspot_map.png
  reports/figures/homicide_zones/homicide_zones_map.png
  reports/figures/homicide_zones/zone_detail_panel.png
  reports/figures/homicide_zones/homicide_zones_interactive.html

Run:
    python3 src/build_homicide_zones.py
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from scipy import ndimage
from scipy.stats import norm
from shapely.geometry import Polygon
from shapely.ops import unary_union

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.size": 14,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    }
)

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[1]
HEX_COUNTS_CSV = (
    ROOT_DIR / "data" / "processed" / "hex" / "chicago_homicides_hex_counts.csv"
)
COMMUNITY_GEOJSON = ROOT_DIR / "data" / "raw" / "chicago_community_areas.geojson"
GI_STAR_CSV = ROOT_DIR / "data" / "processed" / "hex" / "homicide_gi_star_hex.csv"
ZONE_DIR = ROOT_DIR / "reports" / "figures" / "homicide_zones"

# ── Constants ────────────────────────────────────────────────────────────────

HEX_SIZE_M = 500

CHICAGO_BOUNDS = {
    "lat_min": 41.63,
    "lat_max": 42.02,
    "lon_min": -87.94,
    "lon_max": -87.52,
}

# Gi* z-score thresholds for hotspot classification
GI_THRESHOLDS = {
    "Hotspot 99%": 2.576,
    "Hotspot 95%": 1.960,
    "Hotspot 90%": 1.645,
}

# Colours for each tier (warm = hot, cool = cold, grey = not significant)
TIER_COLORS = {
    "Hotspot 99%": "#d73027",
    "Hotspot 95%": "#fc8d59",
    "Hotspot 90%": "#fee08b",
    "Not Significant": "#d9d9d9",
    "Coldspot 90%": "#91bfdb",
    "Coldspot 95%": "#4575b4",
    "Coldspot 99%": "#313695",
}

# Minimum number of hexes for a zone to be retained
MIN_ZONE_HEXES = 3

ZONE_CMAP_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
    "#e6beff",
]


# ── Hex geometry ─────────────────────────────────────────────────────────────


def hex_id_to_qr(hex_id: str) -> tuple[int, int]:
    """Parse hex_id string 'q_r' back to axial coordinates."""
    parts = hex_id.split("_")
    return int(parts[0]), int(parts[1])


def axial_to_center_xy(q: int, r: int, size: float) -> tuple[float, float]:
    """Convert axial hex coords to EPSG:3857 centre."""
    x = size * math.sqrt(3.0) * (q + r / 2.0)
    y = size * 1.5 * r
    return x, y


def hex_polygon_from_center(cx: float, cy: float, size: float) -> Polygon:
    """Build a flat-top hexagon polygon in projected coords."""
    coords = []
    for i in range(6):
        angle = 2 * math.pi * (i + 0.5) / 6.0
        coords.append((cx + size * math.cos(angle), cy + size * math.sin(angle)))
    return Polygon(coords)


def build_hex_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Reconstruct hex polygons from hex_id and return WGS84 GeoDataFrame."""
    polygons = []
    for hex_id in df["hex_id"]:
        q, r = hex_id_to_qr(hex_id)
        cx, cy = axial_to_center_xy(q, r, HEX_SIZE_M)
        polygons.append(hex_polygon_from_center(cx, cy, HEX_SIZE_M))
    gdf = gpd.GeoDataFrame(df.copy(), geometry=polygons, crs="EPSG:3857")
    return gdf.to_crs(epsg=4326)


# ── Spatial weights & Gi* ────────────────────────────────────────────────────


def build_adjacency(gdf_proj: gpd.GeoDataFrame) -> dict[int, list[int]]:
    """Build queen-contiguity adjacency dict from hex polygons (projected CRS).

    Two hexes are neighbours if their polygons touch or overlap.
    Uses a small buffer to handle floating-point boundary precision.
    """
    sindex = gdf_proj.sindex
    neighbours: dict[int, list[int]] = defaultdict(list)
    buffer_distance = HEX_SIZE_M * 0.05  # small tolerance

    for i, geom_i in enumerate(gdf_proj.geometry):
        buffered = geom_i.buffer(buffer_distance)
        candidates = list(sindex.intersection(buffered.bounds))
        for j in candidates:
            if j <= i:
                continue
            if buffered.intersects(gdf_proj.geometry.iloc[j]):
                neighbours[i].append(j)
                neighbours[j].append(i)

    # Ensure every hex appears even if isolated
    for i in range(len(gdf_proj)):
        if i not in neighbours:
            neighbours[i] = []

    return dict(neighbours)


def compute_gi_star(
    values: np.ndarray, adjacency: dict[int, list[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Getis-Ord Gi* statistic for each location.

    Gi* includes the location itself in its local sum (unlike Gi which
    excludes it).  This is the standard formulation for hotspot detection.

    Returns z-scores and two-tailed p-values.
    """
    n = len(values)
    x_bar = values.mean()
    s = values.std(ddof=0)

    if s == 0:
        return np.zeros(n), np.ones(n)

    z_scores = np.zeros(n)
    p_values = np.ones(n)

    for i in range(n):
        # Gi* includes self → w_ij = 1 for j in {i} ∪ neighbours(i)
        neighbours_and_self = [i] + adjacency.get(i, [])
        w_count = len(neighbours_and_self)
        local_sum = values[neighbours_and_self].sum()

        numerator = local_sum - x_bar * w_count
        denominator = s * math.sqrt((n * w_count - w_count**2) / (n - 1))

        if denominator > 0:
            z_scores[i] = numerator / denominator
            p_values[i] = 2 * (1 - norm.cdf(abs(z_scores[i])))

    return z_scores, p_values


def classify_hotspot(z: float) -> str:
    """Classify a hex based on its Gi* z-score."""
    if z >= GI_THRESHOLDS["Hotspot 99%"]:
        return "Hotspot 99%"
    if z >= GI_THRESHOLDS["Hotspot 95%"]:
        return "Hotspot 95%"
    if z >= GI_THRESHOLDS["Hotspot 90%"]:
        return "Hotspot 90%"
    if z <= -GI_THRESHOLDS["Hotspot 99%"]:
        return "Coldspot 99%"
    if z <= -GI_THRESHOLDS["Hotspot 95%"]:
        return "Coldspot 95%"
    if z <= -GI_THRESHOLDS["Hotspot 90%"]:
        return "Coldspot 90%"
    return "Not Significant"


# ── Zone grouping ────────────────────────────────────────────────────────────


def group_zones(
    gdf: gpd.GeoDataFrame,
    adjacency: dict[int, list[int]],
    min_hexes: int = MIN_ZONE_HEXES,
) -> pd.Series:
    """Group adjacent significant hotspot hexes into zones via connected components.

    Only hexes classified as Hotspot 95% or 99% are included.
    Returns a Series mapping each row index to a zone_id (0 = not in any zone).
    """
    is_hot = gdf["hotspot_tier"].isin(["Hotspot 99%", "Hotspot 95%"])
    hot_indices = set(gdf.index[is_hot])

    # Build adjacency matrix for hot hexes only
    n = len(gdf)
    labels_array = np.zeros(n, dtype=int)

    # Use manual BFS for connected components among hot hexes
    visited: set[int] = set()
    zone_id = 0

    for start_idx in sorted(hot_indices):
        if start_idx in visited:
            continue
        # BFS
        zone_id += 1
        component: list[int] = []
        queue = [start_idx]
        visited.add(start_idx)
        while queue:
            current = queue.pop(0)
            component.append(current)
            for neighbour in adjacency.get(current, []):
                if neighbour in hot_indices and neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

        if len(component) >= min_hexes:
            for idx in component:
                labels_array[idx] = zone_id
        # Small fragments → leave as 0

    return pd.Series(labels_array, index=gdf.index, name="zone_id")


# ── Community area labelling ─────────────────────────────────────────────────


def load_community_areas() -> gpd.GeoDataFrame:
    """Load Chicago community area boundaries."""
    ca = gpd.read_file(COMMUNITY_GEOJSON)
    ca["area_numbe"] = pd.to_numeric(ca["area_numbe"], errors="coerce").astype(int)
    ca["community"] = ca["community"].str.title()
    ca = ca.to_crs(epsg=4326)
    return ca[["area_numbe", "community", "geometry"]].copy()


def label_zones(
    gdf: gpd.GeoDataFrame, ca: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Build a zone summary with community area labels.

    For each zone, dissolve its hex polygons, find which community areas
    the zone overlaps, and pick the top 1-2 names by overlap area.
    """
    zones = gdf[gdf["zone_id"] > 0].copy()
    if zones.empty:
        return pd.DataFrame()

    records = []
    for zid in sorted(zones["zone_id"].unique()):
        zone_hexes = zones[zones["zone_id"] == zid]
        zone_polygon = unary_union(zone_hexes.geometry)

        # Spatial intersection with community areas
        overlaps = []
        for _, ca_row in ca.iterrows():
            intersection = zone_polygon.intersection(ca_row.geometry)
            if not intersection.is_empty:
                overlaps.append(
                    {
                        "community": ca_row["community"],
                        "area_numbe": ca_row["area_numbe"],
                        "overlap_area": intersection.area,
                    }
                )

        overlaps_df = pd.DataFrame(overlaps)
        if overlaps_df.empty:
            ca_label = "Unknown"
        else:
            overlaps_df = overlaps_df.sort_values("overlap_area", ascending=False)
            top_cas = overlaps_df.head(2)["community"].tolist()
            ca_label = " / ".join(top_cas)

        centroid = zone_polygon.centroid
        records.append(
            {
                "zone_id": zid,
                "zone_label": ca_label,
                "hex_count": len(zone_hexes),
                "total_homicides": int(zone_hexes["count"].sum()),
                "mean_count": round(zone_hexes["count"].mean(), 1),
                "max_count": int(zone_hexes["count"].max()),
                "centroid_lat": centroid.y,
                "centroid_lon": centroid.x,
            }
        )

    summary = pd.DataFrame(records)
    summary = summary.sort_values("total_homicides", ascending=False).reset_index(
        drop=True
    )
    # Re-rank zone_id by total homicides (Zone 1 = worst)
    summary["zone_rank"] = range(1, len(summary) + 1)
    return summary


# ── Static figures ───────────────────────────────────────────────────────────


def _draw_community_outlines(
    ax: plt.Axes,
    ca: gpd.GeoDataFrame,
    label: bool = False,
    alpha: float = 0.35,
) -> None:
    """Draw community area outlines on an axes."""
    import matplotlib.patheffects as pe
    import textwrap
    
    for _, row in ca.iterrows():
        geom = row.geometry
        if geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            polys = [geom]
        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color="#555555", linewidth=0.5, alpha=alpha)
        if label:
            centroid = geom.centroid
            wrapped_text = "\n".join(textwrap.wrap(row["community"], width=11))
            ax.text(
                centroid.x,
                centroid.y,
                wrapped_text,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                color="#222222",
                alpha=0.9,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white", alpha=0.85)],
            )


def _set_chicago_extent(ax: plt.Axes) -> None:
    """Set axes limits to Chicago bounds."""
    ax.set_xlim(CHICAGO_BOUNDS["lon_min"], CHICAGO_BOUNDS["lon_max"])
    ax.set_ylim(CHICAGO_BOUNDS["lat_min"], CHICAGO_BOUNDS["lat_max"])
    ax.set_aspect("auto")


def plot_gi_star_choropleth(
    gdf: gpd.GeoDataFrame,
    ca: gpd.GeoDataFrame,
    out_path: Path,
) -> None:
    """Choropleth of hex grid coloured by Gi* hotspot tier."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 14), facecolor="#fafafa")
    ax.set_facecolor("#f0ece3")

    # Draw each hex coloured by tier
    for tier, color in TIER_COLORS.items():
        subset = gdf[gdf["hotspot_tier"] == tier]
        if subset.empty:
            continue
        patches = []
        for _, row in subset.iterrows():
            poly = row.geometry
            if poly.is_empty:
                continue
            xs, ys = poly.exterior.xy
            patches.append(MplPolygon(list(zip(xs, ys)), closed=True))
        pc = PatchCollection(
            patches,
            facecolors=color,
            edgecolors="none",
            linewidths=0,
            alpha=0.85,
        )
        ax.add_collection(pc)

    _draw_community_outlines(ax, ca, label=True, alpha=0.25)
    _set_chicago_extent(ax)

    ax.set_title(
        "Getis-Ord Gi* Hotspot Analysis — Homicides\n"
        "(500 m hex grid, Chicago 2001–2026)",
        fontsize=15,
        pad=12,
        color="#222222",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=10)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=color, edgecolor="none", label=tier)
        for tier, color in TIER_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=10,
        title="Hotspot Tier",
        title_fontsize=11,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_zone_map(
    gdf: gpd.GeoDataFrame,
    ca: gpd.GeoDataFrame,
    zone_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """Map of named homicide zones with boundaries and labels."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 14), facecolor="#fafafa")
    ax.set_facecolor("#f0ece3")

    # Community area outlines as base layer
    _draw_community_outlines(ax, ca, label=True, alpha=0.20)

    # Draw non-zone hexes in light grey
    non_zone = gdf[gdf["zone_id"] == 0]
    if not non_zone.empty:
        patches = []
        for _, row in non_zone.iterrows():
            poly = row.geometry
            if poly.is_empty:
                continue
            xs, ys = poly.exterior.xy
            patches.append(MplPolygon(list(zip(xs, ys)), closed=True))
        pc = PatchCollection(
            patches,
            facecolors="#e0e0e0",
            edgecolors="none",
            linewidths=0,
            alpha=0.4,
        )
        ax.add_collection(pc)

    # Build zone_id → rank + colour mapping
    zone_colour_map = {}
    zone_label_map = {}
    for _, zrow in zone_summary.iterrows():
        rank = int(zrow["zone_rank"])
        colour_idx = (rank - 1) % len(ZONE_CMAP_COLORS)
        zone_colour_map[zrow["zone_id"]] = ZONE_CMAP_COLORS[colour_idx]
        zone_label_map[zrow["zone_id"]] = f"Zone {rank}: {zrow['zone_label']}"

    # Draw zone hexes
    zone_hexes = gdf[gdf["zone_id"] > 0]
    for zid in sorted(zone_hexes["zone_id"].unique()):
        subset = zone_hexes[zone_hexes["zone_id"] == zid]
        colour = zone_colour_map.get(zid, "#999999")
        patches = []
        for _, row in subset.iterrows():
            poly = row.geometry
            if poly.is_empty:
                continue
            xs, ys = poly.exterior.xy
            patches.append(MplPolygon(list(zip(xs, ys)), closed=True))
        pc = PatchCollection(
            patches,
            facecolors=colour,
            edgecolors="white",
            linewidths=0.3,
            alpha=0.80,
        )
        ax.add_collection(pc)

        # Draw zone boundary
        zone_polygon = unary_union(subset.geometry)
        if zone_polygon.geom_type == "MultiPolygon":
            boundary_polys = list(zone_polygon.geoms)
        else:
            boundary_polys = [zone_polygon]
        for bp in boundary_polys:
            xs, ys = bp.exterior.xy
            ax.plot(xs, ys, color=colour, linewidth=2.0, alpha=0.9)

    # Label zones
    for _, zrow in zone_summary.iterrows():
        rank = int(zrow["zone_rank"])
        label = f"Z{rank}"
        ax.annotate(
            label,
            xy=(zrow["centroid_lon"], zrow["centroid_lat"]),
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=zone_colour_map.get(zrow["zone_id"], "#333"),
                alpha=0.85,
                edgecolor="white",
                linewidth=1.0,
            ),
        )

    _set_chicago_extent(ax)

    ax.set_title(
        "Homicide Hotspot Zones\n"
        "(Gi* p < 0.05, grouped adjacent hexes, Chicago 2001–2026)",
        fontsize=15,
        pad=12,
        color="#222222",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=10)

    # Legend for top zones
    legend_handles = []
    for _, zrow in zone_summary.head(12).iterrows():
        zid = zrow["zone_id"]
        rank = int(zrow["zone_rank"])
        colour = zone_colour_map.get(zid, "#999")
        n_hom = int(zrow["total_homicides"])
        legend_handles.append(
            mpatches.Patch(
                facecolor=colour,
                edgecolor="white",
                linewidth=0.5,
                label=f"Z{rank}: {zrow['zone_label']} ({n_hom:,})",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=8,
        title="Homicide Zones (total homicides)",
        title_fontsize=9,
        framealpha=0.92,
        edgecolor="#cccccc",
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_zone_detail_panel(
    gdf: gpd.GeoDataFrame,
    ca: gpd.GeoDataFrame,
    zone_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """Multi-panel figure showing zoomed-in views of the top zones."""
    top_n = min(6, len(zone_summary))
    if top_n == 0:
        print("  SKIP zone detail panel — no zones found")
        return

    ncols = 3
    nrows = (top_n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(20, 6 * nrows + 1), facecolor="#fafafa"
    )
    if nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, "__len__") else [row])]

    for panel_idx in range(top_n):
        ax = axes_flat[panel_idx]
        ax.set_facecolor("#f5f3ee")

        zrow = zone_summary.iloc[panel_idx]
        zid = zrow["zone_id"]
        rank = int(zrow["zone_rank"])
        colour = ZONE_CMAP_COLORS[(rank - 1) % len(ZONE_CMAP_COLORS)]

        zone_hexes = gdf[gdf["zone_id"] == zid]
        zone_polygon = unary_union(zone_hexes.geometry)
        minx, miny, maxx, maxy = zone_polygon.bounds
        pad = 0.015
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("auto")

        # Community area outlines in view
        _draw_community_outlines(ax, ca, label=False, alpha=0.3)

        # Nearby non-zone hexes for context (within bounds)
        all_in_view = gdf.cx[minx - pad : maxx + pad, miny - pad : maxy + pad]
        context = all_in_view[all_in_view["zone_id"] != zid]
        if not context.empty:
            patches = []
            for _, row in context.iterrows():
                poly = row.geometry
                if poly.is_empty:
                    continue
                xs, ys = poly.exterior.xy
                patches.append(MplPolygon(list(zip(xs, ys)), closed=True))
            pc = PatchCollection(
                patches,
                facecolors="#e0e0e0",
                edgecolors="#cccccc",
                linewidths=0.3,
                alpha=0.5,
            )
            ax.add_collection(pc)

        # Zone hexes coloured by count
        counts = zone_hexes["count"].values.astype(float)
        vmin_c, vmax_c = counts.min(), max(counts.max(), 1)
        norm_c = mcolors.Normalize(vmin=vmin_c, vmax=vmax_c)
        cmap = plt.get_cmap("YlOrRd")

        patches = []
        colours = []
        for _, row in zone_hexes.iterrows():
            poly = row.geometry
            if poly.is_empty:
                continue
            xs, ys = poly.exterior.xy
            patches.append(MplPolygon(list(zip(xs, ys)), closed=True))
            colours.append(cmap(norm_c(row["count"])))

        pc = PatchCollection(
            patches, facecolors=colours, edgecolors="white", linewidths=0.5
        )
        ax.add_collection(pc)

        # Zone boundary
        if zone_polygon.geom_type == "MultiPolygon":
            for bp in zone_polygon.geoms:
                xs, ys = bp.exterior.xy
                ax.plot(xs, ys, color=colour, linewidth=2.5, alpha=0.9)
        else:
            xs, ys = zone_polygon.exterior.xy
            ax.plot(xs, ys, color=colour, linewidth=2.5, alpha=0.9)

        ax.set_title(
            f"Zone {rank}: {zrow['zone_label']}\n"
            f"{int(zrow['total_homicides']):,} homicides · "
            f"{int(zrow['hex_count'])} hexes · "
            f"mean {zrow['mean_count']}/hex",
            fontsize=12,
            pad=8,
        )
        ax.tick_params(labelsize=8)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm_c)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.03, shrink=0.7)
        cbar.set_label("Homicides", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # Hide empty panels
    for idx in range(top_n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Top Homicide Hotspot Zones — Detail Views\n"
        "(500 m hex grid, Gi* p < 0.05, Chicago 2001–2026)",
        fontsize=16,
        y=1.01,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── Interactive Folium map ───────────────────────────────────────────────────


def build_interactive_map(
    gdf: gpd.GeoDataFrame,
    ca: gpd.GeoDataFrame,
    zone_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """Build an interactive Folium map with zone boundaries and hex popups."""
    center_lat = (CHICAGO_BOUNDS["lat_min"] + CHICAGO_BOUNDS["lat_max"]) / 2
    center_lon = (CHICAGO_BOUNDS["lon_min"] + CHICAGO_BOUNDS["lon_max"]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB dark_matter",
    )

    # Community area outlines layer
    ca_style = {
        "fillColor": "transparent",
        "color": "#888888",
        "weight": 1,
        "fillOpacity": 0,
    }
    folium.GeoJson(
        ca,
        name="Community Areas",
        style_function=lambda feature: ca_style,
        tooltip=folium.GeoJsonTooltip(fields=["community"], aliases=["Community:"]),
    ).add_to(m)

    # Gi* tier layer — all hexes coloured by tier
    gi_layer = folium.FeatureGroup(name="Gi* Hotspot Tiers")
    for _, row in gdf.iterrows():
        tier = row["hotspot_tier"]
        colour = TIER_COLORS.get(tier, "#d9d9d9")
        poly = row.geometry
        if poly.is_empty:
            continue

        coords = [list(reversed(c)) for c in poly.exterior.coords]
        tooltip_text = (
            f"Hex: {row['hex_id']}<br>"
            f"Homicides: {int(row['count'])}<br>"
            f"Gi* z: {row['gi_z']:.2f}<br>"
            f"Tier: {tier}"
        )

        folium.Polygon(
            locations=coords,
            tooltip=tooltip_text,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.7,
            weight=0.5,
            opacity=0.8,
        ).add_to(gi_layer)

    gi_layer.add_to(m)

    # Zone boundaries layer
    zone_colour_map = {}
    for _, zrow in zone_summary.iterrows():
        rank = int(zrow["zone_rank"])
        zone_colour_map[zrow["zone_id"]] = ZONE_CMAP_COLORS[
            (rank - 1) % len(ZONE_CMAP_COLORS)
        ]

    zone_layer = folium.FeatureGroup(name="Homicide Zones")
    for _, zrow in zone_summary.iterrows():
        zid = zrow["zone_id"]
        rank = int(zrow["zone_rank"])
        colour = zone_colour_map.get(zid, "#999999")

        zone_hexes = gdf[gdf["zone_id"] == zid]
        zone_polygon = unary_union(zone_hexes.geometry)

        if zone_polygon.geom_type == "MultiPolygon":
            polys = list(zone_polygon.geoms)
        else:
            polys = [zone_polygon]

        for poly in polys:
            coords = [list(reversed(c)) for c in poly.exterior.coords]
            tooltip_text = (
                f"<b>Zone {rank}: {zrow['zone_label']}</b><br>"
                f"Homicides: {int(zrow['total_homicides']):,}<br>"
                f"Hexes: {int(zrow['hex_count'])}<br>"
                f"Mean/hex: {zrow['mean_count']}"
            )
            folium.Polygon(
                locations=coords,
                tooltip=tooltip_text,
                color=colour,
                fill=True,
                fill_color=colour,
                fill_opacity=0.25,
                weight=3,
                opacity=0.9,
            ).add_to(zone_layer)

        # Zone label marker
        folium.Marker(
            location=[zrow["centroid_lat"], zrow["centroid_lon"]],
            icon=folium.DivIcon(
                html=(
                    f'<div style="'
                    f"background:{colour};"
                    f"color:white;"
                    f"font-weight:bold;"
                    f"font-size:12px;"
                    f"padding:3px 7px;"
                    f"border-radius:4px;"
                    f"border:1px solid white;"
                    f"text-align:center;"
                    f"white-space:nowrap;"
                    f'">Z{rank}</div>'
                ),
                icon_size=(40, 24),
                icon_anchor=(20, 12),
            ),
        ).add_to(zone_layer)

    zone_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_path))
    print(f"  Saved {out_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ZONE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    if not HEX_COUNTS_CSV.exists():
        print(f"ERROR: Missing {HEX_COUNTS_CSV}")
        print("Run  python3 src/build_hex_maps.py  first.")
        return 1

    if not COMMUNITY_GEOJSON.exists():
        print(f"ERROR: Missing {COMMUNITY_GEOJSON}")
        print("Download Chicago community area boundaries first.")
        return 1

    print("Loading homicide hex counts …")
    df = pd.read_csv(HEX_COUNTS_CSV)
    print(f"  {len(df):,} hexes, {int(df['count'].sum()):,} total homicides")

    print("Loading community area boundaries …")
    ca = load_community_areas()
    print(f"  {len(ca)} community areas")

    # ── Reconstruct hex geometries ───────────────────────────────────────
    print("Reconstructing hex geometries …")
    gdf = build_hex_geodataframe(df)
    gdf_proj = gdf.to_crs(epsg=3857)
    print(f"  GeoDataFrame ready ({len(gdf):,} hexes)")

    # ── Build adjacency ─────────────────────────────────────────────────
    print("Building spatial adjacency …")
    adjacency = build_adjacency(gdf_proj)
    total_edges = sum(len(v) for v in adjacency.values()) // 2
    print(f"  {total_edges:,} edges among {len(adjacency):,} hexes")

    # ── Compute Gi* ──────────────────────────────────────────────────────
    print("Computing Getis-Ord Gi* statistics …")
    values = gdf["count"].values.astype(float)
    z_scores, p_values = compute_gi_star(values, adjacency)

    gdf["gi_z"] = z_scores
    gdf["gi_p"] = p_values
    gdf["hotspot_tier"] = [classify_hotspot(z) for z in z_scores]

    tier_counts = gdf["hotspot_tier"].value_counts()
    print("  Tier distribution:")
    for tier in TIER_COLORS:
        n = tier_counts.get(tier, 0)
        print(f"    {tier:20s}  {n:5d} hexes")

    # ── Group into zones ─────────────────────────────────────────────────
    print("Grouping adjacent hotspot hexes into zones …")
    gdf["zone_id"] = group_zones(gdf, adjacency)
    n_zones = gdf[gdf["zone_id"] > 0]["zone_id"].nunique()
    print(f"  {n_zones} zones identified")

    # ── Label zones ──────────────────────────────────────────────────────
    print("Labelling zones by community area …")
    zone_summary = label_zones(gdf, ca)
    if zone_summary.empty:
        print("  WARNING: No zones survived grouping.")
        zone_summary = pd.DataFrame(
            columns=[
                "zone_id",
                "zone_label",
                "hex_count",
                "total_homicides",
                "mean_count",
                "max_count",
                "centroid_lat",
                "centroid_lon",
                "zone_rank",
            ]
        )
    else:
        # Remap zone_ids in gdf to match ranked order
        old_to_new = dict(zip(zone_summary["zone_id"], zone_summary["zone_rank"]))
        # Keep old zone_id in summary for reference, but update gdf
        gdf["zone_id"] = gdf["zone_id"].map(lambda x: old_to_new.get(x, 0))
        zone_summary["zone_id"] = zone_summary["zone_rank"]

        print("  Top zones:")
        for _, zrow in zone_summary.head(10).iterrows():
            print(
                f"    Zone {int(zrow['zone_rank']):2d}: {zrow['zone_label']:30s}  "
                f"{int(zrow['total_homicides']):5d} homicides  "
                f"({int(zrow['hex_count'])} hexes)"
            )

    # ── Save CSV outputs ─────────────────────────────────────────────────
    print("\nSaving CSV outputs …")
    gi_out = gdf[
        ["hex_id", "hex_q", "hex_r", "count", "centroid_lat", "centroid_lon",
         "gi_z", "gi_p", "hotspot_tier", "zone_id"]
    ].copy()
    gi_out.to_csv(GI_STAR_CSV, index=False)
    print(f"  Saved {GI_STAR_CSV.name}")

    zone_summary.to_csv(ZONE_DIR / "zone_summary.csv", index=False)
    print(f"  Saved zone_summary.csv")

    # ── Generate figures ─────────────────────────────────────────────────
    print("\nGenerating Gi* hotspot choropleth …")
    plot_gi_star_choropleth(gdf, ca, ZONE_DIR / "gi_star_hotspot_map.png")

    print("Generating zone map …")
    plot_zone_map(gdf, ca, zone_summary, ZONE_DIR / "homicide_zones_map.png")

    print("Generating zone detail panel …")
    plot_zone_detail_panel(gdf, ca, zone_summary, ZONE_DIR / "zone_detail_panel.png")

    print("Generating interactive Folium map …")
    build_interactive_map(
        gdf, ca, zone_summary, ZONE_DIR / "homicide_zones_interactive.html"
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
