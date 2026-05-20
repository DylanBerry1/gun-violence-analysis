"""
Co-location figures: side-by-side hex choropleths showing homicide density
alongside protective infrastructure density per 500 m hex cell.

Demonstrates that schools, places of worship, playgrounds, community centres,
social facilities, and parks co-locate with high-homicide hex cells — not
because they cause violence, but because these institutions are sited where
need is greatest (South & West Side) and in densely populated residential areas.

Outputs (reports/figures/):
- colocation_<infra_type>_vs_homicide.png    per-type side-by-side
- colocation_panel_protective_vs_homicide.png  combined 3×2 panel

Run:
    python3 src/build_colocation_figures.py
"""

import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon

plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_OUT = ROOT_DIR / "data" / "processed"
FIG_DIR = ROOT_DIR / "reports" / "figures"

MERGED_CSV = DATA_OUT / "crime_infrastructure_hex_merged.csv"

HEX_SIZE_M = 500

# Protective infrastructure types to show, ordered by Spearman ρ with homicide
COLOCATION_TYPES = [
    ("infra_place_of_worship", "Places of Worship"),
    ("infra_school", "Schools"),
    ("infra_playground", "Playgrounds"),
    ("infra_social_facility", "Social Facilities"),
    ("infra_park", "Parks"),
    ("infra_community_centre", "Community Centres"),
]

# Chicago bounding box for the map viewport
CHICAGO_BOUNDS = {
    "lat_min": 41.63,
    "lat_max": 42.02,
    "lon_min": -87.94,
    "lon_max": -87.52,
}


# ── Hex geometry reconstruction ─────────────────────────────────────────────


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
    """Reconstruct hex polygons from hex_id column and return a WGS84 GeoDataFrame."""
    polygons = []
    for hex_id in df["hex_id"]:
        q, r = hex_id_to_qr(hex_id)
        cx, cy = axial_to_center_xy(q, r, HEX_SIZE_M)
        polygons.append(hex_polygon_from_center(cx, cy, HEX_SIZE_M))
    gdf = gpd.GeoDataFrame(df.copy(), geometry=polygons, crs="EPSG:3857")
    return gdf.to_crs(epsg=4326)


# ── Plotting ─────────────────────────────────────────────────────────────────


def _make_hex_choropleth(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    col: str,
    cmap: str,
    title: str,
    label: str,
    vmax_override: float | None = None,
) -> None:
    """Render a hex choropleth on the given axes."""
    values = gdf[col].values.astype(float)
    vmin = 0
    vmax = vmax_override if vmax_override is not None else max(values.max(), 1)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    # Draw each hex as a matplotlib polygon
    patches = []
    colours = []
    for idx, row in gdf.iterrows():
        poly = row.geometry
        if poly.is_empty:
            continue
        xs, ys = poly.exterior.xy
        verts = list(zip(xs, ys))
        patches.append(MplPolygon(verts, closed=True))
        colours.append(colormap(norm(row[col])))

    pc = PatchCollection(patches, facecolors=colours, edgecolors="none", linewidths=0)
    ax.add_collection(pc)

    # Set map extent
    ax.set_xlim(CHICAGO_BOUNDS["lon_min"], CHICAGO_BOUNDS["lon_max"])
    ax.set_ylim(CHICAGO_BOUNDS["lat_min"], CHICAGO_BOUNDS["lat_max"])
    ax.set_aspect("auto")

    # Title and colorbar
    ax.set_title(title, fontsize=14, pad=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04, shrink=0.8)
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=10)

    # Clean up axes
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=10)

    # Light grid
    ax.grid(True, alpha=0.15, linewidth=0.3)


def plot_side_by_side(
    gdf: gpd.GeoDataFrame,
    infra_col: str,
    infra_label: str,
    out_path: Path,
) -> None:
    """Create a side-by-side figure: homicide (left) vs infrastructure (right)."""
    fig, (ax_hom, ax_inf) = plt.subplots(
        1, 2, figsize=(18, 7), facecolor="#fafafa"
    )
    fig.patch.set_facecolor("#fafafa")

    _make_hex_choropleth(
        ax_hom,
        gdf,
        "homicide",
        "YlOrRd",
        "Homicide Count per Hex",
        "Homicide count",
    )

    _make_hex_choropleth(
        ax_inf,
        gdf,
        infra_col,
        "YlGnBu",
        f"{infra_label} Count per Hex",
        f"{infra_label} count",
    )

    fig.suptitle(
        f"Co-location: Homicides  vs.  {infra_label}\n"
        f"(500 m hexagonal grid, Chicago 2001–2026)",
        fontsize=14,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_panel(
    gdf: gpd.GeoDataFrame,
    infra_types: list[tuple[str, str]],
    out_path: Path,
) -> None:
    """Create a 3×2 panel with homicide on the left, 6 infra types on right side."""
    n = len(infra_types)
    nrows = 3
    ncols = 4  # two pairs: each pair = (homicide, infra)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(28, 18), facecolor="#fafafa"
    )
    fig.patch.set_facecolor("#fafafa")

    for i, (infra_col, infra_label) in enumerate(infra_types):
        row = i // 2
        pair_offset = (i % 2) * 2  # 0 for left pair, 2 for right pair
        ax_hom = axes[row, pair_offset]
        ax_inf = axes[row, pair_offset + 1]

        _make_hex_choropleth(
            ax_hom,
            gdf,
            "homicide",
            "YlOrRd",
            "Homicides",
            "Count",
        )

        _make_hex_choropleth(
            ax_inf,
            gdf,
            infra_col,
            "YlGnBu",
            infra_label,
            "Count",
        )

        # Add panel label
        panel_letter = chr(ord("a") + i)
        ax_hom.text(
            -0.02, 1.08,
            f"({panel_letter})",
            transform=ax_hom.transAxes,
            fontsize=14,
            va="top",
        )

    fig.suptitle(
        "Appendix A.1 — Co-location of Homicides and Protective Infrastructure\n"
        "(500 m hexagonal grid, Chicago 2001–2026)",
        fontsize=16,
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not MERGED_CSV.exists():
        print(f"ERROR: Missing {MERGED_CSV}")
        print("Run  python3 src/build_correlation_analysis.py  first.")
        return 1

    print("Loading merged hex table …")
    df = pd.read_csv(MERGED_CSV)
    print(f"  {len(df):,} hex cells loaded")

    print("Reconstructing hex geometries …")
    gdf = build_hex_geodataframe(df)
    print(f"  GeoDataFrame ready ({len(gdf):,} hexes)")

    # Individual side-by-side figures
    print("\nGenerating individual co-location figures …")
    for infra_col, infra_label in COLOCATION_TYPES:
        if infra_col not in gdf.columns:
            print(f"  SKIP {infra_col} (column not found)")
            continue
        slug = infra_col.replace("infra_", "")
        out = FIG_DIR / f"colocation_{slug}_vs_homicide.png"
        plot_side_by_side(gdf, infra_col, infra_label, out)

    # Combined panel figure
    print("\nGenerating combined panel figure …")
    available = [
        (col, label)
        for col, label in COLOCATION_TYPES
        if col in gdf.columns
    ]
    if available:
        plot_panel(gdf, available, FIG_DIR / "colocation_panel_protective_vs_homicide.png")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
