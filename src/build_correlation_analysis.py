"""
Spatial correlation analysis: all crime types and social infrastructure vs.
homicide, aggregated to a 500 m axial hexagonal grid (EPSG:3857).

Reads the full Chicago Crimes dataset, assigns every incident to a hex,
pivots to per-hex counts by Primary Type, then computes Spearman / Pearson
correlations against homicide counts — answering the question "which crime
types co-occur spatially with homicide?"

Outputs (reports/figures/ and data/processed/):
- crime_infrastructure_hex_merged.csv      full merged hex table
- correlation_crime_vs_homicide.png        bar chart of ρ per crime type
- correlation_matrix_all_crimes.png        heatmap across crime types
- correlation_matrix_crime_infrastructure.png
- scatter_<crime>_vs_homicides.png         top-N scatter plots
- scatter_infrastructure_vs_homicides.png
- scatter_protective_vs_homicides.png
- scatter_risk_vs_homicides.png
- top_infrastructure_correlations.png
- correlation_summary.txt

Run:
    python3 src/build_correlation_analysis.py

Input:
    data/raw/Crimes_-_2001_to_Present_20260408.csv
"""

import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_OUT = ROOT_DIR / "data" / "processed"
FIG_DIR = ROOT_DIR / "reports" / "figures"

CRIMES_CSV = DATA_RAW / "Crimes_-_2001_to_Present_20260408.csv"
INFRA_CSV = DATA_RAW / "infrastructure_locations.csv"

HEX_SIZE_M = 500

CHICAGO_BOUNDS = {
    "lat_min": 41.5,
    "lat_max": 42.1,
    "lon_min": -88.0,
    "lon_max": -87.5,
}

PROTECTIVE_TYPES = {
    "library",
    "community_centre",
    "social_facility",
    "school",
    "hospital",
    "clinic",
    "park",
    "playground",
    "recreation_ground",
    "arts_centre",
    "place_of_worship",
    "police",
    "fire_station",
}
RISK_TYPES = {
    "bar",
    "pub",
    "nightclub",
    "stripclub",
    "alcohol",
    "tobacco",
    "e-cigarette",
    "casino",
    "gambling",
    "fuel",
}

# Crime types too rare or too noisy to include in the main analysis
EXCLUDE_CRIME_TYPES = {
    "NON-CRIMINAL",
    "RITUALISM",
    "DOMESTIC VIOLENCE",
    "OTHER NARCOTIC VIOLATION",
}

TOP_N_SCATTER = 6


# ── Hex grid (same axial system as build_hex_maps.py) ────────────────────────


def cube_round(qf: np.ndarray, rf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xf, zf = qf, rf
    yf = -xf - zf
    rx, ry, rz = np.round(xf), np.round(yf), np.round(zf)
    x_diff, y_diff, z_diff = np.abs(rx - xf), np.abs(ry - yf), np.abs(rz - zf)

    x_largest = (x_diff > y_diff) & (x_diff > z_diff)
    y_largest = (~x_largest) & (y_diff > z_diff)

    rx[x_largest] = -ry[x_largest] - rz[x_largest]
    ry[y_largest] = -rx[y_largest] - rz[y_largest]
    rz[~x_largest & ~y_largest] = (
        -rx[~x_largest & ~y_largest] - ry[~x_largest & ~y_largest]
    )
    return rx.astype(int), rz.astype(int)


def assign_hex_ids(
    lats: np.ndarray,
    lons: np.ndarray,
    hex_size_m: float,
) -> pd.DataFrame:
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    x = gdf.geometry.x.to_numpy()
    y = gdf.geometry.y.to_numpy()
    sqrt3 = math.sqrt(3.0)
    qf = ((sqrt3 / 3.0) * x - (1.0 / 3.0) * y) / hex_size_m
    rf = ((2.0 / 3.0) * y) / hex_size_m
    q, r = cube_round(qf, rf)

    return pd.DataFrame(
        {
            "hex_q": q,
            "hex_r": r,
            "hex_id": pd.array([f"{qi}_{ri}" for qi, ri in zip(q, r)]),
        }
    )


# ── Data loading ─────────────────────────────────────────────────────────────


def load_crimes(path: Path) -> pd.DataFrame:
    """Load only the columns we need from the full crimes CSV."""
    usecols = ["Primary Type", "Latitude", "Longitude"]
    print(f"Reading {path.name} (this may take a minute) …")
    df = pd.read_csv(path, usecols=usecols, dtype={"Primary Type": "category"})

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    print(f"  Dropped {before - len(df):,} rows with null coordinates")

    in_bounds = (
        (df["Latitude"] >= CHICAGO_BOUNDS["lat_min"])
        & (df["Latitude"] <= CHICAGO_BOUNDS["lat_max"])
        & (df["Longitude"] >= CHICAGO_BOUNDS["lon_min"])
        & (df["Longitude"] <= CHICAGO_BOUNDS["lon_max"])
    )
    dropped = int((~in_bounds).sum())
    df = df[in_bounds].copy()
    print(f"  Dropped {dropped:,} out-of-bounds rows")
    print(
        f"  Kept {len(df):,} crime incidents across {df['Primary Type'].nunique()} types"
    )
    return df


def pivot_crimes_to_hex(df: pd.DataFrame) -> pd.DataFrame:
    """Assign hex IDs and pivot to one row per hex, one col per crime type."""
    print("Projecting to hex grid …")
    hex_info = assign_hex_ids(
        df["Latitude"].to_numpy(),
        df["Longitude"].to_numpy(),
        HEX_SIZE_M,
    )
    df = pd.concat([df.reset_index(drop=True), hex_info], axis=1)

    print("Pivoting crime counts per hex …")
    pivot = (
        df.groupby(["hex_id", "Primary Type"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    pivot.columns = [col.lower().replace(" ", "_") for col in pivot.columns]
    pivot = pivot.reset_index()
    return pivot


def load_infrastructure(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["latitude", "longitude"])
    in_bounds = (
        (df["latitude"] >= CHICAGO_BOUNDS["lat_min"])
        & (df["latitude"] <= CHICAGO_BOUNDS["lat_max"])
        & (df["longitude"] >= CHICAGO_BOUNDS["lon_min"])
        & (df["longitude"] <= CHICAGO_BOUNDS["lon_max"])
    )
    df = df[in_bounds].copy()
    hex_info = assign_hex_ids(
        df["latitude"].to_numpy(),
        df["longitude"].to_numpy(),
        HEX_SIZE_M,
    )
    df = pd.concat([df.reset_index(drop=True), hex_info], axis=1)
    return df


def pivot_infrastructure(infra_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        infra_df.groupby(["hex_id", "infrastructure_type"]).size().unstack(fill_value=0)
    )
    pivot.columns = [f"infra_{c}" for c in pivot.columns]
    pivot["infra_total"] = pivot.sum(axis=1)

    prot_cols = [
        f"infra_{t}" for t in PROTECTIVE_TYPES if f"infra_{t}" in pivot.columns
    ]
    risk_cols = [f"infra_{t}" for t in RISK_TYPES if f"infra_{t}" in pivot.columns]
    if prot_cols:
        pivot["infra_protective"] = pivot[prot_cols].sum(axis=1)
    if risk_cols:
        pivot["infra_risk"] = pivot[risk_cols].sum(axis=1)
    return pivot.reset_index()


def build_merged(crime_pivot: pd.DataFrame, infra_pivot: pd.DataFrame) -> pd.DataFrame:
    merged = crime_pivot.merge(infra_pivot, on="hex_id", how="outer").fillna(0)
    crime_cols = [c for c in crime_pivot.columns if c != "hex_id"]
    merged["total_crime"] = merged[crime_cols].sum(axis=1)
    return merged


# ── Correlation helpers ──────────────────────────────────────────────────────


def crime_vs_homicide_correlations(
    df: pd.DataFrame,
    crime_cols: list[str],
) -> pd.DataFrame:
    """Spearman ρ of each crime-type column against homicide."""
    rows = []
    for col in crime_cols:
        if col == "homicide":
            continue
        r_s, p_s = stats.spearmanr(df[col], df["homicide"])
        r_p, p_p = stats.pearsonr(df[col], df["homicide"])
        rows.append(
            {
                "crime_type": col.replace("_", " ").title(),
                "col": col,
                "spearman_rho": r_s,
                "spearman_p": p_s,
                "pearson_r": r_p,
                "pearson_p": p_p,
                "total_incidents": int(df[col].sum()),
                "n_hexes_present": int((df[col] > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=False)


# ── Plotting ─────────────────────────────────────────────────────────────────


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_crime_vs_homicide_bars(corr_df: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart: Spearman ρ of each crime type vs homicide."""
    plot_df = corr_df.sort_values("spearman_rho", ascending=True)

    fig, ax = plt.subplots(figsize=(16, max(6, len(plot_df) * 0.38)))
    colors = ["#bf616a" if v > 0 else "#5e81ac" for v in plot_df["spearman_rho"]]
    bars = ax.barh(
        plot_df["crime_type"], plot_df["spearman_rho"], color=colors, height=0.65
    )

    for bar, p in zip(bars, plot_df["spearman_p"]):
        x_pos = bar.get_width()
        offset = 0.008 if x_pos >= 0 else -0.008
        ha = "left" if x_pos >= 0 else "right"
        ax.text(
            x_pos + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{x_pos:.3f} {_sig_stars(p)}",
            va="center",
            ha=ha,
            fontsize=10,
        )

    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("Spearman ρ with Homicide Count per Hex")
    ax.set_title(
        "Crime Type – Homicide Spatial Correlation\n(per 500 m hexagon, all crime types)",
        fontsize=14,
    )
    ax.text(
        0.01,
        -0.04,
        "Significance: * p<.05  ** p<.01  *** p<.001",
        transform=ax.transAxes,
        fontsize=10,
        color="grey",
    )
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_correlation_matrix(
    corr_df: pd.DataFrame,
    title: str,
    out_path: Path,
    label_map: dict | None = None,
    figsize: tuple = (14, 11),
    annot_size: int = 7,
) -> None:
    if label_map:
        renamed = corr_df.rename(index=label_map, columns=label_map)
    else:
        nice = {c: c.replace("_", " ").title() for c in corr_df.columns}
        renamed = corr_df.rename(index=nice, columns=nice)

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(renamed, dtype=bool), k=1)
    sns.heatmap(
        renamed,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7, "label": "Spearman ρ"},
        ax=ax,
        annot_kws={"size": annot_size},
    )
    ax.set_title(title, fontsize=14, pad=16)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(df[x], df[y], alpha=0.35, s=18, edgecolors="none", color="#d95f02")

    r_s, p_s = stats.spearmanr(df[x], df[y])
    r_p, p_p = stats.pearsonr(df[x], df[y])
    stat_text = (
        f"Spearman ρ = {r_s:.3f} (p = {p_s:.2e})\nPearson r = {r_p:.3f} (p = {p_p:.2e})"
    )
    ax.text(
        0.03,
        0.96,
        stat_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85),
    )

    z = np.polyfit(df[x], df[y], 1)
    x_line = np.linspace(df[x].min(), df[x].max(), 200)
    ax.plot(x_line, np.polyval(z, x_line), color="#1b9e77", lw=1.5, ls="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14)
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_top_infrastructure_correlations(
    df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    infra_cols = [
        c
        for c in df.columns
        if c.startswith("infra_")
        and c not in ("infra_total", "infra_protective", "infra_risk")
    ]
    results = []
    for col in infra_cols:
        if int((df[col] > 0).sum()) < 10:
            continue
        r_s, p_s = stats.spearmanr(df[col], df["homicide"])
        results.append(
            {
                "infrastructure": col.replace("infra_", "").replace("_", " ").title(),
                "spearman_rho": r_s,
                "p_value": p_s,
                "n_hexes_present": int((df[col] > 0).sum()),
            }
        )

    res_df = pd.DataFrame(results).sort_values("spearman_rho", ascending=True)

    fig, ax = plt.subplots(figsize=(16, max(6, len(res_df) * 0.35)))
    colors = ["#bf616a" if v > 0 else "#5e81ac" for v in res_df["spearman_rho"]]
    bars = ax.barh(
        res_df["infrastructure"], res_df["spearman_rho"], color=colors, height=0.65
    )
    for bar, p in zip(bars, res_df["p_value"]):
        x_pos = bar.get_width()
        offset = 0.005 if x_pos >= 0 else -0.005
        ha = "left" if x_pos >= 0 else "right"
        ax.text(
            x_pos + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{x_pos:.3f} {_sig_stars(p)}",
            va="center",
            ha=ha,
            fontsize=10,
        )
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("Spearman ρ with Homicide Count")
    ax.set_title(
        "Infrastructure–Homicide Correlation by Type\n(per 500 m hexagon)",
        fontsize=14,
    )
    ax.text(
        0.01,
        -0.06,
        "Significance: * p<.05  ** p<.01  *** p<.001",
        transform=ax.transAxes,
        fontsize=10,
        color="grey",
    )
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return res_df


# ── Summary ──────────────────────────────────────────────────────────────────


def write_summary(
    merged: pd.DataFrame,
    crime_corr_df: pd.DataFrame,
    infra_corr_df: pd.DataFrame,
    out_path: Path,
) -> None:
    n_hex = len(merged)
    n_with_homicide = int((merged["homicide"] > 0).sum())

    lines = [
        "=" * 76,
        "CORRELATION ANALYSIS SUMMARY",
        "All Crime Types & Social Infrastructure vs. Homicide",
        "per 500 m Hexagon — Chicago (2001–2026)",
        "=" * 76,
        "",
        f"Total hexagons:                {n_hex:,}",
        f"Hexagons with ≥1 homicide:     {n_with_homicide:,}",
        f"Crime types analysed:          {len(crime_corr_df)}",
        "",
        "─" * 76,
        "1. CRIME TYPE ↔ HOMICIDE — Spearman ρ (ranked)",
        "─" * 76,
    ]
    for _, row in crime_corr_df.iterrows():
        sig = _sig_stars(row["spearman_p"])
        lines.append(
            f"   {row['crime_type']:<36s}  ρ = {row['spearman_rho']:+.4f} {sig:4s}"
            f"  (n_hex={row['n_hexes_present']:,},  incidents={row['total_incidents']:,})"
        )

    if "infra_protective" in merged.columns:
        active = merged[merged["homicide"] > 0]
        r_prot, p_prot = stats.spearmanr(active["infra_protective"], active["homicide"])
        r_risk, p_risk = stats.spearmanr(active["infra_risk"], active["homicide"])
        lines += [
            "",
            "─" * 76,
            "2. AGGREGATE INFRASTRUCTURE ↔ HOMICIDE (homicide-active hexes)",
            "─" * 76,
            f"   Protective infrastructure   ρ = {r_prot:+.4f}  (p = {p_prot:.2e})",
            f"   Risk-associated             ρ = {r_risk:+.4f}  (p = {p_risk:.2e})",
        ]

    lines += [
        "",
        "─" * 76,
        "3. PER-TYPE INFRASTRUCTURE ↔ HOMICIDE (top 10 by |ρ|)",
        "─" * 76,
    ]
    top10 = infra_corr_df.reindex(
        infra_corr_df["spearman_rho"].abs().sort_values(ascending=False).index
    ).head(10)
    for _, row in top10.iterrows():
        lines.append(
            f"   {row['infrastructure']:<28s}  ρ = {row['spearman_rho']:+.4f} "
            f"{_sig_stars(row['p_value']):4s}  (present in {row['n_hexes_present']:,} hexes)"
        )

    lines += [
        "",
        "─" * 76,
        "4. FULL SPEARMAN MATRIX (homicide + top-5 crime types + infra aggregates)",
        "─" * 76,
    ]
    top5_cols = ["homicide"] + crime_corr_df.head(5)["col"].tolist()
    for c in ("infra_total", "infra_protective", "infra_risk"):
        if c in merged.columns:
            top5_cols.append(c)
    small = merged[top5_cols].corr(method="spearman")
    lines.append(small.to_string())
    lines.append("")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved {out_path}")
    print()
    print(text)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    for d in (DATA_OUT, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    csv_out = DATA_OUT / "crime_infrastructure_hex_merged.csv"
    if csv_out.exists():
        print(f"Loading cached merged table from {csv_out}")
        merged = pd.read_csv(csv_out)
        crime_cols = [
            c for c in merged.columns 
            if c not in ("hex_id", "homicide", "total_crime") and not c.startswith("infra_")
        ]
    else:
        for p in (CRIMES_CSV, INFRA_CSV):
            if not p.exists():
                print(f"ERROR: Missing {p}")
                return 1

        # ── Load & hex-bin all crimes ────────────────────────────────────────
        crimes = load_crimes(CRIMES_CSV)
        crime_pivot = pivot_crimes_to_hex(crimes)
        del crimes  # free ~1 GB

        crime_cols = [c for c in crime_pivot.columns if c != "hex_id"]
        excluded_normalised = {t.lower().replace(" ", "_") for t in EXCLUDE_CRIME_TYPES}
        crime_cols = [c for c in crime_cols if c not in excluded_normalised]

        print(f"Crime type columns ({len(crime_cols)}): {crime_cols}")

        # ── Load & hex-bin infrastructure ────────────────────────────────────
        print("Loading infrastructure …")
        infra_raw = load_infrastructure(INFRA_CSV)
        infra_pivot = pivot_infrastructure(infra_raw)

        # ── Merge ────────────────────────────────────────────────────────────
        merged = build_merged(crime_pivot[["hex_id"] + crime_cols], infra_pivot)
        merged.to_csv(csv_out, index=False)
        print(f"Saved merged table ({len(merged):,} hexagons) → {csv_out}")

    if "homicide" not in merged.columns:
        print(
            "ERROR: 'homicide' column not found after pivot. Check Primary Type values."
        )
        return 1

    active = merged[merged["homicide"] > 0].copy()
    print(f"Hexagons with ≥1 homicide: {len(active):,}")

    # ── Crime ↔ homicide correlations ────────────────────────────────────
    crime_corr = crime_vs_homicide_correlations(merged, crime_cols)
    print("\nCrime-type vs homicide correlations:")
    print(
        crime_corr[["crime_type", "spearman_rho", "spearman_p"]].to_string(index=False)
    )

    plot_crime_vs_homicide_bars(
        crime_corr,
        FIG_DIR / "correlation_crime_vs_homicide.png",
    )

    # Top-N scatter plots for strongest crime–homicide correlations
    top_crimes = crime_corr.head(TOP_N_SCATTER)
    for _, row in top_crimes.iterrows():
        col = row["col"]
        nice = row["crime_type"]
        safe_name = col.replace(" ", "_")
        plot_scatter(
            merged,
            col,
            "homicide",
            f"{nice} Count (per hex)",
            "Homicide Count (per hex)",
            f"{nice} vs. Homicides per 500 m Hexagon",
            FIG_DIR / f"scatter_{safe_name}_vs_homicides.png",
        )

    # Crime-type heatmap (top 15 + homicide to keep legible)
    top15_cols = ["homicide"] + crime_corr.head(15)["col"].tolist()
    top15_spearman = merged[top15_cols].corr(method="spearman")
    plot_correlation_matrix(
        top15_spearman,
        "Spearman Correlation: Top Crime Types per Hex",
        FIG_DIR / "correlation_matrix_all_crimes.png",
        figsize=(14, 12),
        annot_size=8,
    )

    # ── Infrastructure ↔ homicide ────────────────────────────────────────
    infra_corr_df = plot_top_infrastructure_correlations(
        active,
        FIG_DIR / "top_infrastructure_correlations.png",
    )

    # Infrastructure + crime combined heatmap
    agg_infra = [
        c
        for c in ("infra_total", "infra_protective", "infra_risk")
        if c in merged.columns
    ]
    combined_cols = ["homicide"] + crime_corr.head(8)["col"].tolist() + agg_infra
    combined_spearman = merged[combined_cols].corr(method="spearman")
    label_map = {
        c: c.replace("infra_", "").replace("_", " ").title() for c in combined_cols
    }
    label_map["homicide"] = "Homicide"
    plot_correlation_matrix(
        combined_spearman,
        "Spearman Correlation: Crime & Infrastructure per Hex",
        FIG_DIR / "correlation_matrix_crime_infrastructure.png",
        label_map=label_map,
        figsize=(12, 10),
        annot_size=9,
    )

    # Infrastructure scatter plots
    if "infra_total" in active.columns:
        plot_scatter(
            active,
            "infra_total",
            "homicide",
            "Total Infrastructure Count (per hex)",
            "Homicide Count (per hex)",
            "Social Infrastructure vs. Homicides per 500 m Hexagon",
            FIG_DIR / "scatter_infrastructure_vs_homicides.png",
        )
    if "infra_protective" in active.columns:
        plot_scatter(
            active,
            "infra_protective",
            "homicide",
            "Protective Infrastructure (per hex)",
            "Homicide Count (per hex)",
            "Protective Infrastructure vs. Homicides per Hexagon",
            FIG_DIR / "scatter_protective_vs_homicides.png",
        )
    if "infra_risk" in active.columns:
        plot_scatter(
            active,
            "infra_risk",
            "homicide",
            "Risk-Associated Infrastructure (per hex)",
            "Homicide Count (per hex)",
            "Risk-Associated Infrastructure vs. Homicides per Hexagon",
            FIG_DIR / "scatter_risk_vs_homicides.png",
        )

    # ── Summary ──────────────────────────────────────────────────────────
    write_summary(
        merged, crime_corr, infra_corr_df, FIG_DIR / "correlation_summary.txt"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
