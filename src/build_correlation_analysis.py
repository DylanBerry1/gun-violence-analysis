"""
Spatial correlation analysis between crime types and social infrastructure.

Merges homicide and drug-crime hex counts with infrastructure counts per hex
(500 m axial grid, EPSG:3857) and computes Pearson / Spearman correlations.

Outputs (all under reports/figures/ and data/processed/):
- crime_infrastructure_hex_merged.csv
- correlation_matrix_crime_infrastructure.png
- scatter_homicides_vs_drugs.png
- scatter_infrastructure_vs_homicides.png
- top_infrastructure_correlations.png
- correlation_summary.txt

Run:
    python3 src/build_correlation_analysis.py
"""

import math
import sys
import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_HEX = ROOT_DIR / "data" / "processed" / "hex"
DATA_OUT = ROOT_DIR / "data" / "processed"
FIG_DIR = ROOT_DIR / "reports" / "figures"

HOMICIDE_HEX = DATA_HEX / "chicago_homicides_hex_counts.csv"
DRUG_HEX = DATA_HEX / "chicago_drug_hex_counts.csv"
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

MIN_HEX_ACTIVITY = 1


# ── Hex grid helpers (same logic as build_homicides_hex_map.py) ──────────────

def cube_round(qf: np.ndarray, rf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xf, zf = qf, rf
    yf = -xf - zf
    rx, ry, rz = np.round(xf), np.round(yf), np.round(zf)
    x_diff, y_diff, z_diff = np.abs(rx - xf), np.abs(ry - yf), np.abs(rz - zf)

    x_largest = (x_diff > y_diff) & (x_diff > z_diff)
    y_largest = (~x_largest) & (y_diff > z_diff)
    z_largest = (~x_largest) & (~y_largest)

    rx[x_largest] = -ry[x_largest] - rz[x_largest]
    ry[y_largest] = -rx[y_largest] - rz[y_largest]
    rz[z_largest] = -rx[z_largest] - ry[z_largest]
    return rx.astype(int), rz.astype(int)


def assign_hex_ids(
    lats: np.ndarray, lons: np.ndarray, hex_size_m: float
) -> pd.DataFrame:
    """Convert lat/lon arrays to hex_id via EPSG:3857 projection."""
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326"
    ).to_crs(epsg=3857)

    x = gdf.geometry.x.to_numpy()
    y = gdf.geometry.y.to_numpy()
    sqrt3 = math.sqrt(3.0)

    qf = ((sqrt3 / 3.0) * x - (1.0 / 3.0) * y) / hex_size_m
    rf = ((2.0 / 3.0) * y) / hex_size_m
    q, r = cube_round(qf, rf)

    return pd.DataFrame({
        "hex_q": q,
        "hex_r": r,
        "hex_id": [f"{qi}_{ri}" for qi, ri in zip(q, r)],
    })


# ── Loading helpers ──────────────────────────────────────────────────────────

def load_hex_counts(path: Path, count_col_alias: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["hex_id", "count"]].rename(columns={"count": count_col_alias})


def load_infrastructure(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["latitude", "longitude"])
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
    """One row per hex_id, columns = infrastructure type counts."""
    pivot = (
        infra_df.groupby(["hex_id", "infrastructure_type"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.columns = [f"infra_{c}" for c in pivot.columns]
    pivot["infra_total"] = pivot.sum(axis=1)

    protective_cols = [f"infra_{t}" for t in PROTECTIVE_TYPES if f"infra_{t}" in pivot.columns]
    risk_cols = [f"infra_{t}" for t in RISK_TYPES if f"infra_{t}" in pivot.columns]
    if protective_cols:
        pivot["infra_protective"] = pivot[protective_cols].sum(axis=1)
    if risk_cols:
        pivot["infra_risk"] = pivot[risk_cols].sum(axis=1)

    return pivot.reset_index()


# ── Merge ────────────────────────────────────────────────────────────────────

def build_merged_hex_df() -> pd.DataFrame:
    homicide = load_hex_counts(HOMICIDE_HEX, "homicide_count")
    drug = load_hex_counts(DRUG_HEX, "drug_count")
    infra_raw = load_infrastructure(INFRA_CSV)
    infra_pivot = pivot_infrastructure(infra_raw)

    merged = homicide.merge(drug, on="hex_id", how="outer")
    merged = merged.merge(infra_pivot, on="hex_id", how="outer")
    merged = merged.fillna(0)

    merged["has_crime"] = (merged["homicide_count"] + merged["drug_count"]) > 0
    return merged


# ── Correlation helpers ──────────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame, cols: list[str]) -> dict:
    """Return Pearson and Spearman matrices."""
    sub = df[cols]
    pearson = sub.corr(method="pearson")
    spearman = sub.corr(method="spearman")
    return {"pearson": pearson, "spearman": spearman}


def annotated_corr(
    r: float, p: float, n: int
) -> str:
    stars = ""
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    return f"r={r:.3f}{stars} (n={n})"


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_correlation_matrix(
    corr_df: pd.DataFrame, title: str, out_path: Path
) -> None:
    nice_labels = {c: c.replace("infra_", "").replace("_", " ").title() for c in corr_df.columns}
    nice_labels["homicide_count"] = "Homicides"
    nice_labels["drug_count"] = "Drug Crimes"
    renamed = corr_df.rename(index=nice_labels, columns=nice_labels)

    fig, ax = plt.subplots(figsize=(14, 11))
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
        annot_kws={"size": 7},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
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
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[x], df[y], alpha=0.35, s=18, edgecolors="none", color="#d95f02")

    r_s, p_s = stats.spearmanr(df[x], df[y])
    r_p, p_p = stats.pearsonr(df[x], df[y])
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    stat_text = f"Spearman ρ = {r_s:.3f} (p = {p_s:.2e})\nPearson r = {r_p:.3f} (p = {p_p:.2e})"
    ax.text(
        0.03, 0.96, stat_text, transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85),
    )

    z = np.polyfit(df[x], df[y], 1)
    x_line = np.linspace(df[x].min(), df[x].max(), 200)
    ax.plot(x_line, np.polyval(z, x_line), color="#1b9e77", linewidth=1.5, linestyle="--")

    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_top_infrastructure_correlations(
    df: pd.DataFrame, out_path: Path
) -> None:
    """Bar chart of Spearman ρ between each infrastructure type and homicide count."""
    infra_cols = [c for c in df.columns if c.startswith("infra_") and c not in (
        "infra_total", "infra_protective", "infra_risk",
    )]
    results = []
    for col in infra_cols:
        nonzero = df[df[col] > 0]
        if len(nonzero) < 10:
            continue
        r_s, p_s = stats.spearmanr(df[col], df["homicide_count"])
        results.append({
            "infrastructure": col.replace("infra_", "").replace("_", " ").title(),
            "spearman_rho": r_s,
            "p_value": p_s,
            "n_hexes_present": int((df[col] > 0).sum()),
        })

    res_df = pd.DataFrame(results).sort_values("spearman_rho", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(res_df) * 0.35)))
    colors = ["#d95f02" if v > 0 else "#1b9e77" for v in res_df["spearman_rho"]]
    bars = ax.barh(res_df["infrastructure"], res_df["spearman_rho"], color=colors, height=0.65)

    for bar, p in zip(bars, res_df["p_value"]):
        if p < 0.001:
            star = " ***"
        elif p < 0.01:
            star = " **"
        elif p < 0.05:
            star = " *"
        else:
            star = ""
        x_pos = bar.get_width()
        offset = 0.005 if x_pos >= 0 else -0.005
        ha = "left" if x_pos >= 0 else "right"
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                f"{x_pos:.3f}{star}", va="center", ha=ha, fontsize=8)

    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("Spearman ρ with Homicide Count", fontsize=11)
    ax.set_title(
        "Infrastructure–Homicide Correlation by Type\n(per 500 m hexagon)",
        fontsize=13, fontweight="bold",
    )
    note = "Significance: * p<.05  ** p<.01  *** p<.001"
    ax.text(0.01, -0.06, note, transform=ax.transAxes, fontsize=8, color="grey")
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    return res_df


def write_summary(
    merged: pd.DataFrame,
    corr: dict,
    infra_corr_df: pd.DataFrame,
    out_path: Path,
) -> None:
    crime_hex = merged[merged["has_crime"]]
    n_total = len(merged)
    n_crime = len(crime_hex)

    r_s, p_s = stats.spearmanr(crime_hex["drug_count"], crime_hex["homicide_count"])
    r_p, p_p = stats.pearsonr(crime_hex["drug_count"], crime_hex["homicide_count"])

    lines = [
        "=" * 72,
        "CORRELATION ANALYSIS SUMMARY",
        "Crime Types × Social Infrastructure per 500 m Hexagon — Chicago",
        "=" * 72,
        "",
        f"Total hexagons with any data:  {n_total:,}",
        f"Hexagons with ≥1 crime event:  {n_crime:,}",
        f"Hexagons with ≥1 homicide:     {int((merged['homicide_count'] > 0).sum()):,}",
        f"Hexagons with ≥1 drug crime:   {int((merged['drug_count'] > 0).sum()):,}",
        "",
        "─" * 72,
        "1. DRUG CRIME ↔ HOMICIDE CORRELATION",
        "─" * 72,
        f"   Spearman ρ = {r_s:.4f}   (p = {p_s:.2e})",
        f"   Pearson  r = {r_p:.4f}   (p = {p_p:.2e})",
        "",
    ]

    if "infra_protective" in merged.columns:
        r_prot, p_prot = stats.spearmanr(crime_hex["infra_protective"], crime_hex["homicide_count"])
        r_risk, p_risk = stats.spearmanr(crime_hex["infra_risk"], crime_hex["homicide_count"])
        lines += [
            "─" * 72,
            "2. AGGREGATE INFRASTRUCTURE ↔ HOMICIDE (crime-active hexes only)",
            "─" * 72,
            f"   Protective infrastructure  ρ = {r_prot:.4f}  (p = {p_prot:.2e})",
            f"   Risk-associated infrastructure  ρ = {r_risk:.4f}  (p = {p_risk:.2e})",
            "",
        ]

    lines += [
        "─" * 72,
        "3. PER-TYPE INFRASTRUCTURE ↔ HOMICIDE (top 10 by |ρ|)",
        "─" * 72,
    ]
    top10 = infra_corr_df.reindex(
        infra_corr_df["spearman_rho"].abs().sort_values(ascending=False).index
    ).head(10)
    for _, row in top10.iterrows():
        sig = ""
        if row["p_value"] < 0.001:
            sig = "***"
        elif row["p_value"] < 0.01:
            sig = "**"
        elif row["p_value"] < 0.05:
            sig = "*"
        lines.append(
            f"   {row['infrastructure']:<28s}  ρ = {row['spearman_rho']:+.4f}{sig:4s}"
            f"  (present in {row['n_hexes_present']:,} hexes)"
        )

    lines += ["", "─" * 72, "4. FULL SPEARMAN MATRIX (crime + aggregates)", "─" * 72]
    summary_cols = ["homicide_count", "drug_count"]
    for c in ("infra_total", "infra_protective", "infra_risk"):
        if c in merged.columns:
            summary_cols.append(c)
    small_corr = crime_hex[summary_cols].corr(method="spearman")
    lines.append(small_corr.to_string())
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

    for p in (HOMICIDE_HEX, DRUG_HEX, INFRA_CSV):
        if not p.exists():
            print(f"ERROR: Missing {p}")
            return 1

    print("Loading and merging hex data …")
    merged = build_merged_hex_df()

    csv_out = DATA_OUT / "crime_infrastructure_hex_merged.csv"
    merged.to_csv(csv_out, index=False)
    print(f"Saved merged table ({len(merged):,} hexagons) → {csv_out}")

    crime_active = merged[merged["has_crime"]].copy()
    print(f"Crime-active hexagons: {len(crime_active):,}")

    # Correlation columns
    core_cols = ["homicide_count", "drug_count"]
    agg_infra = [c for c in ("infra_total", "infra_protective", "infra_risk") if c in merged.columns]
    type_infra = sorted([
        c for c in merged.columns
        if c.startswith("infra_") and c not in ("infra_total", "infra_protective", "infra_risk")
    ])

    # Full matrix on crime-active hexes
    matrix_cols = core_cols + agg_infra + type_infra
    corr = compute_correlations(crime_active, matrix_cols)

    plot_correlation_matrix(
        corr["spearman"],
        "Spearman Correlation: Crime & Infrastructure per Hex",
        FIG_DIR / "correlation_matrix_crime_infrastructure.png",
    )

    plot_scatter(
        crime_active,
        "drug_count", "homicide_count",
        "Drug Crime Count (per hex)", "Homicide Count (per hex)",
        "Drug Crimes vs. Homicides per 500 m Hexagon",
        FIG_DIR / "scatter_homicides_vs_drugs.png",
    )

    if "infra_total" in crime_active.columns:
        plot_scatter(
            crime_active,
            "infra_total", "homicide_count",
            "Total Infrastructure Count (per hex)", "Homicide Count (per hex)",
            "Social Infrastructure vs. Homicides per 500 m Hexagon",
            FIG_DIR / "scatter_infrastructure_vs_homicides.png",
        )

    if "infra_protective" in crime_active.columns:
        plot_scatter(
            crime_active,
            "infra_protective", "homicide_count",
            "Protective Infrastructure Count (per hex)", "Homicide Count (per hex)",
            "Protective Infrastructure vs. Homicides per Hexagon",
            FIG_DIR / "scatter_protective_vs_homicides.png",
        )

    if "infra_risk" in crime_active.columns:
        plot_scatter(
            crime_active,
            "infra_risk", "homicide_count",
            "Risk-Associated Infrastructure Count (per hex)", "Homicide Count (per hex)",
            "Risk-Associated Infrastructure vs. Homicides per Hexagon",
            FIG_DIR / "scatter_risk_vs_homicides.png",
        )

    infra_corr_df = plot_top_infrastructure_correlations(
        crime_active, FIG_DIR / "top_infrastructure_correlations.png"
    )

    write_summary(
        merged, corr, infra_corr_df,
        FIG_DIR / "correlation_summary.txt",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
