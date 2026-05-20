#!/usr/bin/env python3
"""
PCA-based analysis of hex-level crime, infrastructure, and socioeconomic
features and their relationship to homicide.

Reads:
  - data/processed/crime_infrastructure_hex_merged.csv
  - data/processed/modeling/chicago_hex_modeling_table.csv

Outputs (all under reports/figures/):
  - pca_scree_plot.png
  - pca_loadings_heatmap.png
  - pca_correlation_with_homicide.png
  - pca_biplot_pc1_pc2.png
  - pca_top_loadings_pc*.png
  - pca_regression_summary.csv
  - pca_loadings_table.csv
  - pca_correlation_with_homicide.csv
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "gun-violence-analysis-pca-mplconfig"),
)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MERGED_PATH = ROOT / "data/processed/crime_infrastructure_hex_merged.csv"
MODELING_PATH = ROOT / "data/processed/modeling/chicago_hex_modeling_table.csv"
OUT_DIR = ROOT / "reports/figures"

# Columns to exclude from PCA input (target, aggregates, identifiers).
EXCLUDE = {
    "hex_id",
    "homicide",
    "total_crime",
    "infra_total",
    "infra_protective",
    "infra_risk",
}

SOCIO_CONTROLS = ["per_capita_income", "poverty_pct", "hardship_index"]
TARGET = "homicide"


def load_data() -> tuple[pd.DataFrame, list[str]]:
    """Load and merge the crime/infra table with socioeconomic controls."""
    merged = pd.read_csv(MERGED_PATH)
    socio = pd.read_csv(MODELING_PATH)[["hex_id", *SOCIO_CONTROLS]]

    df = merged.merge(socio, on="hex_id", how="inner")

    # Select numeric feature columns, excluding target and aggregates.
    feature_cols = [
        c
        for c in df.columns
        if c not in EXCLUDE
        and c not in SOCIO_CONTROLS
        and np.issubdtype(df[c].dtype, np.number)
    ]
    feature_cols += SOCIO_CONTROLS

    df = df.dropna(subset=[TARGET, *feature_cols]).copy()
    return df, feature_cols


def run_pca(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[PCA, np.ndarray, StandardScaler]:
    """Standardize features and fit PCA, keeping all components."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    return pca, scores, scaler


def plot_scree(pca: PCA, out_path: Path) -> None:
    """Scree plot: variance explained per component + cumulative."""
    n = min(20, pca.n_components_)
    var_ratio = pca.explained_variance_ratio_[:n]
    cum_var = np.cumsum(pca.explained_variance_ratio_)[:n]
    x = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(14, 4.5))
    ax1.bar(x, var_ratio * 100, color="#5e81ac", alpha=0.85, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_xticks(x)
    ax1.set_title("PCA Scree Plot — Crime, Infrastructure & Socioeconomic Features", fontsize=14)

    ax2 = ax1.twinx()
    ax2.plot(x, cum_var * 100, color="#bf616a", marker="o", linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def plot_loadings_heatmap(
    pca: PCA, feature_cols: list[str], out_path: Path, n_components: int = 8
) -> None:
    """Heatmap of PCA loadings for the top-N components."""
    n_components = min(n_components, pca.n_components_)
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    # Show only features with at least one |loading| > threshold for readability.
    threshold = 0.20
    mask = (loadings.abs() > threshold).any(axis=1)
    loadings_filtered = loadings[mask]

    fig, ax = plt.subplots(figsize=(16, max(7, len(loadings_filtered) * 0.35)))
    im = ax.imshow(
        loadings_filtered.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks(range(n_components))
    ax.set_xticklabels(loadings_filtered.columns)
    ax.set_yticks(range(len(loadings_filtered)))
    ax.set_yticklabels(loadings_filtered.index)
    ax.set_title(f"PCA Loadings (features with |loading| > {threshold})", fontsize=14)
    fig.colorbar(im, ax=ax, label="Loading")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def compute_pc_homicide_correlations(
    scores: np.ndarray, homicide: np.ndarray, n_components: int = 10
) -> pd.DataFrame:
    """Spearman correlation of each PC with homicide."""
    n_components = min(n_components, scores.shape[1])
    rows = []
    for i in range(n_components):
        rho, pval = spearmanr(scores[:, i], homicide, nan_policy="omit")
        rows.append(
            {
                "component": f"PC{i+1}",
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "abs_rho": float(abs(rho)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False)


def plot_pc_homicide_correlation(corr_df: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart of PC ↔ homicide Spearman rho."""
    ordered = corr_df.sort_values("spearman_rho")
    colors = ["#bf616a" if r > 0 else "#5e81ac" for r in ordered["spearman_rho"]]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.barh(ordered["component"], ordered["spearman_rho"], color=colors, edgecolor="white")
    ax.set_xlabel("Spearman ρ with Homicide")
    ax.set_title("Principal Components vs Homicide Correlation (* = p < .05, ** = p < .01, *** = p < .001)", fontsize=14)
    ax.axvline(0, color="black", linewidth=0.8)

    # Annotate significance.
    for idx, row in ordered.iterrows():
        stars = ""
        if row["p_value"] < 0.001:
            stars = "***"
        elif row["p_value"] < 0.01:
            stars = "**"
        elif row["p_value"] < 0.05:
            stars = "*"
        x_pos = row["spearman_rho"]
        offset = 0.01 if x_pos >= 0 else -0.01
        ha = "left" if x_pos >= 0 else "right"
        ax.text(x_pos + offset, row["component"], f"{x_pos:.3f}{stars}", va="center", ha=ha, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def plot_top_loadings_per_pc(
    pca: PCA,
    feature_cols: list[str],
    out_dir: Path,
    n_components: int = 6,
    n_top: int = 10,
) -> None:
    """One bar chart per PC showing top-N loadings, saved as separate files."""
    n_components = min(n_components, pca.n_components_)

    for i in range(n_components):
        loadings = pd.Series(pca.components_[i], index=feature_cols)
        top = loadings.abs().nlargest(n_top)
        ordered = loadings[top.index].sort_values()
        colors = ["#bf616a" if v > 0 else "#5e81ac" for v in ordered]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.barh(ordered.index, ordered.values, color=colors, edgecolor="white")
        var_pct = pca.explained_variance_ratio_[i] * 100
        ax.set_title(
            f"PC{i+1} — Top {n_top} Feature Loadings ({var_pct:.1f}% variance explained)",
            fontsize=14,
        )
        ax.set_xlabel("Loading")
        ax.axvline(0, color="black", linewidth=0.6)

        # Annotate values.
        for idx_name, val in ordered.items():
            offset = 0.005 if val >= 0 else -0.005
            ha = "left" if val >= 0 else "right"
            ax.text(val + offset, idx_name, f"{val:.3f}", va="center", ha=ha, fontsize=10)

        fig.tight_layout()
        out_path = out_dir / f"pca_top_loadings_pc{i+1}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Wrote: {out_path}")


def plot_biplot(
    pca: PCA,
    scores: np.ndarray,
    feature_cols: list[str],
    homicide: np.ndarray,
    out_path: Path,
    n_arrows: int = 15,
) -> None:
    """Biplot of PC1 vs PC2, colored by homicide count, with top-N loading arrows."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Scatter: each hex, colored by log1p(homicide).
    hom_log = np.log1p(homicide)
    sc = ax.scatter(
        scores[:, 0],
        scores[:, 1],
        c=hom_log,
        cmap="YlOrRd",
        alpha=0.55,
        s=15,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log(1 + homicide count)")

    # Loading arrows for the top-N most influential variables.
    loadings = pca.components_[:2].T  # (n_features, 2)
    magnitude = np.sqrt(loadings[:, 0] ** 2 + loadings[:, 1] ** 2)
    top_idx = np.argsort(magnitude)[-n_arrows:]

    # Scale arrows to plot range.
    scale = max(scores[:, 0].max(), scores[:, 1].max()) * 0.7
    for i in top_idx:
        ax.annotate(
            feature_cols[i],
            xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#2e3440", lw=1.2),
            fontsize=9,
            color="#2e3440",
            ha="center",
            va="center",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA Biplot — Hexagons Colored by Homicide Count", fontsize=14)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def run_pc_regression(
    scores: np.ndarray, homicide: np.ndarray, pca: PCA, n_components: int = 10
) -> pd.DataFrame:
    """OLS regression of homicide ~ PC1 + PC2 + ... + PCn."""
    n_components = min(n_components, scores.shape[1])
    X = scores[:, :n_components]
    y = homicide

    reg = LinearRegression()
    reg.fit(X, y)
    r2 = reg.score(X, y)

    var_explained = pca.explained_variance_ratio_[:n_components]

    rows = []
    for i in range(n_components):
        rows.append(
            {
                "component": f"PC{i+1}",
                "variance_explained_pct": float(var_explained[i] * 100),
                "regression_coefficient": float(reg.coef_[i]),
                "abs_coefficient": float(abs(reg.coef_[i])),
            }
        )
    rows.append(
        {
            "component": "MODEL",
            "variance_explained_pct": float(sum(var_explained) * 100),
            "regression_coefficient": float(reg.intercept_),
            "abs_coefficient": np.nan,
        }
    )

    summary = pd.DataFrame(rows)
    # Add R² as metadata in the last row.
    summary.loc[summary["component"] == "MODEL", "abs_coefficient"] = r2

    print(f"\nOLS Regression: homicide ~ PC1..PC{n_components}")
    print(f"  R² = {r2:.4f}")
    print(
        f"  Top predictor: "
        f"{summary[summary['component'] != 'MODEL'].sort_values('abs_coefficient', ascending=False).iloc[0]['component']}"
    )

    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, feature_cols = load_data()
    print(f"  {len(df)} hexagons, {len(feature_cols)} features")

    print("Running PCA...")
    pca, scores, scaler = run_pca(df, feature_cols)

    homicide = df[TARGET].to_numpy(dtype=float)

    # --- Outputs ---

    # 1. Scree plot.
    plot_scree(pca, OUT_DIR / "pca_scree_plot.png")

    # 2. Loadings heatmap.
    plot_loadings_heatmap(pca, feature_cols, OUT_DIR / "pca_loadings_heatmap.png")

    # 3. Full loadings table (CSV).
    n_save = min(10, pca.n_components_)
    loadings_df = pd.DataFrame(
        pca.components_[:n_save].T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(n_save)],
    )
    loadings_df.index.name = "variable"
    loadings_path = OUT_DIR / "pca_loadings_table.csv"
    loadings_df.to_csv(loadings_path)
    print(f"Wrote: {loadings_path}")

    # 4. PC ↔ homicide correlations.
    corr_df = compute_pc_homicide_correlations(scores, homicide, n_components=10)
    corr_path = OUT_DIR / "pca_correlation_with_homicide.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"Wrote: {corr_path}")

    plot_pc_homicide_correlation(corr_df, OUT_DIR / "pca_correlation_with_homicide.png")

    # 5. Biplot.
    plot_biplot(pca, scores, feature_cols, homicide, OUT_DIR / "pca_biplot_pc1_pc2.png")

    # 6. Top loadings per PC (one file each).
    plot_top_loadings_per_pc(pca, feature_cols, OUT_DIR)

    # 7. Regression summary.
    reg_df = run_pc_regression(scores, homicide, pca, n_components=2)
    reg_path = OUT_DIR / "pca_regression_summary.csv"
    reg_df.to_csv(reg_path, index=False)
    print(f"Wrote: {reg_path}")

    # --- Console summary ---
    print("\n=== Top 5 PC ↔ Homicide Correlations ===")
    print(corr_df.head().to_string(index=False))

    print("\n=== PC1 Top Loadings ===")
    pc1 = loadings_df["PC1"].abs().sort_values(ascending=False).head(10)
    for var, val in pc1.items():
        sign = "+" if loadings_df.loc[var, "PC1"] > 0 else "−"
        print(f"  {sign}{val:.3f}  {var}")

    print("\nDone.")


if __name__ == "__main__":
    main()
