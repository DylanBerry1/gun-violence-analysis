#!/usr/bin/env python3
"""
Robustness checks for homicide correlations under socioeconomic confounding.

Outputs:
  - reports/figures/hex_gaussian_z_percentile_spearman_crime_vs_homicide.csv
  - reports/figures/hex_gaussian_z_percentile_spearman_infra_vs_homicide.csv
  - reports/figures/hex_gam_residual_spearman_crime_vs_homicide.csv
  - reports/figures/hex_gam_residual_spearman_infra_vs_homicide.csv
  - reports/figures/hex_gaussian_z_percentile_spearman_top10_crime_vs_homicide.png
  - reports/figures/hex_gaussian_z_percentile_spearman_top10_infra_vs_homicide.png
  - reports/figures/hex_gam_residual_spearman_crime_vs_homicide.png
  - reports/figures/hex_gam_residual_spearman_infra_vs_homicide.png
  - reports/figures/hex_gam_residual_spearman_crime_vs_homicide_dumbbell.png
  - reports/figures/hex_gam_residual_spearman_infra_vs_homicide_dumbbell.png
  - reports/figures/hex_gam_residual_spearman_crime_vs_homicide_side_by_side_delta.png
  - reports/figures/hex_gam_residual_spearman_infra_vs_homicide_side_by_side_delta.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import spearmanr
from matplotlib.colors import TwoSlopeNorm
from sklearn.preprocessing import QuantileTransformer


ROOT = Path(__file__).resolve().parents[1]
MERGED_PATH = ROOT / "data/processed/crime_infrastructure_hex_merged.csv"
MODELING_PATH = ROOT / "data/processed/modeling/chicago_hex_modeling_table.csv"
OUT_DIR = ROOT / "reports/figures"

CONTROLS = ["per_capita_income", "poverty_pct", "hardship_index"]
TARGET = "homicide"
INFRA_EXCLUDE = {"infra_total", "infra_protective", "infra_risk"}


def _gaussianized_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Map controls to approximately Gaussian marginals via rank transform."""
    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, len(df)),
        random_state=42,
    )
    arr = qt.fit_transform(df[CONTROLS].to_numpy(dtype=float))
    return pd.DataFrame(arr, columns=CONTROLS, index=df.index)


def _build_disadvantage_score_from_gaussian_controls(
    gaussian_controls: pd.DataFrame,
) -> pd.Series:
    """Construct transformed common-cause score Z."""
    # Higher score means more disadvantage: low income, high poverty/hardship.
    return (
        -gaussian_controls["per_capita_income"]
        + gaussian_controls["poverty_pct"]
        + gaussian_controls["hardship_index"]
    ) / 3.0


def _design_matrix(controls_df: pd.DataFrame) -> np.ndarray:
    """
    Build additive natural spline basis for GAM-style residualization.
    """
    # Natural cubic regression splines with linear tails.
    formula = (
        "cr(per_capita_income, df=5) + "
        "cr(poverty_pct, df=5) + "
        "cr(hardship_index, df=5)"
    )
    X = dmatrix(formula, controls_df, return_type="dataframe")
    return np.asarray(X)


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    return y - y_hat


def _compute_percentile_stratified(
    df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    strat_rows: list[dict[str, object]] = []
    for var in columns:
        for p, q_df in df.groupby("z_percentile", observed=False):
            if q_df[var].nunique(dropna=True) < 2 or q_df[TARGET].nunique(dropna=True) < 2:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(q_df[var], q_df[TARGET], nan_policy="omit")
            strat_rows.append(
                {
                    "variable": var,
                    "z_percentile": int(p),
                    "n": int(q_df.shape[0]),
                    "rho_spearman": float(rho),
                    "p_value": float(pval),
                }
            )
    return pd.DataFrame(strat_rows)


def _compute_gam_residual(df: pd.DataFrame, columns: list[str], X_spline: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    homicide_resid = _residualize(df[TARGET].to_numpy(dtype=float), X_spline)
    for var in columns:
        x_resid = _residualize(df[var].to_numpy(dtype=float), X_spline)
        rho, pval = spearmanr(x_resid, homicide_resid, nan_policy="omit")
        naive_rho, naive_p = spearmanr(df[var], df[TARGET], nan_policy="omit")
        rows.append(
            {
                "variable": var,
                "n": int(df.shape[0]),
                "naive_spearman": float(naive_rho),
                "naive_p_value": float(naive_p),
                "gam_residual_spearman": float(rho),
                "gam_residual_p_value": float(pval),
                "delta": float(rho - naive_rho),
            }
        )
    return pd.DataFrame(rows).sort_values("gam_residual_spearman", ascending=False)


def _plot_top10_percentile_lines(strat_df: pd.DataFrame, gam_df: pd.DataFrame, out_path: Path, title: str) -> None:
    top10 = gam_df.head(10)["variable"].tolist()
    strat_top10 = (
        strat_df[strat_df["variable"].isin(top10)]
        .copy()
        .sort_values(["variable", "z_percentile"])
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    for variable in top10:
        d = strat_top10[strat_top10["variable"] == variable]
        ax.plot(d["z_percentile"], d["rho_spearman"], label=variable, linewidth=1.8)
    ax.set_title(title)
    ax.set_ylabel("Spearman rho")
    ax.set_xlabel("Gaussian-Z percentile")
    ax.set_xlim(1, 100)
    ax.legend(title="Variable", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_top10_percentile_heatmap(
    strat_df: pd.DataFrame, gam_df: pd.DataFrame, out_path: Path, title: str
) -> None:
    top10 = (
        gam_df.assign(abs_rho=gam_df["gam_residual_spearman"].abs())
        .sort_values("abs_rho", ascending=False)
        .head(10)["variable"]
        .tolist()
    )
    d = strat_df[strat_df["variable"].isin(top10)].copy()
    pivot = d.pivot(index="variable", columns="z_percentile", values="rho_spearman")
    pivot = pivot.reindex(top10)
    vals = np.nan_to_num(pivot.to_numpy(dtype=float), nan=0.0)
    vmax = max(0.2, float(np.nanmax(np.abs(vals))))

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(
        vals,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Gaussian-Z percentile")
    ax.set_ylabel("Infrastructure variable")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    xticks = list(range(0, 100, 10))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x + 1) for x in xticks])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spearman rho")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_overlap_bars(gam_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(gam_df))
    ax.barh(y, gam_df["naive_spearman"], alpha=0.5, label="Naive Spearman")
    ax.barh(y, gam_df["gam_residual_spearman"], alpha=0.8, label="GAM residual Spearman")
    ax.set_yticks(y)
    ax.set_yticklabels(gam_df["variable"])
    ax.invert_yaxis()
    ax.set_xlabel("Spearman rho")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_dumbbell(gam_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    ordered = gam_df.sort_values("delta")
    y = np.arange(len(ordered))
    ax.hlines(y, ordered["naive_spearman"], ordered["gam_residual_spearman"], color="gray", alpha=0.7)
    ax.scatter(ordered["naive_spearman"], y, label="Naive", s=30)
    ax.scatter(ordered["gam_residual_spearman"], y, label="GAM residual", s=30)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["variable"])
    ax.invert_yaxis()
    ax.set_xlabel("Spearman rho")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_side_by_side_delta(gam_df: pd.DataFrame, out_path: Path, title: str) -> None:
    ordered = gam_df.sort_values("delta")
    x = np.arange(len(ordered))
    width = 0.42
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, ordered["naive_spearman"], width=width, label="Naive")
    ax.bar(x + width / 2, ordered["gam_residual_spearman"], width=width, label="GAM residual")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["variable"], rotation=65, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    merged = pd.read_csv(MERGED_PATH)
    socio = pd.read_csv(MODELING_PATH)[["hex_id", *CONTROLS]]

    df = merged.merge(socio, on="hex_id", how="inner")
    df = df.dropna(subset=[TARGET, *CONTROLS]).copy()

    # Restrict to variables used in report-level correlation analysis.
    crime_cols = [
        c
        for c in merged.columns
        if c
        not in {
            "hex_id",
            TARGET,
            "total_crime",
        }
        and not c.startswith("infra_")
        and np.issubdtype(merged[c].dtype, np.number)
    ]
    infra_cols = [
        c
        for c in merged.columns
        if c.startswith("infra_")
        and c not in INFRA_EXCLUDE
        and np.issubdtype(merged[c].dtype, np.number)
    ]

    # ---------- 1) Gaussian-Z percentile Spearman ----------
    gaussian_controls = _gaussianized_controls(df)
    df["z_disadvantage_gaussian"] = _build_disadvantage_score_from_gaussian_controls(
        gaussian_controls
    )
    # Percentile bins 1..100 based on transformed Z.
    z_pct = df["z_disadvantage_gaussian"].rank(method="average", pct=True) * 100.0
    df["z_percentile"] = np.ceil(z_pct).astype(int).clip(1, 100)

    strat_crime_df = _compute_percentile_stratified(df, crime_cols)
    strat_infra_df = _compute_percentile_stratified(df, infra_cols)
    strat_crime_out = OUT_DIR / "hex_gaussian_z_percentile_spearman_crime_vs_homicide.csv"
    strat_infra_out = OUT_DIR / "hex_gaussian_z_percentile_spearman_infra_vs_homicide.csv"
    strat_crime_df.to_csv(strat_crime_out, index=False)
    strat_infra_df.to_csv(strat_infra_out, index=False)

    # ---------- 2) GAM residualization then Spearman on residuals ----------
    X_spline = _design_matrix(df[CONTROLS])
    gam_crime_df = _compute_gam_residual(df, crime_cols, X_spline)
    gam_infra_df = _compute_gam_residual(df, infra_cols, X_spline)
    gam_crime_out = OUT_DIR / "hex_gam_residual_spearman_crime_vs_homicide.csv"
    gam_infra_out = OUT_DIR / "hex_gam_residual_spearman_infra_vs_homicide.csv"
    gam_crime_df.to_csv(gam_crime_out, index=False)
    gam_infra_df.to_csv(gam_infra_out, index=False)

    # ---------- Plots ----------
    _plot_top10_percentile_lines(
        strat_crime_df,
        gam_crime_df,
        OUT_DIR / "hex_gaussian_z_percentile_spearman_top10_crime_vs_homicide.png",
        "Spearman(homicide, crime) by Gaussian-Z percentile (1-100)",
    )
    _plot_top10_percentile_heatmap(
        strat_infra_df,
        gam_infra_df,
        OUT_DIR / "hex_gaussian_z_percentile_spearman_top10_infra_vs_homicide.png",
        "Infrastructure vs homicide by Gaussian-Z percentile (top 10, heatmap)",
    )

    _plot_overlap_bars(
        gam_crime_df,
        OUT_DIR / "hex_gam_residual_spearman_crime_vs_homicide.png",
        "Crime vs homicide: naive vs GAM-residual Spearman",
    )
    _plot_overlap_bars(
        gam_infra_df,
        OUT_DIR / "hex_gam_residual_spearman_infra_vs_homicide.png",
        "Infrastructure vs homicide: naive vs GAM-residual Spearman",
    )

    _plot_dumbbell(
        gam_crime_df,
        OUT_DIR / "hex_gam_residual_spearman_crime_vs_homicide_dumbbell.png",
        "Crime vs homicide: naive vs GAM-residual (dumbbell)",
    )
    _plot_dumbbell(
        gam_infra_df,
        OUT_DIR / "hex_gam_residual_spearman_infra_vs_homicide_dumbbell.png",
        "Infrastructure vs homicide: naive vs GAM-residual (dumbbell)",
    )

    _plot_side_by_side_delta(
        gam_crime_df,
        OUT_DIR / "hex_gam_residual_spearman_crime_vs_homicide_side_by_side_delta.png",
        "Crime vs homicide: sorted side-by-side bars by delta",
    )
    _plot_side_by_side_delta(
        gam_infra_df,
        OUT_DIR / "hex_gam_residual_spearman_infra_vs_homicide_side_by_side_delta.png",
        "Infrastructure vs homicide: sorted side-by-side bars by delta",
    )

    print(f"Wrote: {strat_crime_out}")
    print(f"Wrote: {strat_infra_out}")
    print(f"Wrote: {gam_crime_out}")
    print(f"Wrote: {gam_infra_out}")
    print("Wrote: reports/figures/hex_gaussian_z_percentile_spearman_top10_crime_vs_homicide.png")
    print("Wrote: reports/figures/hex_gaussian_z_percentile_spearman_top10_infra_vs_homicide.png")
    print("Wrote: reports/figures/hex_gam_residual_spearman_crime_vs_homicide.png")
    print("Wrote: reports/figures/hex_gam_residual_spearman_infra_vs_homicide.png")
    print("Wrote: reports/figures/hex_gam_residual_spearman_crime_vs_homicide_dumbbell.png")
    print("Wrote: reports/figures/hex_gam_residual_spearman_infra_vs_homicide_dumbbell.png")
    print("Wrote: reports/figures/hex_gam_residual_spearman_crime_vs_homicide_side_by_side_delta.png")
    print("Wrote: reports/figures/hex_gam_residual_spearman_infra_vs_homicide_side_by_side_delta.png")


if __name__ == "__main__":
    main()
