#!/usr/bin/env python3
"""
Logistic-regression arrest model figures.

Reproduces the two feature-importance bar charts from
notebooks/exploration/arrest_model.ipynb with publication-quality
bold fonts for LaTeX embedding.

Reads:
    data/raw/all_chicago_crimes.parquet

Outputs (reports/figures/log_reg_arrests/):
    most_imp_features_arrests_predict_positive.png
    most_imp_features_arrests_predict_negative.png

Run:
    python3 src/build_arrest_logreg_figures.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "gun-violence-analysis-logreg-mplconfig"),
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT / "data" / "raw" / "all_chicago_crimes.parquet"
OUT_DIR = ROOT / "reports" / "figures" / "log_reg_arrests"

FEATURE_COLS = [
    "Primary Type",
    "Description",
    "Year",
    "Beat",
    "District",
    "Ward",
    "Location Description",
]
TARGET = "Arrest"
TOP_N = 10


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading parquet …")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  {len(df):,} rows")

    X_raw = df[FEATURE_COLS]
    y = df[TARGET]

    print("One-hot encoding …")
    enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    X = enc.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.9, random_state=42,
    )

    print("Fitting logistic regression …")
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"  Test accuracy: {score:.4f}")

    # Build coefficient table.
    categories = enc.categories_
    all_labels = [label for cat_arr in categories for label in cat_arr]
    coef_df = (
        pd.DataFrame({"category": all_labels, "model_coef": model.coef_[0]})
        .sort_values("model_coef", ascending=False)
    )

    # --- Positive plot (top factors for arrest) ---
    top = coef_df.head(TOP_N).sort_values("model_coef")
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.barh(top["category"], top["model_coef"], color="#337ab7")
    ax.set_title("Factors contributing the most to an arrest being made", fontsize=14)
    ax.set_xlabel("Model coefficient")
    ax.set_ylabel("Category")
    fig.tight_layout()
    pos_path = OUT_DIR / "most_imp_features_arrests_predict_positive.png"
    fig.savefig(pos_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {pos_path}")

    # --- Negative plot (top factors against arrest) ---
    bottom = coef_df.tail(TOP_N).sort_values("model_coef")
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.barh(bottom["category"], bottom["model_coef"], color="#337ab7")
    ax.set_title("Factors contributing the most to an arrest not being made", fontsize=14)
    ax.set_xlabel("Model coefficient")
    ax.set_ylabel("Category")
    fig.tight_layout()
    neg_path = OUT_DIR / "most_imp_features_arrests_predict_negative.png"
    fig.savefig(neg_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {neg_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
