"""
Train XGBoost models on Chicago hex cells and export feature importance artifacts.

This script treats each 500m hex cell as one observation and builds features from:
- homicide incidents (target only)
- drug crime incidents
- OSM infrastructure points
- socioeconomic indicators approximated from the dominant community area in each hex

Outputs:
- data/processed/modeling/chicago_hex_modeling_table.csv
- reports/modeling/xgboost_hotspot_metrics.json
- reports/modeling/xgboost_count_metrics.json
- reports/modeling/xgboost_hotspot_holdout_predictions.csv
- reports/modeling/xgboost_count_holdout_predictions.csv
- reports/modeling/xgboost_hotspot_feature_importance.csv
- reports/modeling/xgboost_count_feature_importance.csv
- reports/modeling/xgboost_hotspot_feature_groups.csv
- reports/modeling/xgboost_count_feature_groups.csv
- reports/modeling/xgboost_hotspot_feature_importance.png
- reports/modeling/xgboost_count_feature_importance.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODELING_DIR = DATA_DIR / "processed" / "modeling"
REPORTS_DIR = ROOT_DIR / "reports" / "modeling"

HOMICIDE_CSV = RAW_DIR / "chicago_violence_homicides.csv"
DRUG_CSV = RAW_DIR / "chicago_drug_crimes.csv"
INFRA_CSV = RAW_DIR / "infrastructure_locations.csv"
SOCIO_CSV = RAW_DIR / "chicago_socioeconomic_neighborhoods.csv"

DEFAULT_HEX_SIZE_M = 500
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"
SEASON_ORDER = ["winter", "spring", "summer", "fall"]
TIME_BIN_LABELS = ["night", "morning", "afternoon", "evening"]
TIME_BIN_BINS = [-1, 5, 11, 17, 23]


def configure_matplotlib():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "gun-violence-analysis-mplconfig"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost hotspot and count models on Chicago hex cells."
    )
    parser.add_argument(
        "--task",
        choices=["all", "hotspot", "count"],
        default="all",
        help="Which model task to run.",
    )
    parser.add_argument(
        "--hex-size-m",
        type=int,
        default=DEFAULT_HEX_SIZE_M,
        help="Hex size in meters. Keep at 500 to match the existing maps.",
    )
    parser.add_argument(
        "--hotspot-quantile",
        type=float,
        default=0.75,
        help="Quantile over non-zero homicide counts used to define a hotspot.",
    )
    parser.add_argument(
        "--top-drug-locations",
        type=int,
        default=10,
        help="How many drug-crime location descriptions to expand into features.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Holdout fraction for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and model reproducibility.",
    )
    return parser.parse_args()


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


def assign_hex_ids(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    hex_size_m: int,
) -> gpd.GeoDataFrame:
    clean = df.dropna(subset=[lat_col, lon_col]).copy()
    gdf = gpd.GeoDataFrame(
        clean,
        geometry=gpd.points_from_xy(clean[lon_col], clean[lat_col]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    x = gdf.geometry.x.to_numpy()
    y = gdf.geometry.y.to_numpy()
    sqrt3 = math.sqrt(3.0)
    qf = ((sqrt3 / 3.0) * x - (1.0 / 3.0) * y) / hex_size_m
    rf = ((2.0 / 3.0) * y) / hex_size_m
    q, r = cube_round(qf, rf)

    gdf["hex_q"] = q
    gdf["hex_r"] = r
    gdf["hex_id"] = gdf["hex_q"].astype(str) + "_" + gdf["hex_r"].astype(str)
    return gdf


def axial_to_center_xy(q: int, r: int, hex_size_m: int) -> tuple[float, float]:
    x = hex_size_m * math.sqrt(3.0) * (q + r / 2.0)
    y = hex_size_m * 1.5 * r
    return x, y


def build_hex_index(hex_ids: set[str], hex_size_m: int) -> pd.DataFrame:
    base = pd.DataFrame({"hex_id": sorted(hex_ids)})
    base[["hex_q", "hex_r"]] = base["hex_id"].str.split("_", expand=True).astype(int)

    centers_xy = base.apply(
        lambda row: axial_to_center_xy(
            int(row["hex_q"]), int(row["hex_r"]), hex_size_m
        ),
        axis=1,
        result_type="expand",
    )
    base["center_x"] = centers_xy[0]
    base["center_y"] = centers_xy[1]

    centers = gpd.GeoDataFrame(
        base[["hex_id", "hex_q", "hex_r"]].copy(),
        geometry=[Point(xy) for xy in zip(base["center_x"], base["center_y"])],
        crs="EPSG:3857",
    ).to_crs(epsg=4326)

    base["centroid_lon"] = centers.geometry.x
    base["centroid_lat"] = centers.geometry.y
    return base.drop(columns=["center_x", "center_y"])


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def add_count_feature(
    base: pd.DataFrame,
    df: pd.DataFrame,
    group_col: str,
    prefix: str,
) -> pd.DataFrame:
    counts = df.groupby(group_col).size().rename(prefix)
    base = base.join(counts, on="hex_id")
    base[prefix] = base[prefix].fillna(0).astype(float)
    return base


def add_group_counts(
    base: pd.DataFrame,
    df: pd.DataFrame,
    category_col: str,
    prefix: str,
) -> pd.DataFrame:
    pivot = (
        df.groupby(["hex_id", category_col], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    pivot.columns = [f"{prefix}_{sanitize_name(col)}" for col in pivot.columns]
    return base.join(pivot, on="hex_id")


def add_top_category_counts(
    base: pd.DataFrame,
    df: pd.DataFrame,
    category_col: str,
    prefix: str,
    top_n: int,
) -> pd.DataFrame:
    top_categories = df[category_col].fillna("UNKNOWN").value_counts().head(top_n).index
    filtered = df[df[category_col].fillna("UNKNOWN").isin(top_categories)].copy()
    filtered[category_col] = filtered[category_col].fillna("UNKNOWN")
    return add_group_counts(base, filtered, category_col, prefix)


def mode_or_nan(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    modes = non_null.mode()
    if modes.empty:
        return np.nan
    return float(modes.iloc[0])


def load_feature_table(hex_size_m: int, top_drug_locations: int) -> pd.DataFrame:
    homicide = pd.read_csv(HOMICIDE_CSV)
    homicide["Date"] = pd.to_datetime(homicide["Date"], format=DATE_FORMAT)
    homicide_hex = assign_hex_ids(homicide, "Latitude", "Longitude", hex_size_m)

    drug = pd.read_csv(DRUG_CSV)
    drug["Date"] = pd.to_datetime(drug["Date"], format=DATE_FORMAT)
    drug["season"] = drug["Date"].dt.month.map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "fall",
            10: "fall",
            11: "fall",
        }
    )
    drug["time_bin"] = pd.cut(
        drug["Date"].dt.hour,
        bins=TIME_BIN_BINS,
        labels=TIME_BIN_LABELS,
        ordered=True,
    )
    drug_hex = assign_hex_ids(drug, "Latitude", "Longitude", hex_size_m)

    infrastructure = pd.read_csv(INFRA_CSV).rename(
        columns={"latitude": "Latitude", "longitude": "Longitude"}
    )
    infrastructure["infrastructure_type"] = infrastructure[
        "infrastructure_type"
    ].fillna("unknown")
    infrastructure_hex = assign_hex_ids(
        infrastructure, "Latitude", "Longitude", hex_size_m
    )

    hex_ids = (
        set(homicide_hex["hex_id"])
        | set(drug_hex["hex_id"])
        | set(infrastructure_hex["hex_id"])
    )
    base = build_hex_index(hex_ids, hex_size_m)

    base = add_count_feature(base, homicide_hex, "hex_id", "homicide_count")
    base = add_count_feature(base, drug_hex, "hex_id", "drug_count")
    base = add_count_feature(base, infrastructure_hex, "hex_id", "infrastructure_total")

    drug_hex["Arrest"] = drug_hex["Arrest"].astype(int)
    drug_hex["Domestic"] = drug_hex["Domestic"].astype(int)
    drug_rates = drug_hex.groupby("hex_id").agg(
        drug_arrest_rate=("Arrest", "mean"),
        drug_domestic_rate=("Domestic", "mean"),
        drug_unique_blocks=("Block", "nunique"),
    )
    base = base.join(drug_rates, on="hex_id")

    base = add_group_counts(base, drug_hex, "season", "drug_season")
    base = add_group_counts(base, drug_hex, "time_bin", "drug_time")
    base = add_top_category_counts(
        base,
        drug_hex,
        "Location Description",
        "drug_location",
        top_drug_locations,
    )
    base = add_group_counts(base, infrastructure_hex, "infrastructure_type", "infra")

    community_area_lookup = (
        pd.concat(
            [
                homicide_hex[["hex_id", "Community Area"]],
                drug_hex[["hex_id", "Community Area"]],
            ],
            ignore_index=True,
        )
        .groupby("hex_id")["Community Area"]
        .agg(mode_or_nan)
    )
    base = base.join(
        community_area_lookup.rename("community_area_number"),
        on="hex_id",
    )

    socioeconomic = pd.read_csv(SOCIO_CSV)
    socioeconomic.columns = [col.strip() for col in socioeconomic.columns]
    socioeconomic = socioeconomic.rename(
        columns={
            "Community Area Number": "community_area_number",
            "PERCENT HOUSEHOLDS BELOW POVERTY": "poverty_pct",
            "PER CAPITA INCOME": "per_capita_income",
            "HARDSHIP INDEX": "hardship_index",
        }
    )
    base = base.merge(
        socioeconomic[
            [
                "community_area_number",
                "poverty_pct",
                "per_capita_income",
                "hardship_index",
            ]
        ],
        on="community_area_number",
        how="left",
    )

    base["community_area_missing"] = base["community_area_number"].isna().astype(int)

    count_prefixes = ("drug_", "infra_")
    for column in base.columns:
        if column.startswith(count_prefixes):
            base[column] = base[column].fillna(0)

    base["drug_arrest_rate"] = base["drug_arrest_rate"].fillna(0.0)
    base["drug_domestic_rate"] = base["drug_domestic_rate"].fillna(0.0)
    base["drug_unique_blocks"] = base["drug_unique_blocks"].fillna(0.0)
    base["community_area_number"] = base["community_area_number"].fillna(-1)

    for column in ["poverty_pct", "per_capita_income", "hardship_index"]:
        base[column] = base[column].fillna(base[column].median())

    numeric_columns = base.select_dtypes(include=["number"]).columns
    base[numeric_columns] = base[numeric_columns].fillna(0)

    return base.sort_values("hex_id").reset_index(drop=True)


def feature_family(feature_name: str) -> str:
    if feature_name.startswith("drug_location_"):
        return "drug_location"
    if feature_name.startswith("drug_time_"):
        return "drug_time"
    if feature_name.startswith("drug_season_"):
        return "drug_season"
    if feature_name.startswith("drug_"):
        return "drug_summary"
    if feature_name.startswith("infra_"):
        return "infrastructure"
    if feature_name in {"poverty_pct", "per_capita_income", "hardship_index"}:
        return "socioeconomic"
    if feature_name in {"community_area_number", "community_area_missing"}:
        return "community_area"
    if feature_name in {"hex_q", "hex_r", "centroid_lat", "centroid_lon"}:
        return "spatial"
    return "other"


def build_importance_frame(model, feature_names: list[str]) -> pd.DataFrame:
    booster = model.get_booster()
    importance_frames = []
    for importance_type in ("gain", "weight", "cover"):
        scores = booster.get_score(importance_type=importance_type)
        series = pd.Series(scores, dtype=float).reindex(feature_names).fillna(0.0)
        importance_frames.append(series.rename(importance_type))

    importance = pd.concat(importance_frames, axis=1)
    importance["sklearn_importance"] = model.feature_importances_
    importance = importance.reset_index().rename(columns={"index": "feature"})
    importance["feature_group"] = importance["feature"].map(feature_family)
    importance["gain_rank"] = (
        importance["gain"].rank(method="dense", ascending=False).astype(int)
    )
    return importance.sort_values(
        ["gain", "weight", "feature"], ascending=[False, False, True]
    )


def save_importance_outputs(
    importance: pd.DataFrame,
    task_name: str,
) -> None:
    plt = configure_matplotlib()

    csv_path = REPORTS_DIR / f"xgboost_{task_name}_feature_importance.csv"
    importance.to_csv(csv_path, index=False)

    group_summary = (
        importance.groupby("feature_group", as_index=False)[["gain", "weight", "cover"]]
        .sum()
        .sort_values("gain", ascending=False)
    )
    group_summary.to_csv(
        REPORTS_DIR / f"xgboost_{task_name}_feature_groups.csv",
        index=False,
    )

    top_features = importance.head(15).sort_values("gain", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features["feature"], top_features["gain"], color="#b85c38")
    ax.set_title(f"Top XGBoost Features ({task_name})")
    ax.set_xlabel("Gain")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(
        REPORTS_DIR / f"xgboost_{task_name}_feature_importance.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def train_hotspot_model(
    df: pd.DataFrame,
    features: list[str],
    hotspot_quantile: float,
    test_size: float,
    random_state: int,
) -> dict:
    non_zero_counts = df.loc[df["homicide_count"] > 0, "homicide_count"]
    hotspot_threshold = int(math.ceil(non_zero_counts.quantile(hotspot_quantile)))
    y = (df["homicide_count"] >= hotspot_threshold).astype(int)
    X = df[features]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        df[["hex_id", "centroid_lat", "centroid_lon", "homicide_count"]],
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = negatives / max(positives, 1)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=0,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "task": "hotspot",
        "hotspot_quantile": hotspot_quantile,
        "hotspot_threshold": hotspot_threshold,
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }

    holdout = meta_test.copy()
    holdout["actual_hotspot"] = y_test.to_numpy()
    holdout["predicted_hotspot_probability"] = probabilities
    holdout["predicted_hotspot_label"] = predictions

    return {
        "model": model,
        "metrics": metrics,
        "holdout": holdout.sort_values(
            "predicted_hotspot_probability", ascending=False
        ),
    }


def train_count_model(
    df: pd.DataFrame,
    features: list[str],
    test_size: float,
    random_state: int,
) -> dict:
    X = df[features]
    y = df["homicide_count"]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        df[["hex_id", "centroid_lat", "centroid_lon"]],
        test_size=test_size,
        random_state=random_state,
    )

    model = XGBRegressor(
        objective="count:poisson",
        tree_method="hist",
        n_estimators=450,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=0,
    )
    model.fit(X_train, y_train)

    predictions = np.clip(model.predict(X_test), 1e-6, None)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    metrics = {
        "task": "count",
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "mean_poisson_deviance": float(mean_poisson_deviance(y_test, predictions)),
    }

    holdout = meta_test.copy()
    holdout["actual_homicide_count"] = y_test.to_numpy()
    holdout["predicted_homicide_count"] = predictions
    holdout["absolute_error"] = np.abs(
        holdout["actual_homicide_count"] - holdout["predicted_homicide_count"]
    )

    return {
        "model": model,
        "metrics": metrics,
        "holdout": holdout.sort_values("absolute_error", ascending=False),
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    MODELING_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_table = load_feature_table(args.hex_size_m, args.top_drug_locations)

    non_zero_counts = feature_table.loc[
        feature_table["homicide_count"] > 0, "homicide_count"
    ]
    hotspot_threshold = int(math.ceil(non_zero_counts.quantile(args.hotspot_quantile)))
    feature_table["is_hotspot"] = (
        feature_table["homicide_count"] >= hotspot_threshold
    ).astype(int)
    feature_table.to_csv(
        MODELING_DIR / "chicago_hex_modeling_table.csv",
        index=False,
    )

    feature_columns = [
        column
        for column in feature_table.columns
        if column not in {"hex_id", "homicide_count", "is_hotspot"}
    ]

    metadata = {
        "rows": int(len(feature_table)),
        "feature_count": int(len(feature_columns)),
        "hex_size_m": args.hex_size_m,
        "hotspot_quantile": args.hotspot_quantile,
        "hotspot_threshold": hotspot_threshold,
    }

    if args.task in {"all", "hotspot"}:
        hotspot_result = train_hotspot_model(
            feature_table,
            feature_columns,
            args.hotspot_quantile,
            args.test_size,
            args.random_state,
        )
        hotspot_importance = build_importance_frame(
            hotspot_result["model"],
            feature_columns,
        )
        save_importance_outputs(hotspot_importance, "hotspot")
        hotspot_result["holdout"].to_csv(
            REPORTS_DIR / "xgboost_hotspot_holdout_predictions.csv",
            index=False,
        )
        write_json(
            REPORTS_DIR / "xgboost_hotspot_metrics.json",
            {**metadata, **hotspot_result["metrics"]},
        )
        print(
            "Hotspot model:",
            f"threshold={hotspot_result['metrics']['hotspot_threshold']}",
            f"roc_auc={hotspot_result['metrics']['roc_auc']:.3f}",
            f"average_precision={hotspot_result['metrics']['average_precision']:.3f}",
        )

    if args.task in {"all", "count"}:
        count_result = train_count_model(
            feature_table,
            feature_columns,
            args.test_size,
            args.random_state,
        )
        count_importance = build_importance_frame(
            count_result["model"],
            feature_columns,
        )
        save_importance_outputs(count_importance, "count")
        count_result["holdout"].to_csv(
            REPORTS_DIR / "xgboost_count_holdout_predictions.csv",
            index=False,
        )
        write_json(
            REPORTS_DIR / "xgboost_count_metrics.json",
            {**metadata, **count_result["metrics"]},
        )
        print(
            "Count model:",
            f"rmse={count_result['metrics']['rmse']:.3f}",
            f"mae={count_result['metrics']['mae']:.3f}",
            f"r2={count_result['metrics']['r2']:.3f}",
        )


if __name__ == "__main__":
    main()
