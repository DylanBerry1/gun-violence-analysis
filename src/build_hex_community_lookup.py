"""
Build hex_id -> dominant Community Area from the full Chicago crimes parquet.

Each incident row supplies a Community Area code; we assign the incident to
the same 500 m hex grid used by build_correlation_analysis.py, then take the
modal community area per hex (ties broken by pandas .mode()).

Output:
    data/processed/hex_dominant_community_area.csv

Columns:
    hex_id, community_area_number, n_incidents_used

Run (from repo root) when the full crimes file is present:
    python3 src/build_hex_community_lookup.py

Input:
    data/raw/all_chicago_crimes.parquet
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_OUT = ROOT_DIR / "data" / "processed"

CRIMES_PARQUET = DATA_RAW / "all_chicago_crimes.parquet"
OUT_PARQUET = DATA_OUT / "hex_dominant_community_area.parquet"

HEX_SIZE_M = 500.0

CHICAGO_BOUNDS = {
    "lat_min": 41.5,
    "lat_max": 42.1,
    "lon_min": -88.0,
    "lon_max": -87.5,
}


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


def main() -> None:
    if not CRIMES_PARQUET.is_file():
        print(f"Missing {CRIMES_PARQUET}; nothing to do.", file=sys.stderr)
        sys.exit(1)

    counters: dict[str, Counter] = {}

    print(f"Reading {CRIMES_PARQUET.name} …")
    frame = pd.read_parquet(CRIMES_PARQUET, columns=["Latitude", "Longitude", "Community Area"])
    frame["Latitude"] = pd.to_numeric(frame["Latitude"], errors="coerce")
    frame["Longitude"] = pd.to_numeric(frame["Longitude"], errors="coerce")
    frame = frame.dropna(subset=["Latitude", "Longitude", "Community Area"])
    in_bounds = (
        (frame["Latitude"] >= CHICAGO_BOUNDS["lat_min"])
        & (frame["Latitude"] <= CHICAGO_BOUNDS["lat_max"])
        & (frame["Longitude"] >= CHICAGO_BOUNDS["lon_min"])
        & (frame["Longitude"] <= CHICAGO_BOUNDS["lon_max"])
    )
    frame = frame.loc[in_bounds]

    hex_info = assign_hex_ids(
        frame["Latitude"].to_numpy(),
        frame["Longitude"].to_numpy(),
        HEX_SIZE_M,
    )
    frame = frame.reset_index(drop=True)
    frame["hex_id"] = hex_info["hex_id"].to_numpy()
    frame["Community Area"] = frame["Community Area"].astype(int)

    for hid, sub in frame.groupby("hex_id", sort=False)["Community Area"]:
        ctr = counters.setdefault(hid, Counter())
        ctr.update(sub.tolist())

    rows = []
    for hid, ctr in counters.items():
        ca, n = ctr.most_common(1)[0]
        rows.append(
            {
                "hex_id": hid,
                "community_area_number": ca,
                "n_incidents_used": n,
            }
        )

    out = pd.DataFrame(rows).sort_values("hex_id").reset_index(drop=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET} ({len(out):,} hex rows).")


if __name__ == "__main__":
    main()
