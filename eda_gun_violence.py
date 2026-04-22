"""
EDA: Gun Violence in Chicago — broad exploration of correlates.

Focus:
- What relates to gun violence beyond social infrastructure?
- Temporal: hour of day, day of week, month (hotspots?)
- Location: where does GV occur (Location Description, Community Area)?
- Social infrastructure: density/proximity vs GV by community area
- Other factors: e.g. liquor-related locations (TAVERN/LIQUOR STORE)
- Correlations and optional GAM partial dependence

Gun violence definition: HOMICIDE, WEAPONS VIOLATION, or Description contains
GUN/FIREARM/HANDGUN/SHOOT (covers fatal and non-fatal shootings, weapon offenses).
"""

import os
import warnings
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message=".*Operation between Series with different indexes.*",
    category=FutureWarning,
)

# Paths
CRIMES_CSV = "Crimes_-_2001_to_Present_20260220.csv"
OUTPUT_DIR = Path("eda_gun_violence_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load crimes with Dask and filter to gun violence
# ---------------------------------------------------------------------------
def load_gun_violence(use_sample_if_large=True, sample_years=(2015, 2025)):
    """Load crimes from CSV, filter to gun violence, return in-memory DataFrame."""
    if not os.path.isfile(CRIMES_CSV):
        raise FileNotFoundError(
            f"Crimes file not found: {CRIMES_CSV}. Place the CSV in the project root."
        )

    try:
        import dask.dataframe as dd
    except ImportError:
        raise ImportError("Install dask: pip install dask")

    dtype_dict = {
        "ID": "object",
        "Case Number": "object",
        "Date": "object",
        "Block": "object",
        "IUCR": "object",
        "Primary Type": "object",
        "Description": "object",
        "Location Description": "object",
        "Arrest": "object",
        "Domestic": "object",
        "Beat": "object",
        "District": "object",
        "Ward": "object",
        "Community Area": "object",
        "FBI Code": "object",
        "X Coordinate": "float64",
        "Y Coordinate": "float64",
        "Year": "int64",
        "Updated On": "object",
        "Latitude": "float64",
        "Longitude": "float64",
        "Location": "object",
    }

    cols = [
        "Date", "Primary Type", "Description", "Location Description",
        "Community Area", "Latitude", "Longitude", "Year",
    ]
    df = dd.read_csv(CRIMES_CSV, dtype=dtype_dict, assume_missing=True)[cols]

    # Restrict to years of interest to reduce size
    df = df[(df["Year"] >= sample_years[0]) & (df["Year"] <= sample_years[1])]

    # Define gun violence: HOMICIDE, WEAPONS VIOLATION, or gun/firearm in description
    # Cast to bool to avoid Dask FutureWarning (Series with different indexes/dtypes)
    is_homicide = (df["Primary Type"].str.upper() == "HOMICIDE").astype(bool)
    is_weapons = (df["Primary Type"].str.upper() == "WEAPONS VIOLATION").astype(bool)
    desc = df["Description"].fillna("").str.upper()
    is_gun_desc = (
        desc.str.contains("GUN|FIREARM|HANDGUN|SHOOT", regex=True, na=False).astype(bool)
    )
    is_gv = is_homicide | is_weapons | is_gun_desc
    gv = df[is_gv]

    n_gv = gv.shape[0].compute()
    print(f"Gun violence incidents (approx): {n_gv:,}")

    # Bring into memory (GV is a fraction of total, should be manageable)
    gv = gv.compute()
    return gv


def parse_date(gv: pd.DataFrame) -> pd.DataFrame:
    """Parse Date and add hour, day_of_week, month."""
    gv = gv.copy()
    gv["Date"] = pd.to_datetime(gv["Date"], errors="coerce")
    gv = gv.dropna(subset=["Date"])
    gv["hour"] = gv["Date"].dt.hour
    gv["day_of_week"] = gv["Date"].dt.dayofweek  # 0=Mon
    gv["month"] = gv["Date"].dt.month
    gv["year"] = gv["Date"].dt.year
    return gv


# ---------------------------------------------------------------------------
# 2. Social infrastructure and community-area aggregates
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def load_social_infrastructure():
    """Load libraries, parks, community centers; return dict of (name, lat, lon)."""
    infra = {}
    for name, f in [
        ("libraries", "chicago_libraries_osm.csv"),
        ("parks", "chicago_parks_osm.csv"),
        ("community_centers", "chicago_community_centers_osm.csv"),
    ]:
        if not os.path.isfile(f):
            continue
        df = pd.read_csv(f)
        if "latitude" in df.columns and "longitude" in df.columns:
            df = df.dropna(subset=["latitude", "longitude"])
            infra[name] = df[["latitude", "longitude"]].values
        else:
            infra[name] = np.empty((0, 2))
    return infra


def community_area_centroids(gv: pd.DataFrame) -> pd.DataFrame:
    """Mean lat/lon per Community Area from GV data."""
    gv_geo = gv.dropna(subset=["Community Area", "Latitude", "Longitude"]).copy()
    gv_geo["Community Area"] = pd.to_numeric(gv_geo["Community Area"], errors="coerce")
    gv_geo = gv_geo[gv_geo["Community Area"].between(1, 77)]
    ca = (
        gv_geo.groupby("Community Area", as_index=False)[["Latitude", "Longitude"]]
        .mean()
        .rename(columns={"Latitude": "ca_lat", "Longitude": "ca_lon"})
    )
    return ca


def count_infra_within_radius(ca_lat, ca_lon, infra_latlon, radius_km=1.61):
    """Count infra points within radius_km of (ca_lat, ca_lon). 1.61 km ~ 1 mile."""
    if len(infra_latlon) == 0:
        return 0
    d = haversine_km(
        np.full(len(infra_latlon), ca_lat),
        np.full(len(infra_latlon), ca_lon),
        infra_latlon[:, 0],
        infra_latlon[:, 1],
    )
    return int((d <= radius_km).sum())


def build_ca_dataset(gv: pd.DataFrame, infra: dict, radius_km=1.61) -> pd.DataFrame:
    """Community-area level: GV count and infra counts within radius."""
    ca_centroids = community_area_centroids(gv)
    gv_geo = gv.dropna(subset=["Community Area", "Latitude", "Longitude"]).copy()
    gv_geo["Community Area"] = pd.to_numeric(gv_geo["Community Area"], errors="coerce")
    gv_geo = gv_geo[gv_geo["Community Area"].between(1, 77)]
    ca_counts = (
        gv_geo.groupby("Community Area", as_index=False)
        .size()
        .rename(columns={"size": "gv_count"})
    )
    # Merge so every centroid has a count (left join from centroids)
    ca_centroids = ca_centroids.merge(ca_counts, on="Community Area", how="left")
    ca_centroids["gv_count"] = ca_centroids["gv_count"].fillna(0).astype(int)

    rows = []
    for _, r in ca_centroids.iterrows():
        ca_id = r["Community Area"]
        lat, lon = r["ca_lat"], r["ca_lon"]
        gv_count = int(r["gv_count"])
        row = {
            "Community Area": ca_id,
            "ca_lat": lat,
            "ca_lon": lon,
            "gv_count": gv_count,
        }
        for name, points in infra.items():
            row[f"{name}_within_{radius_km:.2f}km"] = count_infra_within_radius(
                lat, lon, points, radius_km
            )
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------
def run_temporal_eda(gv: pd.DataFrame):
    """Plots: GV by hour, day of week, month."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping temporal plots.")
        return

    gv = parse_date(gv)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Hour of day
    ax = axes[0, 0]
    hour_counts = gv["hour"].value_counts().sort_index()
    hour_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="gray", alpha=0.8)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Gun violence incidents")
    ax.set_title("GV by hour of day (hotspots)")
    ax.tick_params(axis="x", rotation=0)

    # Day of week
    ax = axes[0, 1]
    dow = gv["day_of_week"].value_counts().sort_index()
    dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(dow)]
    dow.plot(kind="bar", ax=ax, color="coral", edgecolor="gray", alpha=0.8)
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Gun violence incidents")
    ax.set_title("GV by day of week")

    # Month
    ax = axes[1, 0]
    month_counts = gv["month"].value_counts().sort_index()
    month_counts.plot(kind="bar", ax=ax, color="seagreen", edgecolor="gray", alpha=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Gun violence incidents")
    ax.set_title("GV by month")
    ax.tick_params(axis="x", rotation=0)

    # Year trend
    ax = axes[1, 1]
    year_counts = gv["year"].value_counts().sort_index()
    year_counts.plot(ax=ax, marker="o", color="purple", linewidth=2, markersize=6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Gun violence incidents")
    ax.set_title("GV by year (trend)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "temporal_gv.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def run_location_eda(gv: pd.DataFrame):
    """Location Description and Community Area distribution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Top location descriptions for GV
    loc_desc = gv["Location Description"].fillna("(missing)").value_counts().head(15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    loc_desc.plot(kind="barh", ax=ax, color="teal", alpha=0.8)
    ax.set_xlabel("Gun violence incidents")
    ax.set_title("GV by location type (top 15)")
    ax.invert_yaxis()

    # Community area (top 20)
    gv_ca = gv.dropna(subset=["Community Area"]).copy()
    gv_ca["Community Area"] = pd.to_numeric(gv_ca["Community Area"], errors="coerce")
    gv_ca = gv_ca[gv_ca["Community Area"].between(1, 77)]
    ca_counts = gv_ca["Community Area"].value_counts().head(20)

    ax = axes[1]
    ca_counts.plot(kind="barh", ax=ax, color="darkorange", alpha=0.8)
    ax.set_xlabel("Gun violence incidents")
    ax.set_title("GV by Community Area (top 20)")
    ax.invert_yaxis()

    plt.tight_layout()
    out = OUTPUT_DIR / "location_gv.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def run_correlation_eda(ca_df: pd.DataFrame):
    """Correlation of GV count with social infra and simple scatter."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    infra_cols = [c for c in ca_df.columns if c.startswith(("libraries", "parks", "community_centers")) and "within" in c]
    if not infra_cols:
        return

    X = ca_df[["gv_count"] + infra_cols].dropna()
    corr = X.corr()["gv_count"].drop("gv_count").sort_values(key=abs, ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    corr.plot(kind="barh", ax=ax, color=["green" if x < 0 else "red" for x in corr], alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Correlation with GV count")
    ax.set_title("Community area: GV count vs social infrastructure")
    ax.invert_yaxis()

    # Scatter: e.g. libraries vs GV
    ax = axes[1]
    lib_col = next((c for c in infra_cols if "libraries" in c), infra_cols[0])
    ax.scatter(ca_df[lib_col], ca_df["gv_count"], alpha=0.6, s=40, color="steelblue")
    ax.set_xlabel(lib_col.replace("_", " ").replace("within 1.61km", "(within ~1 mi)"))
    ax.set_ylabel("GV count")
    ax.set_title("GV count vs infrastructure (community area)")
    plt.tight_layout()
    out = OUTPUT_DIR / "correlation_gv_infra.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    print("Correlations with gv_count:")
    print(corr.to_string())

    # Correlation heatmap (CA-level)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(X.corr(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(X.columns)))
        ax.set_yticks(range(len(X.columns)))
        ax.set_xticklabels([c[:20] for c in X.columns], rotation=45, ha="right")
        ax.set_yticklabels([c[:20] for c in X.columns])
        plt.colorbar(im, ax=ax, label="Correlation")
        ax.set_title("Community area: correlation matrix (GV + infra)")
        plt.tight_layout()
        out = OUTPUT_DIR / "correlation_heatmap.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")
    except Exception as e:
        print(f"Correlation heatmap skipped: {e}")


def run_gam_pdp(ca_df: pd.DataFrame):
    """GAM partial dependence (optional)."""
    try:
        from pygam import LinearGAM, s
    except ImportError:
        print("pygam not installed. Install with: pip install pygam")
        return

    infra_cols = [
        c for c in ca_df.columns
        if "within" in c and ca_df[c].dtype in (np.int64, np.float64)
    ]
    if len(infra_cols) < 1:
        return

    n_terms = min(3, len(infra_cols))
    infra_cols = infra_cols[:n_terms]
    X = ca_df[infra_cols].fillna(0).values
    y = ca_df["gv_count"].values
    y_log = np.log1p(y)

    terms = [s(i) for i in range(n_terms)]
    gam = LinearGAM(reduce(lambda a, b: a + b, terms))
    gam.fit(X, y_log)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, n_terms, figsize=(4 * n_terms, 4))
        if n_terms == 1:
            axes = [axes]
        for i in range(n_terms):
            XX = gam.generate_X_grid(term=i)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
            pdep = np.asarray(pdep).ravel()
            confi = np.asarray(confi)
            ax = axes[i]
            ax.plot(XX[:, i], pdep, color="steelblue", linewidth=2, label="Partial dependence")
            ax.fill_between(
                XX[:, i],
                confi[:, 0],
                confi[:, 1],
                color="steelblue",
                alpha=0.25,
                label="95% CI",
            )
            ax.set_xlabel(infra_cols[i].replace("_", " ")[:30])
            ax.set_ylabel("Partial dependence (log GV)")
            ax.set_title(f"GAM partial dependence (term {i+1})")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = OUTPUT_DIR / "gam_partial_dependence.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")
    except Exception as e:
        print(f"GAM PDP plot skipped: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("EDA: Gun violence and correlates (broad)")
    print("=" * 60)

    gv = load_gun_violence(sample_years=(2015, 2025))
    gv = parse_date(gv)
    print(f"Gun violence rows in memory: {len(gv):,}")

    # Quick summary: Primary Type and Description (verify GV definition)
    print("\n--- GV by Primary Type (sample) ---")
    print(gv["Primary Type"].value_counts().head(10).to_string())
    print("\n--- GV by Description (top 10) ---")
    print(gv["Description"].value_counts().head(10).to_string())

    # Temporal
    print("\n--- Temporal EDA ---")
    run_temporal_eda(gv)

    # Location
    print("\n--- Location EDA ---")
    run_location_eda(gv)

    # Social infrastructure + CA
    print("\n--- Social infrastructure & community areas ---")
    infra = load_social_infrastructure()
    radius_km = 1.61  # ~1 mile
    ca_df = build_ca_dataset(gv, infra, radius_km=radius_km)
    ca_df.to_csv(OUTPUT_DIR / "community_area_gv_infra.csv", index=False)
    print(f"Saved community_area_gv_infra.csv ({len(ca_df)} community areas)")

    run_correlation_eda(ca_df)

    # GAM (optional)
    print("\n--- GAM partial dependence ---")
    run_gam_pdp(ca_df)

    print("\n" + "=" * 60)
    print("EDA outputs in:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
