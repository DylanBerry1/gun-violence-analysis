"""
Builds the rank order plots for homicides within a hex area
Make crime density a parameter (e.g., places with high homicide% vs low/no homicide%)
Pareto statistic to fit the data
Rank order handles heavy-tailed phenomena with scale (alpha) and shape (xmin) parameters
Plot the log-log complementary CDF (1-CDF)
The log-log is the log of rank, and the log of probability
Slope of the line gives the shape (xmin)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import powerlaw
import plotly.express as px


def plot_real_crime_rank_order():
    csv_path = Path("data/processed/hex/chicago_homicides_hex_counts.csv")
    df = pd.read_csv(csv_path)

    crime_counts = df["count"].sort_values(ascending=False).values

    # calculates ranks and the CCDF (prob of having > X crimes)
    ranks = np.arange(1, len(crime_counts) + 1)
    ccdf = ranks / len(crime_counts)

    # log-log plot
    plt.figure()
    plt.loglog(
        crime_counts,
        ccdf,
        marker="o",
        linestyle="none",
        markersize=5,
        alpha=0.7,
        color="#f16913",
    )  # matching your map's orange/red
    plt.title("Log-Log Rank Order Graph of Chicago Homicides (500m Hexes)")
    plt.xlabel("# of Homicides in Hex (Log)")
    plt.ylabel(r"$\mathbb{P}$(Hex Having > X Homicides) (Log)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    out_path = Path("reports/figures/homicide_rank_order_plot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.show()


def analyze_and_plot_heavy_tail():
    csv_path = Path("data/processed/hex/chicago_homicides_hex_counts.csv")
    df = pd.read_csv(csv_path)

    df_active = df[df["count"] > 0].copy()
    df_active = df_active.sort_values(by="count", ascending=False).reset_index(
        drop=True
    )
    df_active["rank"] = df_active.index + 1
    df_active["ccdf"] = df_active["rank"] / len(df_active)  ## p > X

    crime_counts = df_active["count"].values

    # powerlaw finds the optimal x_min that minimizes the K-S distance between the data and the fit
    fit = powerlaw.Fit(crime_counts, discrete=True, verbose=False)

    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin

    ## log-log plot
    fig = px.scatter(
        df_active,
        x="count",
        y="ccdf",
        hover_data={"count": True, "ccdf": True, "hex_id": True, "rank": False},
        log_x=True,
        log_y=True,
        title=rf"Log-Log Rank Order of Chicago Homicides (500m Hexes) (alpha={alpha:.2f}, x_min={xmin})",
        labels={
            "count": "# of Homicides in Hex (Log)",
            "ccdf": "P(Hex Having > X Homicides) (Log)",
            "hex_id": "Hexagon ID",
        },
        color_discrete_sequence=["#f16913"],
    )
    ## the line where the power law starts (x_min)
    fig.add_vline(
        x=xmin,
        line_dash="dash",
        line_color="gray",
        annotation_text="  x_min threshold",
        annotation_position="top right",
    )
    fig.update_layout(template="plotly_white", font=dict(size=14), hovermode="closest")
    max_count = df_active["count"].max()
    max_exponent = np.log10(max_count) + 0.1
    fig.update_xaxes(range=[-0.05, max_exponent])
    out_path = Path("reports/figures/interactive_rank_order.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)
    fig.show()


if __name__ == "__main__":
    plot_real_crime_rank_order()
    analyze_and_plot_heavy_tail()
