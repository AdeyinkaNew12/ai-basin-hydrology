#!/usr/bin/env python3
"""
Water-distance ECDF for rivers, lakes, coastlines, and nearest water feature.

This script compares AI, Power, and TRI facility distances with basin-matched
random baselines using precomputed distance CSV files.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_paths import DATA_FOLDERS, output_folder


DATA_ROOT = DATA_FOLDERS["water_project"]
OUTDIR = Path(__file__).resolve().parents[2] / "final_results" / "water_distance_ecdf"
os.makedirs(OUTDIR, exist_ok=True)

PROX_DIR = "/mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/water_proximity_threshold_ord5"

FILES = {
    "AI": {
        "sector": os.path.join(PROX_DIR, "data_center_sector_nearest_water_BASIN_FILTER_ORD5.csv"),
        "random": os.path.join(PROX_DIR, "data_center_random_nearest_water_BASIN_FILTER_ORD5.csv"),
    },
    "Power": {
        "sector": os.path.join(PROX_DIR, "power_plant_sector_nearest_water_BASIN_FILTER_ORD5.csv"),
        "random": os.path.join(PROX_DIR, "power_plant_random_nearest_water_BASIN_FILTER_ORD5.csv"),
    },
    "TRI": {
        "sector": os.path.join(PROX_DIR, "industry_TRI_sector_nearest_water_BASIN_FILTER_ORD5.csv"),
        "random": os.path.join(PROX_DIR, "industry_TRI_random_nearest_water_BASIN_FILTER_ORD5.csv"),
    },
}

OUT_PNG = os.path.join(OUTDIR, "WaterDistance_ECDF_allfeatures.png")
OUT_PDF = os.path.join(OUTDIR, "WaterDistance_ECDF_allfeatures.pdf")
OUT_SUMMARY = os.path.join(OUTDIR, "WaterDistance_ECDF_input_summary.csv")

SECTORS = ["AI", "Power", "TRI"]
COLORS = {"AI": "#1f78b4", "Power": "#d95f02", "TRI": "#1b9e77"}


def load_distance_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required input file: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    required = ["dist_river_km", "dist_lake_km", "dist_coast_km"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "dist_any_km" not in df.columns:
        df["dist_any_km"] = df[required].min(axis=1)
    else:
        df["dist_any_km"] = pd.to_numeric(df["dist_any_km"], errors="coerce")

    return df


def ecdf(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    x = np.sort(x)

    if x.size == 0:
        return np.array([np.nan]), np.array([np.nan])

    y = np.arange(1, x.size + 1) / x.size
    return x, y


def main():
    print("[INFO] Water-distance ECDF: rivers, lakes, coastlines, nearest water")
    print("[INFO] DATA_ROOT:", DATA_ROOT)
    print("[INFO] OUTDIR:", OUTDIR)
    print("[INFO] PROX_DIR:", PROX_DIR)

    stores = {}
    summary_rows = []

    for sector in SECTORS:
        stores[sector] = {
            "sector": load_distance_file(FILES[sector]["sector"]),
            "random": load_distance_file(FILES[sector]["random"]),
        }

        for group in ["sector", "random"]:
            df = stores[sector][group]
            summary_rows.append({
                "sector": sector,
                "group": group,
                "input_file": FILES[sector][group],
                "N": len(df),
                "median_dist_river_km": df["dist_river_km"].median(),
                "median_dist_lake_km": df["dist_lake_km"].median(),
                "median_dist_coast_km": df["dist_coast_km"].median(),
                "median_dist_any_km": df["dist_any_km"].median(),
            })

        print(f"[FOUND] {sector} sector:", FILES[sector]["sector"])
        print(f"[FOUND] {sector} random:", FILES[sector]["random"])
        print(
            f"[INFO] {sector}: sector N={len(stores[sector]['sector']):,}, "
            f"random N={len(stores[sector]['random']):,}"
        )

    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    print("[SAVED]", OUT_SUMMARY)

    panels = [
        ("river", "Major rivers (ORD_STRA ≥ 5)"),
        ("lake", "Lakes (HydroLAKES)"),
        ("coast", "Coastline (Natural Earth)"),
        ("any", "Any water (river/lake/coast)"),
    ]

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })

    fig, axes = plt.subplots(
        2, 2,
        figsize=(6.0, 5.2),
        dpi=300,
        constrained_layout=False
    )
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, panels):
        col = f"dist_{key}_km"

        for sector in SECTORS:
            sector_df = stores[sector]["sector"]
            random_df = stores[sector]["random"]

            xs, ys = ecdf(sector_df[col].values)
            xr, yr = ecdf(random_df[col].values)

            ax.plot(xs, ys, color=COLORS[sector], lw=1.7, ls="-")
            ax.plot(xr, yr, color=COLORS[sector], lw=1.7, ls="--", alpha=0.70)

        ax.set_xscale("log")
        ax.set_xlabel("Distance (km, log scale)", fontsize=8.5, fontweight="normal", labelpad=3)
        ax.set_ylabel("ECDF", fontsize=8.5, fontweight="normal", labelpad=3)
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=2)
        ax.grid(True, linewidth=0.6, alpha=0.30)

        vals = []
        for sector in SECTORS:
            vals.append(stores[sector]["sector"][col].values)
            vals.append(stores[sector]["random"][col].values)

        vals = np.concatenate(vals).astype(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]

        if vals.size > 0:
            xmin = max(np.quantile(vals, 0.001), 0.1)
            xmax = np.quantile(vals, 0.995)
            ax.set_xlim(xmin, max(xmax, xmin * 10))

    style_handles = [
        Line2D([0], [0], color="black", lw=2.6, ls="-", label="Sector"),
        Line2D([0], [0], color="black", lw=2.6, ls="--", label="Basin-matched random"),
    ]

    color_handles = []
    for sector in SECTORS:
        n_sector = len(stores[sector]["sector"])
        n_random = len(stores[sector]["random"])
        color_handles.append(
            Line2D(
                [0], [0],
                color=COLORS[sector],
                lw=3.2,
                label=f"{sector} (N={n_sector:,}; random N={n_random:,})",
            )
        )

    legend = fig.legend(
        handles=style_handles + color_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.22,
        top=0.95,
        wspace=0.22,
        hspace=0.48
    )

    fig.savefig(
        OUT_PNG,
        dpi=350,
        bbox_inches="tight",
        bbox_extra_artists=(legend,)
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
        bbox_extra_artists=(legend,)
    )
    plt.close(fig)

    print("[SAVED]", OUT_PNG)
    print("[SAVED]", OUT_PDF)


if __name__ == "__main__":
    main()
