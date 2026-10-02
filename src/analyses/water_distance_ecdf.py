#!/usr/bin/env python3
"""
Generate water-distance ECDFs and Table 2 statistics for AI, Power,
and TRI facilities relative to basin-matched random locations.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# PATHS
# ============================================================

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_paths import DATA_FOLDERS


DATA_ROOT = DATA_FOLDERS["water_project"]

OUTDIR = (
    Path(__file__).resolve().parents[2]
    / "final_results"
    / "water_distance_ecdf"
)

os.makedirs(OUTDIR, exist_ok=True)


PROX_DIR = (
    "/mnt/disk3/aoolaseinde/projects/"
    "integrated_dc_water_pathways/results/"
    "water_proximity_threshold_ord5"
)


FILES = {
    "AI": {
        "sector": os.path.join(
            PROX_DIR,
            "data_center_sector_nearest_water_BASIN_FILTER_ORD5.csv",
        ),
        "random": os.path.join(
            PROX_DIR,
            "data_center_random_nearest_water_BASIN_FILTER_ORD5.csv",
        ),
    },
    "Power": {
        "sector": os.path.join(
            PROX_DIR,
            "power_plant_sector_nearest_water_BASIN_FILTER_ORD5.csv",
        ),
        "random": os.path.join(
            PROX_DIR,
            "power_plant_random_nearest_water_BASIN_FILTER_ORD5.csv",
        ),
    },
    "TRI": {
        "sector": os.path.join(
            PROX_DIR,
            "industry_TRI_sector_nearest_water_BASIN_FILTER_ORD5.csv",
        ),
        "random": os.path.join(
            PROX_DIR,
            "industry_TRI_random_nearest_water_BASIN_FILTER_ORD5.csv",
        ),
    },
}


OUT_PNG = os.path.join(
    OUTDIR,
    "WaterDistance_ECDF_allfeatures.png",
)

OUT_PDF = os.path.join(
    OUTDIR,
    "WaterDistance_ECDF_allfeatures.pdf",
)

OUT_SUMMARY = os.path.join(
    OUTDIR,
    "WaterDistance_ECDF_input_summary.csv",
)

OUT_TABLE = os.path.join(
    OUTDIR,
    "WaterDistance_Table2_statistics.csv",
)


SECTORS = ["AI", "Power", "TRI"]

COLORS = {
    "AI": "#1f78b4",
    "Power": "#d95f02",
    "TRI": "#1b9e77",
}


FEATURES = [
    ("River", "dist_river_km"),
    ("Lake", "dist_lake_km"),
    ("Coastline", "dist_coast_km"),
    ("Nearest surface water", "dist_any_km"),
]


# ============================================================
# LOAD DATA
# ============================================================

def load_distance_file(path):
    """
    Load one ORD5 distance file and calculate nearest
    surface-water distance.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required input file: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    required = [
        "dist_river_km",
        "dist_lake_km",
        "dist_coast_km",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path} missing columns: {missing}"
        )

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Recalculate this consistently every time.
    df["dist_any_km"] = (
        df[required]
        .min(axis=1)
    )

    return df


# ============================================================
# CLEAN DISTANCE VECTOR
# ============================================================

def clean_vec(values):
    """
    Retain finite, non-negative distances.

    Zero is retained because a zero-distance observation
    represents a valid spatial intersection with a mapped
    water feature.
    """

    x = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(dtype=float)

    x = x[np.isfinite(x)]

    x = x[x >= 0]

    return x


# ============================================================
# ECDF
# ============================================================

def ecdf(values):

    x = clean_vec(values)

    x = np.sort(x)

    if x.size == 0:
        return (
            np.array([np.nan]),
            np.array([np.nan]),
        )

    y = (
        np.arange(1, x.size + 1)
        / x.size
    )

    return x, y


# ============================================================
# CLIFF'S DELTA
# ============================================================

def cliffs_delta(x, y):
    """
    Calculate Cliff's delta.

    Negative values indicate that sector distances tend
    to be smaller than basin-matched random distances.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if len(x) == 0 or len(y) == 0:
        return np.nan

    n = len(x)
    m = len(y)

    gt = 0
    lt = 0

    # Chunking avoids constructing an excessively
    # large comparison matrix for Power and TRI.
    chunk = max(
        1,
        2_000_000 // max(1, m),
    )

    for i in range(
        0,
        n,
        chunk,
    ):

        xx = x[i:i + chunk]

        diff = (
            xx[:, None]
            - y[None, :]
        )

        gt += np.sum(
            diff > 0
        )

        lt += np.sum(
            diff < 0
        )

    delta = (
        (gt - lt)
        / (n * m)
    )

    return float(delta)


# ============================================================
# BENJAMINI-HOCHBERG FDR
# ============================================================

def bh_fdr(pvals):
    """
    Benjamini-Hochberg false discovery rate correction.
    """

    pvals = np.asarray(
        pvals,
        dtype=float,
    )

    result = np.full_like(
        pvals,
        np.nan,
        dtype=float,
    )

    valid = np.where(
        np.isfinite(pvals)
    )[0]

    if len(valid) == 0:
        return result

    pv = pvals[valid]

    order = np.argsort(pv)

    sorted_p = pv[order]

    n = len(sorted_p)

    q = (
        sorted_p
        * n
        / np.arange(1, n + 1)
    )

    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.clip(
        q,
        0,
        1,
    )

    result[
        valid[order]
    ] = q

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "[INFO] Water-distance ECDF and statistical analysis"
    )

    print(
        "[INFO] DATA_ROOT:",
        DATA_ROOT,
    )

    print(
        "[INFO] OUTDIR:",
        OUTDIR,
    )

    print(
        "[INFO] PROX_DIR:",
        PROX_DIR,
    )


    # ========================================================
    # READ ALL SIX INPUT FILES
    # ========================================================

    stores = {}

    summary_rows = []

    for sector in SECTORS:

        stores[sector] = {
            "sector": load_distance_file(
                FILES[sector]["sector"]
            ),
            "random": load_distance_file(
                FILES[sector]["random"]
            ),
        }


        for group in [
            "sector",
            "random",
        ]:

            df = stores[sector][group]

            summary_rows.append(
                {
                    "sector":
                        sector,

                    "group":
                        group,

                    "input_file":
                        FILES[sector][group],

                    "N":
                        len(df),

                    "median_dist_river_km":
                        df[
                            "dist_river_km"
                        ].median(),

                    "median_dist_lake_km":
                        df[
                            "dist_lake_km"
                        ].median(),

                    "median_dist_coast_km":
                        df[
                            "dist_coast_km"
                        ].median(),

                    "median_dist_any_km":
                        df[
                            "dist_any_km"
                        ].median(),
                }
            )


        print(
            f"[FOUND] {sector} sector: "
            f"{FILES[sector]['sector']}"
        )

        print(
            f"[FOUND] {sector} random: "
            f"{FILES[sector]['random']}"
        )

        print(
            f"[INFO] {sector}: "
            f"sector N={len(stores[sector]['sector']):,}, "
            f"random N={len(stores[sector]['random']):,}"
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    print(
        "[SAVED]",
        OUT_SUMMARY,
    )


    # ========================================================
    # TABLE 2 STATISTICS
    # ========================================================

    table_rows = []


    for feature_name, col in FEATURES:

        feature_rows = []


        for sector in SECTORS:

            x = clean_vec(
                stores[
                    sector
                ]["sector"][col]
            )

            y = clean_vec(
                stores[
                    sector
                ]["random"][col]
            )


            ks_result = ks_2samp(
                x,
                y,
                alternative="two-sided",
                method="auto",
            )


            feature_rows.append(
                {
                    "Water Feature":
                        feature_name,

                    "Sector":
                        sector,

                    "N sector":
                        len(x),

                    "N random":
                        len(y),

                    "Median (km)":
                        float(
                            np.median(x)
                        ),

                    "Random Median (km)":
                        float(
                            np.median(y)
                        ),

                    "KS D":
                        float(
                            ks_result.statistic
                        ),

                    "KS p":
                        float(
                            ks_result.pvalue
                        ),

                    "Cliff's delta":
                        cliffs_delta(
                            x,
                            y,
                        ),
                }
            )


        # BH-FDR correction is performed across the
        # AI, Power, and TRI comparisons for this feature.
        pvalues = [
            row["KS p"]
            for row in feature_rows
        ]

        qvalues = bh_fdr(
            pvalues
        )


        for row, qvalue in zip(
            feature_rows,
            qvalues,
        ):

            row["KS q(FDR)"] = float(
                qvalue
            )

            table_rows.append(
                row
            )


    table_df = pd.DataFrame(
        table_rows
    )


    table_df = table_df[
        [
            "Water Feature",
            "Sector",
            "N sector",
            "N random",
            "Median (km)",
            "Random Median (km)",
            "KS D",
            "KS p",
            "KS q(FDR)",
            "Cliff's delta",
        ]
    ]


    table_df.to_csv(
        OUT_TABLE,
        index=False,
    )


    print(
        "[SAVED]",
        OUT_TABLE,
    )


    # ========================================================
    # PRINT TABLE 2 RESULTS
    # ========================================================

    print()

    print(
        "=" * 110
    )

    print(
        "TABLE 2 — STATISTICS WITH ZERO DISTANCES RETAINED"
    )

    print(
        "=" * 110
    )


    for _, row in table_df.iterrows():

        feature = row[
            "Water Feature"
        ]

        sector = row[
            "Sector"
        ]

        median_sector = row[
            "Median (km)"
        ]

        median_random = row[
            "Random Median (km)"
        ]

        ks_d = row[
            "KS D"
        ]

        ks_q = row[
            "KS q(FDR)"
        ]

        cliff = row[
            "Cliff's delta"
        ]


        print(
            f"{feature:22s} "
            f"{sector:6s} | "
            f"Median={median_sector:8.3f} | "
            f"Random={median_random:8.3f} | "
            f"KS D={ks_d:.6f} | "
            f"KS q={ks_q:.12g} | "
            f"Cliff={cliff:.6f}"
        )


    # ========================================================
    # ECDF FIGURE
    # ========================================================

    panels = [
        (
            "river",
            "Major rivers (ORD_STRA ≥ 5)",
        ),
        (
            "lake",
            "Lakes (HydroLAKES)",
        ),
        (
            "coast",
            "Coastline (Natural Earth)",
        ),
        (
            "any",
            "Any water (river/lake/coast)",
        ),
    ]


    plt.rcParams.update(
        {
            "font.size":
                8.5,

            "axes.titleweight":
                "bold",

            "axes.labelweight":
                "bold",

            "axes.linewidth":
                1.0,

            "xtick.major.width":
                1.0,

            "ytick.major.width":
                1.0,
        }
    )


    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            6.0,
            5.2,
        ),
        dpi=300,
        constrained_layout=False,
    )


    axes = axes.flatten()


    for ax, (
        key,
        title,
    ) in zip(
        axes,
        panels,
    ):

        col = (
            f"dist_{key}_km"
        )


        for sector in SECTORS:

            sector_df = stores[
                sector
            ]["sector"]

            random_df = stores[
                sector
            ]["random"]


            xs, ys = ecdf(
                sector_df[
                    col
                ].values
            )

            xr, yr = ecdf(
                random_df[
                    col
                ].values
            )


            ax.plot(
                xs,
                ys,
                color=COLORS[
                    sector
                ],
                lw=1.7,
                ls="-",
            )


            ax.plot(
                xr,
                yr,
                color=COLORS[
                    sector
                ],
                lw=1.7,
                ls="--",
                alpha=0.70,
            )


        ax.set_xscale(
            "log"
        )


        ax.set_xlabel(
            "Distance (km, log scale)",
            fontsize=8.5,
            fontweight="normal",
            labelpad=3,
        )


        ax.set_ylabel(
            "ECDF",
            fontsize=8.5,
            fontweight="normal",
            labelpad=3,
        )


        ax.set_title(
            title,
            fontsize=8.5,
            fontweight="bold",
            pad=2,
        )


        ax.grid(
            True,
            linewidth=0.6,
            alpha=0.30,
        )


        # Positive distances are used only for determining
        # logarithmic plotting limits.
        #
        # Zero-distance observations remain included in
        # the statistical analysis above.

        vals = []


        for sector in SECTORS:

            vals.append(
                stores[
                    sector
                ]["sector"][
                    col
                ].values
            )

            vals.append(
                stores[
                    sector
                ]["random"][
                    col
                ].values
            )


        vals = np.concatenate(
            vals
        ).astype(float)


        vals = vals[
            np.isfinite(vals)
            & (vals > 0)
        ]


        if vals.size > 0:

            xmin = max(
                np.quantile(
                    vals,
                    0.001,
                ),
                0.1,
            )

            xmax = np.quantile(
                vals,
                0.995,
            )


            ax.set_xlim(
                xmin,
                max(
                    xmax,
                    xmin * 10,
                ),
            )


    # ========================================================
    # LEGEND
    # ========================================================

    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.6,
            ls="-",
            label="Sector",
        ),

        Line2D(
            [0],
            [0],
            color="black",
            lw=2.6,
            ls="--",
            label="Basin-matched random",
        ),
    ]


    color_handles = []


    for sector in SECTORS:

        n_sector = len(
            stores[
                sector
            ]["sector"]
        )

        n_random = len(
            stores[
                sector
            ]["random"]
        )


        color_handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[
                    sector
                ],
                lw=3.2,
                label=(
                    f"{sector} "
                    f"(N={n_sector:,}; "
                    f"random N={n_random:,})"
                ),
            )
        )


    legend = fig.legend(
        handles=(
            style_handles
            + color_handles
        ),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(
            0.5,
            -0.02,
        ),
        frameon=False,
    )


    plt.tight_layout(
        rect=[
            0,
            0.06,
            1,
            0.96,
        ]
    )


    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.22,
        top=0.95,
        wspace=0.22,
        hspace=0.48,
    )


    fig.savefig(
        OUT_PNG,
        dpi=350,
        bbox_inches="tight",
        bbox_extra_artists=(
            legend,
        ),
    )


    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
        bbox_extra_artists=(
            legend,
        ),
    )


    plt.close(
        fig
    )


    print()

    print(
        "[SAVED]",
        OUT_PNG,
    )

    print(
        "[SAVED]",
        OUT_PDF,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
