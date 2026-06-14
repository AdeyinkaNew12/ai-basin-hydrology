#!/usr/bin/env python3
"""
Groundwater-DC paper-ready figure generator.

Produces four paper-ready figures from the integrated data-center water-pathway
outputs:

1. Hydrologic pathway classification
2. Reservoir/lake distance distribution
3. Aquifer overrepresentation
4. Sector composition in top HUC8 basins

This script does not recompute the full spatial analysis. It reads existing
tables produced by the integrated water-pathways workflow.
"""

import os
import textwrap
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common_paths import output_folder


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
RESULTS_ROOT = output_folder("")
INTEGRATED_DIR = output_folder("integrated_dc_water_pathways")
TABLEDIR = os.path.join(INTEGRATED_DIR, "tables")

OUTDIR = output_folder("groundwater_dc_figures")
FIGDIR = os.path.join(OUTDIR, "figures")
TABLE_OUTDIR = os.path.join(OUTDIR, "tables")

os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABLE_OUTDIR, exist_ok=True)

print("[INFO] Groundwater-DC figure generator")
print("[INFO] Integrated results folder:", INTEGRATED_DIR)
print("[INFO] Table folder:", TABLEDIR)
print("[INFO] Output folder:", OUTDIR)


# ------------------------------------------------------------
# Plot style
# ------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 450,
    "font.size": 11,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save_fig(fig, filename):
    path = os.path.join(FIGDIR, filename)
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", path)
    return path


def wrap_labels(labels, width=32):
    return ["\n".join(textwrap.wrap(str(x), width=width)) for x in labels]


def find_table(possible_names):
    search_dirs = [
        TABLEDIR,
        INTEGRATED_DIR,
        RESULTS_ROOT,
    ]

    for folder in search_dirs:
        if not os.path.exists(folder):
            continue

        for name in possible_names:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                print("[READ]", path)
                return pd.read_csv(path)

    raise FileNotFoundError(
        "Could not find any of these tables:\n"
        + "\n".join(possible_names)
        + "\nSearched in:\n"
        + "\n".join(search_dirs)
    )


def first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ------------------------------------------------------------
# Load required tables
# ------------------------------------------------------------
dc = find_table([
    "data_centers_integrated_water_pathways.csv",
    "dc_water_supply_pathways.csv",
])

master = find_table([
    "master_huc8_integrated_table.csv",
    "huc8_dc_counts.csv",
])

# Aquifer enrichment may already exist. If not, use aquifer count table.
try:
    aq = find_table([
        "aquifer_enrichment_table.csv",
        "aquifer_dc_counts.csv",
    ])
except FileNotFoundError:
    aq = None


# ------------------------------------------------------------
# Figure 1: Hydrologic pathway classification
# ------------------------------------------------------------
pathway_col = first_existing_column(dc, [
    "combined_supply_pathway_proxy",
    "supply_pathway_class",
    "hydrologic_pathway_classification",
    "pathway_classification",
])

if pathway_col is None:
    raise ValueError(
        "Could not find pathway-classification column in data-center table. "
        f"Available columns: {list(dc.columns)}"
    )

pathway = (
    dc[pathway_col]
    .fillna("Mixed or uncertain")
    .astype(str)
    .value_counts()
    .rename_axis("Hydrologic pathway classification")
    .reset_index(name="Count of AI data centers")
)

label_clean = {
    "Surface-water-leaning": "Surface-water leaning",
    "Surface-water leaning": "Surface-water leaning",
    "Mixed: groundwater + nearby surface water": "Mixed groundwater and nearby surface water",
    "Mixed groundwater and nearby surface water": "Mixed groundwater and nearby surface water",
    "Surface-water county, not near mapped source": "Surface-water county, no nearby mapped source",
    "Surface-water county, no nearby mapped source": "Surface-water county, no nearby mapped source",
    "Groundwater-leaning": "Groundwater leaning",
    "Groundwater leaning": "Groundwater leaning",
    "Mixed / uncertain": "Mixed or uncertain",
    "Mixed or uncertain": "Mixed or uncertain",
}

pathway["Hydrologic pathway classification"] = pathway[
    "Hydrologic pathway classification"
].replace(label_clean)

pathway = pathway.sort_values("Count of AI data centers", ascending=True)

pathway.to_csv(
    os.path.join(TABLE_OUTDIR, "GroundwaterDC_pathway_classification_counts.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.barh(
    pathway["Hydrologic pathway classification"],
    pathway["Count of AI data centers"],
)
ax.set_xlabel("Count of AI data centers")
ax.set_ylabel("Hydrologic pathway classification")
ax.grid(axis="x", alpha=0.20)
plt.tight_layout()
save_fig(fig, "GroundwaterDC_pathway_classification.png")


# ------------------------------------------------------------
# Figure 2: Reservoir/lake distance distribution
# ------------------------------------------------------------
dist_col = first_existing_column(dc, [
    "dist_to_reservoir_km",
    "dist_reservoir_km",
    "dist_lake_km",
    "nearest_reservoir_lake_km",
])

if dist_col is None:
    print("[WARNING] Reservoir/lake distance column not found. Skipping Figure 2.")
    print("[WARNING] Available columns:", list(dc.columns))
else:
    dist = pd.to_numeric(dc[dist_col], errors="coerce").dropna()
    dist = dist[dist >= 0]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.hist(dist, bins=30)
    ax.set_xlabel("Distance to nearest reservoir or lake (km)")
    ax.set_ylabel("Number of AI data centers")
    ax.grid(alpha=0.20)
    plt.tight_layout()
    save_fig(fig, "GroundwaterDC_reservoir_lake_distance_distribution.png")


# ------------------------------------------------------------
# Figure 3: Aquifer overrepresentation
# ------------------------------------------------------------
if aq is None:
    print("[WARNING] No aquifer table found. Skipping Figure 3.")
else:
    aq = aq.copy()

    aquifer_col = first_existing_column(aq, [
        "Aquifer",
        "AQUIFER_NAME",
        "aquifer",
        "aquifer_name",
    ])

    ratio_col = first_existing_column(aq, [
        "Odds_ratio",
        "odds_ratio",
        "enrichment_ratio",
        "Enrichment_ratio",
        "OR",
    ])

    count_col = first_existing_column(aq, [
        "dc_count",
        "count",
        "observed_dc_count",
        "Observed",
    ])

    if aquifer_col is None:
        print("[WARNING] Aquifer-name column not found. Skipping Figure 3.")
        print("[WARNING] Available columns:", list(aq.columns))
    else:
        # If enrichment ratio exists, use it.
        # If not, use dc_count as a fallback ranking but label the x-axis as count.
        if ratio_col is not None:
            aq_plot = aq.dropna(subset=[ratio_col]).copy()
            aq_plot[ratio_col] = pd.to_numeric(aq_plot[ratio_col], errors="coerce")
            aq_plot = aq_plot.dropna(subset=[ratio_col])
            aq_plot = aq_plot.sort_values(ratio_col, ascending=True).tail(10)

            x_col = ratio_col
            x_label = "Enrichment ratio of AI data-center occurrence"
            table_name = "GroundwaterDC_aquifer_overrepresentation.csv"
            fig_name = "GroundwaterDC_aquifer_overrepresentation.png"

        elif count_col is not None:
            aq_plot = aq.dropna(subset=[count_col]).copy()
            aq_plot[count_col] = pd.to_numeric(aq_plot[count_col], errors="coerce")
            aq_plot = aq_plot.dropna(subset=[count_col])
            aq_plot = aq_plot.sort_values(count_col, ascending=True).tail(10)

            x_col = count_col
            x_label = "Count of AI data centers"
            table_name = "GroundwaterDC_top_aquifers_by_dc_count.csv"
            fig_name = "GroundwaterDC_top_aquifers_by_dc_count.png"

        else:
            aq_plot = None
            print("[WARNING] No enrichment-ratio or count column found. Skipping Figure 3.")
            print("[WARNING] Available columns:", list(aq.columns))

        if aq_plot is not None and len(aq_plot) > 0:
            aq_plot.to_csv(os.path.join(TABLE_OUTDIR, table_name), index=False)

            fig, ax = plt.subplots(figsize=(9.5, 6.2))
            ax.barh(
                wrap_labels(aq_plot[aquifer_col], width=34),
                aq_plot[x_col],
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel("Aquifer system")
            ax.grid(axis="x", alpha=0.20)
            plt.tight_layout()
            save_fig(fig, fig_name)


# ------------------------------------------------------------
# Figure 4: Sector composition in top HUC8 basins
# ------------------------------------------------------------
master = master.copy()

huc_name_col = first_existing_column(master, [
    "HUC8_NAME",
    "huc8_name",
    "name",
    "Name",
])

dc_count_col = first_existing_column(master, [
    "dc_count",
    "n_dc",
    "AI_count",
])

has_dc_col = first_existing_column(master, [
    "has_dc",
    "dc_present",
    "AI_present",
])

sector_cols = {
    "Public supply": first_existing_column(master, ["PS_share", "public_supply_share"]),
    "Domestic": first_existing_column(master, ["DO_share", "domestic_share"]),
    "Industrial": first_existing_column(master, ["IN_share", "industrial_share"]),
    "Irrigation": first_existing_column(master, ["IR_share", "irrigation_share"]),
}

sector_cols = {k: v for k, v in sector_cols.items() if v is not None}

if huc_name_col is None or dc_count_col is None or len(sector_cols) == 0:
    print("[WARNING] Required columns for sector-composition figure not found. Skipping Figure 4.")
    print("[WARNING] Available columns:", list(master.columns))
else:
    master[dc_count_col] = pd.to_numeric(master[dc_count_col], errors="coerce").fillna(0)

    if has_dc_col is not None:
        if master[has_dc_col].dtype == bool:
            top = master[master[has_dc_col] == True].copy()
        else:
            top = master[pd.to_numeric(master[has_dc_col], errors="coerce").fillna(0) > 0].copy()
    else:
        top = master[master[dc_count_col] > 0].copy()

    top = top.sort_values(dc_count_col, ascending=False).head(15).copy()

    if len(top) == 0:
        print("[WARNING] No HUC8 basins with data centers found. Skipping Figure 4.")
    else:
        top.to_csv(
            os.path.join(TABLE_OUTDIR, "GroundwaterDC_top_HUC8_sector_composition.csv"),
            index=False
        )

        fig, ax = plt.subplots(figsize=(11.5, 7.0))

        x = np.arange(len(top))
        bottom = np.zeros(len(top))

        for label, col in sector_cols.items():
            vals = pd.to_numeric(top[col], errors="coerce").fillna(0).to_numpy()
            ax.bar(x, vals, bottom=bottom, label=label)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(
            top[huc_name_col].astype(str),
            rotation=75,
            ha="right",
        )

        ax.set_ylabel("Sector share")
        ax.set_xlabel("Top HUC8 basins containing AI data centers")
        ax.legend(frameon=True, loc="lower left")
        ax.grid(axis="y", alpha=0.20)

        plt.tight_layout()
        save_fig(fig, "GroundwaterDC_sector_composition_top_HUC8_basins.png")


print("\nDONE.")
print("Groundwater-DC figures saved in:", FIGDIR)
print("Groundwater-DC tables saved in:", TABLE_OUTDIR)
