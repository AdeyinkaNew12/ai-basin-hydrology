#!/usr/bin/env python3
"""
Groundwater-DC paper figure generator.

This script reproduces the four manuscript-style Groundwater-DC figures
from final processed tables.

Figures:
1. Hydrologic pathway classification
2. Reservoir/lake distance distribution
3. Aquifer enrichment ratio
4. Sector composition in top HUC8 basins
"""

import os
import textwrap
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common_paths import DATA_FOLDERS, output_folder


# ============================================================
# PATHS
# ============================================================

# Read all four final manuscript tables from one folder
TABLEDIR = output_folder("paper_tables")

DC_TABLE = os.path.join(TABLEDIR, "data_centers_integrated_water_pathways.csv")
MASTER_TABLE = os.path.join(TABLEDIR, "master_huc8_integrated_table.csv")
PATHWAY_TABLE = os.path.join(TABLEDIR, "combined_supply_pathway_counts.csv")
AQUIFER_TABLE = os.path.join(TABLEDIR, "aquifer_enrichment_table.csv")

OUTDIR = output_folder("groundwater_dc_paper_figures")
FIGDIR = os.path.join(OUTDIR, "figures")
TABLE_OUTDIR = os.path.join(OUTDIR, "tables")

os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABLE_OUTDIR, exist_ok=True)

print("[INFO] Groundwater-DC paper figures")
print("[INFO] DC table:", DC_TABLE)
print("[INFO] Master HUC8 table:", MASTER_TABLE)
print("[INFO] Pathway table:", PATHWAY_TABLE)
print("[INFO] Aquifer table:", AQUIFER_TABLE)
print("[INFO] Output folder:", OUTDIR)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 450,
    "font.size": 11,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.labelweight": "normal",
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print("[READ]", path)
    return pd.read_csv(path)


def save_fig(fig, filename):
    path = os.path.join(FIGDIR, filename)
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", path)


def wrap_labels(labels, width=34):
    return ["\n".join(textwrap.wrap(str(x), width=width)) for x in labels]


# ============================================================
# LOAD TABLES
# ============================================================

dc = require(DC_TABLE)
master = require(MASTER_TABLE)
pathway = require(PATHWAY_TABLE)
aq = require(AQUIFER_TABLE)


# ============================================================
# FIGURE 1: HYDROLOGIC PATHWAY CLASSIFICATION
# ============================================================

cat_col = "combined_supply_pathway_proxy"
count_col = "dc_count"

if cat_col not in pathway.columns:
    if "supply_pathway_class" in pathway.columns:
        cat_col = "supply_pathway_class"
    else:
        raise ValueError(f"Pathway class column missing. Columns: {list(pathway.columns)}")

if count_col not in pathway.columns:
    count_col = "Count of AI data centers" if "Count of AI data centers" in pathway.columns else None
    if count_col is None:
        raise ValueError(f"Pathway count column missing. Columns: {list(pathway.columns)}")

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

pathway = pathway.copy()
pathway[cat_col] = pathway[cat_col].astype(str).replace(label_clean)
pathway[count_col] = pd.to_numeric(pathway[count_col], errors="coerce")
pathway = pathway.dropna(subset=[count_col]).sort_values(count_col, ascending=True)

pathway.to_csv(
    os.path.join(TABLE_OUTDIR, "GroundwaterDC_pathway_classification_counts.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.barh(pathway[cat_col], pathway[count_col])
ax.set_xlabel("Count of AI data centers")
ax.set_ylabel("Hydrologic pathway classification")
ax.grid(axis="x", alpha=0.20)
plt.tight_layout()
save_fig(fig, "GroundwaterDC_01_pathway_classification.png")


# ============================================================
# FIGURE 2: RESERVOIR / LAKE DISTANCE DISTRIBUTION
# ============================================================

dist_col = None
for c in [
    "dist_to_reservoir_km",
    "dist_reservoir_km",
    "dist_lake_km",
    "nearest_reservoir_lake_km",
]:
    if c in dc.columns:
        dist_col = c
        break

if dist_col is None:
    raise ValueError(f"Reservoir/lake distance column missing. Columns: {list(dc.columns)}")

dist = pd.to_numeric(dc[dist_col], errors="coerce").dropna()
dist = dist[dist >= 0]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.hist(dist, bins=30)
ax.set_xlabel("Distance to nearest reservoir or lake (km)")
ax.set_ylabel("Number of AI data centers")
ax.grid(alpha=0.20)
plt.tight_layout()
save_fig(fig, "GroundwaterDC_02_reservoir_lake_distance_distribution.png")


# ============================================================
# FIGURE 3: AQUIFER ENRICHMENT RATIO
# ============================================================

aq = aq.copy()

rename_map = {
    "AQUIFER_NAME": "Aquifer",
    "aquifer_name": "Aquifer",
    "observed_dc_count": "Observed",
    "expected_dc_count": "Expected",
    "enrichment_ratio": "Odds_ratio",
    "OR": "Odds_ratio",
}

aq = aq.rename(columns={k: v for k, v in rename_map.items() if k in aq.columns})

if "Aquifer" not in aq.columns:
    raise ValueError(f"Aquifer name column missing. Columns: {list(aq.columns)}")

if "Odds_ratio" not in aq.columns:
    raise ValueError(f"Enrichment ratio column missing. Columns: {list(aq.columns)}")

aq["Odds_ratio"] = pd.to_numeric(aq["Odds_ratio"], errors="coerce")
aq_plot = aq.dropna(subset=["Odds_ratio"]).copy()

# remove broad non-aquifer category if present
aq_plot = aq_plot[
    ~aq_plot["Aquifer"].astype(str).str.lower().isin(["other rocks", "other rock", "nan", "none"])
].copy()

aq_plot = aq_plot.sort_values("Odds_ratio", ascending=True).tail(10)

aq_plot.to_csv(
    os.path.join(TABLE_OUTDIR, "GroundwaterDC_aquifer_enrichment_ratio.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(9.5, 6.2))
ax.barh(
    wrap_labels(aq_plot["Aquifer"], width=34),
    aq_plot["Odds_ratio"],
)
ax.set_xlabel("Enrichment ratio of AI data-center occurrence")
ax.set_ylabel("Aquifer system")
ax.grid(axis="x", alpha=0.20)
plt.tight_layout()
save_fig(fig, "GroundwaterDC_03_aquifer_enrichment_ratio.png")


# ============================================================
# FIGURE 4: SECTOR COMPOSITION — TOP HUC8 BASINS
# ============================================================

required = ["HUC8_NAME", "dc_count", "has_dc", "PS_share", "DO_share", "IN_share", "IR_share"]
missing = [c for c in required if c not in master.columns]
if missing:
    raise ValueError(f"Master HUC8 table missing columns: {missing}. Columns: {list(master.columns)}")

top = master.copy()

# handle boolean stored as text
if top["has_dc"].dtype == bool:
    top = top[top["has_dc"] == True].copy()
else:
    top = top[top["has_dc"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

top["dc_count"] = pd.to_numeric(top["dc_count"], errors="coerce").fillna(0)
top = top.sort_values("dc_count", ascending=False).head(15)

sector_cols = {
    "Public supply": "PS_share",
    "Domestic": "DO_share",
    "Industrial": "IN_share",
    "Irrigation": "IR_share",
}

top.to_csv(
    os.path.join(TABLE_OUTDIR, "GroundwaterDC_top_HUC8_sector_composition.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(11, 7))

x = np.arange(len(top))
bottom = np.zeros(len(top))

for label, col in sector_cols.items():
    vals = pd.to_numeric(top[col], errors="coerce").fillna(0).values
    ax.bar(x, vals, bottom=bottom, label=label)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(top["HUC8_NAME"].astype(str), rotation=75, ha="right")
ax.set_ylabel("Sector share")
ax.set_xlabel("Top HUC8 basins containing AI data centers")
ax.legend(frameon=True, loc="lower left")
ax.grid(axis="y", alpha=0.20)

plt.tight_layout()
save_fig(fig, "GroundwaterDC_04_sector_composition_top_HUC8_basins.png")


print("\nDONE.")
print("Figures saved in:", FIGDIR)
print("Tables saved in:", TABLE_OUTDIR)
