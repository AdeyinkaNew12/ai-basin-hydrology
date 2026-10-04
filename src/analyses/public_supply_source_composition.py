#!/usr/bin/env python3
"""
Generate Figure 5 and statistical comparisons of public-supply
source composition for AI, Power, and TRI facilities.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

INPUT_FILE = Path(
    "/mnt/disk3/aoolaseinde/projects/ai-basin-hydrology/"
    "latest_ERL_results/groundwater_dc/three_sector_pathways/"
    "three_sector_pathway_assignments.csv"
)

OUT_DIR = Path(
    "/mnt/disk3/aoolaseinde/projects/ai-basin-hydrology/"
    "final_results/public_supply_source_composition"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_5_public_supply_source_composition.png"
OUT_PDF = OUT_DIR / "Figure_5_public_supply_source_composition.pdf"


# ------------------------------------------------------------
# Read data
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE, low_memory=False)

gw = df["PS_gw_fraction"]
sw = df["PS_sw_fraction"]


# ------------------------------------------------------------
# Public-supply source classification
# ------------------------------------------------------------

df["public_supply_source"] = "Unclassified"

df.loc[
    sw >= 0.50,
    "public_supply_source"
] = "Surface water dominated"

df.loc[
    gw > 0.50,
    "public_supply_source"
] = "Groundwater dominated"


# ------------------------------------------------------------
# Categories and sectors
# ------------------------------------------------------------

SECTORS = ["AI", "Power", "TRI"]

CATEGORIES = [
    "Surface water dominated",
    "Groundwater dominated",
    "Unclassified",
]

SHORT_LABELS = [
    "Surface water\ndominated",
    "Groundwater\ndominated",
    "Unclassified",
]

SECTOR_COLOR = {
    "AI": "#1f78b4",
    "Power": "#d95f02",
    "TRI": "#1b9e77",
}

LEGEND_LABELS = {
    "AI": "AI",
    "Power": "Power plants",
    "TRI": "TRI",
}


# ------------------------------------------------------------
# Counts and percentages
# ------------------------------------------------------------

counts = pd.crosstab(
    df["sector"],
    df["public_supply_source"]
).reindex(
    index=SECTORS,
    columns=CATEGORIES,
    fill_value=0
)

percentages = counts.div(
    counts.sum(axis=1),
    axis=0
) * 100


# ------------------------------------------------------------
# Chi-square and Cramer's V
# ------------------------------------------------------------

chi2, p, dof, expected = chi2_contingency(counts)

n = counts.to_numpy().sum()
r, k = counts.shape

cramers_v = np.sqrt(
    chi2 / (n * min(r - 1, k - 1))
)


# ------------------------------------------------------------
# Save numerical results
# ------------------------------------------------------------

counts.to_csv(
    OUT_DIR / "public_supply_source_counts.csv"
)

percentages.to_csv(
    OUT_DIR / "public_supply_source_percentages.csv"
)

statistics = pd.DataFrame(
    {
        "Chi-square": [chi2],
        "df": [dof],
        "p-value": [p],
        "Cramer's V": [cramers_v],
        "N": [n],
    }
)

statistics.to_csv(
    OUT_DIR / "public_supply_source_statistics.csv",
    index=False,
)

print("[SAVED]", OUT_DIR / "public_supply_source_counts.csv")
print("[SAVED]", OUT_DIR / "public_supply_source_percentages.csv")
print("[SAVED]", OUT_DIR / "public_supply_source_statistics.csv")


# ------------------------------------------------------------
# Verification output
# ------------------------------------------------------------

print("\nCOUNTS")
print(counts.to_string())

print("\nPERCENTAGES")
print(percentages.round(2).to_string())

print("\nSTATISTICAL RESULTS")
print(f"Chi-square = {chi2:.4f}")
print(f"df = {dof}")
print(f"p-value = {p:.6g}")
print(f"Cramer's V = {cramers_v:.4f}")
print(f"N = {n}")

print("\nUNCLASSIFIED CHECK")
print(
    "Both public-supply fractions missing:",
    (gw.isna() & sw.isna()).sum()
)
print(
    "Facilities classified as Unclassified:",
    (df["public_supply_source"] == "Unclassified").sum()
)


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

x = np.arange(len(CATEGORIES))
width = 0.23

fig, ax = plt.subplots(
    figsize=(6.0, 3.0),
    dpi=450
)

offsets = {
    "AI": -width,
    "Power": 0,
    "TRI": width,
}

for sec in SECTORS:
    vals = percentages.loc[sec, CATEGORIES].values

    ax.bar(
        x + offsets[sec],
        vals,
        width=width,
        color=SECTOR_COLOR[sec],
        edgecolor="black",
        linewidth=0.6,
        label=LEGEND_LABELS[sec],
        zorder=3,
    )


# ------------------------------------------------------------
# Axes and formatting
# ------------------------------------------------------------

ax.set_ylabel("Facilities (%)", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(SHORT_LABELS)

ax.tick_params(
    axis="both",
    labelsize=9,
    width=0.8,
    length=3,
)

ax.set_ylim(0, 70)
ax.set_yticks(np.arange(0, 71, 10))

ax.grid(
    axis="y",
    linewidth=0.5,
    alpha=0.25,
    zorder=0,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=False,
    fontsize=9,
    loc="upper right",
)

fig.subplots_adjust(
    left=0.11,
    right=0.98,
    bottom=0.25,
    top=0.97,
)


# ------------------------------------------------------------
# Save Figure 5
# ------------------------------------------------------------

fig.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03,
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.close(fig)

print("\nSaved:", OUT_PNG)
print("Saved:", OUT_PDF)
print("Figure size = 6.0 x 3.0 inches")
print("Fonts = 9 pt")
