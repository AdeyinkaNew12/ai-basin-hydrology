from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(
    "/mnt/disk3/aoolaseinde/projects/"
    "integrated_dc_water_pathways/results/"
    "groundwater_dc/three_sector_pathways"
)

PCT_FILE = ROOT / "three_sector_pathway_percentages_STRICT_CONUS.csv"

OUT_PNG = ROOT / "three_sector_water_supply_pathways_FINAL.png"
OUT_PDF = ROOT / "three_sector_water_supply_pathways_FINAL.pdf"

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

SECTORS = ["AI", "Power", "TRI"]

PATHWAYS = [
    "Surface-water leaning",
    "Mixed groundwater and nearby surface water",
    "Surface-water county, no nearby mapped source",
    "Groundwater leaning",
    "Mixed or uncertain",
]

SHORT_LABELS = [
    "Surface-water\nleaning",
    "Mixed GW +\nnearby SW",
    "SW county,\nno nearby source",
    "Groundwater\nleaning",
    "Mixed or\nuncertain",
]

df = pd.read_csv(PCT_FILE).set_index("sector")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

x = np.arange(len(PATHWAYS))
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
    vals = df.loc[sec, PATHWAYS].astype(float).values

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

ax.set_ylabel("Facilities (%)", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(SHORT_LABELS)

ax.tick_params(
    axis="both",
    labelsize=9,
    width=0.8,
    length=3,
)

ax.set_ylim(0, 65)
ax.set_yticks(np.arange(0, 61, 10))

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
    bottom=0.26,
    top=0.97,
)

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

print("Saved:", OUT_PNG)
print("Saved:", OUT_PDF)
print("Figure size = 6.0 x 3.0 inches")
print("Fonts = 9 pt")
print("AI = #1f78b4")
print("Power = #d95f02")
print("TRI = #1b9e77")
