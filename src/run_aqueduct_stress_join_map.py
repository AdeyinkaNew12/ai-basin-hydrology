#!/usr/bin/env python3
"""
Aqueduct Stress Join Verification Map

This script verifies the spatial join between AI data-center locations and Aqueduct baseline water-stress attributes. It creates a two-panel map showing the selected AI facility, surrounding basin context, Aqueduct stress classification, and local comparison points.

Input files used:
- DC_CONUS.csv
- hybas_na_lev08_v1c.shp
- Aqueduct40_baseline_monthly_y2023m07d05.csv

Server/GitHub version:
- reads input files from DEFAULT_DATA_ROOT
- default input folder: /mnt/disk3/aoolaseinde/data/AqueductStress
- writes outputs to DEFAULT_OUTPUT_ROOT
- default output folder: /mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/aqueduct_stress_join_map
- can be run independently or through src/run_all.py
"""

from __future__ import annotations

import os
from common_paths import DATA_FOLDERS, output_folder

DEFAULT_DATA_ROOT = DATA_FOLDERS["aqueduct_stress"]
DEFAULT_OUTPUT_ROOT = output_folder("aqueduct_stress_join_map")
os.makedirs(DEFAULT_OUTPUT_ROOT, exist_ok=True)

# Project-style variables retained for compatibility with notebook-derived code.
PROJECT_DIR = DEFAULT_DATA_ROOT

print("[INFO] Analysis: Aqueduct Stress Join Verification Map")
print("[INFO] DEFAULT_DATA_ROOT =", DEFAULT_DATA_ROOT)
print("[INFO] DEFAULT_OUTPUT_ROOT =", DEFAULT_OUTPUT_ROOT)


import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import Point
import contextily as cx
from shapely.geometry import Point

PROJECT_DIR = DEFAULT_DATA_ROOT
AI_FILE = os.path.join(DEFAULT_DATA_ROOT, "DC_CONUS.csv")
BASINS_SHP = os.path.join(DEFAULT_DATA_ROOT, "hybas_na_lev08_v1c.shp")
AQUEDUCT_CSV = os.path.join(PROJECT_DIR, "Aqueduct40_baseline_monthly_y2023m07d05.csv")

OUT_MAP = os.path.join(DEFAULT_OUTPUT_ROOT, 'verify_AqueductStress_join_2panel.png')

PROJ_CRS   = "EPSG:5070"
WEB_CRS    = "EPSG:3857"
STRESS_COL = "bws_01_raw"

AI_INDEX   = 0
REGION_KM  = 150
RADIUS_KM  = 10

PLOT_LOCAL_RANDOM = True
N_LOCAL_RANDOM    = 2500
SEED = 7

def pick_col(cols, keys):
    cols_l = {str(c).lower(): c for c in cols}
    for k in keys:
        if k in cols_l:
            return cols_l[k]
    for c in cols:
        cl = str(c).lower()
        if any(k in cl for k in keys):
            return c
    return None

def load_points_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    lat = pick_col(df.columns, ["latitude","lat","y"])
    lon = pick_col(df.columns, ["longitude","lon","lng","long","x"])
    if lat is None or lon is None:
        raise RuntimeError(f"Could not find lat/lon. cols[:60]={list(df.columns)[:60]}")
    df[lon] = pd.to_numeric(df[lon], errors="coerce")
    df[lat] = pd.to_numeric(df[lat], errors="coerce")
    df = df.dropna(subset=[lon, lat]).copy()
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")

def sample_points_in_circle(x0, y0, r, n, rng):
    ang = rng.uniform(0, 2*np.pi, n)
    rad = r * np.sqrt(rng.uniform(0, 1, n))
    xs = x0 + rad * np.cos(ang)
    ys = y0 + rad * np.sin(ang)
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=PROJ_CRS)

def add_north_arrow(ax, x=0.95, y=0.88, size=0.08):
    ax.annotate("N", xy=(x, y), xytext=(x, y-size),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", lw=1.6))

def add_scalebar(ax, label="5 km", x0=0.08, y0=0.06, x1=0.26, lw=5):
    ax.plot([x0, x1], [y0, y0], transform=ax.transAxes, lw=lw, color="k", solid_capstyle="butt")
    ax.text(x0, y0+0.02, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, weight="bold")

for p in [AI_FILE, BASINS_SHP, AQUEDUCT_CSV]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")

ai = load_points_csv(AI_FILE).to_crs(PROJ_CRS)
if AI_INDEX >= len(ai):
    raise ValueError(f"AI_INDEX={AI_INDEX} too large. len(AI)={len(ai)}")

ai_one = ai.iloc[[AI_INDEX]].copy()
x0, y0 = float(ai_one.geometry.iloc[0].x), float(ai_one.geometry.iloc[0].y)

bas = gpd.read_file(BASINS_SHP, engine="pyogrio")
bas.columns = bas.columns.astype(str).str.strip()
if bas.crs is None:
    bas = bas.set_crs("EPSG:4326")
if "PFAF_ID" not in bas.columns:
    raise RuntimeError("Expected PFAF_ID in basins shapefile.")
bas["pfaf6"] = (pd.to_numeric(bas["PFAF_ID"], errors="coerce") // 100).astype("Int64")

aq = pd.read_csv(AQUEDUCT_CSV, low_memory=False)
aq.columns = aq.columns.astype(str).str.strip()
if ("pfaf_id" not in aq.columns) or (STRESS_COL not in aq.columns):
    raise RuntimeError("Aqueduct CSV missing pfaf_id or bws_01_raw.")
aq_small = aq[["pfaf_id", STRESS_COL]].copy()
aq_small["pfaf_id"] = pd.to_numeric(aq_small["pfaf_id"], errors="coerce").astype("Int64")
aq_small[STRESS_COL] = pd.to_numeric(aq_small[STRESS_COL], errors="coerce")

bas2 = bas[["pfaf6","geometry"]].merge(aq_small, left_on="pfaf6", right_on="pfaf_id", how="left")
bas2 = bas2.rename(columns={STRESS_COL:"stress"}).drop(columns=["pfaf_id"], errors="ignore")
bas2 = bas2.dropna(subset=["stress"]).copy().to_crs(PROJ_CRS)

stress_base = bas2["stress"].to_numpy(dtype=float)
t1, t2 = np.quantile(stress_base, [1/3, 2/3]).astype(float)

bas2["stress_bin"] = pd.cut(
    bas2["stress"],
    bins=[-np.inf, t1, t2, np.inf],
    labels=["Low","Medium","High"],
    include_lowest=True
)

ai_one_w = gpd.sjoin(ai_one, bas2[["pfaf6","stress","stress_bin","geometry"]], how="left", predicate="within")
if ai_one_w["pfaf6"].isna().all():
    raise RuntimeError("Selected AI point not in a stress basin. Try another AI_INDEX.")

pfaf6     = int(ai_one_w["pfaf6"].iloc[0])
ai_stress = float(ai_one_w["stress"].iloc[0])
ai_bin    = str(ai_one_w["stress_bin"].iloc[0])

bas_one = bas2.loc[bas2["pfaf6"] == pfaf6].copy()

region_buf = Point(x0, y0).buffer(REGION_KM*1000.0)
bas_region = bas2[bas2.intersects(region_buf)].copy()

circle10 = gpd.GeoDataFrame(geometry=[Point(x0, y0).buffer(RADIUS_KM*1000.0)], crs=PROJ_CRS)

local_pts = None
if PLOT_LOCAL_RANDOM:
    rng = np.random.default_rng(SEED)
    cand = sample_points_in_circle(x0, y0, RADIUS_KM*1000.0, N_LOCAL_RANDOM, rng)
    local_pts = gpd.sjoin(cand, bas2[["stress_bin","geometry"]], how="left", predicate="within").dropna()

bin_order = ["Low","Medium","High"]
bin_to_int = {b:i for i,b in enumerate(bin_order)}
cmap = mpl.colors.ListedColormap(["#2b8cbe", "#41ab5d", "#de2d26"])
norm = mpl.colors.BoundaryNorm([0,1,2,3], cmap.N)
bas_region["bin_i"] = bas_region["stress_bin"].map(bin_to_int).astype(int)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(16.8, 8.4))
fig.subplots_adjust(left=0.33, right=0.98, bottom=0.06, top=0.92, wspace=0.06)

bas_region_web = bas_region.to_crs(WEB_CRS)
bas_one_web    = bas_one.to_crs(WEB_CRS)
ai_web         = ai_one.to_crs(WEB_CRS)

bas_region_web.plot(ax=axA, column="bin_i", cmap=cmap, norm=norm,
                    linewidth=0.35, edgecolor="white", alpha=0.72)
bas_one_web.plot(ax=axA, facecolor="none", edgecolor="black", linewidth=2.8, zorder=4)
ai_web.plot(ax=axA, color="red", markersize=95, edgecolor="black", linewidth=1.2, zorder=5)
cx.add_basemap(axA, source=cx.providers.CartoDB.Positron)
axA.set_axis_off()
axA.set_title("A) Regional Aqueduct water-stress bins (HydroBASINS)\nSelected basin outlined; AI facility shown",
              weight="bold", fontsize=14)
add_north_arrow(axA)
add_scalebar(axA, "20 km")

bas_region.to_crs(WEB_CRS).plot(ax=axB,
                               column=bas_region["stress_bin"].map(bin_to_int).astype(int),
                               cmap=cmap, norm=norm, linewidth=0.25, edgecolor="white", alpha=0.32)
circle10.to_crs(WEB_CRS).plot(ax=axB, facecolor="none", edgecolor="red", linewidth=2.3, zorder=4)
ai_web.plot(ax=axB, color="red", markersize=105, edgecolor="black", linewidth=1.2, zorder=5)

if PLOT_LOCAL_RANDOM and local_pts is not None and len(local_pts) > 0:
    local_web = local_pts.to_crs(WEB_CRS)
    local_web["bin_i"] = local_web["stress_bin"].map(bin_to_int).astype(int)
    local_web.plot(ax=axB, markersize=9, column="bin_i", cmap=cmap, norm=norm, alpha=0.55, linewidth=0)

minx, miny, maxx, maxy = circle10.to_crs(WEB_CRS).total_bounds
pad = 2500
axB.set_xlim(minx-pad, maxx+pad)
axB.set_ylim(miny-pad, maxy+pad)

cx.add_basemap(axB, source=cx.providers.CartoDB.Positron)
axB.set_axis_off()
axB.set_title(f"B) Local verification within {RADIUS_KM:.0f} km\nStress bins + {RADIUS_KM:.0f} km radius",
              weight="bold", fontsize=14)
add_north_arrow(axB)
add_scalebar(axB, "5 km")

info = (
    f"PFAF6={pfaf6}\n"
    f"Aqueduct stress={ai_stress:.3f}\n"
    f"Bin={ai_bin}\n\n"
    f"GLOBAL tertile cutoffs:\n"
    f"Low < {t1:.3f}\n"
    f"Medium < {t2:.3f}\n"
    f"High ≥ {t2:.3f}"
)
fig.text(0.02, 0.86, info, ha="left", va="top",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", alpha=0.98),
         fontsize=12, weight="bold")

handles = [
    mpl.lines.Line2D([0],[0], marker="s", linestyle="None", markerfacecolor=cmap(0),
                     markeredgecolor="black", markersize=12, label="Low"),
    mpl.lines.Line2D([0],[0], marker="s", linestyle="None", markerfacecolor=cmap(1),
                     markeredgecolor="black", markersize=12, label="Medium"),
    mpl.lines.Line2D([0],[0], marker="s", linestyle="None", markerfacecolor=cmap(2),
                     markeredgecolor="black", markersize=12, label="High"),
    mpl.lines.Line2D([0],[0], marker="o", linestyle="None", markerfacecolor="red",
                     markeredgecolor="black", markersize=10, label="AI facility"),
    mpl.lines.Line2D([0],[0], color="black", lw=2.8, label=f"Selected basin (PFAF6={pfaf6})"),
]
fig.legend(handles=handles,
           loc="upper left",
           bbox_to_anchor=(0.02, 0.34),
           frameon=True, framealpha=0.98,
           fontsize=12,
           title=None)

plt.savefig(OUT_MAP, dpi=350, bbox_inches="tight")
plt.show()

print("✅ Saved:", OUT_MAP)
print(f"AI_INDEX={AI_INDEX} | PFAF6={pfaf6} | stress={ai_stress:.6f} | bin={ai_bin}")
print(f"Tertile cutoffs: t1={t1:.6f}, t2={t2:.6f}")
