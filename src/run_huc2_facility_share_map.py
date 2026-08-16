#!/usr/bin/env python3
"""
HUC2 Facility Share Choropleth Map

This script aggregates AI data centers, power plants, and TRI facilities to HUC2 regions and creates a three-panel choropleth map showing each sector’s share of national facilities by hydrologic region.

Input files used:
- DC_CONUS_STRICT.csv
- Power_Unique_Site_CONUS_STRICT.xlsx
- TRI_2024_Unique_Site_CONUS_STRICT.csv
- WBD_National_GPKG.gpkg

Server/GitHub version:
- reads input files from DEFAULT_DATA_ROOT
- default input folder: /mnt/disk3/aoolaseinde/data/WaterProject
- writes outputs to DEFAULT_OUTPUT_ROOT
- default output folder: /mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/huc2_facility_share_map
- can be run independently or through src/run_all.py
"""

from __future__ import annotations

import os
from common_paths import DATA_FOLDERS, output_folder

DEFAULT_DATA_ROOT = DATA_FOLDERS["water_project"]
DEFAULT_OUTPUT_ROOT = output_folder("huc2_facility_share_map")
os.makedirs(DEFAULT_OUTPUT_ROOT, exist_ok=True)

# Project-style variables retained for compatibility with notebook-derived code.
PROJECT_DIR = DEFAULT_DATA_ROOT

print("[INFO] Analysis: HUC2 Facility Share Choropleth Map")
print("[INFO] DEFAULT_DATA_ROOT =", DEFAULT_DATA_ROOT)
print("[INFO] DEFAULT_OUTPUT_ROOT =", DEFAULT_OUTPUT_ROOT)



import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import fiona
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

PROJECT_DIR = DEFAULT_DATA_ROOT
AI_FILE = os.path.join(DEFAULT_DATA_ROOT, "DC_CONUS_STRICT.csv")
POWER_FILE = os.path.join(DEFAULT_DATA_ROOT, "Power_Unique_Site_CONUS_STRICT.xlsx")
TRI_FILE = os.path.join(DEFAULT_DATA_ROOT, "TRI_2024_Unique_Site_CONUS_STRICT.csv")
HYBAS_FILE = os.path.join(DEFAULT_DATA_ROOT, "hybas_na_lev08_v1c.shp")
HUC_FILE     = os.path.join(PROJECT_DIR, "WBD_National_GPKG", "WBD_National_GPKG.gpkg")

OUT_FIG = os.path.join(DEFAULT_OUTPUT_ROOT, 'figure1_facility_distribution_water_stress_huc.png')

N_LABELS = 10
LABEL_MIN_PCT = 3.0
CANDIDATE_POOL = 40

LABEL_FONTSIZE = 10
LABEL_WEIGHT = "semibold"
TEXT_EFFECTS = [pe.withStroke(linewidth=1.2, foreground="white")]

EDGE_COLOR = "black"
EDGE_LW = 1.0

NUDGE_RINGS = [0, 60000, 110000, 160000, 220000]
DIRS = [
    (0, 0),
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (-1, 1), (1, -1), (-1, -1),
    (2, 0), (-2, 0), (0, 2), (0, -2)
]

BINS = [0, 0.000001, 1, 3, 6, 10, 100]
BIN_LABELS = ["0%", "0–1%", "1–3%", "3–6%", "6–10%", ">10%"]

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

def find_excel_header_row(path, sheet_name=None, max_rows=80):
    xls = pd.ExcelFile(path)
    sheets = [sheet_name] if sheet_name else xls.sheet_names
    for sh in sheets:
        preview = pd.read_excel(path, sheet_name=sh, header=None, nrows=max_rows)
        for r in range(len(preview)):
            row_vals = preview.iloc[r].astype(str).str.lower().tolist()
            has_lat = any(("latitude" in v) or (v.strip() == "lat") for v in row_vals)
            has_lon = any(("longitude" in v) or (v.strip() in ["lon","long","lng"]) for v in row_vals)
            if has_lat and has_lon:
                return r, sh
    raise RuntimeError("Could not find Latitude/Longitude header row in Power_Unique_Site_CONUS_STRICT.xlsx")


def filter_to_hydrobasins_domain(gdf, hybas_file, label):
    """Keep only points inside the HydroBASINS CONUS analysis domain."""
    basins = gpd.read_file(hybas_file).to_crs(gdf.crs)
    before = len(gdf)

    joined = gpd.sjoin(
        gdf,
        basins[["HYBAS_ID", "geometry"]],
        how="inner",
        predicate="within"
    )

    joined = joined.drop(columns=[c for c in ["index_right", "HYBAS_ID"] if c in joined.columns])
    after = len(joined)

    print(f"✅ CONUS HydroBASINS filter: {label} {before} -> {after}")
    return joined

def load_points_csv(path, name, force_lat=None, force_lon=None):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    if force_lat and force_lon and force_lat in df.columns and force_lon in df.columns:
        lat, lon = force_lat, force_lon
    else:
        lat = pick_col(df.columns, ["lat","latitude","y"])
        lon = pick_col(df.columns, ["lon","longitude","lng","long","x"])

    if lat is None or lon is None:
        raise RuntimeError(f"{name}: could not detect lat/lon. Columns: {df.columns[:80].tolist()}")

    df[lat] = pd.to_numeric(df[lat], errors="coerce")
    df[lon] = pd.to_numeric(df[lon], errors="coerce")
    df = df.dropna(subset=[lat, lon]).copy()

    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")

def load_points_power_excel(path):
    """
    Robust loader for cleaned Power_Unique_Site_CONUS_STRICT.xlsx.
    Handles Sheet1/Plant sheet names and normal Latitude/Longitude columns.
    """
    import pandas as pd
    import geopandas as gpd

    path = str(path)
    xls = pd.ExcelFile(path)
    sheet = "Plant" if "Plant" in xls.sheet_names else xls.sheet_names[0]

    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(options):
        lookup = {c.lower(): c for c in df.columns}
        for opt in options:
            if opt.lower() in lookup:
                return lookup[opt.lower()]
        raise ValueError(
            f"Could not find any of these columns: {options}\n"
            f"Available columns are: {list(df.columns)}"
        )

    lat_col = find_col(["Latitude", "latitude", "lat", "LAT"])
    lon_col = find_col(["Longitude", "longitude", "lon", "LON", "long"])

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()

    print(f"[INFO] Loaded Power Excel sheet: {sheet}")
    print(f"[INFO] Power rows with coordinates: {len(df)}")

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )
def pct_classify(pct_series):
    v = pd.Series(pct_series).astype(float)
    cats = pd.cut(v, bins=BINS, labels=BIN_LABELS, include_lowest=True, right=True).astype(str)
    cats[v == 0] = "0%"
    return cats

def make_huc2_pct_map(points_gdf, huc2_gdf, HUC2_ID):
    # Use the HydroBASINS-filtered facility count as the denominator
    # so the HUC2 map matches the ECDF / major-river / water-proximity analyses.
    total = int(len(points_gdf))

    j = gpd.sjoin(points_gdf, huc2_gdf[[HUC2_ID, "geometry"]], how="inner", predicate="within")
    counts = j.groupby(HUC2_ID).size().rename("count").reset_index()

    m = huc2_gdf[[HUC2_ID, "geometry"]].merge(counts, on=HUC2_ID, how="left")
    m["count"] = m["count"].fillna(0).astype(int)

    assigned = int(m["count"].sum())
    if assigned != total:
        print(f"⚠️ HUC2 assigned {assigned:,} of {total:,}; percentages use HydroBASINS-filtered denominator.")

    m["pct"] = 0.0 if total == 0 else (100.0 * m["count"] / total)
    m["class"] = pct_classify(m["pct"])
    return m, total

def legend_handles(color_map):
    return [Patch(facecolor=color_map[lbl], edgecolor=EDGE_COLOR, label=lbl) for lbl in BIN_LABELS]

def _bboxes_intersect(b1, b2, pad=2.0):
    return not (b1[2] + pad < b2[0] or b1[0] - pad > b2[2] or b1[3] + pad < b2[1] or b1[1] - pad > b2[3])

def place_labels_bbox(ax, gdf, n_labels, min_pct, candidate_pool):
    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    candidates = (
        gdf[gdf["pct"] >= min_pct]
        .sort_values("pct", ascending=False)
        .head(candidate_pool)
        .copy()
    )

    kept = 0
    kept_bboxes = []

    for _, row in candidates.iterrows():
        if kept >= n_labels:
            break

        rp = row.geometry.representative_point()
        base_x, base_y = float(rp.x), float(rp.y)
        label = f"{float(row['pct']):.1f}%"
        placed = False

        for r in NUDGE_RINGS:
            for dx, dy in DIRS:
                x = base_x + dx * r
                y = base_y + dy * r

                t = ax.text(
                    x, y, label,
                    ha="center", va="center",
                    fontsize=LABEL_FONTSIZE,
                    fontweight=LABEL_WEIGHT,
                    color="black",
                    path_effects=TEXT_EFFECTS,
                    zorder=10
                )

                fig.canvas.draw()
                bb = t.get_window_extent(renderer=renderer)
                bbox = (bb.x0, bb.y0, bb.x1, bb.y1)

                collision = any(_bboxes_intersect(bbox, bprev) for bprev in kept_bboxes)
                if collision:
                    t.remove()
                else:
                    kept_bboxes.append(bbox)
                    kept += 1
                    placed = True
                    break

            if placed:
                break

        if not placed:
            continue

def draw_panel(ax, gdf, color_map, panel_letter, title, total):
    for lbl in BIN_LABELS:
        sub = gdf[gdf["class"] == lbl]
        if len(sub) > 0:
            sub.plot(ax=ax, color=color_map[lbl], edgecolor=EDGE_COLOR, linewidth=EDGE_LW)

    ax.text(0.03, 0.97, panel_letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, fontweight="normal")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=4)

    place_labels_bbox(ax, gdf, N_LABELS, LABEL_MIN_PCT, CANDIDATE_POOL)

    ax.text(0.99, 0.01, f"N={total:,}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9)
    ax.set_axis_off()

layers = fiona.listlayers(HUC_FILE)
if "WBDHU2" not in layers:
    raise RuntimeError(f"WBDHU2 not found. Layers: {layers}")

huc2 = gpd.read_file(HUC_FILE, layer="WBDHU2")
HUC2_ID = pick_col(huc2.columns, ["huc2"])
if HUC2_ID is None:
    raise RuntimeError(f"Could not find HUC2 id field. Columns: {huc2.columns.tolist()}")

huc2[HUC2_ID] = huc2[HUC2_ID].astype(str).str.zfill(2)
huc2 = huc2[~huc2[HUC2_ID].isin(["19","20","21","22"])].copy()
huc2 = huc2.to_crs("EPSG:5070")

ai  = load_points_csv(AI_FILE, "AI").to_crs("EPSG:5070")
pwr = load_points_power_excel(POWER_FILE).to_crs("EPSG:5070")
tri = load_points_csv(TRI_FILE, "TRI", force_lat="12. LATITUDE", force_lon="13. LONGITUDE").to_crs("EPSG:5070")

print("✅ Loaded before CONUS filter:", "AI", len(ai), "| Power", len(pwr), "| TRI", len(tri))

ai  = filter_to_hydrobasins_domain(ai, HYBAS_FILE, "AI")
pwr = filter_to_hydrobasins_domain(pwr, HYBAS_FILE, "Power")
tri = filter_to_hydrobasins_domain(tri, HYBAS_FILE, "TRI")

print("✅ Loaded after CONUS filter:", "AI", len(ai), "| Power", len(pwr), "| TRI", len(tri))

m_ai,  tot_ai  = make_huc2_pct_map(ai,  huc2, HUC2_ID)
m_pwr, tot_pwr = make_huc2_pct_map(pwr, huc2, HUC2_ID)
m_tri, tot_tri = make_huc2_pct_map(tri, huc2, HUC2_ID)

blue_map = {
    "0%":"#E6E6E6", "0–1%":"#DEEBF7", "1–3%":"#9ECAE1", "3–6%":"#6BAED6", "6–10%":"#3182BD", ">10%":"#08519C"
}
orng_map = {
    "0%":"#E6E6E6", "0–1%":"#FEE6CE", "1–3%":"#FDBE85", "3–6%":"#FD8D3C", "6–10%":"#E6550D", ">10%":"#A63603"
}
grn_map = {
    "0%":"#E6E6E6", "0–1%":"#E5F5E0", "1–3%":"#A1D99B", "3–6%":"#74C476", "6–10%":"#31A354", ">10%":"#006D2C"
}

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, axes = plt.subplots(
    3, 1,
    figsize=(6.5, 10.0),
    dpi=600
)

draw_panel(axes[0], m_ai,  blue_map, "a", "AI Data Centers",  tot_ai)
draw_panel(axes[1], m_pwr, orng_map, "b", "Power Facilities", tot_pwr)
draw_panel(axes[2], m_tri, grn_map,  "c", "TRI Facilities",   tot_tri)

axes[0].legend(handles=legend_handles(blue_map), title="Share of national facilities (%)",
               loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9, title_fontsize=9)
axes[1].legend(handles=legend_handles(orng_map), title="Share of national facilities (%)",
               loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9, title_fontsize=9)
axes[2].legend(handles=legend_handles(grn_map),  title="Share of national facilities (%)",
               loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9, title_fontsize=9)

plt.subplots_adjust(
    left=0.05,
    right=0.95,
    top=0.96,
    bottom=0.05,
    hspace=0.65
)
plt.savefig(OUT_FIG, dpi=600, bbox_inches="tight", pad_inches=0.10)
plt.show()

print("✅ Saved:", OUT_FIG)
print(f"Labels: up to {N_LABELS} per panel | min {LABEL_MIN_PCT:.2f}% | candidates {CANDIDATE_POOL}")

print(f"✅ Final journal HUC2 figure saved: {OUT_FIG}")
