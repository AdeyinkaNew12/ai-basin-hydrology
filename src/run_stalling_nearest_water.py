#!/usr/bin/env python3
"""
Stallings Center Nearest Water Analysis

This script computes the nearest major water features to the Stallings Center location and generates a proof map. It evaluates distances to nearby river, coastline, and HydroLAKES waterbody features for a single site.

Input files used:
- Stalling.csv
- HydroRIVERS_v10_na.shp
- ne_10m_coastline.shp
- HydroLAKES_polys_v10.gdb

Server/GitHub version:
- reads input files from DEFAULT_DATA_ROOT
- default input folder: /mnt/disk3/aoolaseinde/data/StallingNearestWater
- writes outputs to DEFAULT_OUTPUT_ROOT
- default output folder: /mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/stalling_nearest_water
- can be run independently or through src/run_all.py
"""

from __future__ import annotations

import os
from common_paths import DATA_FOLDERS, output_folder

DEFAULT_DATA_ROOT = DATA_FOLDERS["stalling_nearest_water"]
DEFAULT_OUTPUT_ROOT = output_folder("stalling_nearest_water")
os.makedirs(DEFAULT_OUTPUT_ROOT, exist_ok=True)

# Project-style variables retained for compatibility with notebook-derived code.
PROJECT_DIR = DEFAULT_DATA_ROOT

print("[INFO] Analysis: Stallings Center Nearest Water Analysis")
print("[INFO] DEFAULT_DATA_ROOT =", DEFAULT_DATA_ROOT)
print("[INFO] DEFAULT_OUTPUT_ROOT =", DEFAULT_OUTPUT_ROOT)



import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import fiona

from shapely.ops import nearest_points

warnings.filterwarnings("ignore")

try:
    import contextily as cx
except:
    import contextily as cx

PROJECT_DIR = DEFAULT_DATA_ROOT
OUT_DIR = DEFAULT_OUTPUT_ROOT
os.makedirs(OUT_DIR, exist_ok=True)

DC_PATH    = os.path.join(PROJECT_DIR, "Stalling.csv")

RIVERS_SHP = os.path.join(DEFAULT_DATA_ROOT, "HydroRIVERS_v10_na.shp")
COAST_SHP  = os.path.join(PROJECT_DIR, "ne_10m_coastline.shp")
LAKES_GDB  = os.path.join(PROJECT_DIR, "_HYDROLAKES_GDB", "HydroLAKES_polys_v10.gdb")
LAKES_SHP  = os.path.join(PROJECT_DIR, "HydroLAKES_polys_v10.shp")

OUT_CSV = os.path.join(OUT_DIR, "stalling_nearest_water.csv")
OUT_MAP = os.path.join(OUT_DIR, "stalling_proof_map.png")

RIVER_STREAMORDER_THRESHOLD = 3
STREAM_ORDER_COL_CANDIDATES = ["ORD_STRA","ORD_FLOW","STREAMORDE","STRM_ORDER","ORD_ST"]

def detect_lat_lon_columns(df):
    cols = {c.lower().strip(): c for c in df.columns}
    return cols["latitude"], cols["longitude"]

def utm_epsg_from_lonlat(lon, lat):
    zone = int(np.floor((lon + 180) / 6) + 1)
    return 32600 + zone

def nearest_bruteforce(point_geom, gdf):
    if len(gdf) == 0: return np.nan, None
    dmin, gmin = np.inf, None
    for g in gdf.geometry.values:
        d = point_geom.distance(g)
        if d < dmin: dmin, gmin = d, g
    _, near = nearest_points(point_geom, gmin)
    return float(dmin), near

print("Loading Stalling.csv…")
dc = pd.read_csv(DC_PATH)
lat_col, lon_col = detect_lat_lon_columns(dc)

lon = float(dc.iloc[0][lon_col])
lat = float(dc.iloc[0][lat_col])
print("Point:", lat, lon)

print("Loading water layers…")
rivers = gpd.read_file(RIVERS_SHP).to_crs("EPSG:4326")
coast  = gpd.read_file(COAST_SHP).to_crs("EPSG:4326")
lakes = gpd.read_file(LAKES_SHP).to_crs("EPSG:4326")

stream_col = next(c for c in STREAM_ORDER_COL_CANDIDATES if c in rivers.columns)
rivers = rivers[rivers[stream_col] >= RIVER_STREAMORDER_THRESHOLD]

utm = utm_epsg_from_lonlat(lon, lat)
p = gpd.GeoSeries(gpd.points_from_xy([lon],[lat]), crs="EPSG:4326").to_crs(f"EPSG:{utm}").iloc[0]
rivers_utm = rivers.to_crs(f"EPSG:{utm}")
lakes_utm  = lakes.to_crs(f"EPSG:{utm}")
coast_utm  = coast.to_crs(f"EPSG:{utm}")

d_r, _ = nearest_bruteforce(p, rivers_utm)
d_l, _ = nearest_bruteforce(p, lakes_utm.boundary)
d_c, _ = nearest_bruteforce(p, coast_utm)

nearest = min([("River",d_r),("Lake",d_l),("Coast",d_c)], key=lambda x: x[1])

out = {
    "latitude": lat,
    "longitude": lon,
    "d_river_km": d_r/1000,
    "d_lake_km": d_l/1000,
    "d_coast_km": d_c/1000,
    "nearest": nearest[0]
}

pd.DataFrame([out]).to_csv(OUT_CSV, index=False)
print("Saved CSV:", OUT_CSV)

p_web = gpd.GeoSeries([p], crs=f"EPSG:{utm}").to_crs(3857)

fig, ax = plt.subplots(figsize=(8,8))
cx.add_basemap(ax)
p_web.plot(ax=ax, color="red", markersize=120)

ax.set_title("Stallings Center — Location")
ax.axis("off")

plt.savefig(OUT_MAP, dpi=300)
plt.close()

print("Saved map:", OUT_MAP)
print("DONE ✅")
