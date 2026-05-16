#!/usr/bin/env python3
"""
AI, Power, and TRI Logistic Regression Against Matched Random Sites

This script compares AI data centers, power plants, and TRI facilities with basin-matched random sites using logistic regression. It evaluates whether proximity to major rivers predicts facility occurrence and generates regression tables and predicted-probability curves.

Input files used:
- DC_CONUS.csv
- Power.xlsx
- TRI_2024.csv
- hybas_na_lev08_v1c.shp
- HydroRIVERS_v10_na.shp

Server/GitHub version:
- reads input files from DEFAULT_DATA_ROOT
- default input folder: /mnt/disk3/aoolaseinde/data/WaterProject
- writes outputs to DEFAULT_OUTPUT_ROOT
- default output folder: /mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/logistic_ai_power_tri
- can be run independently or through src/run_all.py
"""

from __future__ import annotations

import os
from common_paths import DATA_FOLDERS, output_folder

DEFAULT_DATA_ROOT = DATA_FOLDERS["water_project"]
DEFAULT_OUTPUT_ROOT = output_folder("logistic_ai_power_tri")
os.makedirs(DEFAULT_OUTPUT_ROOT, exist_ok=True)

# Project-style variables retained for compatibility with notebook-derived code.
PROJECT_DIR = DEFAULT_DATA_ROOT

print("[INFO] Analysis: AI, Power, and TRI Logistic Regression Against Matched Random Sites")
print("[INFO] DEFAULT_DATA_ROOT =", DEFAULT_DATA_ROOT)
print("[INFO] DEFAULT_OUTPUT_ROOT =", DEFAULT_OUTPUT_ROOT)


import os, warnings, numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
from shapely.strtree import STRtree
import statsmodels.api as sm

warnings.filterwarnings("ignore")

AI_FILE = os.path.join(DEFAULT_DATA_ROOT, "DC_CONUS.csv")
POWER_FILE = os.path.join(DEFAULT_DATA_ROOT, "Power.xlsx")
TRI_FILE = os.path.join(DEFAULT_DATA_ROOT, "TRI_2024.csv")
BASINS_SHP = os.path.join(DEFAULT_DATA_ROOT, "hybas_na_lev08_v1c.shp")
RIVERS_SHP = os.path.join(DEFAULT_DATA_ROOT, "HydroRIVERS_v10_na.shp")

OUT_DIR = DEFAULT_OUTPUT_ROOT
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TABLE  = os.path.join(OUT_DIR, "logit_results_AI_Power_TRI_vs_random_majorRivers.csv")
OUT_FIG    = os.path.join(OUT_DIR, "predicted_probability_curves_AI_Power_TRI_vs_random_majorRivers.png")

ORD_COL = "ORD_STRA"
STREAM_ORDER_MIN = 5
PROJ_CRS = "EPSG:5070"
SIMPLIFY_RIVER_M = 200

SEED = 7
SEED_MAP = {"AI": 101, "Power": 202, "TRI": 303}
RANDOM_BATCH = 40000
NEAREST_CHUNK = 20000

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

def safe_sjoin(left_gdf, right_gdf, how="left", predicate="within", rsuffix="bas"):
    L = left_gdf.copy()
    R = right_gdf.copy()
    for c in ["index_right", "index_left", f"index_{rsuffix}", "index_bas"]:
        if c in L.columns: L = L.drop(columns=[c], errors="ignore")
        if c in R.columns: R = R.drop(columns=[c], errors="ignore")
    L = L.reset_index(drop=True)
    R = R.reset_index(drop=True)
    return gpd.sjoin(L, R, how=how, predicate=predicate, rsuffix=rsuffix)

def generate_random_points_in_basins(bas_gdf_proj, n, seed=0, batch=40000, max_tries=700):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = bas_gdf_proj.total_bounds
    kept = []
    total = 0
    tries = 0
    while total < n and tries < max_tries:
        tries += 1
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=bas_gdf_proj.crs)
        j = safe_sjoin(pts, bas_gdf_proj[["geometry"]], how="inner", predicate="within", rsuffix="bas")
        if len(j) == 0:
            continue
        need = n - total
        kept.append(j.iloc[:need].copy())
        total += len(kept[-1])
    if total < n:
        raise RuntimeError(f"Random generation failed: got {total}/{n}")
    out = pd.concat(kept, ignore_index=True).iloc[:n].copy()
    return gpd.GeoDataFrame(out, geometry="geometry", crs=bas_gdf_proj.crs)

def build_tree(geoms):
    geoms = list(geoms)
    return STRtree(geoms), geoms

def nearest_dist_km(points_proj, tree, target_geoms, chunk=20000):
    pts = list(points_proj)
    out = np.empty(len(pts), dtype=float)
    for start in range(0, len(pts), chunk):
        sub = pts[start:start+chunk]
        nearest = tree.nearest(sub)
        if len(nearest) > 0 and isinstance(nearest[0], (int, np.integer)):
            near_geoms = [target_geoms[i] for i in nearest]
        else:
            near_geoms = nearest
        for i, (p, g) in enumerate(zip(sub, near_geoms)):
            out[start+i] = p.distance(g) / 1000.0
    return out

def load_ai_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    lat = pick_col(df.columns, ["latitude", "lat", "y"])
    lon = pick_col(df.columns, ["longitude", "lon", "lng", "long", "x"])
    if lat is None or lon is None:
        raise RuntimeError(f"AI: could not detect lat/lon in {list(df.columns)[:60]}")
    df[lon] = pd.to_numeric(df[lon], errors="coerce")
    df[lat] = pd.to_numeric(df[lat], errors="coerce")
    df = df.dropna(subset=[lon, lat]).copy()
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")

def load_power_xlsx(path):
    df = pd.read_excel(path, header=1)
    df.columns = df.columns.astype(str).str.strip()
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        lat = pick_col(df.columns, ["latitude", "lat", "y"])
        lon = pick_col(df.columns, ["longitude", "lon", "lng", "long", "x"])
    else:
        lat, lon = "Latitude", "Longitude"
    df[lon] = pd.to_numeric(df[lon], errors="coerce")
    df[lat] = pd.to_numeric(df[lat], errors="coerce")
    df = df.dropna(subset=[lon, lat]).copy()
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")

def load_tri_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    if "12. LATITUDE" in df.columns and "13. LONGITUDE" in df.columns:
        lat, lon = "12. LATITUDE", "13. LONGITUDE"
    else:
        lat = pick_col(df.columns, ["latitude", "lat", "y"])
        lon = pick_col(df.columns, ["longitude", "lon", "lng", "long", "x"])
    if lat is None or lon is None:
        raise RuntimeError(f"TRI: could not detect lat/lon in {list(df.columns)[:60]}")
    df[lon] = pd.to_numeric(df[lon], errors="coerce")
    df[lat] = pd.to_numeric(df[lat], errors="coerce")
    df = df.dropna(subset=[lon, lat]).copy()
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")

def fit_logit_and_curve(dist_sector_km, dist_random_km):
    df = pd.DataFrame({
        "presence": np.r_[np.ones(len(dist_sector_km), dtype=int),
                          np.zeros(len(dist_random_km), dtype=int)],
        "dist_km":  np.r_[dist_sector_km, dist_random_km]
    })
    df["log_dist"] = np.log(df["dist_km"] + 1.0)
    X = sm.add_constant(df[["log_dist"]])
    y = df["presence"]
    res = sm.Logit(y, X).fit(disp=False)
    coef = float(res.params["log_dist"])
    OR = float(np.exp(coef))
    ci = res.conf_int()
    OR_lo = float(np.exp(ci.loc["log_dist", 0]))
    OR_hi = float(np.exp(ci.loc["log_dist", 1]))
    pval = float(res.pvalues["log_dist"])
    return res, OR, OR_lo, OR_hi, pval, df

need = [AI_FILE, POWER_FILE, TRI_FILE, BASINS_SHP, RIVERS_SHP]
missing = [p for p in need if not os.path.exists(p)]
if missing:
    raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

bas = gpd.read_file(BASINS_SHP, engine="pyogrio")
bas.columns = bas.columns.astype(str).str.strip()
if bas.crs is None:
    bas = bas.set_crs("EPSG:4326")
bas = bas.to_crs("EPSG:4326")
bas_proj = bas.to_crs(PROJ_CRS)

rivers = gpd.read_file(RIVERS_SHP, engine="pyogrio")
rivers.columns = rivers.columns.astype(str).str.strip()
if ORD_COL not in rivers.columns:
    raise RuntimeError(f"Expected {ORD_COL} in HydroRIVERS.")
rivers[ORD_COL] = pd.to_numeric(rivers[ORD_COL], errors="coerce")
rmaj = rivers.loc[rivers[ORD_COL].notna() & (rivers[ORD_COL] >= STREAM_ORDER_MIN), ["geometry"]].copy()
print(f"✅ Major river segments (ORD≥{STREAM_ORDER_MIN}): {len(rmaj):,}")

rmaj_proj = rmaj.to_crs(PROJ_CRS).copy()
rmaj_proj["geometry"] = rmaj_proj["geometry"].simplify(SIMPLIFY_RIVER_M)
tree_riv, geoms_riv = build_tree(rmaj_proj.geometry)

ai  = load_ai_csv(AI_FILE);          ai["sector"]="AI"
pwr = load_power_xlsx(POWER_FILE);   pwr["sector"]="Power"
tri = load_tri_csv(TRI_FILE);        tri["sector"]="TRI"
sectors = [("AI", ai), ("Power", pwr), ("TRI", tri)]

results_rows = []
curves = {}

for sec, gdf in sectors:
    gdf_ll = gdf.to_crs(bas.crs)
    g_in = safe_sjoin(gdf_ll, bas[["geometry"]], how="inner", predicate="within", rsuffix="bas")
    N = len(g_in)
    print(f"✅ {sec} points inside basins: {N:,}")
    if N == 0:
        continue

    seed = SEED + SEED_MAP.get(sec, 999)
    rd = generate_random_points_in_basins(bas_proj, n=N, seed=seed, batch=RANDOM_BATCH)

    g_proj = g_in.to_crs(PROJ_CRS)
    dist_sec  = nearest_dist_km(g_proj.geometry, tree_riv, geoms_riv, chunk=NEAREST_CHUNK)
    dist_rand = nearest_dist_km(rd.geometry,     tree_riv, geoms_riv, chunk=NEAREST_CHUNK)

    res, OR, OR_lo, OR_hi, pval, df_model = fit_logit_and_curve(dist_sec, dist_rand)

    dmax = float(np.nanpercentile(df_model["dist_km"], 99.5))
    dist_grid = np.linspace(0, max(50.0, dmax), 300)
    log_grid = np.log(dist_grid + 1.0)
    X_pred = sm.add_constant(pd.DataFrame({"log_dist": log_grid}))
    prob = res.predict(X_pred)
    curves[sec] = {"dist_grid": dist_grid, "prob": prob}

    results_rows.append({
        "sector": sec,
        "N_sector": int(N),
        "N_random": int(N),
        "coef_log_dist": float(res.params["log_dist"]),
        "OR_log_dist": OR,
        "OR_2.5%": OR_lo,
        "OR_97.5%": OR_hi,
        "p_value": pval,
        "pseudo_R2": float(res.prsquared),
        "LLR_pvalue": float(res.llr_pvalue),
    })

df_out = pd.DataFrame(results_rows).sort_values("sector")
df_out.to_csv(OUT_TABLE, index=False)
print("\n✅ Saved logit results:", OUT_TABLE)

plt.figure(figsize=(8.2, 5.8))
for sec in ["AI", "Power", "TRI"]:
    if sec in curves:
        plt.plot(curves[sec]["dist_grid"], curves[sec]["prob"], linewidth=2, label=sec)

plt.xlabel(f"Distance to Major Rivers (km) — HydroRIVERS {ORD_COL} ≥ {STREAM_ORDER_MIN}")
plt.ylabel("Predicted Probability of Sector Presence\n(vs N-matched random)")
plt.title("Predicted Probability Curves from Logistic Regression")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=350)
plt.show()

print("✅ Saved probability curve figure:", OUT_FIG)
