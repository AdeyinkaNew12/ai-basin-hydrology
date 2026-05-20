#!/usr/bin/env python3
"""
NWM Hydro-Regime Analysis

This script builds a hydro-regime analysis using National Water Model discharge information and facility presence data. It produces basin-level hydro-regime characterictics, ECDF outputs, and optional odds-ratio/logistic summaries for AI, Power, and TRI siting.

Input files used:
- DC_CONUS.csv
- Power.xlsx
- TRI_2024.csv
- WBD_National_GPKG.gpkg
- NWM discharge/Zarr inputs as configured in script

Server/GitHub version:
- reads input files from DEFAULT_DATA_ROOT
- default input folder: /mnt/disk3/aoolaseinde/data/NWM_HydroRegime
- writes outputs to DEFAULT_OUTPUT_ROOT
- default output folder: /mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/nwm_hydroregime
- can be run independently or through src/run_all.py
"""

from __future__ import annotations

import os
from common_paths import DATA_FOLDERS, output_folder

DEFAULT_DATA_ROOT = DATA_FOLDERS["nwm_hydroregime"]
DEFAULT_OUTPUT_ROOT = output_folder("nwm_hydroregime")
os.makedirs(DEFAULT_OUTPUT_ROOT, exist_ok=True)

# Project-style variables retained for compatibility with notebook-derived code.
PROJECT_DIR = DEFAULT_DATA_ROOT

print("[INFO] Analysis: NWM Hydro-Regime Analysis")
print("[INFO] DEFAULT_DATA_ROOT =", DEFAULT_DATA_ROOT)
print("[INFO] DEFAULT_OUTPUT_ROOT =", DEFAULT_OUTPUT_ROOT)

               "fsspec==2026.2.0" "s3fs==2026.2.0" \
               xarray zarr "dask[complete]" \
               pandas openpyxl scikit-learn matplotlib pyarrow

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xarray as xr
import s3fs
from dask.diagnostics import ProgressBar

PROJECT_DIR = DEFAULT_DATA_ROOT
AI_FILE = os.path.join(DEFAULT_DATA_ROOT, "DC_CONUS.csv")
POWER_FILE = os.path.join(DEFAULT_DATA_ROOT, "Power.xlsx")
TRI_FILE = os.path.join(DEFAULT_DATA_ROOT, "TRI_2024.csv")
BASINS_SHP = os.path.join(DEFAULT_DATA_ROOT, "hybas_na_lev08_v1c.shp")

OUTDIR = os.path.join(PROJECT_DIR, "NWM_HydroRegime_Output")
os.makedirs(OUTDIR, exist_ok=True)

START_DATE = "2003-01-01"
END_DATE   = "2020-12-31"
ZARR_PATH = "s3://noaa-nwm-retrospective-3-0-pds/zarr/CONUS/chrtout.zarr"

REACHES_PER_BASIN = 1
TIME_CHUNK = 720
FEAT_CHUNK = 20000
MAP_CHUNK  = 200000

def find_lat_lon_columns(df):
    cols = list(df.columns)
    cols_low = [str(c).strip().lower() for c in cols]
    lat_col = None
    lon_col = None
    for c, cl in zip(cols, cols_low):
        if cl in {"lat", "latitude", "y"}:
            lat_col = c
        if cl in {"lon", "long", "longitude", "x"}:
            lon_col = c
    if lat_col is None:
        for c, cl in zip(cols, cols_low):
            if "latitude" in cl or cl.startswith("lat"):
                lat_col = c
                break
    if lon_col is None:
        for c, cl in zip(cols, cols_low):
            if "longitude" in cl or " long" in cl or cl.startswith("lon"):
                lon_col = c
                break
    return lat_col, lon_col

def load_points_csv_robust(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    lat_col, lon_col = find_lat_lon_columns(df)
    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not detect lat/lon in {path}. Columns sample: {list(df.columns)[:80]}")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=4326)
    print(f"Loaded {os.path.basename(path)} lat='{lat_col}' lon='{lon_col}' rows={len(gdf):,}")
    return gdf

def load_power_excel_fixed(path):
    xls = pd.ExcelFile(path, engine="openpyxl")
    sheet = None
    for s in xls.sheet_names:
        if "plant" in s.lower():
            sheet = s
            break
    if sheet is None:
        sheet = xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet, header=1, engine="openpyxl")
    df.columns = df.columns.astype(str).str.strip()
    lat_col, lon_col = find_lat_lon_columns(df)
    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not detect lat/lon in Power sheet '{sheet}'. Columns sample: {list(df.columns)[:80]}")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=4326)
    print(f"Loaded Power.xlsx sheet='{sheet}' lat='{lat_col}' lon='{lon_col}' rows={len(gdf):,}")
    return gdf

def assign_basins(points_gdf, basins_gdf, basin_id_field):
    j = gpd.sjoin(points_gdf, basins_gdf[[basin_id_field, "geometry"]], how="left", predicate="within")
    j = j.drop(columns=[c for c in j.columns if c.startswith("index_")], errors="ignore")
    return j

def ecdf_vals(x):
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce").dropna())
    if x.size == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

def median_iqr(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    return s.quantile(0.50), s.quantile(0.25), s.quantile(0.75)

def fmt_median_iqr(med, q1, q3, nd=3):
    if pd.isna(med):
        return ""
    return f"{med:.{nd}f} [{q1:.{nd}f}, {q3:.{nd}f}]"

need = [AI_FILE, POWER_FILE, TRI_FILE, BASINS_SHP]
missing = [p for p in need if not os.path.exists(p)]
if missing:
    raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

bas = gpd.read_file(BASINS_SHP, engine="pyogrio").to_crs(4326)
id_field = "HYBAS_ID" if "HYBAS_ID" in bas.columns else None
if id_field is None:
    raise ValueError(f"Could not find HYBAS_ID in basins. Columns sample: {list(bas.columns)[:40]}")
area_field = "SUB_AREA" if "SUB_AREA" in bas.columns else None

ai_gdf    = load_points_csv_robust(AI_FILE)
tri_gdf   = load_points_csv_robust(TRI_FILE)
power_gdf = load_power_excel_fixed(POWER_FILE)

ai_gdf    = assign_basins(ai_gdf, bas, id_field).dropna(subset=[id_field]).copy()
tri_gdf   = assign_basins(tri_gdf, bas, id_field).dropna(subset=[id_field]).copy()
power_gdf = assign_basins(power_gdf, bas, id_field).dropna(subset=[id_field]).copy()

ai_gdf[id_field]    = ai_gdf[id_field].astype(np.int64)
tri_gdf[id_field]   = tri_gdf[id_field].astype(np.int64)
power_gdf[id_field] = power_gdf[id_field].astype(np.int64)

def basin_presence_counts(gdf, basin_id_field, prefix):
    out = gdf.groupby(basin_id_field).size().rename(f"n_{prefix}").to_frame().reset_index()
    out[f"{prefix}_present"] = 1
    return out

ai_b = basin_presence_counts(ai_gdf, id_field, "AI")
pw_b = basin_presence_counts(power_gdf, id_field, "Power")
tr_b = basin_presence_counts(tri_gdf, id_field, "TRI")

bas_master = bas[[id_field]].drop_duplicates().copy()
if area_field:
    bas_master["area_km2"] = pd.to_numeric(bas[area_field], errors="coerce")

bm = bas_master.merge(ai_b, on=id_field, how="left") \
               .merge(pw_b, on=id_field, how="left") \
               .merge(tr_b, on=id_field, how="left")

for c in ["AI_present","Power_present","TRI_present"]:
    bm[c] = bm[c].fillna(0).astype(int)
for c in ["n_AI","n_Power","n_TRI"]:
    bm[c] = bm[c].fillna(0).astype(int)

PRES_OUT = os.path.join(OUTDIR, "basin_presence_counts.csv")
bm.to_csv(PRES_OUT, index=False)
print("Saved:", PRES_OUT)

print("\nOpening NWM Zarr (metadata only)...")
fs = s3fs.S3FileSystem(anon=True)
mapper = s3fs.S3Map(root=ZARR_PATH, s3=fs, check=False)
try:
    ds = xr.open_zarr(mapper, consolidated=True)
except Exception:
    ds = xr.open_zarr(mapper, consolidated=False)

ds = ds.chunk({"time": TIME_CHUNK, "feature_id": FEAT_CHUNK})

MAP_FILE = os.path.join(OUTDIR, "feature_to_basin.parquet")
if os.path.exists(MAP_FILE):
    print("\nLoading cached reach->basin map:", MAP_FILE)
    feat2bas = pd.read_parquet(MAP_FILE)
else:
    print("\nBuilding reach->basin map (ONE-TIME heavy step)...")
    nfeat = ds["feature_id"].shape[0]
    maps = []
    for i0 in range(0, nfeat, MAP_CHUNK):
        i1 = min(i0 + MAP_CHUNK, nfeat)
        print(f"  mapping reaches {i0:,}..{i1:,}")
        fid  = ds["feature_id"].isel(feature_id=slice(i0, i1)).values.astype(np.int64)
        lat  = ds["latitude"].isel(feature_id=slice(i0, i1)).values.astype(float)
        lon  = ds["longitude"].isel(feature_id=slice(i0, i1)).values.astype(float)
        ordv = ds["order"].isel(feature_id=slice(i0, i1)).values.astype(float)

        pts = gpd.GeoDataFrame(
            {"feature_id": fid, "order": ordv},
            geometry=gpd.points_from_xy(lon, lat),
            crs=4326
        )

        j = gpd.sjoin(pts, bas[[id_field, "geometry"]], how="left", predicate="within")
        j = j.drop(columns=[c for c in j.columns if c.startswith("index_")], errors="ignore")
        j = j.dropna(subset=[id_field])
        j[id_field] = j[id_field].astype(np.int64)
        maps.append(j[["feature_id", id_field, "order"]])

    feat2bas = pd.concat(maps, ignore_index=True).drop_duplicates(subset=["feature_id"])
    feat2bas.to_parquet(MAP_FILE, index=False)
    print("Saved:", MAP_FILE)

feat2bas_sorted = feat2bas.sort_values("order", ascending=False)
top_reach = feat2bas_sorted.groupby(id_field, as_index=False).head(REACHES_PER_BASIN).copy()
sel_features = top_reach["feature_id"].astype(np.int64).unique()

print("\nSubsetting streamflow to time window + selected reaches...")
ds_sub = ds.sel(time=slice(START_DATE, END_DATE))
fid_index = pd.Index(ds_sub["feature_id"].values.astype(np.int64))
keep_pos = fid_index.get_indexer(sel_features)
keep_pos = keep_pos[keep_pos >= 0]
ds_sub = ds_sub.isel(feature_id=keep_pos)

q = ds_sub["streamflow"]
q_daily = q.resample(time="1D").mean()

q_mean = q_daily.mean("time")
q_std  = q_daily.std("time")
q_cv   = q_std / xr.where(q_mean == 0, np.nan, q_mean)
q_q10 = q_daily.quantile(0.10, dim="time")
q_q90 = q_daily.quantile(0.90, dim="time")

dq = q_daily.diff("time").abs()
rbi = dq.sum("time") / xr.where(q_daily.sum("time") == 0, np.nan, q_daily.sum("time"))

q_mclim = q_daily.groupby("time.month").mean("time")
top3 = q_mclim.sortby(q_mclim, ascending=False).isel(month=slice(0,3)).sum("month")
ann  = q_mclim.sum("month")
season_conc = top3 / xr.where(ann == 0, np.nan, ann)

feat_metrics = xr.Dataset({
    "meanQ": q_mean,
    "stdQ": q_std,
    "CVQ": q_cv,
    "Q10": q_q10,
    "Q90": q_q90,
    "RBI": rbi,
    "season_conc": season_conc
})

print("\nComputing metrics (heavy)...")
with ProgressBar():
    feat_metrics_df = feat_metrics.compute().to_dataframe().reset_index()

feat_metrics_df["feature_id"] = feat_metrics_df["feature_id"].astype(np.int64)
top_reach_small = top_reach[[id_field, "feature_id"]].copy()
top_reach_small["feature_id"] = top_reach_small["feature_id"].astype(np.int64)

feat_metrics_df = feat_metrics_df.merge(top_reach_small, on="feature_id", how="left").dropna(subset=[id_field])
feat_metrics_df[id_field] = feat_metrics_df[id_field].astype(np.int64)

bas_metrics = feat_metrics_df.groupby(id_field).agg({
    "meanQ": "mean",
    "stdQ": "mean",
    "CVQ": "mean",
    "Q10": "mean",
    "Q90": "mean",
    "RBI": "mean",
    "season_conc": "mean"
}).reset_index()

HYDRO_OUT = os.path.join(OUTDIR, "basin_hydrologic_metrics.csv")
bas_metrics.to_csv(HYDRO_OUT, index=False)
print("Saved:", HYDRO_OUT)

master = bm.merge(bas_metrics, on=id_field, how="left")
MASTER_OUT = os.path.join(OUTDIR, "basin_master_presence_hydro.csv")
master.to_csv(MASTER_OUT, index=False)
print("Saved:", MASTER_OUT)

metrics = ["meanQ", "CVQ", "Q10", "Q90", "RBI", "season_conc"]
metrics = [m for m in metrics if m in master.columns]

groups = {
    "AI basins":     master.loc[master["AI_present"]==1],
    "Non-AI basins": master.loc[master["AI_present"]==0],
    "Power basins":  master.loc[master["Power_present"]==1],
    "TRI basins":    master.loc[master["TRI_present"]==1],
}

rows = []
for m in metrics:
    row = {"Metric": m}
    for gname, gdf in groups.items():
        med, q1, q3 = median_iqr(gdf[m])
        row[gname] = fmt_median_iqr(med, q1, q3, nd=3)
    rows.append(row)

table1 = pd.DataFrame(rows)
TABLE_OUT = os.path.join(OUTDIR, "Table_1_hydro_regime_by_sector.csv")
table1.to_csv(TABLE_OUT, index=False)
print("Saved:", TABLE_OUT)

g_ai   = master.loc[master["AI_present"] == 1].copy()
g_pw   = master.loc[master["Power_present"] == 1].copy()
g_tri  = master.loc[master["TRI_present"] == 1].copy()
g_none = master.loc[
    (master["AI_present"] == 0) &
    (master["Power_present"] == 0) &
    (master["TRI_present"] == 0)
].copy()

plot_metrics = [m for m in ["CVQ", "Q10", "Q90", "RBI", "season_conc"] if m in master.columns]
if len(plot_metrics) == 0:
    plot_metrics = [m for m in ["meanQ", "stdQ"] if m in master.columns]

plt.style.use("default")
plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 350,
    "font.size": 12,
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

COLORS = {"AI":"#1f77b4","Power":"#2ca02c","TRI":"#9467bd","None":"#000000"}

fig_h = 3.35 * len(plot_metrics)
fig, axes = plt.subplots(len(plot_metrics), 1, figsize=(9.6, fig_h), dpi=220)
if len(plot_metrics) == 1:
    axes = [axes]

for i, m in enumerate(plot_metrics):
    ax = axes[i]
    xa, ya = ecdf_vals(g_ai[m])
    xp, yp = ecdf_vals(g_pw[m])
    xt, yt = ecdf_vals(g_tri[m])
    xn, yn = ecdf_vals(g_none[m])

    if xa.size: ax.plot(xa, ya, lw=3.0, color=COLORS["AI"],    label=f"AI (N={len(g_ai):,})")
    if xp.size: ax.plot(xp, yp, lw=3.0, color=COLORS["Power"], label=f"Power (N={len(g_pw):,})")
    if xt.size: ax.plot(xt, yt, lw=3.0, color=COLORS["TRI"],   label=f"TRI (N={len(g_tri):,})")
    if xn.size: ax.plot(xn, yn, lw=2.4, color=COLORS["None"],  ls="--", alpha=0.85, label=f"None (N={len(g_none):,})")

    ax.set_xlabel(m)
    ax.set_ylabel("ECDF")
    ax.grid(True, which="major", linewidth=0.6, alpha=0.20)
    ax.grid(False, which="minor")
    ax.spines["top"].set_alpha(0.5)
    ax.spines["right"].set_alpha(0.5)

    if i == 0:
        ax.set_title("Hydrologic regime ECDF by basin class (NWM v3.0 retrospective)")
        ax.legend(ncol=2, frameon=False)

plt.tight_layout()
FIG_ECDF = os.path.join(OUTDIR, "Figure_ECDF_hydro_regime_AI_Power_TRI_None.png")
plt.savefig(FIG_ECDF, dpi=350, bbox_inches="tight")
plt.show()
print("Saved:", FIG_ECDF)

model_df = master[["AI_present"] + metrics].dropna().copy()
model_df["AI_present"] = model_df["AI_present"].astype(int)

if len(model_df) > 50 and len(metrics) > 0:
    X = model_df[metrics].values
    y = model_df["AI_present"].values
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=4000))])
    pipe.fit(X, y)

    coef = pipe.named_steps["clf"].coef_[0]
    odds = np.exp(coef)

    or_df = pd.DataFrame({"predictor": metrics, "coef": coef, "odds_ratio": odds}) \
             .sort_values("odds_ratio", ascending=False)

    OR_CSV = os.path.join(OUTDIR, "OddsRatios_AI_present.csv")
    or_df.to_csv(OR_CSV, index=False)
    print("Saved:", OR_CSV)

    plt.figure(figsize=(8.6, 5.0))
    plt.bar(or_df["predictor"], or_df["odds_ratio"])
    plt.axhline(1.0, linestyle="--")
    plt.ylabel("Odds ratio (AI_present)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    FIG_OR = os.path.join(OUTDIR, "Figure_OddsRatios_AI_present.png")
    plt.savefig(FIG_OR, dpi=350, bbox_inches="tight")
    plt.show()
    print("Saved:", FIG_OR)
else:
    print("Skipping odds ratio model (not enough rows after dropna or no metrics).")

print("\nDONE ✅ Outputs in:", OUTDIR)
