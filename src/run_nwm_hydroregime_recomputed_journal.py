#!/usr/bin/env python3
"""
NWM Hydro-Regime Recompute + Journal Figures
"""

from __future__ import annotations

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import xarray as xr
import s3fs
from dask.diagnostics import ProgressBar

from common_paths import DATA_FOLDERS, output_folder


PROJECT_DIR = DATA_FOLDERS["nwm_hydroregime"]

AI_FILE = "/mnt/disk3/aoolaseinde/data/WaterProject/DC_CONUS_STRICT.csv"
POWER_FILE = "/mnt/disk3/aoolaseinde/data/WaterProject/Power_Unique_Site_CONUS_STRICT.xlsx"
TRI_FILE = "/mnt/disk3/aoolaseinde/data/WaterProject/TRI_2024_Unique_Site_CONUS_STRICT.csv"
BASINS_SHP = os.path.join(PROJECT_DIR, "hybas_na_lev08_v1c.shp")

OUTDIR = "/mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/nwm_hydroregime"

os.makedirs(OUTDIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--recompute",
    action="store_true",
    help="Recompute NWM hydrologic metrics from raw Zarr instead of using cached outputs."
)
args = parser.parse_args()
RECOMPUTE_NWM = args.recompute


ZARR_PATH = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"

START_DATE = "2003-01-01"
END_DATE   = "2020-12-31"

REACHES_PER_BASIN = 1
TIME_CHUNK = 720
FEAT_CHUNK = 20000
MAP_CHUNK  = 200000

print("Output folder:", OUTDIR)


def find_lat_lon_columns(df):
    cols = list(df.columns)
    low = [str(c).strip().lower() for c in cols]
    lat_col, lon_col = None, None

    for c, cl in zip(cols, low):
        if cl in {"lat", "latitude", "y"}:
            lat_col = c
        if cl in {"lon", "long", "longitude", "lng", "x"}:
            lon_col = c

    if lat_col is None:
        for c, cl in zip(cols, low):
            if "latitude" in cl or cl.startswith("lat"):
                lat_col = c
                break

    if lon_col is None:
        for c, cl in zip(cols, low):
            if "longitude" in cl or cl.startswith("lon") or "long" in cl:
                lon_col = c
                break

    return lat_col, lon_col


def load_points_csv(path):
    df0 = pd.read_csv(path, low_memory=False)
    df0.columns = df0.columns.astype(str).str.strip()

    lat_col, lon_col = find_lat_lon_columns(df0)
    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not find lat/lon columns in {path}")

    df0[lat_col] = pd.to_numeric(df0[lat_col], errors="coerce")
    df0[lon_col] = pd.to_numeric(df0[lon_col], errors="coerce")
    df0 = df0.dropna(subset=[lat_col, lon_col]).copy()

    return gpd.GeoDataFrame(
        df0,
        geometry=gpd.points_from_xy(df0[lon_col], df0[lat_col]),
        crs="EPSG:4326",
    )


def load_power_excel(path):
    df0 = pd.read_excel(path, engine="openpyxl")
    df0.columns = df0.columns.astype(str).str.strip()

    lat_col, lon_col = find_lat_lon_columns(df0)
    if lat_col is None or lon_col is None:
        raise ValueError("Could not find lat/lon columns in Power_Unique_Site_CONUS.xlsx")

    df0[lat_col] = pd.to_numeric(df0[lat_col], errors="coerce")
    df0[lon_col] = pd.to_numeric(df0[lon_col], errors="coerce")
    df0 = df0.dropna(subset=[lat_col, lon_col]).copy()

    return gpd.GeoDataFrame(
        df0,
        geometry=gpd.points_from_xy(df0[lon_col], df0[lat_col]),
        crs="EPSG:4326",
    )


def ecdf_xy(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().sort_values().to_numpy()
    if len(x) == 0:
        return None, None
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def ecdf_on_grid(values, grid):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    values = np.sort(values)
    return np.searchsorted(values, grid, side="right") / len(values)


def xlim_quantiles(df, metric, qlo=0.01, qhi=0.995):
    x = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(x) == 0:
        return None
    return float(x.quantile(qlo)), float(x.quantile(qhi))


for p in [AI_FILE, POWER_FILE, TRI_FILE, BASINS_SHP]:
    if not os.path.exists(p):
        raise FileNotFoundError(p)

bas = gpd.read_file(BASINS_SHP, engine="pyogrio")

if bas.crs is None:
    bas = bas.set_crs("EPSG:4326")

bas = bas.to_crs("EPSG:4326")

if "HYBAS_ID" not in bas.columns:
    raise ValueError("HYBAS_ID missing from HydroBASINS file.")

id_field = "HYBAS_ID"

ai_gdf = load_points_csv(AI_FILE)
power_gdf = load_power_excel(POWER_FILE)
tri_gdf = load_points_csv(TRI_FILE)


def assign_basins(points_gdf, basins_gdf):
    joined = gpd.sjoin(
        points_gdf,
        basins_gdf[[id_field, "geometry"]],
        how="inner",
        predicate="within",
    )
    joined = joined.drop(columns=[c for c in joined.columns if c.startswith("index_")], errors="ignore")
    joined[id_field] = joined[id_field].astype(np.int64)
    return joined


ai_gdf = assign_basins(ai_gdf, bas)
power_gdf = assign_basins(power_gdf, bas)
tri_gdf = assign_basins(tri_gdf, bas)


def basin_presence_counts(gdf, prefix):
    out = gdf.groupby(id_field).size().rename(f"n_{prefix}").reset_index()
    out[f"{prefix}_present"] = 1
    return out


ai_b = basin_presence_counts(ai_gdf, "AI")
pw_b = basin_presence_counts(power_gdf, "Power")
tr_b = basin_presence_counts(tri_gdf, "TRI")

bm = bas[[id_field]].drop_duplicates().copy()
bm = bm.merge(ai_b, on=id_field, how="left")
bm = bm.merge(pw_b, on=id_field, how="left")
bm = bm.merge(tr_b, on=id_field, how="left")

for c in ["AI_present", "Power_present", "TRI_present"]:
    bm[c] = bm[c].fillna(0).astype(int)

for c in ["n_AI", "n_Power", "n_TRI"]:
    bm[c] = bm[c].fillna(0).astype(int)

PRES_OUT = os.path.join(OUTDIR, "basin_presence_counts.csv")
bm.to_csv(PRES_OUT, index=False)
print("Saved:", PRES_OUT)



print("Opening NWM Zarr metadata...")

fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-east-1"})
mapper = s3fs.S3Map(root=ZARR_PATH, s3=fs, check=False)

try:
    ds = xr.open_zarr(mapper, consolidated=True)
    print("Opened with consolidated=True")
except Exception:
    ds = xr.open_zarr(mapper, consolidated=False)
    print("Opened with consolidated=False")

print(ds)

chunk_dict = {}
if "time" in ds.dims:
    chunk_dict["time"] = TIME_CHUNK
if "feature_id" in ds.dims:
    chunk_dict["feature_id"] = FEAT_CHUNK

ds = ds.chunk(chunk_dict)

print("Data variables:", list(ds.data_vars))
print("Coordinates:", list(ds.coords))
print("Dimensions:", dict(ds.dims))


MAP_FILE = os.path.join(OUTDIR, "feature_to_basin.parquet")

if os.path.exists(MAP_FILE):
    print("Loading cached feature-to-basin map:", MAP_FILE)
    feat2bas = pd.read_parquet(MAP_FILE)
else:
    print("Building feature-to-basin map from NWM reach coordinates and HydroBASINS...")

    nfeat = ds["feature_id"].shape[0]
    maps = []

    lat_name = "latitude" if "latitude" in ds else "lat"
    lon_name = "longitude" if "longitude" in ds else "lon"

    for i0 in range(0, nfeat, MAP_CHUNK):
        i1 = min(i0 + MAP_CHUNK, nfeat)
        print(f"Mapping reaches {i0:,} to {i1:,}")

        fid = ds["feature_id"].isel(feature_id=slice(i0, i1)).values.astype(np.int64)
        lat = ds[lat_name].isel(feature_id=slice(i0, i1)).values.astype(float)
        lon = ds[lon_name].isel(feature_id=slice(i0, i1)).values.astype(float)

        if "order" in ds:
            ordv = ds["order"].isel(feature_id=slice(i0, i1)).values.astype(float)
        elif "stream_order" in ds:
            ordv = ds["stream_order"].isel(feature_id=slice(i0, i1)).values.astype(float)
        else:
            ordv = np.full(len(fid), np.nan)

        pts = gpd.GeoDataFrame(
            {"feature_id": fid, "order": ordv},
            geometry=gpd.points_from_xy(lon, lat),
            crs="EPSG:4326",
        )

        j = gpd.sjoin(
            pts,
            bas[[id_field, "geometry"]],
            how="inner",
            predicate="within",
        )

        j = j.drop(columns=[c for c in j.columns if c.startswith("index_")], errors="ignore")
        j[id_field] = j[id_field].astype(np.int64)
        maps.append(j[["feature_id", id_field, "order"]])

    feat2bas = pd.concat(maps, ignore_index=True)
    feat2bas = feat2bas.drop_duplicates(subset=["feature_id"])
    feat2bas.to_parquet(MAP_FILE, index=False)
    print("Saved:", MAP_FILE)

feat2bas["order"] = pd.to_numeric(feat2bas["order"], errors="coerce")

if feat2bas["order"].notna().sum() > 0:
    feat2bas_sorted = feat2bas.sort_values("order", ascending=False)
else:
    feat2bas_sorted = feat2bas.copy()

top_reach = (
    feat2bas_sorted
    .groupby(id_field, as_index=False)
    .head(REACHES_PER_BASIN)
    .copy()
)

sel_features = top_reach["feature_id"].astype(np.int64).unique()

selected_csv = os.path.join(OUTDIR, "selected_reach_per_basin.csv")
top_reach.to_csv(selected_csv, index=False)
print("Saved:", selected_csv)
print("Selected reaches:", len(sel_features))



# ============================================================
# 4. COMPUTE HYDROLOGIC METRICS IN SMALL FEATURE BATCHES
# ============================================================

print("Subsetting NWM streamflow in batches...")

BATCH_SIZE = 250
BATCH_DIR = os.path.join(OUTDIR, "metric_batches")
os.makedirs(BATCH_DIR, exist_ok=True)

HYDRO_OUT = os.path.join(OUTDIR, "basin_hydrologic_metrics.csv")

if (not RECOMPUTE_NWM) and os.path.exists(HYDRO_OUT) and os.path.getsize(HYDRO_OUT) > 0:
    print(f"Using cached hydrologic metrics: {HYDRO_OUT}")
    bas_metrics = pd.read_csv(HYDRO_OUT)

else:
    print("Opening NWM Zarr metadata...")

    ds_sub = ds.sel(time=slice(START_DATE, END_DATE))

    fid_index = pd.Index(ds_sub["feature_id"].values.astype(np.int64))
    sel_features = np.asarray(sel_features, dtype=np.int64)

    keep_pos_all = fid_index.get_indexer(sel_features)
    valid_mask = keep_pos_all >= 0

    keep_pos_all = keep_pos_all[valid_mask]
    sel_features_valid = sel_features[valid_mask]

    print("Total selected reaches found in NWM:", len(sel_features_valid))
    print("Batch size:", BATCH_SIZE)

    batch_files = []

    for b0 in range(0, len(keep_pos_all), BATCH_SIZE):
        b1 = min(b0 + BATCH_SIZE, len(keep_pos_all))
        batch_id = b0 // BATCH_SIZE + 1

        batch_file = os.path.join(BATCH_DIR, f"metrics_batch_{batch_id:04d}.parquet")

        if os.path.exists(batch_file):
            print(f"Skipping existing batch {batch_id}: {batch_file}")
            batch_files.append(batch_file)
            continue

        print(f"\nComputing batch {batch_id}: reaches {b0:,} to {b1:,}")

        ds_b = ds_sub.isel(feature_id=keep_pos_all[b0:b1])

        if "streamflow" in ds_b.data_vars:
            q = ds_b["streamflow"]
        elif "q" in ds_b.data_vars:
            q = ds_b["q"]
        else:
            raise ValueError("Could not find streamflow variable in NWM Zarr.")

        q_daily = q.resample(time="1D").mean()

        q_mean = q_daily.mean("time").rename("meanQ")
        q_std = q_daily.std("time").rename("stdQ")
        CVQ = (q_std / xr.where(q_mean == 0, np.nan, q_mean)).rename("CVQ")

        Q10 = q_daily.quantile(0.10, dim="time").squeeze(drop=True).rename("Q10")
        Q90 = q_daily.quantile(0.90, dim="time").squeeze(drop=True).rename("Q90")

        q_sum = q_daily.sum("time")

        RBI = (
            np.abs(q_daily.diff("time")).sum("time") /
            xr.where(q_sum == 0, np.nan, q_sum)
        ).rename("RBI")

        q_mclim = q_daily.groupby("time.month").mean("time").chunk({"month": -1})

        q_sorted = xr.apply_ufunc(
            np.sort,
            q_mclim,
            input_core_dims=[["month"]],
            output_core_dims=[["month"]],
            dask="parallelized",
            output_dtypes=[q_mclim.dtype],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )

        top3 = q_sorted.isel(month=slice(-3, None)).sum("month")
        ann = q_mclim.sum("month")

        season_conc = (
            top3 / xr.where(ann == 0, np.nan, ann)
        ).rename("season_conc")

        feat_metrics = xr.merge(
            [q_mean, q_std, CVQ, Q10, Q90, RBI, season_conc],
            compat="override",
        )

        with ProgressBar():
            feat_metrics_df = feat_metrics.compute().to_dataframe().reset_index()

        feat_metrics_df["feature_id"] = feat_metrics_df["feature_id"].astype(np.int64)
        feat_metrics_df = feat_metrics_df.drop(columns=["quantile"], errors="ignore")

        feat_metrics_df.to_parquet(batch_file, index=False)
        batch_files.append(batch_file)

        print("Saved batch:", batch_file)

    print("\nCombining metric batches...")

    batch_files = sorted([
        os.path.join(BATCH_DIR, f)
        for f in os.listdir(BATCH_DIR)
        if f.startswith("metrics_batch_") and f.endswith(".parquet")
    ])

    if len(batch_files) == 0:
        raise RuntimeError("No metric batch files found.")

    feat_metrics_df = pd.concat(
        [pd.read_parquet(f) for f in batch_files],
        ignore_index=True
    )

    feat_metrics_df["feature_id"] = feat_metrics_df["feature_id"].astype(np.int64)

    top_reach_small = top_reach[[id_field, "feature_id"]].copy()
    top_reach_small["feature_id"] = top_reach_small["feature_id"].astype(np.int64)

    feat_metrics_df = feat_metrics_df.merge(top_reach_small, on="feature_id", how="left")
    feat_metrics_df = feat_metrics_df.dropna(subset=[id_field])
    feat_metrics_df[id_field] = feat_metrics_df[id_field].astype(np.int64)

    bas_metrics = feat_metrics_df.groupby(id_field).agg({
        "meanQ": "mean",
        "stdQ": "mean",
        "CVQ": "mean",
        "Q10": "mean",
        "Q90": "mean",
        "RBI": "mean",
        "season_conc": "mean",
    }).reset_index()

    bas_metrics.to_csv(HYDRO_OUT, index=False)
    print("Saved:", HYDRO_OUT)

print("Saved:", HYDRO_OUT)


master = bm.merge(bas_metrics, on=id_field, how="inner")

master["any_infra"] = (
    (master["AI_present"] == 1) |
    (master["Power_present"] == 1) |
    (master["TRI_present"] == 1)
).astype(int)

master["group_excl"] = np.where(master["any_infra"] == 0, "None", "Infrastructure")
master = master.dropna(subset=["RBI", "season_conc", "CVQ"]).copy()

MASTER_OUT = os.path.join(OUTDIR, "basin_master_presence_hydro.csv")
master.to_csv(MASTER_OUT, index=False)
print("Saved:", MASTER_OUT)

df = master.copy()

print("Valid basins:", len(df))
print("None:", (df["group_excl"] == "None").sum())
print("AI:", df["AI_present"].sum())
print("Power:", df["Power_present"].sum())
print("TRI:", df["TRI_present"].sum())


SECTORS = ["AI", "Power", "TRI"]
METRICS = ["RBI", "season_conc", "CVQ"]

COLOR = {
    "AI": "#1f78b4",
    "Power": "#d95f02",
    "TRI": "#1b9e77",
}

title_map = {
    "RBI": "RBI",
    "season_conc": "Seasonality concentration",
    "CVQ": "CVQ",
}


def sector_mask(df, sector):
    return df[f"{sector}_present"] == 1


B = 500
RNG = np.random.default_rng(42)

matched_cache = {}
effect_rows = []

none_df = df[df["group_excl"] == "None"].copy()

for sector in SECTORS:
    sec_df = df[sector_mask(df, sector)].copy()

    for metric in METRICS:
        sec_vals = pd.to_numeric(sec_df[metric], errors="coerce").dropna()
        none_vals = pd.to_numeric(none_df[metric], errors="coerce").dropna()

        if len(sec_vals) < 3 or len(none_vals) < 3:
            matched_cache[(sector, metric)] = None
            continue

        lim = xlim_quantiles(df, metric, 0.01, 0.99 if metric == "CVQ" else 0.995)
        x_grid = np.linspace(lim[0], lim[1], 250)

        g_ecdf = ecdf_on_grid(sec_vals, x_grid)

        boot_ecdfs = []
        boot_median_diffs = []

        n = len(sec_vals)

        for _ in range(B):
            sample = RNG.choice(none_vals.to_numpy(), size=n, replace=True)
            boot_ecdfs.append(ecdf_on_grid(sample, x_grid))
            boot_median_diffs.append(np.median(sec_vals) - np.median(sample))

        boot_ecdfs = np.vstack(boot_ecdfs)
        boot_median_diffs = np.array(boot_median_diffs)

        matched_cache[(sector, metric)] = {
            "x_grid": x_grid,
            "g_ecdf": g_ecdf,
            "b_med": np.nanmedian(boot_ecdfs, axis=0),
            "b_lo": np.nanquantile(boot_ecdfs, 0.025, axis=0),
            "b_hi": np.nanquantile(boot_ecdfs, 0.975, axis=0),
        }

        effect_rows.append({
            "Sector": sector,
            "Metric": metric,
            "N_sector": len(sec_vals),
            "N_none_pool": len(none_vals),
            "Median_sector": np.median(sec_vals),
            "Median_matched_baseline": np.median(none_vals),
            "MedianDiff_sector_minus_matched": np.median(boot_median_diffs),
            "CI95_low": np.quantile(boot_median_diffs, 0.025),
            "CI95_high": np.quantile(boot_median_diffs, 0.975),
        })

effects = pd.DataFrame(effect_rows)

EFFECTS_OUT = os.path.join(OUTDIR, "NWM_matched_baseline_effect_sizes.csv")
effects.to_csv(EFFECTS_OUT, index=False)
print("Saved:", EFFECTS_OUT)


GRAY_GRID = "#E6E6E6"
GRAY_ZERO = "#8A8A8A"
GRAY_FILL = "#BDBDBD"
BLACKISH = "#222222"
NONE_COLOR = "#B3B3B3"

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 700,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

# ============================================================
# EXPLORATORY FIGURE 1 REMOVED FROM PRODUCTION WORKFLOW
# The manuscript uses only the matched-baseline effect-size figure.
# ============================================================

FIG2 = os.path.join(OUTDIR, "FIG_2_NWM_HydroRegime_EffectSizes.png")

fig2, ax = plt.subplots(figsize=(6.0, 3.4), dpi=450)
fig2.subplots_adjust(left=0.24, right=0.98, bottom=0.16, top=0.88)

metric_order = ["RBI", "season_conc", "CVQ"]
metric_labels = {
    "RBI": "RBI",
    "season_conc": "Seasonality concentration",
    "CVQ": "CVQ",
}

base_y = np.array([2.0, 1.0, 0.0])
offset = {"AI": 0.18, "Power": 0.00, "TRI": -0.18}

for sector in SECTORS:
    sub = effects[effects["Sector"] == sector].copy()
    sub = sub.set_index("Metric").reindex(metric_order).reset_index()

    x = sub["MedianDiff_sector_minus_matched"].astype(float).values
    lo = sub["CI95_low"].astype(float).values
    hi = sub["CI95_high"].astype(float).values
    y = base_y + offset[sector]

    ok = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
    xerr = np.vstack([x[ok] - lo[ok], hi[ok] - x[ok]])

    ax.errorbar(
        x[ok],
        y[ok],
        xerr=xerr,
        fmt="o",
        color=COLOR[sector],
        ecolor=COLOR[sector],
        elinewidth=2.0,
        capsize=5,
        capthick=1.6,
        markersize=11,
        markeredgecolor="white",
        markeredgewidth=0.9,
        label=sector,
    )

ax.axvline(0, color=GRAY_ZERO, lw=1.4, ls=(0, (4, 3)))

for yy in base_y:
    ax.axhline(yy, color="#F2F2F2", lw=1.0)

all_lo = pd.to_numeric(effects["CI95_low"], errors="coerce")
all_hi = pd.to_numeric(effects["CI95_high"], errors="coerce")
ax.set_xlim(np.nanmin(all_lo) - 0.015, np.nanmax(all_hi) + 0.015)

ax.set_yticks(base_y)
ax.set_yticklabels([metric_labels[m] for m in metric_order], fontweight="bold")
ax.set_xlabel(
    "Median difference (sector - matched baseline)",
    fontsize=9
)
ax.tick_params(axis="both", labelsize=9)

ax.set_title(
    "Matched-baseline effect sizes",
    fontsize=10,
    fontweight="bold",
    pad=4
)
ax.grid(True, axis="x", color=GRAY_GRID, linewidth=0.8)
ax.grid(False, axis="y")
ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    fontsize=9
)

fig2.savefig(FIG2, bbox_inches="tight")
plt.close()
print("Saved:", FIG2)


# ============================================================
# EXPLORATORY FIGURE 3 REMOVED FROM PRODUCTION WORKFLOW
# The manuscript uses only the matched-baseline effect-size figure.
# ============================================================

print("\nDONE")
print("All outputs saved in:", OUTDIR)
