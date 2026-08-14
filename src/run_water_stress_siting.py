#!/usr/bin/env python3
"""
Water-Stress Siting Analysis for AI, Power, and TRI Facilities

This script evaluates whether AI data centers, power plants, and TRI facilities are preferentially located in water-stressed basins. It supports all_basins and basin-matched baselines, generates statistical tables, and summarizes water-stress siting patterns.

Input files used:
- DC_CONUS.csv
- Power_Unique_Site_CONUS.xlsx
- TRI_2024_Unique_Site_CONUS.csv
- Aqueduct40_baseline_monthly_y2023m07d05.csv
- hybas_na_lev08_v1c.shp

Server/GitHub version:
- reads input files from DEFAULT_DATA_ROOT
- default input folder: /mnt/disk3/aoolaseinde/data/AqueductStress
- writes outputs to DEFAULT_OUTPUT_ROOT
- default output folder: /mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/water_stress_siting
- can be run independently or through src/run_all.py
"""

from __future__ import annotations

import os
from common_paths import DATA_FOLDERS, output_folder

DEFAULT_DATA_ROOT = DATA_FOLDERS["aqueduct_stress"]
DEFAULT_OUTPUT_ROOT = output_folder("water_stress_siting")
os.makedirs(DEFAULT_OUTPUT_ROOT, exist_ok=True)

# Project-style variables retained for compatibility with notebook-derived code.
PROJECT_DIR = DEFAULT_DATA_ROOT

print("[INFO] Analysis: Water-Stress Siting Analysis for AI, Power, and TRI Facilities")
print("[INFO] DEFAULT_DATA_ROOT =", DEFAULT_DATA_ROOT)
print("[INFO] DEFAULT_OUTPUT_ROOT =", DEFAULT_OUTPUT_ROOT)

# ============================================================
#  STRESS SITING (AI / Power / TRI)
# ALL-BASINS BASELINE ONLY

# HOW TO RUN:
# 1) Set PROJECT_DIR below to the folder that contains your input files.
# 2) Ensure the files listed in INPUT FILENAMES exist in that folder.
# 3) Run this cell. Outputs will be created in:
#     
# ============================================================




import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import chi2_contingency

# -------------------------
# PATHS (EDIT IF NEEDED)
# -------------------------
PROJECT_DIR = Path(DATA_FOLDERS["water_project"])

AI_FILE       = PROJECT_DIR / "DC_CONUS.csv"
POWER_FILE    = PROJECT_DIR / "Power_Unique_Site_CONUS.xlsx"
TRI_FILE      = PROJECT_DIR / "TRI_2024_Unique_Site_CONUS.csv"
BASINS_SHP    = PROJECT_DIR / "hybas_na_lev08_v1c.shp"
AQUEDUCT_CSV  = Path(DATA_FOLDERS["aqueduct_stress"]) / "Aqueduct40_baseline_annual_y2023m07d05.csv"

OUT_DIR = Path(output_folder("water_stress_siting"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# PARAMS
# -------------------------
SECTORS    = ["AI","Power","TRI"]
STRESS_COL = "bws_raw"

BASE_SEED = 7
SEED_MAP  = {"AI": 101, "Power": 202, "TRI": 303}
MODE_SEED = {"ALL_BASINS": 10000}

POOL_FACTOR = 5.0
POOL_CAP    = 300000
BATCH       = 70000
MAX_TRIES   = 60

SHOW_PERCENT_LABELS = True
PCT_DECIMALS        = 1

# -------------------------
# STYLE (clean + professional)
# -------------------------
mpl.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 350,
    "font.size": 12,
    "axes.titleweight": "semibold",
    "axes.labelweight": "semibold",
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
})

COL3 = {"low":"#0072B2", "medium":"#009E73", "high":"#CC79A7"}
COL4 = {"Q1":"#56B4E9", "Q2":"#009E73", "Q3":"#E69F00", "Q4":"#D55E00"}

need = [AI_FILE, POWER_FILE, TRI_FILE, BASINS_SHP, AQUEDUCT_CSV]
missing = [p for p in need if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required inputs:\n" + "\n".join(str(p) for p in missing))

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

def safe_sjoin(left_gdf, right_gdf, how="left", predicate="within"):
    L = left_gdf.copy()
    R = right_gdf.copy()
    for c in ["index_right", "index_left"]:
        if c in L.columns: L = L.drop(columns=[c], errors="ignore")
        if c in R.columns: R = R.drop(columns=[c], errors="ignore")
    L = L.reset_index(drop=True)
    R = R.reset_index(drop=True)
    return gpd.sjoin(L, R, how=how, predicate=predicate)

def load_points_any(path, sector_name, force_lat=None, force_lon=None, excel_header=0):
    path = Path(path)
    if str(path).lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(path, header=excel_header)
    else:
        df = pd.read_csv(path, low_memory=False)

    df.columns = df.columns.astype(str).str.strip()

    if force_lat and force_lon:
        lat_col, lon_col = force_lat, force_lon
    else:
        lat_col = pick_col(df.columns, ["latitude","lat","y"])
        lon_col = pick_col(df.columns, ["longitude","lon","lng","long","x"])

    if lat_col is None or lon_col is None:
        raise RuntimeError(f"{sector_name}: could not detect lat/lon columns.")

    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df = df.dropna(subset=[lon_col, lat_col]).copy()

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
    gdf["sector"] = sector_name
    return gdf

def generate_random_points_in_polygonset(poly_gdf, n, seed, batch=BATCH, max_tries=MAX_TRIES):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = poly_gdf.total_bounds
    kept = []
    total = 0
    tries = 0
    while total < n and tries < max_tries:
        tries += 1
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=poly_gdf.crs)
        j = safe_sjoin(pts, poly_gdf[["geometry"]], how="inner", predicate="within")
        if len(j) == 0:
            continue
        need_now = n - total
        kept.append(j.iloc[:need_now].copy())
        total += len(kept[-1])
    if total < n:
        raise RuntimeError(f"Random generation failed: got {total}/{n} points within polygons.")
    out = pd.concat(kept, ignore_index=True).iloc[:n].copy()
    return gpd.GeoDataFrame(out, geometry="geometry", crs=poly_gdf.crs)

def odds_ratio_2x2(a, b, c, d):
    a,b,c,d = float(a),float(b),float(c),float(d)
    if min(a,b,c,d) == 0:
        a += 0.5; b += 0.5; c += 0.5; d += 0.5
    OR = (a/b) / (c/d)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = 1.96
    lo = np.exp(np.log(OR) - z*se)
    hi = np.exp(np.log(OR) + z*se)
    return float(OR), float(lo), float(hi)

def make_all_basins_bins_from_basins(stress_series):
    s = pd.to_numeric(stress_series, errors="coerce")
    t1, t2 = s.quantile([1/3, 2/3]).values
    q1, q2, q3 = s.quantile([0.25, 0.50, 0.75]).values

    def tert(v):
        if pd.isna(v): return np.nan
        if v <= t1: return "low"
        if v <= t2: return "medium"
        return "high"

    def quart(v):
        if pd.isna(v): return np.nan
        if v <= q1: return "Q1"
        if v <= q2: return "Q2"
        if v <= q3: return "Q3"
        return "Q4"

    return s.apply(tert), s.apply(quart)

def stable_sort_for_repro(gdf):
    out = gdf.copy()
    out["_wkb"] = out.geometry.apply(lambda g: g.wkb_hex if g is not None else "")
    out = out.sort_values(["_wkb"], kind="mergesort").drop(columns=["_wkb"])
    return out

def clean_axis(ax):
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, axis="y", linewidth=0.7, alpha=0.18)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for side in ["left","bottom","right","top"]:
        ax.spines[side].set_linewidth(0.9)

def plot_grouped_percent_only(ax, classes, obs_counts, rnd_counts, color_map, title):
    obs_total = int(obs_counts.sum())
    rnd_total = int(rnd_counts.sum())
    p_obs = (obs_counts / max(1, obs_total)).values
    p_rnd = (rnd_counts / max(1, rnd_total)).values

    x = np.arange(len(classes))
    w = 0.36

    for i, cls in enumerate(classes):
        col = color_map[cls]
        ax.bar(x[i]-w/2, p_obs[i], width=w, color=col, alpha=0.95, edgecolor="black", linewidth=0.9)
        ax.bar(x[i]+w/2, p_rnd[i], width=w, color=col, alpha=0.55, edgecolor="black", linewidth=0.9)

        if SHOW_PERCENT_LABELS:
            ax.text(x[i]-w/2, p_obs[i]+0.015, f"{p_obs[i]*100:.{PCT_DECIMALS}f}%", ha="center", va="bottom", fontsize=10)
            ax.text(x[i]+w/2, p_rnd[i]+0.015, f"{p_rnd[i]*100:.{PCT_DECIMALS}f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_title(title)
    clean_axis(ax)

# -------------------------
# Load basins + Aqueduct stress
# -------------------------
bas = gpd.read_file(BASINS_SHP, engine="pyogrio")
bas.columns = bas.columns.astype(str).str.strip()
if bas.crs is None:
    bas = bas.set_crs("EPSG:4326")
if "PFAF_ID" not in bas.columns:
    raise RuntimeError("Expected PFAF_ID in HydroBASINS shapefile.")

bas["PFAF_ID"] = pd.to_numeric(bas["PFAF_ID"], errors="coerce").astype("Int64")
bas["pfaf6"] = (bas["PFAF_ID"] // 100).astype("Int64")
bas8 = bas[["PFAF_ID","pfaf6","geometry"]].copy()

aq = pd.read_csv(AQUEDUCT_CSV, low_memory=False)
aq.columns = aq.columns.astype(str).str.strip()
if ("pfaf_id" not in aq.columns) or (STRESS_COL not in aq.columns):
    raise RuntimeError(f"Aqueduct CSV must contain columns: pfaf_id and {STRESS_COL}.")

aq_small = aq[["pfaf_id", STRESS_COL]].copy()
aq_small["pfaf_id"] = pd.to_numeric(aq_small["pfaf_id"], errors="coerce").astype("Int64")
aq_small[STRESS_COL] = pd.to_numeric(aq_small[STRESS_COL], errors="coerce")

# Retain one annual Aqueduct stress value per Level-6 basin.
# Duplicate pfaf_id records have identical bws_raw values.
aq_small = (
    aq_small
    .dropna(subset=["pfaf_id"])
    .query("pfaf_id != -9999")
    .drop_duplicates(subset=["pfaf_id"])
    .copy()
)

bas_stress = bas8.merge(aq_small, left_on="pfaf6", right_on="pfaf_id", how="left")
bas_stress = bas_stress.rename(columns={STRESS_COL:"stress_value"}).drop(columns=["pfaf_id"], errors="ignore")

terts, quarts = make_all_basins_bins_from_basins(bas_stress["stress_value"])
bas_stress["stress_tertile"]  = terts
bas_stress["stress_quartile"] = quarts

# -------------------------
# Load sectors + attach basin + stress
# -------------------------
ai  = load_points_any(AI_FILE, "AI")

tri_try = pd.read_csv(TRI_FILE, low_memory=False)
tri_cols = [c.strip() for c in tri_try.columns.astype(str)]
if ("12. LATITUDE" in tri_cols) and ("13. LONGITUDE" in tri_cols):
    tri = load_points_any(TRI_FILE, "TRI", force_lat="12. LATITUDE", force_lon="13. LONGITUDE")
else:
    tri = load_points_any(TRI_FILE, "TRI")

pwr = load_points_any(POWER_FILE, "Power", force_lat="Latitude", force_lon="Longitude")

fac_all = []

for gdf in [ai, pwr, tri]:
    sec = gdf["sector"].iloc[0]
    joined = safe_sjoin(
        gdf.to_crs(bas_stress.crs),
        bas_stress[["PFAF_ID","stress_value","stress_tertile","stress_quartile","geometry"]],
        how="left", predicate="within"
    ).rename(columns={"PFAF_ID":"pfaf_id_lev08"})

    joined = joined.dropna(subset=["pfaf_id_lev08","stress_value","stress_tertile","stress_quartile"]).copy()
    joined["sector"] = sec
    fac_all.append(joined)

fac = pd.concat(fac_all, ignore_index=True)

def run_mode(mode_name):
    mode = mode_name.upper()
    if mode != "ALL_BASINS":
        raise ValueError("mode must be 'ALL_BASINS'")

    OUT_FIG    = OUT_DIR / f"analysis4_fig_{mode}_tertiles_quartiles_3x2.png"
    OUT_JOINED = OUT_DIR / f"analysis4_facilities_with_stress_{mode}.csv"
    OUT_RANDOM = OUT_DIR / f"analysis4_random_with_stress_{mode}.csv"
    OUT_STATS  = OUT_DIR / f"analysis4_stats_chi2_OR_{mode}.csv"

    fac.drop(columns=["index_right"], errors="ignore").to_csv(OUT_JOINED, index=False)

    # ---------------------------------------------------------
    # Common CONUS equal-area reference distribution
    # ---------------------------------------------------------
    # Generate one common spatial reference for all sectors.
    # Random locations are generated in CONUS Albers Equal Area
    # (EPSG:5070) so that equal geographic area has equal
    # probability of being sampled.

    reference_n = POOL_CAP

    seed = (
        BASE_SEED
        + MODE_SEED[mode]
    )

    # Project Level-8 basin polygons to an equal-area CRS before
    # generating spatially uniform random reference locations.
    bas_stress_equal_area = bas_stress.to_crs("EPSG:5070")

    pool_equal_area = generate_random_points_in_polygonset(
        bas_stress_equal_area,
        n=reference_n,
        seed=seed,
    )

    # Attach the Level-8 basin and Aqueduct stress attributes
    # directly in the same equal-area CRS.
    rand = safe_sjoin(
        pool_equal_area,
        bas_stress_equal_area[
            [
                "PFAF_ID",
                "stress_value",
                "stress_tertile",
                "stress_quartile",
                "geometry",
            ]
        ],
        how="left",
        predicate="within",
    ).rename(
        columns={"PFAF_ID": "pfaf_id_lev08"}
    )

    rand = rand.dropna(
        subset=[
            "stress_value",
            "stress_tertile",
            "stress_quartile",
        ]
    ).copy()

    rand = stable_sort_for_repro(rand)
    rand["sector"] = "Reference"

    rand.drop(
        columns=["index_right"],
        errors="ignore"
    ).to_csv(
        OUT_RANDOM,
        index=False
    )

    print(
        f"{mode}: common CONUS equal-area reference "
        f"n={len(rand):,}, CRS={rand.crs}"
    )

    order3 = ["low","medium","high"]
    order4 = ["Q1","Q2","Q3","Q4"]

    fig, axes = plt.subplots(3, 2, figsize=(15.2, 9.6), dpi=160, sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.90, bottom=0.10, wspace=0.12, hspace=0.30)

    try:
        fig.supylabel("Proportion of facilities", x=0.055, fontsize=13, fontweight="semibold")
    except Exception:
        fig.text(0.055, 0.5, "Proportion of facilities", va="center", rotation="vertical", fontsize=13, fontweight="semibold")

    for r in range(3):
        axes[r, 1].tick_params(labelleft=False)

    stats_rows = []

    for i, sec in enumerate(SECTORS):
        obs = fac[fac["sector"] == sec].copy()
        rr  = rand.copy()
        N = len(obs)
        if N == 0:
            axes[i,0].axis("off"); axes[i,1].axis("off")
            continue

        obs3 = obs["stress_tertile"].value_counts().reindex(order3, fill_value=0)
        rnd3 = rr["stress_tertile"].value_counts().reindex(order3, fill_value=0)
        chi2_3, p_3, _, _ = chi2_contingency(np.vstack([obs3.values, rnd3.values]))

        a = int(obs3["high"]); b = int(obs3["low"] + obs3["medium"])
        c = int(rnd3["high"]); d = int(rnd3["low"] + rnd3["medium"])
        ORh, ORh_lo, ORh_hi = odds_ratio_2x2(a,b,c,d)

        plot_grouped_percent_only(axes[i,0], order3, obs3, rnd3, COL3, f"{sec}: Stress tertiles")
        axes[i,0].text(
            0.02, 0.98,
            f"$\\chi^2$={chi2_3:.1f}, p={p_3:.1e}\nOR(high)={ORh:.2f} [{ORh_lo:.2f},{ORh_hi:.2f}]",
            transform=axes[i,0].transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.95, edgecolor="black", linewidth=0.9),
            fontsize=9
        )

        obs4 = obs["stress_quartile"].value_counts().reindex(order4, fill_value=0)
        rnd4 = rr["stress_quartile"].value_counts().reindex(order4, fill_value=0)
        chi2_4, p_4, _, _ = chi2_contingency(np.vstack([obs4.values, rnd4.values]))

        a4 = int(obs4["Q4"]); b4 = int(obs4["Q1"] + obs4["Q2"] + obs4["Q3"])
        c4 = int(rnd4["Q4"]); d4 = int(rnd4["Q1"] + rnd4["Q2"] + rnd4["Q3"])
        OR4, OR4_lo, OR4_hi = odds_ratio_2x2(a4,b4,c4,d4)

        plot_grouped_percent_only(axes[i,1], order4, obs4, rnd4, COL4, f"{sec}: Stress quartiles")
        axes[i,1].text(
            0.02, 0.98,
            f"$\\chi^2$={chi2_4:.1f}, p={p_4:.1e}\nOR(Q4)={OR4:.2f} [{OR4_lo:.2f},{OR4_hi:.2f}]",
            transform=axes[i,1].transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.95, edgecolor="black", linewidth=0.9),
            fontsize=9
        )

        stats_rows.append({
            "Mode": mode,
            "Sector": sec,
            "N_with_stress": int(N),
            "Chi2_tertiles": float(chi2_3),
            "p_tertiles": float(p_3),
            "OR_high": float(ORh),
            "OR_high_lo": float(ORh_lo),
            "OR_high_hi": float(ORh_hi),
            "Chi2_quartiles": float(chi2_4),
            "p_quartiles": float(p_4),
            "OR_Q4": float(OR4),
            "OR_Q4_lo": float(OR4_lo),
            "OR_Q4_hi": float(OR4_hi),
        })

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.95, label="Facilities"),
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.55, label="Basin-matched random")
    ], loc="lower center", ncol=2, frameon=False)

    fig.tight_layout(
        rect=(0.055, 0.065, 0.995, 0.995),
        pad=0.5,
        h_pad=0.6,
        w_pad=0.6
    )
    fig.savefig(OUT_FIG, dpi=350, bbox_inches="tight", pad_inches=0.03)
    plt.show()


    # DIRECT_SECTOR_COMPARISON_START
    # ============================================================
    # Direct comparison among AI, Power, and TRI facilities
    #
    # Test 1: Full stress distribution
    #         Low vs Medium vs High
    #
    # Test 2: High water stress vs Not High
    # ============================================================

    direct_rows = []

    for sec in SECTORS:
        sec_data = fac.loc[
            (fac["sector"] == sec)
            & fac["stress_tertile"].notna()
        ].copy()

        low_n = int((sec_data["stress_tertile"] == "low").sum())
        medium_n = int((sec_data["stress_tertile"] == "medium").sum())
        high_n = int((sec_data["stress_tertile"] == "high").sum())

        not_high_n = low_n + medium_n
        total_n = low_n + medium_n + high_n

        direct_rows.append({
            "Mode": mode,
            "Sector": sec,
            "Low": low_n,
            "Medium": medium_n,
            "High": high_n,
            "Not_high": not_high_n,
            "Total": total_n,
            "Low_percent": (
                100.0 * low_n / total_n if total_n else np.nan
            ),
            "Medium_percent": (
                100.0 * medium_n / total_n if total_n else np.nan
            ),
            "High_percent": (
                100.0 * high_n / total_n if total_n else np.nan
            ),
        })

    direct_df = pd.DataFrame(direct_rows)

    # Full 3 sectors x 3 stress categories test
    table_full = direct_df[
        ["Low", "Medium", "High"]
    ].to_numpy()

    chi2_full, p_full, dof_full, expected_full = chi2_contingency(
        table_full
    )

    n_full = int(table_full.sum())
    denominator_full = n_full * min(
        table_full.shape[0] - 1,
        table_full.shape[1] - 1,
    )

    cramers_v_full = (
        np.sqrt(chi2_full / denominator_full)
        if denominator_full > 0
        else np.nan
    )

    # High versus Not High test
    table_high = direct_df[
        ["High", "Not_high"]
    ].to_numpy()

    chi2_high, p_high, dof_high, expected_high = chi2_contingency(
        table_high
    )

    n_high = int(table_high.sum())
    denominator_high = n_high * min(
        table_high.shape[0] - 1,
        table_high.shape[1] - 1,
    )

    cramers_v_high = (
        np.sqrt(chi2_high / denominator_high)
        if denominator_high > 0
        else np.nan
    )

    print("\n" + "=" * 74)
    print("DIRECT WATER-STRESS COMPARISON: AI vs POWER vs TRI")
    print("=" * 74)

    print(
        direct_df[
            [
                "Sector",
                "Low",
                "Medium",
                "High",
                "Not_high",
                "Total",
                "High_percent",
            ]
        ].to_string(
            index=False,
            formatters={
                "High_percent": lambda value: f"{value:.2f}%"
            },
        )
    )

    print("\nFull distribution: Low vs Medium vs High")
    print(table_full)
    print(f"Chi-square = {chi2_full:.4f}")
    print(f"Degrees of freedom = {dof_full}")
    print(f"P-value = {p_full:.10g}")
    print(f"Cramer's V = {cramers_v_full:.4f}")

    if p_full < 0.05:
        print(
            "Result: Overall water-stress distributions differ "
            "significantly among sectors."
        )
    else:
        print(
            "Result: Overall water-stress distributions do not "
            "differ significantly among sectors."
        )

    print("\nHigh water stress vs Not High")
    print(table_high)
    print(f"Chi-square = {chi2_high:.4f}")
    print(f"Degrees of freedom = {dof_high}")
    print(f"P-value = {p_high:.10g}")
    print(f"Cramer's V = {cramers_v_high:.4f}")

    if p_high < 0.05:
        print(
            "Result: High water-stress proportions differ "
            "significantly among sectors."
        )
    else:
        print(
            "Result: High water-stress proportions do not differ "
            "significantly among sectors."
        )

    if cramers_v_high < 0.10:
        print("Effect size for High vs Not High: very small.")
    elif cramers_v_high < 0.30:
        print("Effect size for High vs Not High: small.")
    elif cramers_v_high < 0.50:
        print("Effect size for High vs Not High: moderate.")
    else:
        print("Effect size for High vs Not High: large.")

    # Add statistics to the sector table
    direct_df["Chi2_full_distribution"] = float(chi2_full)
    direct_df["df_full_distribution"] = int(dof_full)
    direct_df["p_full_distribution"] = float(p_full)
    direct_df["Cramers_V_full_distribution"] = float(
        cramers_v_full
    )

    direct_df["Chi2_high_vs_not_high"] = float(chi2_high)
    direct_df["df_high_vs_not_high"] = int(dof_high)
    direct_df["p_high_vs_not_high"] = float(p_high)
    direct_df["Cramers_V_high_vs_not_high"] = float(
        cramers_v_high
    )

    direct_df["High_test_significant_0.05"] = bool(
        p_high < 0.05
    )

    OUT_DIRECT = (
        OUT_DIR
        / f"analysis4_direct_sector_water_stress_{mode}.csv"
    )

    direct_df.to_csv(OUT_DIRECT, index=False)

    print(f"\nSaved direct sector results: {OUT_DIRECT}")

    # Add a summary row to the existing statistics CSV
    stats_rows.append({
        "Mode": mode,
        "Sector": "AI_vs_Power_vs_TRI",
        "N_with_stress": int(n_high),
        "Chi2_direct_full": float(chi2_full),
        "df_direct_full": int(dof_full),
        "p_direct_full": float(p_full),
        "Cramers_V_direct_full": float(cramers_v_full),
        "Chi2_direct_high": float(chi2_high),
        "df_direct_high": int(dof_high),
        "p_direct_high": float(p_high),
        "Cramers_V_direct_high": float(cramers_v_high),
    })

    # DIRECT_SECTOR_COMPARISON_END

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUT_STATS, index=False)

    return str(OUT_FIG), str(OUT_JOINED), str(OUT_RANDOM), str(OUT_STATS), stats_df

# Primary inference is the direct comparison among AI, Power, and TRI.
# The ALL_BASINS analysis is retained as a secondary national null model.
fig_all, join_all, rand_all, stats_all, df_all = run_mode("ALL_BASINS")

# Exact Level-8 basin matching is not used for water-stress inference
# because stress values and categories are constant within each basin.
# Matching observed and random locations within the same basin therefore
# produces identical category distributions by construction.
