#!/usr/bin/env python3
"""
Proximity of Infrastructure Facilities to Major Rivers

This script evaluates whether AI data centers, power plants, and TRI facilities
are located closer to major river corridors than expected under basin-matched
random placement.

The analysis uses HydroRIVERS Strahler stream order to identify major rivers
and computes nearest-river distances for observed facilities and matched random
points generated within HydroBASINS watershed boundaries.

Main outputs include:
- A summary table of near-river proportions, odds ratios, confidence intervals,
  chi-square statistics, and permutation-based p-values.
- A publication-ready figure comparing observed facilities with matched random
  baselines at a 10 km threshold, with sensitivity checks at 25 km and 50 km.

Input and output paths are managed through common_paths.py so the script can be
run consistently within the project repository environment.
"""

from __future__ import annotations

import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from shapely.strtree import STRtree

from common_paths import DATA_FOLDERS, output_folder


DEFAULT_DATA_ROOT = DATA_FOLDERS["water_project"]
DEFAULT_OUTPUT_ROOT = output_folder("major_river_proximity")
os.makedirs(DEFAULT_OUTPUT_ROOT, exist_ok=True)

AI_FILE = os.path.join(DEFAULT_DATA_ROOT, "DC_CONUS.csv")
POWER_FILE = os.path.join(DEFAULT_DATA_ROOT, "Power.xlsx")
TRI_FILE = os.path.join(DEFAULT_DATA_ROOT, "TRI_2024.csv")
BASINS_SHP = os.path.join(DEFAULT_DATA_ROOT, "hybas_na_lev08_v1c.shp")
RIVERS_SHP = os.path.join(DEFAULT_DATA_ROOT, "HydroRIVERS_v10_na.shp")

OUT_FIG = os.path.join(
    DEFAULT_OUTPUT_ROOT,
    "MajorRiverProximity_ORD_STRA_within10km_AI_Power_TRI.png",
)
OUT_TABLE = os.path.join(
    DEFAULT_OUTPUT_ROOT,
    "MajorRiverProximity_summary_ORD_STRA.csv",
)

SEED = 7
ORD_COL = "ORD_STRA"
STREAM_ORDER_MIN = 5
THRESH_KM = 10.0
SENS_THRESHOLDS_KM = [10, 25, 50]
BOOTSTRAP_B = 2000
PERM_B = 5000
CI = 0.95
SEED_MAP = {"AI": 101, "Power": 202, "TRI": 303}
PROJ_CRS = "EPSG:5070"
SIMPLIFY_M = 200

t0 = time.time()


def mark(msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}")


def pick_col(cols, keys):
    cols_l = {str(c).lower(): c for c in cols}
    for key in keys:
        if key in cols_l:
            return cols_l[key]
    for col in cols:
        col_l = str(col).lower()
        if any(key in col_l for key in keys):
            return col
    return None


def load_points_any(path, sector_name, force_lat=None, force_lon=None, excel_header=0):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file for {sector_name}: {path}")

    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, header=excel_header, engine="openpyxl")
    else:
        df = pd.read_csv(path, low_memory=False)

    df.columns = df.columns.astype(str).str.strip()

    if force_lat and force_lon:
        lat_col, lon_col = force_lat, force_lon
    else:
        lat_col = pick_col(df.columns, ["latitude", "lat", "y"])
        lon_col = pick_col(df.columns, ["longitude", "lon", "lng", "long", "x"])

    if lat_col is None or lon_col is None:
        raise RuntimeError(
            f"{sector_name}: Could not detect lat/lon columns. "
            f"Columns seen: {list(df.columns)[:80]}"
        )

    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df = df.dropna(subset=[lon_col, lat_col]).copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
    gdf["sector"] = sector_name

    print(f"[INFO] Loaded {sector_name}: {len(gdf):,} rows")
    return gdf


def generate_random_points_in_basins(bas_gdf, n, seed=0, batch=25000):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = bas_gdf.total_bounds

    kept = []
    total_kept = 0
    tries = 0

    while total_kept < n and tries < 250:
        tries += 1
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)

        pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(xs, ys),
            crs=bas_gdf.crs,
        )

        joined = gpd.sjoin(
            pts,
            bas_gdf[["geometry"]],
            how="inner",
            predicate="within",
        )

        if len(joined) == 0:
            continue

        needed = n - total_kept
        joined = joined.iloc[:needed].copy()
        kept.append(joined)
        total_kept += len(joined)

    if total_kept < n:
        raise RuntimeError(f"Random generation failed: got {total_kept}/{n} points.")

    return pd.concat(kept, ignore_index=True).iloc[:n].copy()


def bootstrap_prop_binary_fast(is_near, B=2000, seed=0, ci=0.95):
    rng = np.random.default_rng(seed)
    x = np.asarray(pd.Series(is_near).dropna().astype(int).values, dtype=np.int8)
    n = x.size

    if n == 0:
        return np.nan, np.nan, np.nan

    p_hat = x.mean()
    idx = rng.integers(0, n, size=(B, n), endpoint=False)
    boots = x[idx].mean(axis=1)

    alpha = (1 - ci) / 2
    return (
        float(p_hat),
        float(np.quantile(boots, alpha)),
        float(np.quantile(boots, 1 - alpha)),
    )


def odds_ratio_2x2(a, b, c, d):
    a, b, c, d = float(a), float(b), float(c), float(d)

    if min(a, b, c, d) == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    odds_ratio = (a / b) / (c / d)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)

    return (
        float(odds_ratio),
        float(np.exp(np.log(odds_ratio) - 1.96 * se)),
        float(np.exp(np.log(odds_ratio) + 1.96 * se)),
    )


def safe_chi2_2x2(a, b, c, d):
    if (a + c == 0) or (b + d == 0):
        return np.nan, np.nan

    chi2, p_value, _, _ = chi2_contingency([[a, b], [c, d]])
    return float(chi2), float(p_value)


def permutation_pvalue_diff_fast(ai_near, rd_near, B=5000, seed=0):
    rng = np.random.default_rng(seed)

    ai = np.asarray(pd.Series(ai_near).dropna().astype(int).values, dtype=np.int8)
    rd = np.asarray(pd.Series(rd_near).dropna().astype(int).values, dtype=np.int8)

    n_ai, n_rd = ai.size, rd.size
    if n_ai == 0 or n_rd == 0:
        return np.nan, np.nan

    diff_obs = ai.mean() - rd.mean()
    total = np.concatenate([ai, rd])
    ones = int(total.sum())
    total_n = int(total.size)
    zeros = total_n - ones

    k = rng.hypergeometric(ngood=ones, nbad=zeros, nsample=n_ai, size=B)
    p_ai = k / n_ai
    p_rd = (ones - k) / n_rd
    diffs = p_ai - p_rd

    p_value = (np.sum(np.abs(diffs) >= abs(diff_obs)) + 1) / (B + 1)
    return float(diff_obs), float(p_value)


def build_river_strtree(rmaj_proj):
    river_geoms = list(rmaj_proj.geometry)
    tree = STRtree(river_geoms)
    return tree, river_geoms


def nearest_dist_km(points_geoseries, tree, river_geoms, chunk=20000):
    pts_all = list(points_geoseries)
    out = np.empty(len(pts_all), dtype=float)

    for start in range(0, len(pts_all), chunk):
        pts = pts_all[start:start + chunk]
        nearest = tree.nearest(pts)

        if len(nearest) > 0 and isinstance(nearest[0], (int, np.integer)):
            nearest_geoms = [river_geoms[i] for i in nearest]
        else:
            nearest_geoms = nearest

        for i, (point, geom) in enumerate(zip(pts, nearest_geoms)):
            out[start + i] = point.distance(geom) / 1000.0

    return out


def main():
    print("[INFO] Major river proximity")
    print("[INFO] DATA ROOT:", DEFAULT_DATA_ROOT)
    print("[INFO] OUTPUT ROOT:", DEFAULT_OUTPUT_ROOT)

    for path in [AI_FILE, POWER_FILE, TRI_FILE, BASINS_SHP, RIVERS_SHP]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        print("[FOUND]", path)

    mark("Loading basins")
    bas = gpd.read_file(BASINS_SHP, engine="pyogrio")
    bas.columns = bas.columns.astype(str).str.strip()
    if bas.crs is None:
        bas = bas.set_crs("EPSG:4326")
    mark(f"Basins loaded: {len(bas):,}")

    mark("Loading HydroRIVERS")
    rivers = gpd.read_file(RIVERS_SHP, engine="pyogrio")
    rivers.columns = rivers.columns.astype(str).str.strip()

    if ORD_COL not in rivers.columns:
        raise RuntimeError(f"Expected {ORD_COL} in HydroRIVERS. Columns: {list(rivers.columns)}")

    rivers[ORD_COL] = pd.to_numeric(rivers[ORD_COL], errors="coerce")
    rmaj = rivers.dropna(subset=[ORD_COL]).copy()
    rmaj = rmaj[rmaj[ORD_COL] >= STREAM_ORDER_MIN].copy()
    rmaj = rmaj[[ORD_COL, "geometry"]].copy()

    print(f"[INFO] HydroRIVERS segments: {len(rivers):,}")
    print(f"[INFO] Major rivers {ORD_COL} >= {STREAM_ORDER_MIN}: {len(rmaj):,}")

    mark("Projecting and simplifying major rivers")
    rmaj_proj = rmaj.to_crs(PROJ_CRS).copy()
    rmaj_proj["geometry"] = rmaj_proj["geometry"].simplify(SIMPLIFY_M)

    mark("Building STRtree")
    tree, river_geoms = build_river_strtree(rmaj_proj)
    mark("River index ready")

    mark("Loading sector points")
    ai = load_points_any(AI_FILE, "AI")

    tri_probe = pd.read_csv(TRI_FILE, low_memory=False)
    tri_cols = [c.strip() for c in tri_probe.columns.astype(str)]

    if ("12. LATITUDE" in tri_cols) and ("13. LONGITUDE" in tri_cols):
        tri = load_points_any(
            TRI_FILE,
            "TRI",
            force_lat="12. LATITUDE",
            force_lon="13. LONGITUDE",
        )
    else:
        tri = load_points_any(TRI_FILE, "TRI")

    pwr = load_points_any(
        POWER_FILE,
        "Power",
        force_lat="Latitude",
        force_lon="Longitude",
        excel_header=1,
    )

    print("[INFO] Raw counts:", "AI", len(ai), "Power", len(pwr), "TRI", len(tri))

    results = []
    sector_store = {}

    for gdf in [ai, pwr, tri]:
        sector = gdf["sector"].iloc[0]

        mark(f"{sector}: spatial join to basins")
        g_in = gdf.to_crs(bas.crs)
        joined = gpd.sjoin(g_in, bas[["geometry"]], how="inner", predicate="within")
        g_in = joined.drop(columns=["index_right"], errors="ignore").copy()
        n = len(g_in)

        if n == 0:
            print(f"[WARN] {sector}: 0 points inside basins; skipping.")
            continue

        mark(f"{sector}: generating matched random baseline N={n:,}")
        rd = generate_random_points_in_basins(
            bas,
            n=n,
            seed=SEED + SEED_MAP.get(sector, 999),
        )
        rd = gpd.GeoDataFrame(rd, geometry="geometry", crs=bas.crs)

        g_in_proj = g_in.to_crs(PROJ_CRS).copy()
        rd_proj = rd.to_crs(PROJ_CRS).copy()

        mark(f"{sector}: computing nearest distances")
        g_in["dist_river_km"] = nearest_dist_km(g_in_proj.geometry, tree, river_geoms)
        rd["dist_river_km"] = nearest_dist_km(rd_proj.geometry, tree, river_geoms)

        g_in["near10"] = (g_in["dist_river_km"] <= THRESH_KM).astype(np.int8)
        rd["near10"] = (rd["dist_river_km"] <= THRESH_KM).astype(np.int8)

        a = int(g_in["near10"].sum())
        b = int(n - a)
        c = int(rd["near10"].sum())
        d = int(n - c)

        chi2, p_chi = safe_chi2_2x2(a, b, c, d)
        odds_ratio, odds_lo, odds_hi = odds_ratio_2x2(a, b, c, d)

        diff, p_perm = permutation_pvalue_diff_fast(
            g_in["near10"],
            rd["near10"],
            B=PERM_B,
            seed=SEED + 500 + SEED_MAP.get(sector, 0),
        )

        p_sector, lo_sector, hi_sector = bootstrap_prop_binary_fast(
            g_in["near10"],
            B=BOOTSTRAP_B,
            seed=SEED + 1000 + SEED_MAP.get(sector, 0),
            ci=CI,
        )

        p_random, lo_random, hi_random = bootstrap_prop_binary_fast(
            rd["near10"],
            B=BOOTSTRAP_B,
            seed=SEED + 2000 + SEED_MAP.get(sector, 0),
            ci=CI,
        )

        results.append(
            {
                "sector": sector,
                "N": n,
                "order_field": ORD_COL,
                "order_min": STREAM_ORDER_MIN,
                "threshold_km": THRESH_KM,
                "p_near10_sector": p_sector,
                "ci_lo_sector": lo_sector,
                "ci_hi_sector": hi_sector,
                "p_near10_random": p_random,
                "ci_lo_random": lo_random,
                "ci_hi_random": hi_random,
                "a_near_sector": a,
                "b_far_sector": b,
                "c_near_random": c,
                "d_far_random": d,
                "chi2": chi2,
                "p_chi": p_chi,
                "OR_near": odds_ratio,
                "OR_lo": odds_lo,
                "OR_hi": odds_hi,
                "diff_near_pp": diff * 100.0,
                "p_perm": p_perm,
            }
        )

        sector_store[sector] = (g_in, rd)
        mark(f"{sector}: done")

    res_df = pd.DataFrame(results).sort_values("sector")
    res_df.to_csv(OUT_TABLE, index=False)
    print("[SAVED]", OUT_TABLE)
    print(res_df)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    sectors_order = ["AI", "Power", "TRI"]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11), dpi=150)

    for row_idx, sector in enumerate(sectors_order):
        if sector not in sector_store:
            axes[row_idx, 0].axis("off")
            axes[row_idx, 1].axis("off")
            continue

        g_in, rd = sector_store[sector]
        n = len(g_in)
        ax = axes[row_idx, 0]

        p_sector = g_in["near10"].mean()
        p_random = rd["near10"].mean()

        p_sector_hat, lo_sector, hi_sector = bootstrap_prop_binary_fast(
            g_in["near10"],
            B=BOOTSTRAP_B,
            seed=SEED + 3000 + row_idx,
            ci=CI,
        )
        p_random_hat, lo_random, hi_random = bootstrap_prop_binary_fast(
            rd["near10"],
            B=BOOTSTRAP_B,
            seed=SEED + 4000 + row_idx,
            ci=CI,
        )

        categories = [f"Near <= {int(THRESH_KM)} km", f"Far > {int(THRESH_KM)} km"]
        vals_sector = [p_sector, 1 - p_sector]
        vals_random = [p_random, 1 - p_random]

        x = np.arange(2)
        width = 0.35

        ax.bar(
            x - width / 2,
            vals_sector,
            width=width,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.95,
            label=f"{sector} N={n:,}",
        )
        ax.bar(
            x + width / 2,
            vals_random,
            width=width,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.55,
            label=f"Random N={n:,}",
        )

        ax.errorbar(
            x[0] - width / 2,
            p_sector_hat,
            yerr=[[p_sector_hat - lo_sector], [hi_sector - p_sector_hat]],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        ax.errorbar(
            x[0] + width / 2,
            p_random_hat,
            yerr=[[p_random_hat - lo_random], [hi_random - p_random_hat]],
            fmt="none",
            ecolor="black",
            capsize=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Proportion of facilities")
        ax.set_title(f"{sector}: Proximity to major rivers ({ORD_COL} >= {STREAM_ORDER_MIN})")
        ax.yaxis.grid(True, linewidth=0.6)
        ax.set_axisbelow(True)

        result_row = res_df[res_df["sector"] == sector].iloc[0]
        chi_txt = (
            "chi2=NA, p=NA"
            if pd.isna(result_row["chi2"])
            else f"chi2={result_row['chi2']:.1f}, p={result_row['p_chi']:.1e}"
        )

        ax.text(
            0.02,
            0.98,
            f"{chi_txt}\n"
            f"Perm p={result_row['p_perm']:.1e}\n"
            f"OR near={result_row['OR_near']:.2f} "
            f"[{result_row['OR_lo']:.2f},{result_row['OR_hi']:.2f}]\n"
            f"Delta near={result_row['diff_near_pp']:.1f} pp",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.90,
                edgecolor="none",
            ),
        )

        ax.legend(loc="upper right", frameon=False)

        ax2 = axes[row_idx, 1]
        x2 = np.arange(len(SENS_THRESHOLDS_KM))
        sector_props = [(g_in["dist_river_km"] <= t).mean() for t in SENS_THRESHOLDS_KM]
        random_props = [(rd["dist_river_km"] <= t).mean() for t in SENS_THRESHOLDS_KM]

        ax2.plot(x2, sector_props, marker="o", label=sector)
        ax2.plot(x2, random_props, marker="o", label="Random")
        ax2.set_xticks(x2)
        ax2.set_xticklabels([f"{t} km" for t in SENS_THRESHOLDS_KM])
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Proportion within distance")
        ax2.set_title(f"{sector}: Sensitivity")
        ax2.yaxis.grid(True, linewidth=0.6)
        ax2.set_axisbelow(True)
        ax2.legend(frameon=False, loc="lower right")

    fig.suptitle(
        f"Proximity to major rivers HydroRIVERS {ORD_COL} >= {STREAM_ORDER_MIN}\n"
        f"Near vs far at {int(THRESH_KM)} km with matched random baselines",
        y=1.01,
        fontsize=15,
    )

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close()

    print("[SAVED]", OUT_FIG)
    mark("ALL DONE")


if __name__ == "__main__":
    main()
