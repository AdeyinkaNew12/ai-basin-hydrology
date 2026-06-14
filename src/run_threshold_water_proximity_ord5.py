#!/usr/bin/env python3
"""
Threshold water-proximity analysis from raw facility inventories.

Workflow:
1. Load raw facility files: DC_CONUS.csv, Power.xlsx, TRI_2024.csv
2. Convert valid latitude/longitude records to projected points
3. Filter facilities to the HydroBASINS HUC8 study domain
4. Restrict HydroRIVERS to major rivers using ORD_STRA >= 5
5. Generate basin-matched random baselines
6. Compute nearest distance to major rivers, lakes, and coast
7. Export sector/random distance CSVs, odds-ratio CSV, and 3-panel figure
"""

import os
from common_paths import DATA_FOLDERS, output_folder
import time
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from shapely.geometry import Point


# ============================================================
# SERVER PATHS
# ============================================================

DATA_ROOT = DATA_FOLDERS["base_data"] if "base_data" in DATA_FOLDERS else os.path.dirname(DATA_FOLDERS["water_project"])

PROJECT_DATA_DIR = os.path.join(DATA_ROOT, "WaterProject")
GROUNDWATER_DIR = os.path.join(DATA_ROOT, "Groundwater")

OUT_DIR = os.path.join(
    PROJECT_DATA_DIR,
    "Water_Proximity_FROM_RAW_BASIN_FILTER_ORD5"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

PROJ_CRS = "EPSG:5070"
ORD_MIN = 5
THRESHOLDS = [1, 3, 5, 10]
RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

t0 = time.time()

def mark(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def find_file(filename):
    roots = [
        PROJECT_DATA_DIR,
        os.path.join(PROJECT_DATA_DIR, "Data for Analysis"),
        GROUNDWATER_DIR,
        DATA_ROOT,
    ]

    matches = []

    for root in roots:
        if not os.path.exists(root):
            continue

        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower() == filename.lower():
                    matches.append(os.path.join(dirpath, f))

    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {roots}")

    matches = sorted(set(matches))
    print(f"FOUND: {filename} -> {matches[0]}", flush=True)
    return matches[0]


def find_col(df, options):
    cols = {str(c).lower().strip(): c for c in df.columns}

    for opt in options:
        key = opt.lower().strip()
        if key in cols:
            return cols[key]

    raise ValueError(f"Could not find any of these columns: {options}")


def read_and_project(path):
    gdf = gpd.read_file(path)

    if gdf.crs is None:
        print("Missing CRS, setting EPSG:4326 for:", path, flush=True)
        gdf = gdf.set_crs("EPSG:4326")

    return gdf.to_crs(PROJ_CRS)


def make_points(df, lat_col, lon_col):
    temp = df.copy()

    temp[lat_col] = pd.to_numeric(temp[lat_col], errors="coerce")
    temp[lon_col] = pd.to_numeric(temp[lon_col], errors="coerce")

    temp = temp.dropna(subset=[lat_col, lon_col]).copy()

    temp = temp[
        (temp[lat_col].between(-90, 90)) &
        (temp[lon_col].between(-180, 180))
    ].copy()

    temp = temp.reset_index(drop=True)
    temp["facility_uid"] = np.arange(len(temp))

    gdf = gpd.GeoDataFrame(
        temp,
        geometry=gpd.points_from_xy(temp[lon_col], temp[lat_col]),
        crs="EPSG:4326",
    )

    return gdf.to_crs(PROJ_CRS)


def add_nearest_distance(points_gdf, target_gdf, colname):
    joined = gpd.sjoin_nearest(
        points_gdf,
        target_gdf[["geometry"]],
        how="left",
        distance_col=colname + "_m",
    )

    joined[colname] = joined[colname + "_m"] / 1000.0

    joined = joined.sort_values(colname).drop_duplicates(
        subset="facility_uid",
        keep="first",
    )

    joined = joined.drop(
        columns=["index_right", colname + "_m"],
        errors="ignore",
    )

    joined = joined.sort_values("facility_uid").reset_index(drop=True)

    return joined


def make_random_points_matched(facility_gdf, basin_gdf, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    joined = gpd.sjoin(
        facility_gdf[["facility_uid", "geometry"]],
        basin_gdf[["geometry"]],
        how="left",
        predicate="within",
    )

    joined = joined.drop_duplicates(subset="facility_uid", keep="first")

    records = []

    for _, row in joined.iterrows():
        basin_index = row.get("index_right")
        uid = row["facility_uid"]

        if pd.isna(basin_index):
            continue

        poly = basin_gdf.loc[basin_index, "geometry"]

        if poly is None or poly.is_empty:
            continue

        minx, miny, maxx, maxy = poly.bounds

        for _ in range(10000):
            p = Point(
                rng.uniform(minx, maxx),
                rng.uniform(miny, maxy),
            )

            if poly.contains(p):
                records.append({"facility_uid": uid, "geometry": p})
                break

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=facility_gdf.crs,
    )


def odds_ratio_ci(a, b, c, d):
    # Haldane-Anscombe correction
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    odds_ratio = (a / b) / (c / d)
    se = np.sqrt((1 / a) + (1 / b) + (1 / c) + (1 / d))
    log_or = np.log(odds_ratio)

    ci_low = np.exp(log_or - 1.96 * se)
    ci_high = np.exp(log_or + 1.96 * se)

    return odds_ratio, ci_low, ci_high


# ============================================================
# MAIN
# ============================================================

def main():
    print("Outputs will save to:", OUT_DIR, flush=True)

    # -------------------------
    # Locate inputs
    # -------------------------
    ai_file = find_file("DC_CONUS.csv")
    power_file = find_file("Power.xlsx")
    tri_file = find_file("TRI_2024.csv")

    basins_shp = find_file("hybas_na_lev08_v1c.shp")
    rivers_shp = find_file("HydroRIVERS_v10_na.shp")
    lakes_shp = find_file("HydroLAKES_polys_v10.shp")
    coast_shp = find_file("ne_10m_coastline.shp")

    # -------------------------
    # Load raw facilities
    # -------------------------
    mark("Loading raw facility files...")

    dc = pd.read_csv(ai_file)
    power = pd.read_excel(power_file, header=1)
    tri = pd.read_csv(tri_file, low_memory=False)

    dc.columns = dc.columns.str.strip()
    power.columns = power.columns.str.strip()
    tri.columns = tri.columns.str.strip()

    print("\nRaw rows:", flush=True)
    print("DC:", len(dc), flush=True)
    print("Power:", len(power), flush=True)
    print("TRI:", len(tri), flush=True)

    # -------------------------
    # Coordinate columns
    # -------------------------
    dc_lat = find_col(dc, ["latitude", "Latitude", "lat", "LAT"])
    dc_lon = find_col(dc, ["longitude", "Longitude", "lon", "lng", "LON"])

    power_lat = find_col(power, ["Latitude", "latitude", "lat", "LAT"])
    power_lon = find_col(power, ["Longitude", "longitude", "lon", "lng", "LON"])

    tri_lat = find_col(tri, ["12. LATITUDE", "LATITUDE", "latitude", "lat"])
    tri_lon = find_col(tri, ["13. LONGITUDE", "LONGITUDE", "longitude", "lon", "lng"])

    print("\nUsing columns:", flush=True)
    print("DC:", dc_lat, dc_lon, flush=True)
    print("Power:", power_lat, power_lon, flush=True)
    print("TRI:", tri_lat, tri_lon, flush=True)

    # -------------------------
    # Convert to points
    # -------------------------
    mark("Converting facilities to projected points...")

    facilities = {
        "data_center": make_points(dc, dc_lat, dc_lon),
        "power_plant": make_points(power, power_lat, power_lon),
        "industry_TRI": make_points(tri, tri_lat, tri_lon),
    }

    print("\nValid-coordinate facilities before basin filter:", flush=True)
    for sector, gdf in facilities.items():
        print(sector, len(gdf), flush=True)

    # -------------------------
    # Load spatial layers
    # -------------------------
    mark("Loading basins and hydrographic layers...")

    basins = read_and_project(basins_shp)
    rivers = read_and_project(rivers_shp)
    lakes = read_and_project(lakes_shp)
    coast = read_and_project(coast_shp)

    print("\nLayer sizes before river filter:", flush=True)
    print("Basins:", len(basins), flush=True)
    print("Rivers:", len(rivers), flush=True)
    print("Lakes:", len(lakes), flush=True)
    print("Coast:", len(coast), flush=True)

    # -------------------------
    # Filter facilities to HydroBASINS domain
    # -------------------------
    mark("Filtering facilities to HydroBASINS domain...")

    filtered = {}

    for sector, gdf in facilities.items():
        joined = gpd.sjoin(
            gdf,
            basins[["geometry"]],
            how="inner",
            predicate="within",
        )

        joined = joined.drop(columns=["index_right"], errors="ignore")
        joined = joined.reset_index(drop=True)
        joined["facility_uid"] = np.arange(len(joined))

        filtered[sector] = joined

    facilities = filtered

    print("\nFacilities inside HydroBASINS domain:", flush=True)
    for sector, gdf in facilities.items():
        print(sector, len(gdf), flush=True)

    # -------------------------
    # Major rivers only
    # -------------------------
    mark("Filtering HydroRIVERS to major reaches...")

    if "ORD_STRA" not in rivers.columns:
        raise ValueError("ORD_STRA column not found in HydroRIVERS.")

    rivers["ORD_STRA"] = pd.to_numeric(rivers["ORD_STRA"], errors="coerce")
    rivers = rivers[rivers["ORD_STRA"] >= ORD_MIN].copy()

    print(f"\nUsing major rivers only: ORD_STRA >= {ORD_MIN}", flush=True)
    print("Major river reaches:", len(rivers), flush=True)

    print("\nLayer sizes after river filter:", flush=True)
    print("Basins:", len(basins), flush=True)
    print("Rivers ORD_STRA >= 5:", len(rivers), flush=True)
    print("Lakes:", len(lakes), flush=True)
    print("Coast:", len(coast), flush=True)

    # -------------------------
    # Distance calculation
    # -------------------------
    features = {
        "river": "dist_river_km",
        "lake": "dist_lake_km",
        "coast": "dist_coast_km",
    }

    sector_files = {}
    random_files = {}

    for sector, gdf in facilities.items():
        mark(f"Processing {sector}...")

        real = gdf.copy()
        real = add_nearest_distance(real, rivers, "dist_river_km")
        real = add_nearest_distance(real, lakes, "dist_lake_km")
        real = add_nearest_distance(real, coast, "dist_coast_km")

        rand = make_random_points_matched(gdf, basins, seed=RANDOM_SEED)
        rand = add_nearest_distance(rand, rivers, "dist_river_km")
        rand = add_nearest_distance(rand, lakes, "dist_lake_km")
        rand = add_nearest_distance(rand, coast, "dist_coast_km")

        real_csv = os.path.join(
            OUT_DIR,
            f"{sector}_sector_nearest_water_BASIN_FILTER_ORD5.csv",
        )

        rand_csv = os.path.join(
            OUT_DIR,
            f"{sector}_random_nearest_water_BASIN_FILTER_ORD5.csv",
        )

        real.drop(columns="geometry").to_csv(real_csv, index=False)
        rand.drop(columns="geometry").to_csv(rand_csv, index=False)

        sector_files[sector] = real_csv
        random_files[sector] = rand_csv

        print("Real rows:", len(real), flush=True)
        print("Random rows:", len(rand), flush=True)
        print("Saved:", real_csv, flush=True)
        print("Saved:", rand_csv, flush=True)

    # -------------------------
    # Odds-ratio table
    # -------------------------
    mark("Calculating odds ratios...")

    rows = []
    sectors = ["data_center", "power_plant", "industry_TRI"]

    for sector in sectors:
        sector_df = pd.read_csv(sector_files[sector], low_memory=False)
        random_df = pd.read_csv(random_files[sector], low_memory=False)

        for feature, col in features.items():
            sector_dist = pd.to_numeric(sector_df[col], errors="coerce").dropna()
            random_dist = pd.to_numeric(random_df[col], errors="coerce").dropna()

            for threshold in THRESHOLDS:
                a = int((sector_dist <= threshold).sum())
                b = int((sector_dist > threshold).sum())

                c = int((random_dist <= threshold).sum())
                d = int((random_dist > threshold).sum())

                odds_ratio, ci_low, ci_high = odds_ratio_ci(a, b, c, d)

                rows.append(
                    {
                        "sector": sector,
                        "feature": feature,
                        "threshold_km": threshold,
                        "sector_within": a,
                        "sector_outside": b,
                        "random_within": c,
                        "random_outside": d,
                        "odds_ratio": odds_ratio,
                        "CI_low": ci_low,
                        "CI_high": ci_high,
                    }
                )

    res = pd.DataFrame(rows)

    odds_csv = os.path.join(
        OUT_DIR,
        "WaterProximity_threshold_odds_ratios_ORD5.csv",
    )

    res.to_csv(odds_csv, index=False)

    print("\nSaved odds-ratio CSV:", odds_csv, flush=True)
    print(res.head(12), flush=True)

    # -------------------------
    # Plot
    # -------------------------
    mark("Creating figure...")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
        }
    )

    sector_color = {
        "data_center": "tab:blue",
        "power_plant": "tab:orange",
        "industry_TRI": "tab:green",
    }

    sector_marker = {
        "data_center": "o",
        "power_plant": "s",
        "industry_TRI": "^",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)

    for ax, feat in zip(axes, features.keys()):
        sub = res[res["feature"] == feat].copy()

        for sector in sectors:
            s = sub[sub["sector"] == sector].sort_values("threshold_km")

            x = s["threshold_km"].to_numpy()
            y = s["odds_ratio"].to_numpy()
            lo = s["CI_low"].to_numpy()
            hi = s["CI_high"].to_numpy()

            ax.errorbar(
                x,
                y,
                yerr=[y - lo, hi - y],
                color=sector_color[sector],
                marker=sector_marker[sector],
                linestyle="-",
                linewidth=2.2,
                markersize=6,
                capsize=2,
                elinewidth=1.0,
                alpha=0.95,
                zorder=3,
                label=sector,
            )

        ax.axhline(
            1,
            linestyle="--",
            color="black",
            linewidth=1.2,
            alpha=0.75,
            zorder=2,
        )

        ax.set_title(feat)
        ax.set_xlabel("Distance threshold (km)")
        ax.grid(True, alpha=0.18, zorder=1)
        ax.set_xticks(THRESHOLDS)

    axes[0].set_ylabel("Odds ratio vs basin-matched random")

    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(0.4, 3.2)
        ax.yaxis.set_major_locator(mticker.FixedLocator([0.5, 1.0, 2.0, 3.0]))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:g}"))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        title="Sector",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
    )

    fig.suptitle(
        "Infrastructure proximity to water: threshold exceedance analysis",
        y=1.03,
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    out_png = os.path.join(
        OUT_DIR,
        "WaterProximity_threshold_plot_panels_ORD5.png",
    )

    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved figure:", out_png, flush=True)
    print("Saved CSV:", odds_csv, flush=True)
    print("Output folder:", OUT_DIR, flush=True)

    mark("ALL DONE")


if __name__ == "__main__":
    main()
