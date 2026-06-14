#!/usr/bin/env python3
"""
Groundwater-DC analysis.

Creates Groundwater-DC tables and four paper-ready figures:

1. Hydrologic pathway classification
2. Reservoir/lake distance distribution
3. Aquifer enrichment ratio
4. Sector composition in top HUC8 basins

Outputs:
results/groundwater_dc/
"""

import os
import warnings
import textwrap

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common_paths import DATA_FOLDERS, output_folder


# ============================================================
# PATHS
# ============================================================

WATER_ROOT = DATA_FOLDERS["water_project"]
GW_ROOT = DATA_FOLDERS["groundwater"]

OUTDIR = output_folder("groundwater_dc")
TABLEDIR = os.path.join(OUTDIR, "tables")
FIGDIR = os.path.join(OUTDIR, "figures")

os.makedirs(TABLEDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

TARGET_CRS = "EPSG:5070"
MAJOR_RIVER_ORD_STRA = 5

print("[INFO] Groundwater-DC analysis")
print("[INFO] WATER_ROOT:", WATER_ROOT)
print("[INFO] GW_ROOT:", GW_ROOT)
print("[INFO] OUTDIR:", OUTDIR)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 450,
    "font.size": 11,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ============================================================
# HELPERS
# ============================================================

def require(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    print("[FOUND]", label, "->", path)
    return path


def pick_column(df, candidates, required=True):
    low = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    if required:
        raise ValueError(f"Missing columns {candidates}. Available: {list(df.columns)}")
    return None


def clean_geometry(gdf, label):
    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf[~gdf.geometry.isna()]
    gdf = gdf[gdf.geometry.is_valid]
    gdf = gdf[~gdf.geometry.is_empty]
    print(f"[CLEANED] {label}: {len(gdf):,}")
    return gdf


def load_usgs(path):
    for header in range(7):
        df = pd.read_csv(path, header=header, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        if {"STATEFIPS", "COUNTYFIPS", "PS-WFrTo", "DO-WFrTo", "IN-WFrTo", "IR-WFrTo"}.issubset(df.columns):
            print("[USGS LOAD SUCCESS] header =", header)
            return df
    raise ValueError("Could not identify USGS water-use header row.")


def safe_frac(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return np.where(den > 0, num / den, np.nan)


def km(series):
    return pd.to_numeric(series, errors="coerce") / 1000.0


def wrap_labels(labels, width=34):
    return ["\n".join(textwrap.wrap(str(x), width=width)) for x in labels]


def save_fig(fig, filename):
    path = os.path.join(FIGDIR, filename)
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", path)


def classify_pathway(row):
    river_km = row.get("dist_to_major_river_km", np.nan)
    lake_km = row.get("dist_to_reservoir_km", np.nan)
    ps_gw = row.get("county_PS_gw_fraction", np.nan)
    ps_sw = row.get("county_PS_sw_fraction", np.nan)

    near_surface = (
        (pd.notna(river_km) and river_km <= 10) or
        (pd.notna(lake_km) and lake_km <= 10)
    )

    gw_dom = pd.notna(ps_gw) and ps_gw >= 0.5
    sw_dom = pd.notna(ps_sw) and ps_sw >= 0.5

    if sw_dom and near_surface:
        return "Surface-water leaning"
    if gw_dom and near_surface:
        return "Mixed groundwater and nearby surface water"
    if sw_dom and not near_surface:
        return "Surface-water county, no nearby mapped source"
    if gw_dom and not near_surface:
        return "Groundwater leaning"
    return "Mixed or uncertain"


# ============================================================
# INPUTS
# ============================================================

DC_PATH = require(os.path.join(WATER_ROOT, "DC_CONUS.csv"), "DC_CONUS.csv")
USGS_PATH = require(os.path.join(GW_ROOT, "usco2015v2.0.csv"), "USGS county water-use CSV")
HUC8_PATH = require(os.path.join(GW_ROOT, "WBD_National_GPKG.gpkg"), "WBD National GPKG")
COUNTY_PATH = require(os.path.join(GW_ROOT, "tl_2019_us_county.shp"), "County shapefile")
AQUIFER_PATH = require(os.path.join(GW_ROOT, "us_aquifers.shp"), "Aquifer shapefile")
RIVER_PATH = require(os.path.join(GW_ROOT, "HydroRIVERS_v10_na.shp"), "HydroRIVERS")
LAKE_PATH = require(
    os.path.join(GW_ROOT, "HydroLAKES_polys_v10_shp", "HydroLAKES_polys_v10.shp"),
    "HydroLAKES"
)


# ============================================================
# LOAD DATA
# ============================================================

dc_raw = pd.read_csv(DC_PATH)
usgs_raw = load_usgs(USGS_PATH)

huc8 = gpd.read_file(HUC8_PATH, layer="WBDHU8")
counties = gpd.read_file(COUNTY_PATH)
aquifers = gpd.read_file(AQUIFER_PATH)
rivers = gpd.read_file(RIVER_PATH)
lakes = gpd.read_file(LAKE_PATH)

huc8 = clean_geometry(huc8, "HUC8")
counties = clean_geometry(counties, "Counties")
aquifers = clean_geometry(aquifers, "Aquifers")
rivers = clean_geometry(rivers, "Rivers")
lakes = clean_geometry(lakes, "HydroLAKES")


# ============================================================
# DATA CENTERS
# ============================================================

lat_col = pick_column(dc_raw, ["latitude", "lat"])
lon_col = pick_column(dc_raw, ["longitude", "lon", "long"])

dc_raw[lat_col] = pd.to_numeric(dc_raw[lat_col], errors="coerce")
dc_raw[lon_col] = pd.to_numeric(dc_raw[lon_col], errors="coerce")
dc_raw = dc_raw.dropna(subset=[lat_col, lon_col]).copy()
dc_raw["dc_id"] = np.arange(1, len(dc_raw) + 1)

dc = gpd.GeoDataFrame(
    dc_raw,
    geometry=gpd.points_from_xy(dc_raw[lon_col], dc_raw[lat_col]),
    crs="EPSG:4326"
)

print("[INFO] Data centers:", len(dc))


# ============================================================
# PREP SPATIAL LAYERS
# ============================================================

huc8_id = pick_column(huc8, ["HUC8", "huc8", "HUC_8"])
huc8_name = pick_column(huc8, ["NAME", "Name", "HUC8_NAME", "hucname"], required=False)
if huc8_name is None:
    huc8["HUC8_NAME"] = huc8[huc8_id].astype(str)
    huc8_name = "HUC8_NAME"

huc8 = huc8[[huc8_id, huc8_name, "geometry"]].rename(
    columns={huc8_id: "HUC8", huc8_name: "HUC8_NAME"}
)
huc8["HUC8"] = huc8["HUC8"].astype(str).str.zfill(8)

geoid = pick_column(counties, ["GEOID", "GEOID20", "GEOID10"])
county_name = pick_column(counties, ["NAME", "NAMELSAD"], required=False)
counties["FIPS"] = counties[geoid].astype(str).str.zfill(5)
counties["COUNTY_NAME"] = counties[county_name].astype(str) if county_name else counties["FIPS"]
counties = counties[["FIPS", "COUNTY_NAME", "geometry"]]

aq_name = pick_column(
    aquifers,
    ["AQUIFER_NAME", "aq_name", "AQ_NAME", "NAME", "name", "principal_aq", "aquifer"],
    required=False
)
if aq_name is None:
    aq_name = [c for c in aquifers.columns if c != "geometry"][0]
aquifers = aquifers[[aq_name, "geometry"]].rename(columns={aq_name: "AQUIFER_NAME"})
aquifers["AQUIFER_NAME"] = aquifers["AQUIFER_NAME"].astype(str).str.strip()

ord_col = pick_column(rivers, ["ORD_STRA", "ord_stra", "stream_order"], required=False)
if ord_col:
    rivers[ord_col] = pd.to_numeric(rivers[ord_col], errors="coerce")
    rivers = rivers[rivers[ord_col] >= MAJOR_RIVER_ORD_STRA].copy()
rivers = rivers[["geometry"]].copy()

lakes = lakes[["geometry"]].copy()

dc_p = dc.to_crs(TARGET_CRS)
huc8_p = huc8.to_crs(TARGET_CRS)
counties_p = counties.to_crs(TARGET_CRS)
aquifers_p = aquifers.to_crs(TARGET_CRS)
rivers_p = rivers.to_crs(TARGET_CRS)
lakes_p = lakes.to_crs(TARGET_CRS)


# ============================================================
# USGS WATER USE
# ============================================================

usgs = usgs_raw.copy()
usgs["FIPS"] = usgs["STATEFIPS"].astype(str).str.zfill(2) + usgs["COUNTYFIPS"].astype(str).str.zfill(3)

if "YEAR" in usgs.columns:
    usgs["YEAR"] = pd.to_numeric(usgs["YEAR"], errors="coerce")
    usgs = usgs[usgs["YEAR"] == 2015].copy()

rename = {
    "PS-WFrTo": "PS_WFrTo",
    "DO-WFrTo": "DO_WFrTo",
    "IN-WFrTo": "IN_WFrTo",
    "IR-WFrTo": "IR_WFrTo",
    "PS-WGWFr": "PS_WGWFr",
    "PS-WSWFr": "PS_WSWFr",
    "DO-WGWFr": "DO_WGWFr",
    "DO-WSWFr": "DO_WSWFr",
    "IN-WGWFr": "IN_WGWFr",
    "IN-WSWFr": "IN_WSWFr",
    "IR-WGWFr": "IR_WGWFr",
    "IR-WSWFr": "IR_WSWFr",
}
keep = ["FIPS"] + [c for c in rename if c in usgs.columns]
usgs = usgs[keep].rename(columns=rename)

for c in usgs.columns:
    if c != "FIPS":
        usgs[c] = pd.to_numeric(usgs[c], errors="coerce").fillna(0)

counties_water = counties.merge(usgs, on="FIPS", how="left")
for c in counties_water.columns:
    if c not in ["FIPS", "COUNTY_NAME", "geometry"]:
        counties_water[c] = pd.to_numeric(counties_water[c], errors="coerce").fillna(0)

counties_water_p = counties_water.to_crs(TARGET_CRS)


# ============================================================
# SPATIAL JOINS
# ============================================================

print("[INFO] Spatial joins")

dc_huc = gpd.sjoin(
    dc_p[["dc_id", "geometry"]],
    huc8_p[["HUC8", "HUC8_NAME", "geometry"]],
    how="left",
    predicate="within"
).drop(columns="index_right", errors="ignore")

dc_county = gpd.sjoin(
    dc_p[["dc_id", "geometry"]],
    counties_water_p,
    how="left",
    predicate="within"
).drop(columns="index_right", errors="ignore")

dc_aq = gpd.sjoin(
    dc_p[["dc_id", "geometry"]],
    aquifers_p[["AQUIFER_NAME", "geometry"]],
    how="left",
    predicate="within"
).drop(columns="index_right", errors="ignore")

dc_aq_one = (
    dc_aq.sort_values(["dc_id", "AQUIFER_NAME"])
    .drop_duplicates("dc_id")
    [["dc_id", "AQUIFER_NAME"]]
)

dc_table = dc_p.drop(columns="geometry").copy()
dc_table = dc_table.merge(dc_huc.drop(columns="geometry"), on="dc_id", how="left")
dc_table = dc_table.merge(dc_county.drop(columns="geometry"), on="dc_id", how="left")
dc_table = dc_table.merge(dc_aq_one, on="dc_id", how="left")


# ============================================================
# NEAREST RIVER AND LAKE DISTANCES
# ============================================================

print("[INFO] Nearest river/lake distances")

river_near = gpd.sjoin_nearest(
    dc_p[["dc_id", "geometry"]],
    rivers_p,
    how="left",
    distance_col="dist_to_major_river_m"
).drop(columns="index_right", errors="ignore")

lake_near = gpd.sjoin_nearest(
    dc_p[["dc_id", "geometry"]],
    lakes_p,
    how="left",
    distance_col="dist_to_reservoir_m"
).drop(columns="index_right", errors="ignore")

dc_table = dc_table.merge(
    river_near[["dc_id", "dist_to_major_river_m"]],
    on="dc_id",
    how="left"
)
dc_table = dc_table.merge(
    lake_near[["dc_id", "dist_to_reservoir_m"]],
    on="dc_id",
    how="left"
)

dc_table["dist_to_major_river_km"] = km(dc_table["dist_to_major_river_m"])
dc_table["dist_to_reservoir_km"] = km(dc_table["dist_to_reservoir_m"])


# ============================================================
# COUNTY WATER ATTRIBUTES AND PATHWAY
# ============================================================

dc_table["county_selected_total_MGD"] = (
    dc_table.get("PS_WFrTo", 0).fillna(0)
    + dc_table.get("DO_WFrTo", 0).fillna(0)
    + dc_table.get("IN_WFrTo", 0).fillna(0)
    + dc_table.get("IR_WFrTo", 0).fillna(0)
)

for s in ["PS", "DO", "IN", "IR"]:
    if f"{s}_WFrTo" in dc_table.columns:
        dc_table[f"county_{s}_share"] = safe_frac(dc_table[f"{s}_WFrTo"], dc_table["county_selected_total_MGD"])
    if f"{s}_WGWFr" in dc_table.columns and f"{s}_WFrTo" in dc_table.columns:
        dc_table[f"county_{s}_gw_fraction"] = safe_frac(dc_table[f"{s}_WGWFr"], dc_table[f"{s}_WFrTo"])
    if f"{s}_WSWFr" in dc_table.columns and f"{s}_WFrTo" in dc_table.columns:
        dc_table[f"county_{s}_sw_fraction"] = safe_frac(dc_table[f"{s}_WSWFr"], dc_table[f"{s}_WFrTo"])

dc_table["combined_supply_pathway_proxy"] = dc_table.apply(classify_pathway, axis=1)


# ============================================================
# HUC8 WATER USE SUMMARY
# ============================================================

print("[INFO] HUC8 water-use overlay")

counties_water_p["county_area_m2"] = counties_water_p.geometry.area
overlay = gpd.overlay(
    counties_water_p,
    huc8_p[["HUC8", "HUC8_NAME", "geometry"]],
    how="intersection"
)
overlay["area_weight"] = overlay.geometry.area / overlay["county_area_m2"]

water_cols = [c for c in usgs.columns if c != "FIPS"]
for c in water_cols:
    overlay[f"{c}_aw"] = overlay[c] * overlay["area_weight"]

huc_water = overlay.groupby(["HUC8", "HUC8_NAME"])[[f"{c}_aw" for c in water_cols]].sum().reset_index()
huc_water = huc_water.rename(columns={f"{c}_aw": c for c in water_cols})

dc_counts = dc_table.groupby(["HUC8", "HUC8_NAME"]).size().reset_index(name="dc_count")

master = huc8_p[["HUC8", "HUC8_NAME", "geometry"]].drop_duplicates()
master = master.merge(huc_water, on=["HUC8", "HUC8_NAME"], how="left")
master = master.merge(dc_counts, on=["HUC8", "HUC8_NAME"], how="left")

for c in water_cols:
    master[c] = pd.to_numeric(master[c], errors="coerce").fillna(0)

master["dc_count"] = master["dc_count"].fillna(0).astype(int)
master["has_dc"] = master["dc_count"] > 0
master["selected_total_MGD"] = (
    master.get("PS_WFrTo", 0)
    + master.get("DO_WFrTo", 0)
    + master.get("IN_WFrTo", 0)
    + master.get("IR_WFrTo", 0)
)

for s in ["PS", "DO", "IN", "IR"]:
    master[f"{s}_share"] = safe_frac(master.get(f"{s}_WFrTo", 0), master["selected_total_MGD"])


# ============================================================
# AQUIFER ENRICHMENT
# ============================================================

print("[INFO] Aquifer enrichment")

aq_area = aquifers_p.copy()
aq_area["aq_area_m2"] = aq_area.geometry.area
aq_area = aq_area.groupby("AQUIFER_NAME", as_index=False)["aq_area_m2"].sum()

dc_aq_counts = (
    dc_table.dropna(subset=["AQUIFER_NAME"])
    .groupby("AQUIFER_NAME")
    .size()
    .reset_index(name="observed_dc_count")
)

aq_enrich = aq_area.merge(dc_aq_counts, on="AQUIFER_NAME", how="left")
aq_enrich["observed_dc_count"] = aq_enrich["observed_dc_count"].fillna(0)

total_dc = len(dc_table)
total_area = aq_enrich["aq_area_m2"].sum()

aq_enrich["expected_dc_count"] = total_dc * aq_enrich["aq_area_m2"] / total_area
aq_enrich["enrichment_ratio"] = safe_frac(
    aq_enrich["observed_dc_count"],
    aq_enrich["expected_dc_count"]
)

aq_enrich = aq_enrich[
    ~aq_enrich["AQUIFER_NAME"].str.lower().isin(["other rocks", "other rock", "nan", "none"])
].copy()

aq_enrich = aq_enrich.sort_values("enrichment_ratio", ascending=False)


# ============================================================
# SAVE TABLES
# ============================================================

dc_table.to_csv(os.path.join(TABLEDIR, "data_centers_groundwater_dc.csv"), index=False)
master.drop(columns="geometry").to_csv(os.path.join(TABLEDIR, "master_huc8_groundwater_dc.csv"), index=False)
aq_enrich.to_csv(os.path.join(TABLEDIR, "aquifer_enrichment_table.csv"), index=False)

pathway_counts = (
    dc_table["combined_supply_pathway_proxy"]
    .value_counts()
    .rename_axis("combined_supply_pathway_proxy")
    .reset_index(name="dc_count")
)
pathway_counts.to_csv(os.path.join(TABLEDIR, "combined_supply_pathway_counts.csv"), index=False)

print("[SAVED TABLES]", TABLEDIR)


# ============================================================
# FIGURE 1: PATHWAY
# ============================================================

pathway = pathway_counts.copy()
pathway = pathway.sort_values("dc_count", ascending=True)

fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.barh(pathway["combined_supply_pathway_proxy"], pathway["dc_count"])
ax.set_xlabel("Count of AI data centers")
ax.set_ylabel("Hydrologic pathway classification")
ax.grid(axis="x", alpha=0.20)
plt.tight_layout()
save_fig(fig, "GroundwaterDC_01_pathway_classification.png")


# ============================================================
# FIGURE 2: RESERVOIR / LAKE DISTANCE
# ============================================================

dist = pd.to_numeric(dc_table["dist_to_reservoir_km"], errors="coerce").dropna()
dist = dist[dist >= 0]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.hist(dist, bins=30)
ax.set_xlabel("Distance to nearest reservoir or lake (km)")
ax.set_ylabel("Number of AI data centers")
ax.grid(alpha=0.20)
plt.tight_layout()
save_fig(fig, "GroundwaterDC_02_reservoir_lake_distance_distribution.png")


# ============================================================
# FIGURE 3: AQUIFER ENRICHMENT
# ============================================================

aq_plot = aq_enrich.dropna(subset=["enrichment_ratio"]).copy()
aq_plot = aq_plot.sort_values("enrichment_ratio", ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(9.5, 6.2))
ax.barh(
    wrap_labels(aq_plot["AQUIFER_NAME"], width=34),
    aq_plot["enrichment_ratio"]
)
ax.set_xlabel("Enrichment ratio of AI data-center occurrence")
ax.set_ylabel("Aquifer system")
ax.grid(axis="x", alpha=0.20)
plt.tight_layout()
save_fig(fig, "GroundwaterDC_03_aquifer_enrichment_ratio.png")


# ============================================================
# FIGURE 4: SECTOR COMPOSITION
# ============================================================

top = master[master["has_dc"] == True].copy()
top = top.sort_values("dc_count", ascending=False).head(15)

sector_cols = {
    "Public supply": "PS_share",
    "Domestic": "DO_share",
    "Industrial": "IN_share",
    "Irrigation": "IR_share",
}

fig, ax = plt.subplots(figsize=(11, 7))

x = np.arange(len(top))
bottom = np.zeros(len(top))

for label, col in sector_cols.items():
    vals = pd.to_numeric(top[col], errors="coerce").fillna(0).values
    ax.bar(x, vals, bottom=bottom, label=label)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(top["HUC8_NAME"].astype(str), rotation=75, ha="right")
ax.set_ylabel("Sector share")
ax.set_xlabel("Top HUC8 basins containing AI data centers")
ax.legend(frameon=True, loc="lower left")
ax.grid(axis="y", alpha=0.20)

plt.tight_layout()
save_fig(fig, "GroundwaterDC_04_sector_composition_top_HUC8_basins.png")


print("\nDONE.")
print("Tables:", TABLEDIR)
print("Figures:", FIGDIR)
