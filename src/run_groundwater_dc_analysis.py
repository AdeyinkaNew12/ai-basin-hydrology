# ============================================================
# INTEGRATED DATA CENTER WATER SUPPLY PATHWAYS ANALYSIS
# FULL COLAB-READY PAPER-READY VERSION
# Generates NEW tables + NEW paper figures from raw inputs
# Figures start from Figure 10
# ============================================================

# -------------------------
# 0. INSTALL / IMPORT
# -------------------------
import os
import glob
import textwrap
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, binomtest

# -------------------------
# 2. USER SETTINGS
# -------------------------
ROOT_DRIVE = "/mnt/disk3/aoolaseinde/data"
PROJECT_DIR = "/mnt/disk3/aoolaseinde/data/WaterProject"
TARGET_CRS = "EPSG:5070"   # CONUS Albers Equal Area

# NEW OUTPUT FOLDER: this does NOT use old tables
OUTDIR = "/mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/groundwater_dc"
FIGDIR = os.path.join(OUTDIR, "paper_figures")
TABLEDIR = os.path.join(OUTDIR, "tables")
CAPTIONDIR = os.path.join(OUTDIR, "figure_captions")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABLEDIR, exist_ok=True)
os.makedirs(CAPTIONDIR, exist_ok=True)

MAJOR_RIVER_ORD_STRA = 5
DIST_BINS_KM = [5, 10, 25, 50]
USE_HYDROLAKES_MIN_AREA_FILTER = False
HYDROLAKES_MIN_AREA_KM2 = 1.0

print("Main output folder:", OUTDIR)
print("New tables folder:", TABLEDIR)
print("New figures folder:", FIGDIR)

# -------------------------
# 3. HELPER FUNCTIONS
# -------------------------
def recursive_find(root, patterns):
    for pattern in patterns:
        matches = glob.glob(os.path.join(root, "**", pattern), recursive=True)
        if matches:
            print(f"[FOUND by search] {pattern}: {matches[0]}")
            return matches[0]
    return None


def find_first_existing(candidates, label="file", required=True):
    for p in candidates:
        if p is not None and os.path.exists(p):
            print(f"[FOUND] {label}: {p}")
            return p
    if required:
        raise FileNotFoundError("Could not find " + label + ". Checked:\n" + "\n".join(map(str, candidates)))
    print(f"[WARNING] Could not find {label}.")
    return None


def pick_column(df, candidates, required=True):
    lowmap = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowmap:
            return lowmap[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}\nAvailable: {list(df.columns)}")
    return None


def guess_lat_lon_columns(df):
    cols = {str(c).lower(): c for c in df.columns}
    lat_candidates = ["latitude", "lat", "y", "dec_lat", "dec_lat_va"]
    lon_candidates = ["longitude", "lon", "long", "x", "dec_long", "dec_long_va"]
    lat_col = next((cols[c] for c in lat_candidates if c in cols), None)
    lon_col = next((cols[c] for c in lon_candidates if c in cols), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not identify lat/lon columns. Available columns: {list(df.columns)}")
    return lat_col, lon_col


def safe_read_vector(path, label="vector", layer=None):
    if layer is not None:
        gdf = gpd.read_file(path, layer=layer)
    else:
        gdf = gpd.read_file(path)
    print(f"[LOADED] {label}: {path}" + (f" | layer={layer}" if layer else ""))
    return gdf


def clean_geometry(gdf, label="layer"):
    gdf = gdf.copy()
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[gdf.geometry.is_valid].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    print(f"[ED] {label}: {len(gdf)} features retained")
    return gdf


def ensure_crs(gdf, fallback="EPSG:4326"):
    if gdf.crs is None:
        gdf = gdf.set_crs(fallback)
    return gdf


def load_usgs_wateruse_csv(path):
    header_candidates = [0, 1, 2, 3, 4, 5, 6]
    expected_any = ["FIPS", "STATEFIPS", "COUNTYFIPS", "PS-WFrTo", "DO-WFrTo", "IN-WFrTo", "IR-WFrTo"]
    for h in header_candidates:
        try:
            df = pd.read_csv(path, header=h, dtype=str)
            df.columns = [str(c).strip() for c in df.columns]
            found = sum(col in df.columns for col in expected_any)
            if found >= 3:
                print(f"[USGS LOAD SUCCESS] header={h}")
                return df
        except Exception:
            pass
    raise ValueError("Could not identify the correct header row in the USGS file.")


def fdr_bh(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def mw_test(series_a, series_b):
    a = pd.to_numeric(series_a, errors="coerce").dropna()
    b = pd.to_numeric(series_b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, len(a), len(b)
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return stat, p, len(a), len(b)


def safe_frac(num, den):
    return np.where(pd.to_numeric(den, errors="coerce") > 0, pd.to_numeric(num, errors="coerce") / pd.to_numeric(den, errors="coerce"), np.nan)


def km_from_m(series):
    return pd.to_numeric(series, errors="coerce") / 1000.0


def add_threshold_flags(df, source_col, prefix, thresholds_km):
    for t in thresholds_km:
        df[f"{prefix}_within_{t}km"] = pd.to_numeric(df[source_col], errors="coerce") <= t
    return df


def identify_huc8_columns(huc8):
    huc8_id_col = pick_column(huc8, ["huc8", "huc_8", "huc8_id", "huc8code", "huc_code"], required=False)
    huc8_name_col = pick_column(huc8, ["name", "name_1", "hucname", "hu_name"], required=False)
    if huc8_id_col is None:
        for c in huc8.columns:
            if str(c).upper() in ["HUC8", "HUC_8"]:
                huc8_id_col = c
                break
    if huc8_name_col is None:
        for c in huc8.columns:
            if str(c).upper() in ["NAME", "HU_8_NAME", "HUCNAME"]:
                huc8_name_col = c
                break
    if huc8_id_col is None:
        raise ValueError(f"Could not identify HUC8 ID column. Columns: {list(huc8.columns)}")
    if huc8_name_col is None:
        huc8["huc8_name_tmp"] = huc8[huc8_id_col].astype(str)
        huc8_name_col = "huc8_name_tmp"
    return huc8_id_col, huc8_name_col


def choose_aquifer_name_col(aquifers):
    candidates = ["aq_name", "aqname", "aquifer", "aquifer_name", "name", "unit_name", "display_name", "principal_aq", "principal_aquifer", "fullname"]
    col = pick_column(aquifers, candidates, required=False)
    if col is not None:
        return col
    for c in aquifers.columns:
        if c != "geometry" and aquifers[c].dtype == "object":
            return c
    raise ValueError("Could not identify aquifer name column.")


def choose_river_name_col(gdf):
    col = pick_column(gdf, ["name", "gnis_name", "river_name", "stream_name"], required=False)
    if col is not None:
        return col
    for c in gdf.columns:
        if c != "geometry" and gdf[c].dtype == "object":
            return c
    return None


def choose_reservoir_name_col(gdf):
    col = pick_column(gdf, ["gnis_name", "name", "wb_name", "res_name", "feature_name", "waterbody_name", "lake_name", "HYLAK_NAME"], required=False)
    if col is not None:
        return col
    for c in gdf.columns:
        if c != "geometry" and gdf[c].dtype == "object":
            return c
    return None


def nearest_join_points_to_features(points_gdf, features_gdf, feature_cols, distance_col_m):
    use_cols = [c for c in feature_cols if c in features_gdf.columns] + ["geometry"]
    feat = features_gdf[use_cols].copy()
    joined = gpd.sjoin_nearest(points_gdf, feat, how="left", distance_col=distance_col_m)
    return joined.drop(columns=["index_right"], errors="ignore")


def dominant_sector_row(row, sector_share_cols):
    vals = row[sector_share_cols]
    if vals.isna().all():
        return np.nan
    return vals.idxmax().replace("_share", "")


def classify_supply_pathway(row):
    river_km = row.get("dist_to_major_river_km", np.nan)
    res_km = row.get("dist_to_reservoir_km", np.nan)
    ps_gw = row.get("county_PS_gw_fraction", np.nan)
    ps_sw = row.get("county_PS_sw_fraction", np.nan)
    near_river = pd.notna(river_km) and river_km <= 10
    near_res = pd.notna(res_km) and res_km <= 10
    gw_dom = pd.notna(ps_gw) and ps_gw >= 0.5
    sw_dom = pd.notna(ps_sw) and ps_sw >= 0.5
    if gw_dom and not (near_river or near_res):
        return "Groundwater leaning"
    elif sw_dom and (near_river or near_res):
        return "Surface-water leaning"
    elif gw_dom and (near_river or near_res):
        return "Mixed groundwater and nearby surface water"
    elif sw_dom and not (near_river or near_res):
        return "Surface-water county, no nearby mapped source"
    else:
        return "Mixed or uncertain"


def save_fig(fig, filename, dpi=350):
    outpath = os.path.join(FIGDIR, filename)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    print("[SAVED FIGURE]", outpath)
    return outpath


def save_caption(filename, caption):
    outpath = os.path.join(CAPTIONDIR, filename)
    with open(outpath, "w") as f:
        f.write(caption.strip() + "\n")
    print("[SAVED CAPTION]", outpath)

# -------------------------
# 4. FIND INPUT PATHS FROM RAW DATA
# -------------------------
dc_candidates = [
    os.path.join(ROOT_DRIVE, "DC_CONUS.csv"),
    os.path.join(PROJECT_DIR, "DC_CONUS.csv"),
    os.path.join(PROJECT_DIR, "Groundwater", "DC_CONUS.csv"),
]
dc_search = recursive_find(ROOT_DRIVE, ["DC_CONUS.csv", "*DC*CONUS*.csv"])
if dc_search:
    dc_candidates.insert(0, dc_search)
DC_PATH = find_first_existing(dc_candidates, "DC CSV")

usgs_candidates = [
    os.path.join(PROJECT_DIR, "usco2015v2.0.csv"),
    os.path.join(ROOT_DRIVE, "usco2015v2.0.csv"),
]
usgs_search = recursive_find(ROOT_DRIVE, ["usco2015v2.0.csv", "*usco2015*.csv", "*water*use*.csv"])
if usgs_search:
    usgs_candidates.insert(0, usgs_search)
USGS_PATH = find_first_existing(usgs_candidates, "USGS county water-use CSV")

huc8_candidates = [
    os.path.join(PROJECT_DIR, "WBD_National_GPKG", "WBD_National_GPKG.gpkg"),
    os.path.join(PROJECT_DIR, "WBD_National_GPKG.gpkg"),
    os.path.join(ROOT_DRIVE, "WBD_National_GPKG.gpkg"),
]
huc8_search = recursive_find(ROOT_DRIVE, ["WBD_National_GPKG.gpkg", "*WBD*.gpkg"])
if huc8_search:
    huc8_candidates.insert(0, huc8_search)
HUC8_GPKG_PATH = find_first_existing(huc8_candidates, "HUC8 geopackage")

county_candidates = [
    os.path.join(PROJECT_DIR, "counties", "tl_2019_us_county.shp"),
    os.path.join(PROJECT_DIR, "counties", "tl_2023_us_county.shp"),
    os.path.join(ROOT_DRIVE, "tl_2023_us_county.shp"),
]
county_search = recursive_find(ROOT_DRIVE, ["tl_*_us_county.shp", "cb_*_us_county_500k.shp", "*county*.shp"])
if county_search:
    county_candidates.insert(0, county_search)
COUNTY_PATH = find_first_existing(county_candidates, "County shapefile")

aquifer_candidates = [
    os.path.join(PROJECT_DIR, "Groundwater", "us_aquifers.shp"),
    os.path.join(PROJECT_DIR, "us_aquifers.shp"),
]
aquifer_search = recursive_find(ROOT_DRIVE, ["*aquifer*.shp", "*Aquifer*.shp", "*principal*aquifer*.shp"])
if aquifer_search:
    aquifer_candidates.insert(0, aquifer_search)
AQUIFER_PATH = find_first_existing(aquifer_candidates, "Aquifer shapefile")

river_candidates = [
    os.path.join(ROOT_DRIVE, "Copy of HydroRIVERS_v10_na.shp"),
    os.path.join(PROJECT_DIR, "HydroRIVERS_v10_na.shp"),
    os.path.join(PROJECT_DIR, "HydroRIVERS", "HydroRIVERS_v10_na.shp"),
]
river_search = recursive_find(ROOT_DRIVE, ["*HydroRIVERS*.shp", "*hydrorivers*.shp", "*river*.shp"])
if river_search:
    river_candidates.insert(0, river_search)
RIVER_PATH = find_first_existing(river_candidates, "River shapefile")

reservoir_candidates = [
    os.path.join(PROJECT_DIR, "Groundwater", "HydroLAKES_polys_v10_shp", "HydroLAKES_polys_v10.shp"),
    os.path.join(PROJECT_DIR, "Groundwater", "HydroLAKES_polys_v10.shp"),
    os.path.join(PROJECT_DIR, "HydroLAKES_polys_v10_shp", "HydroLAKES_polys_v10.shp"),
    os.path.join(PROJECT_DIR, "HydroLAKES_polys_v10.shp"),
    os.path.join(ROOT_DRIVE, "HydroLAKES_polys_v10_shp", "HydroLAKES_polys_v10.shp"),
]
reservoir_search = recursive_find(ROOT_DRIVE, ["HydroLAKES_polys_v10.shp", "*HydroLAKES*.shp", "*hydrolakes*.shp", "*lake*.shp", "*waterbod*.shp", "*reservoir*.shp"])
if reservoir_search:
    reservoir_candidates.insert(0, reservoir_search)
RESERVOIR_PATH = find_first_existing(reservoir_candidates, "Reservoir/Waterbody dataset", required=False)

# -------------------------
# 5. LOAD DATA
# -------------------------
print("\nLoading datasets...")
dc_raw = pd.read_csv(DC_PATH)
usgs_raw = load_usgs_wateruse_csv(USGS_PATH)
huc8_raw = safe_read_vector(HUC8_GPKG_PATH, "HUC8 watersheds", layer="WBDHU8")
counties_raw = safe_read_vector(COUNTY_PATH, "County boundaries")
aquifers_raw = safe_read_vector(AQUIFER_PATH, "Aquifers")
rivers_raw = safe_read_vector(RIVER_PATH, "Rivers")
reservoirs_raw = safe_read_vector(RESERVOIR_PATH, "Reservoir/Waterbody") if RESERVOIR_PATH else None

# -------------------------
# 6. PREP DATA CENTERS
# -------------------------
lat_col, lon_col = guess_lat_lon_columns(dc_raw)
dc_raw = dc_raw.copy()
dc_raw[lat_col] = pd.to_numeric(dc_raw[lat_col], errors="coerce")
dc_raw[lon_col] = pd.to_numeric(dc_raw[lon_col], errors="coerce")
dc_raw = dc_raw.dropna(subset=[lat_col, lon_col]).copy()

dc = gpd.GeoDataFrame(dc_raw, geometry=gpd.points_from_xy(dc_raw[lon_col], dc_raw[lat_col]), crs="EPSG:4326")
provider_col = pick_column(dc, ["provider", "company", "operator"], required=False)
facility_col = pick_column(dc, ["facility", "site", "name"], required=False)
address_col = pick_column(dc, ["address", "addr", "location"], required=False)
if provider_col is None:
    dc["provider"] = "Unknown"; provider_col = "provider"
if facility_col is None:
    dc["facility"] = "Unknown"; facility_col = "facility"
if address_col is None:
    dc["address"] = ""; address_col = "address"
dc["dc_id"] = np.arange(1, len(dc) + 1)
print(f"Valid data center points: {len(dc):,}")

# -------------------------
# 7. PREP HUC8, COUNTIES, AQUIFERS, RIVERS, RESERVOIRS
# -------------------------
huc8 = ensure_crs(clean_geometry(huc8_raw, "HUC8"))
huc8_id_col, huc8_name_col = identify_huc8_columns(huc8)
huc8 = huc8[[huc8_id_col, huc8_name_col, "geometry"]].copy().rename(columns={huc8_id_col: "HUC8", huc8_name_col: "HUC8_NAME"})
huc8["HUC8"] = huc8["HUC8"].astype(str).str.zfill(8)

counties = ensure_crs(clean_geometry(counties_raw, "Counties"))
statefp_col = pick_column(counties, ["STATEFP", "STATEFP20", "STATEFP10"], required=False)
countyfp_col = pick_column(counties, ["COUNTYFP", "COUNTYFP20", "COUNTYFP10"], required=False)
geoid_col = pick_column(counties, ["GEOID", "GEOID20", "GEOID10"], required=False)
name_col = pick_column(counties, ["NAME", "NAMELSAD"], required=False)
if geoid_col is None:
    counties["FIPS"] = counties[statefp_col].astype(str).str.zfill(2) + counties[countyfp_col].astype(str).str.zfill(3)
else:
    counties["FIPS"] = counties[geoid_col].astype(str).str.zfill(5)
counties["COUNTY_NAME"] = counties[name_col].astype(str) if name_col else counties["FIPS"]
counties = counties[["FIPS", "COUNTY_NAME", "geometry"]].copy()

aquifers = ensure_crs(clean_geometry(aquifers_raw, "Aquifers"))
aq_name_col = choose_aquifer_name_col(aquifers)
aquifers = aquifers[[aq_name_col, "geometry"]].copy().rename(columns={aq_name_col: "AQUIFER_NAME"})
aquifers["AQUIFER_NAME"] = aquifers["AQUIFER_NAME"].astype(str).str.strip()

rivers = ensure_crs(clean_geometry(rivers_raw, "Rivers"))
ord_col = pick_column(rivers, ["ORD_STRA", "ord_stra", "strahler", "streamorde", "stream_order"], required=False)
river_name_col = choose_river_name_col(rivers)
if ord_col is not None:
    rivers["ORD_STRA_TMP"] = pd.to_numeric(rivers[ord_col], errors="coerce")
    major_rivers = rivers[rivers["ORD_STRA_TMP"] >= MAJOR_RIVER_ORD_STRA].copy()
else:
    rivers["ORD_STRA_TMP"] = np.nan
    major_rivers = rivers.copy()
major_rivers["RIVER_NAME"] = major_rivers[river_name_col].astype(str) if river_name_col else "Unnamed river"
major_rivers = major_rivers[["RIVER_NAME", "ORD_STRA_TMP", "geometry"]].copy()

reservoirs = None
if reservoirs_raw is not None:
    reservoirs = ensure_crs(clean_geometry(reservoirs_raw, "Reservoirs/Waterbodies"))
    area_col = pick_column(reservoirs, ["Lake_area", "LAKE_AREA", "lake_area"], required=False)
    if USE_HYDROLAKES_MIN_AREA_FILTER and area_col is not None:
        reservoirs[area_col] = pd.to_numeric(reservoirs[area_col], errors="coerce")
        reservoirs = reservoirs[reservoirs[area_col] >= HYDROLAKES_MIN_AREA_KM2].copy()
    res_name_col = choose_reservoir_name_col(reservoirs)
    reservoirs["RESERVOIR_NAME"] = reservoirs[res_name_col].astype(str) if res_name_col else "Unnamed reservoir"
    reservoirs = reservoirs[["RESERVOIR_NAME", "geometry"]].copy()

# -------------------------
# 8. PREP USGS WATER USE
# -------------------------
usgs = usgs_raw.copy()
if "FIPS" not in usgs.columns:
    state_col = pick_column(usgs, ["STATEFIPS"], required=True)
    county_col = pick_column(usgs, ["COUNTYFIPS"], required=True)
    usgs["FIPS"] = usgs[state_col].astype(str).str.zfill(2) + usgs[county_col].astype(str).str.zfill(3)
else:
    usgs["FIPS"] = usgs["FIPS"].astype(str).str.zfill(5)
if "YEAR" in usgs.columns:
    usgs["YEAR"] = pd.to_numeric(usgs["YEAR"], errors="coerce")
    usgs = usgs[usgs["YEAR"] == 2015].copy()

sector_map = {"PS_WFrTo": "PS-WFrTo", "DO_WFrTo": "DO-WFrTo", "IN_WFrTo": "IN-WFrTo", "IR_WFrTo": "IR-WFrTo"}
optional_cols = {
    "PS_WGWFr": "PS-WGWFr", "PS_WSWFr": "PS-WSWFr",
    "DO_WGWFr": "DO-WGWFr", "DO_WSWFr": "DO-WSWFr",
    "IN_WGWFr": "IN-WGWFr", "IN_WSWFr": "IN-WSWFr",
    "IR_WGWFr": "IR-WGWFr", "IR_WSWFr": "IR-WSWFr",
    "PS_TOPop": "PS-TOPop", "PS_GWPop": "PS-GWPop", "PS_SWPop": "PS-SWPop",
}
missing = [v for v in sector_map.values() if v not in usgs.columns]
if missing:
    raise ValueError(f"Missing required sector columns in USGS file: {missing}")
needed_cols = ["FIPS"] + list(sector_map.values()) + [v for v in optional_cols.values() if v in usgs.columns]
usgs = usgs[needed_cols].copy()
for c in usgs.columns:
    if c != "FIPS":
        usgs[c] = pd.to_numeric(usgs[c], errors="coerce").fillna(0)
rename_dict = {v: k for k, v in sector_map.items()}
rename_dict.update({v: k for k, v in optional_cols.items() if v in usgs.columns})
usgs = usgs.rename(columns=rename_dict)

# -------------------------
# 9. JOIN AND PROJECT DATA
# -------------------------
counties_usgs = counties.merge(usgs, on="FIPS", how="left")
water_cols = [c for c in counties_usgs.columns if c not in ["FIPS", "COUNTY_NAME", "geometry"]]
for c in water_cols:
    counties_usgs[c] = pd.to_numeric(counties_usgs[c], errors="coerce").fillna(0)

dc_p = dc.to_crs(TARGET_CRS)
huc8_p = huc8.to_crs(TARGET_CRS)
counties_p = counties_usgs.to_crs(TARGET_CRS)
aquifers_p = aquifers.to_crs(TARGET_CRS)
major_rivers_p = major_rivers.to_crs(TARGET_CRS)
reservoirs_p = reservoirs.to_crs(TARGET_CRS) if reservoirs is not None else None

# -------------------------
# 10. SPATIAL ASSIGNMENTS
# -------------------------
print("\nAssigning data centers to HUC8, counties, and aquifers...")
dc_huc8 = gpd.sjoin(dc_p, huc8_p[["HUC8", "HUC8_NAME", "geometry"]], how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
dc_county = gpd.sjoin(dc_p, counties_p[["FIPS", "COUNTY_NAME", "geometry"]], how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
dc_aquifer = gpd.sjoin(dc_p, aquifers_p[["AQUIFER_NAME", "geometry"]], how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
dc_aquifer_one = dc_aquifer.sort_values(["dc_id", "AQUIFER_NAME"]).drop_duplicates(subset="dc_id", keep="first").copy()

dc_unique = dc_p.copy()
dc_unique = dc_unique.join(dc_huc8[["HUC8", "HUC8_NAME"]], how="left")
dc_unique = dc_unique.join(dc_county[["FIPS", "COUNTY_NAME"]], how="left")
dc_unique = dc_unique.join(dc_aquifer_one[["AQUIFER_NAME"]], how="left")
dc_unique["provider"] = dc_unique[provider_col].astype(str)
dc_unique["facility"] = dc_unique[facility_col].astype(str)
dc_unique["address"] = dc_unique[address_col].astype(str)
dc_unique = dc_unique.sort_values("dc_id").drop_duplicates(subset="dc_id", keep="first").copy()
print("Unique data-center rows:", len(dc_unique))

# -------------------------
# 11. NEAREST MAJOR RIVER AND RESERVOIR/LAKE
# -------------------------
print("\nComputing nearest major-river distances...")
river_nearest = nearest_join_points_to_features(dc_unique[["dc_id", "geometry"]].copy(), major_rivers_p, ["RIVER_NAME", "ORD_STRA_TMP"], "dist_to_major_river_m")
river_nearest = river_nearest.rename(columns={"RIVER_NAME": "NEAREST_MAJOR_RIVER", "ORD_STRA_TMP": "NEAREST_RIVER_ORD_STRA"})
dc_unique = dc_unique.merge(river_nearest.drop(columns="geometry"), on="dc_id", how="left")
dc_unique["dist_to_major_river_km"] = km_from_m(dc_unique["dist_to_major_river_m"])
dc_unique = add_threshold_flags(dc_unique, "dist_to_major_river_km", "river", DIST_BINS_KM)

if reservoirs_p is not None and len(reservoirs_p) > 0:
    print("\nComputing nearest reservoir/lake distances...")
    reservoir_nearest = nearest_join_points_to_features(dc_unique[["dc_id", "geometry"]].copy(), reservoirs_p, ["RESERVOIR_NAME"], "dist_to_reservoir_m")
    reservoir_nearest = reservoir_nearest.rename(columns={"RESERVOIR_NAME": "NEAREST_RESERVOIR"})
    dc_unique = dc_unique.merge(reservoir_nearest.drop(columns="geometry"), on="dc_id", how="left")
    dc_unique["dist_to_reservoir_km"] = km_from_m(dc_unique["dist_to_reservoir_m"])
    dc_unique = add_threshold_flags(dc_unique, "dist_to_reservoir_km", "reservoir", DIST_BINS_KM)
else:
    dc_unique["NEAREST_RESERVOIR"] = np.nan
    dc_unique["dist_to_reservoir_m"] = np.nan
    dc_unique["dist_to_reservoir_km"] = np.nan
    for t in DIST_BINS_KM:
        dc_unique[f"reservoir_within_{t}km"] = np.nan

# -------------------------
# 12. COUNTY WATER SUPPLY ATTRIBUTES TO DC TABLE
# -------------------------
county_supply_cols = ["FIPS"]
for c in ["PS_WFrTo", "DO_WFrTo", "IN_WFrTo", "IR_WFrTo", "PS_WGWFr", "PS_WSWFr", "DO_WGWFr", "DO_WSWFr", "IN_WGWFr", "IN_WSWFr", "IR_WGWFr", "IR_WSWFr", "PS_TOPop", "PS_GWPop", "PS_SWPop"]:
    if c in counties_usgs.columns:
        county_supply_cols.append(c)
county_supply = counties_usgs[county_supply_cols].drop_duplicates().copy()
dc_unique = dc_unique.merge(county_supply, on="FIPS", how="left")

dc_unique["county_selected_total_MGD"] = dc_unique.get("PS_WFrTo", 0).fillna(0) + dc_unique.get("DO_WFrTo", 0).fillna(0) + dc_unique.get("IN_WFrTo", 0).fillna(0) + dc_unique.get("IR_WFrTo", 0).fillna(0)
for sector in ["PS", "DO", "IN", "IR"]:
    if f"{sector}_WFrTo" in dc_unique.columns:
        dc_unique[f"county_{sector}_share"] = safe_frac(dc_unique[f"{sector}_WFrTo"], dc_unique["county_selected_total_MGD"])
    if f"{sector}_WGWFr" in dc_unique.columns and f"{sector}_WFrTo" in dc_unique.columns:
        dc_unique[f"county_{sector}_gw_fraction"] = safe_frac(dc_unique[f"{sector}_WGWFr"], dc_unique[f"{sector}_WFrTo"])
    if f"{sector}_WSWFr" in dc_unique.columns and f"{sector}_WFrTo" in dc_unique.columns:
        dc_unique[f"county_{sector}_sw_fraction"] = safe_frac(dc_unique[f"{sector}_WSWFr"], dc_unique[f"{sector}_WFrTo"])

dc_unique["combined_supply_pathway_proxy"] = dc_unique.apply(classify_supply_pathway, axis=1)

# -------------------------
# 13. COUNTY -> HUC8 AREA-WEIGHTED WATER USE
# -------------------------
print("\nRunning county-to-HUC8 area-weighted overlay...")
counties_p["county_area_m2"] = counties_p.geometry.area
overlay_county_huc8 = gpd.overlay(counties_p, huc8_p[["HUC8", "HUC8_NAME", "geometry"]], how="intersection")
overlay_county_huc8["intersect_area_m2"] = overlay_county_huc8.geometry.area
overlay_county_huc8["area_weight"] = overlay_county_huc8["intersect_area_m2"] / overlay_county_huc8["county_area_m2"]
weighted_cols = []
for c in water_cols:
    newc = f"{c}_aw"
    overlay_county_huc8[newc] = overlay_county_huc8[c] * overlay_county_huc8["area_weight"]
    weighted_cols.append(newc)
huc8_water = overlay_county_huc8.groupby(["HUC8", "HUC8_NAME"])[weighted_cols].sum().reset_index()
huc8_water = huc8_water.rename(columns={f"{c}_aw": c for c in water_cols})

# -------------------------
# 14. HUC8 SUMMARY
# -------------------------
dc_count_by_huc8 = dc_unique.groupby(["HUC8", "HUC8_NAME"]).size().reset_index(name="dc_count")
huc8_summary = huc8_p[["HUC8", "HUC8_NAME", "geometry"]].drop_duplicates().merge(huc8_water, on=["HUC8", "HUC8_NAME"], how="left").merge(dc_count_by_huc8, on=["HUC8", "HUC8_NAME"], how="left")
for c in water_cols:
    if c in huc8_summary.columns:
        huc8_summary[c] = huc8_summary[c].fillna(0)
huc8_summary["dc_count"] = huc8_summary["dc_count"].fillna(0).astype(int)
huc8_summary["has_dc"] = huc8_summary["dc_count"] > 0
huc8_summary["selected_total_MGD"] = huc8_summary.get("PS_WFrTo", 0) + huc8_summary.get("DO_WFrTo", 0) + huc8_summary.get("IN_WFrTo", 0) + huc8_summary.get("IR_WFrTo", 0)
for sector in ["PS", "DO", "IN", "IR"]:
    huc8_summary[f"{sector}_share"] = safe_frac(huc8_summary.get(f"{sector}_WFrTo", 0), huc8_summary["selected_total_MGD"])
    if f"{sector}_WGWFr" in huc8_summary.columns and f"{sector}_WFrTo" in huc8_summary.columns:
        huc8_summary[f"{sector}_gw_fraction"] = safe_frac(huc8_summary[f"{sector}_WGWFr"], huc8_summary[f"{sector}_WFrTo"])
    if f"{sector}_WSWFr" in huc8_summary.columns and f"{sector}_WFrTo" in huc8_summary.columns:
        huc8_summary[f"{sector}_sw_fraction"] = safe_frac(huc8_summary[f"{sector}_WSWFr"], huc8_summary[f"{sector}_WFrTo"])
share_cols = ["PS_share", "DO_share", "IN_share", "IR_share"]
huc8_summary["dominant_sector"] = huc8_summary.apply(lambda row: dominant_sector_row(row, share_cols), axis=1)

huc8_proximity = dc_unique.groupby(["HUC8", "HUC8_NAME"], as_index=False).agg(
    mean_dist_to_major_river_km=("dist_to_major_river_km", "mean"),
    median_dist_to_major_river_km=("dist_to_major_river_km", "median"),
    min_dist_to_major_river_km=("dist_to_major_river_km", "min"),
    mean_dist_to_reservoir_km=("dist_to_reservoir_km", "mean"),
    median_dist_to_reservoir_km=("dist_to_reservoir_km", "median"),
    min_dist_to_reservoir_km=("dist_to_reservoir_km", "min"),
    pct_dc_within_10km_river=("river_within_10km", lambda s: np.nanmean(pd.to_numeric(s, errors="coerce"))),
    pct_dc_within_10km_reservoir=("reservoir_within_10km", lambda s: np.nanmean(pd.to_numeric(s, errors="coerce"))),
)
for c in ["pct_dc_within_10km_river", "pct_dc_within_10km_reservoir"]:
    huc8_proximity[c] = huc8_proximity[c] * 100.0
huc8_summary = huc8_summary.merge(huc8_proximity, on=["HUC8", "HUC8_NAME"], how="left")

# -------------------------
# 15. DOMINANT AQUIFER PER HUC8 AND AQUIFER ENRICHMENT
# -------------------------
print("\nCalculating dominant aquifer and aquifer enrichment...")
overlay_aq_huc8 = gpd.overlay(aquifers_p[["AQUIFER_NAME", "geometry"]], huc8_p[["HUC8", "HUC8_NAME", "geometry"]], how="intersection")
overlay_aq_huc8["aq_area_m2"] = overlay_aq_huc8.geometry.area
huc8_aq_area = overlay_aq_huc8.groupby(["HUC8", "HUC8_NAME", "AQUIFER_NAME"], as_index=False)["aq_area_m2"].sum()
dominant_aquifer_per_huc8 = huc8_aq_area.sort_values(["HUC8", "aq_area_m2"], ascending=[True, False]).groupby("HUC8", as_index=False).first().rename(columns={"AQUIFER_NAME": "DOM_AQUIFER", "aq_area_m2": "DOM_AQUIFER_AREA_M2"})
huc8_summary = huc8_summary.merge(dominant_aquifer_per_huc8[["HUC8", "DOM_AQUIFER", "DOM_AQUIFER_AREA_M2"]], on="HUC8", how="left")

aquifers_p["aq_area_m2"] = aquifers_p.geometry.area
total_aquifer_area = aquifers_p["aq_area_m2"].sum()
aq_area = aquifers_p.groupby("AQUIFER_NAME", as_index=False)["aq_area_m2"].sum()
aq_area["area_fraction"] = aq_area["aq_area_m2"] / total_aquifer_area
dc_aq_counts = dc_unique.groupby("AQUIFER_NAME", as_index=False).size().rename(columns={"size": "observed_dc_count"})
dc_total_with_aquifer = dc_aq_counts["observed_dc_count"].sum()
aq_enrichment = aq_area.merge(dc_aq_counts, on="AQUIFER_NAME", how="left")
aq_enrichment["observed_dc_count"] = aq_enrichment["observed_dc_count"].fillna(0).astype(int)
aq_enrichment["observed_dc_fraction"] = np.where(dc_total_with_aquifer > 0, aq_enrichment["observed_dc_count"] / dc_total_with_aquifer, np.nan)
aq_enrichment["expected_dc_count"] = aq_enrichment["area_fraction"] * dc_total_with_aquifer
aq_enrichment["enrichment_ratio"] = np.where(aq_enrichment["expected_dc_count"] > 0, aq_enrichment["observed_dc_count"] / aq_enrichment["expected_dc_count"], np.nan)
pvals = []
for _, row in aq_enrichment.iterrows():
    n = int(dc_total_with_aquifer)
    k = int(row["observed_dc_count"])
    p0 = float(row["area_fraction"])
    pvals.append(binomtest(k, n, p=p0, alternative="two-sided").pvalue if n > 0 and 0 < p0 < 1 else np.nan)
aq_enrichment["p_value"] = pvals
aq_enrichment["p_fdr"] = fdr_bh(pd.Series(aq_enrichment["p_value"]).fillna(1.0).values)
aq_enrichment["significant_fdr_0.05"] = aq_enrichment["p_fdr"] < 0.05
aq_enrichment = aq_enrichment.sort_values("enrichment_ratio", ascending=False)
aq_enrichment_nonzero = aq_enrichment[aq_enrichment["observed_dc_count"] > 0].copy()

# -------------------------
# 16. COUNTY SUPPLIER TABLES AND STATS
# -------------------------
dc_count_by_county = dc_unique.groupby(["FIPS", "COUNTY_NAME"], as_index=False).size().rename(columns={"size": "dc_count"})
county_supplier_table = counties_usgs.drop(columns="geometry").merge(dc_count_by_county, on=["FIPS", "COUNTY_NAME"], how="left")
county_supplier_table["dc_count"] = county_supplier_table["dc_count"].fillna(0).astype(int)
county_supplier_table["has_dc"] = county_supplier_table["dc_count"] > 0
county_supplier_table["selected_total_MGD"] = county_supplier_table.get("PS_WFrTo", 0) + county_supplier_table.get("DO_WFrTo", 0) + county_supplier_table.get("IN_WFrTo", 0) + county_supplier_table.get("IR_WFrTo", 0)
for sector in ["PS", "DO", "IN", "IR"]:
    county_supplier_table[f"{sector}_share"] = safe_frac(county_supplier_table.get(f"{sector}_WFrTo", 0), county_supplier_table["selected_total_MGD"])
    if f"{sector}_WGWFr" in county_supplier_table.columns:
        county_supplier_table[f"{sector}_gw_fraction"] = safe_frac(county_supplier_table[f"{sector}_WGWFr"], county_supplier_table[f"{sector}_WFrTo"])
    if f"{sector}_WSWFr" in county_supplier_table.columns:
        county_supplier_table[f"{sector}_sw_fraction"] = safe_frac(county_supplier_table[f"{sector}_WSWFr"], county_supplier_table[f"{sector}_WFrTo"])
top_county_suppliers = county_supplier_table[county_supplier_table["has_dc"]].sort_values(["dc_count", "PS_WFrTo", "selected_total_MGD"], ascending=[False, False, False])

master_huc8_table = huc8_summary.drop(columns="geometry").copy()
dc_huc8_aquifer_counts = dc_unique.groupby(["HUC8", "HUC8_NAME", "AQUIFER_NAME"], as_index=False).size().rename(columns={"size": "dc_in_aquifer_count"})
dominant_dc_aquifer_per_huc8 = dc_huc8_aquifer_counts.sort_values(["HUC8", "dc_in_aquifer_count"], ascending=[True, False]).groupby("HUC8", as_index=False).first().rename(columns={"AQUIFER_NAME": "TOP_DC_AQUIFER", "dc_in_aquifer_count": "TOP_DC_AQUIFER_COUNT"})
master_huc8_table = master_huc8_table.merge(dominant_dc_aquifer_per_huc8[["HUC8", "TOP_DC_AQUIFER", "TOP_DC_AQUIFER_COUNT"]], on="HUC8", how="left")
top_dc_huc8 = master_huc8_table[master_huc8_table["has_dc"]].sort_values(["dc_count", "selected_total_MGD"], ascending=[False, False]).head(25)

dc_group = master_huc8_table[master_huc8_table["has_dc"]].copy()
non_dc_group = master_huc8_table[~master_huc8_table["has_dc"]].copy()
stats_vars = ["selected_total_MGD", "PS_share", "DO_share", "IN_share", "IR_share"]
for c in ["PS_gw_fraction", "PS_sw_fraction", "IN_gw_fraction", "IN_sw_fraction", "DO_gw_fraction", "DO_sw_fraction", "IR_gw_fraction", "IR_sw_fraction"]:
    if c in master_huc8_table.columns:
        stats_vars.append(c)
stats_results = []
for var in stats_vars:
    stat, p, n1, n2 = mw_test(dc_group[var], non_dc_group[var])
    stats_results.append({
        "variable": var,
        "dc_median": pd.to_numeric(dc_group[var], errors="coerce").median(),
        "non_dc_median": pd.to_numeric(non_dc_group[var], errors="coerce").median(),
        "dc_mean": pd.to_numeric(dc_group[var], errors="coerce").mean(),
        "non_dc_mean": pd.to_numeric(non_dc_group[var], errors="coerce").mean(),
        "mannwhitney_u": stat,
        "p_value": p,
        "n_dc": n1,
        "n_non_dc": n2,
    })
stats_results = pd.DataFrame(stats_results)
stats_results["p_fdr"] = fdr_bh(stats_results["p_value"].fillna(1.0).values)
stats_results["significant_fdr_0.05"] = stats_results["p_fdr"] < 0.05

# -------------------------
# 17. SUMMARY TABLES
# -------------------------
dc_proximity_summary = pd.DataFrame({
    "metric": [
        "mean_dist_to_major_river_km", "median_dist_to_major_river_km",
        "mean_dist_to_reservoir_km", "median_dist_to_reservoir_km",
        "pct_within_5km_river", "pct_within_10km_river", "pct_within_25km_river", "pct_within_50km_river",
        "pct_within_5km_reservoir", "pct_within_10km_reservoir", "pct_within_25km_reservoir", "pct_within_50km_reservoir",
    ],
    "value": [
        pd.to_numeric(dc_unique["dist_to_major_river_km"], errors="coerce").mean(),
        pd.to_numeric(dc_unique["dist_to_major_river_km"], errors="coerce").median(),
        pd.to_numeric(dc_unique["dist_to_reservoir_km"], errors="coerce").mean(),
        pd.to_numeric(dc_unique["dist_to_reservoir_km"], errors="coerce").median(),
        pd.to_numeric(dc_unique.get("river_within_5km", np.nan), errors="coerce").mean() * 100,
        pd.to_numeric(dc_unique.get("river_within_10km", np.nan), errors="coerce").mean() * 100,
        pd.to_numeric(dc_unique.get("river_within_25km", np.nan), errors="coerce").mean() * 100,
        pd.to_numeric(dc_unique.get("river_within_50km", np.nan), errors="coerce").mean() * 100,
        pd.to_numeric(dc_unique.get("reservoir_within_5km", np.nan), errors="coerce").mean() * 100,
        pd.to_numeric(dc_unique.get("reservoir_within_10km", np.nan), errors="coerce").mean() * 100,
        pd.to_numeric(dc_unique.get("reservoir_within_25km", np.nan), errors="coerce").mean() * 100,
        pd.to_numeric(dc_unique.get("reservoir_within_50km", np.nan), errors="coerce").mean() * 100,
    ]
})
pathway_counts = dc_unique["combined_supply_pathway_proxy"].value_counts(dropna=False).rename_axis("combined_supply_pathway_proxy").reset_index(name="dc_count")

# -------------------------
# 18. SAVE NEW TABLES
# -------------------------
master_csv = os.path.join(TABLEDIR, "master_huc8_integrated_table.csv")
top_huc8_csv = os.path.join(TABLEDIR, "top_dc_huc8_integrated_table.csv")
county_csv = os.path.join(TABLEDIR, "county_supplier_proxy_table.csv")
top_county_csv = os.path.join(TABLEDIR, "top_dc_counties_supplier_proxy.csv")
aq_csv = os.path.join(TABLEDIR, "aquifer_enrichment_table.csv")
stats_csv = os.path.join(TABLEDIR, "dc_vs_non_dc_huc8_stats.csv")
dc_assign_csv = os.path.join(TABLEDIR, "data_centers_integrated_water_pathways.csv")
dc_summary_csv = os.path.join(TABLEDIR, "dc_proximity_summary.csv")
pathway_csv = os.path.join(TABLEDIR, "combined_supply_pathway_counts.csv")

master_huc8_table.to_csv(master_csv, index=False)
top_dc_huc8.to_csv(top_huc8_csv, index=False)
county_supplier_table.to_csv(county_csv, index=False)
top_county_suppliers.to_csv(top_county_csv, index=False)
aq_enrichment_nonzero.to_csv(aq_csv, index=False)
stats_results.to_csv(stats_csv, index=False)
dc_proximity_summary.to_csv(dc_summary_csv, index=False)
# ---------------------------------------------------------------------
# FINAL SAFETY DEDUPLICATION
# Ensure one output row per unique AI data-center site before saving
# and before computing pathway counts/figures.
# ---------------------------------------------------------------------
if "coord_key" in dc_unique.columns:
    before = len(dc_unique)
    dc_unique = dc_unique.drop_duplicates(subset=["coord_key"]).copy()
    after = len(dc_unique)
    print(f"[DEDUP] dc_unique by coord_key: {before:,} -> {after:,}")
else:
    before = len(dc_unique)
    dc_unique = dc_unique.drop_duplicates(subset=["Latitude", "Longitude"]).copy()
    after = len(dc_unique)
    print(f"[DEDUP] dc_unique by Latitude/Longitude: {before:,} -> {after:,}")

pathway_counts = (
    dc_unique["combined_supply_pathway_proxy"]
    .fillna("Mixed or uncertain")
    .value_counts()
    .rename_axis("combined_supply_pathway_proxy")
    .reset_index(name="dc_count")
)

pathway_counts.to_csv(pathway_csv, index=False)
dc_unique.to_crs("EPSG:4326").drop(columns="geometry").to_csv(dc_assign_csv, index=False)

print("\n[SAVED NEW TABLES]")
for p in [master_csv, top_huc8_csv, county_csv, top_county_csv, aq_csv, stats_csv, dc_assign_csv, dc_summary_csv, pathway_csv]:
    print(p)

# -------------------------
# 19.  PAPER FIGURES ONLY: 4 GroundwaterDC figures
# -------------------------
plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 400,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def wrap_labels(labels, width=32):
    return ["\n".join(textwrap.wrap(str(x), width=width)) for x in labels]

# Figure 1: Hydrologic pathway classification
fig1_data = pathway_counts.copy().sort_values("dc_count", ascending=True)
fig1_csv = os.path.join(TABLEDIR, "GroundwaterDC_pathway_classification_counts.csv")
fig1_data.to_csv(fig1_csv, index=False)

fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.barh(fig1_data["combined_supply_pathway_proxy"], fig1_data["dc_count"])
ax.set_xlabel("Count of AI data centers")
ax.set_ylabel("Hydrologic pathway classification")
ax.grid(axis="x", alpha=0.20)
fig.tight_layout()
save_fig(fig, "GroundwaterDC_01_pathway_classification.png", dpi=400)
plt.close(fig)

# Reservoir/lake distance, aquifer-enrichment, and HUC8 sector-composition figures were archived and removed from the active workflow.

# -------------------------
# 20. WRITE SUMMARY REPORT
# -------------------------
summary_txt = os.path.join(OUTDIR, "summary_report.txt")
with open(summary_txt, "w") as f:
    f.write("INTEGRATED DATA CENTER WATER SUPPLY PATHWAYS ANALYSIS\n")
    f.write("GROUNDWATER-DC ANALYSIS: FIGURES AND TABLES\n")
    f.write("============================================================\n\n")
    f.write(f"Total data centers: {dc_unique['dc_id'].nunique()}\n")
    f.write(f"HUC8s with data centers: {master_huc8_table['has_dc'].sum()}\n")
    f.write(f"HUC8s without data centers: {(~master_huc8_table['has_dc']).sum()}\n")
    f.write(f"Counties with data centers: {top_county_suppliers['FIPS'].nunique()}\n")
    f.write(f"Aquifers containing data centers: {aq_enrichment_nonzero['AQUIFER_NAME'].nunique()}\n\n")
    f.write("COMBINED SUPPLY PATHWAY COUNTS\n------------------------------\n")
    f.write(pathway_counts.to_string(index=False))
    f.write("\n\nTOP AQUIFERS BY ENRICHMENT\n--------------------------\n")
    f.write(aq_enrichment_nonzero.head(15).to_string(index=False))
    f.write("\n")

print("\nDONE. NEW outputs created from raw inputs.")
print("Main folder:", OUTDIR)
print("Tables folder:", TABLEDIR)
print("Figures folder:", FIGDIR)
print("Captions folder:", CAPTIONDIR)
print("Summary report:", summary_txt)

