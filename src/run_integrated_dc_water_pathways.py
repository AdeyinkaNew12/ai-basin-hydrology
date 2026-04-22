#!/usr/bin/env python3
"""
Integrated Data Center Water Supply Pathways Analysis

This script links U.S. data center point locations to HUC8 watersheds,
counties, aquifers, major rivers, and HydroLAKES waterbodies, then combines
those spatial relationships with county-level USGS water-use attributes to
produce integrated tables, summary statistics, figures, and a plain-text
summary report.

Inputs are discovered from a user-supplied data root, with defaults aligned to
common server layouts.
"""

from __future__ import annotations

import argparse
import glob
import os
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

warnings.filterwarnings("ignore")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu


TARGET_CRS = "EPSG:5070"
DEFAULT_DATA_ROOT = "/mnt/disk3/aoolaseinde/data/Groundwater"
DEFAULT_OUTPUT_ROOT = "/mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results"
MAJOR_RIVER_ORD_STRA = 5
DIST_BINS_KM = [5, 10, 25, 50]
USE_HYDROLAKES_MIN_AREA_FILTER = False
HYDROLAKES_MIN_AREA_KM2 = 1.0


@dataclass
class ResolvedPaths:
    dc: str
    usgs: str
    huc8: str
    county: str
    aquifer: str
    river: str
    reservoir: Optional[str]


# ============================================================
# HELPERS
# ============================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def find_all_matches(root: str, patterns: Sequence[str]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    seen = set()
    ordered = []
    for match in matches:
        if match not in seen:
            seen.add(match)
            ordered.append(match)
    return ordered


def choose_best_match(matches: Sequence[str], preferred_substrings: Optional[Sequence[str]] = None) -> Optional[str]:
    if not matches:
        return None
    if not preferred_substrings:
        return matches[0]

    ranked = []
    for match in matches:
        score = 0
        lower = match.lower()
        for token in preferred_substrings:
            if token.lower() in lower:
                score += 1
        ranked.append((score, match))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1]


def find_file_smart(
    root: str,
    exact_candidates: Optional[Sequence[str]] = None,
    search_patterns: Optional[Sequence[str]] = None,
    label: str = "file",
    required: bool = True,
    preferred_substrings: Optional[Sequence[str]] = None,
) -> Optional[str]:
    exact_candidates = exact_candidates or []
    search_patterns = search_patterns or []

    for candidate in exact_candidates:
        if candidate and os.path.exists(candidate):
            print(f"[FOUND] {label}: {candidate}")
            return candidate

    matches = find_all_matches(root, search_patterns)
    if matches:
        best = choose_best_match(matches, preferred_substrings=preferred_substrings)
        print(f"[FOUND by search] {label}: {best}")
        return best

    if required:
        checked = "\n".join(list(exact_candidates) + list(search_patterns))
        raise FileNotFoundError(f"Could not find {label}.\nChecked/search patterns:\n{checked}")

    print(f"[WARNING] Could not find {label}.")
    return None


def pick_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def guess_lat_lon_columns(df: pd.DataFrame) -> tuple[str, str]:
    lower_map = {str(col).lower(): col for col in df.columns}
    lat_candidates = ["latitude", "lat", "y", "dec_lat", "dec_lat_va"]
    lon_candidates = ["longitude", "lon", "long", "x", "dec_long", "dec_long_va"]

    lat_col = next((lower_map[c] for c in lat_candidates if c in lower_map), None)
    lon_col = next((lower_map[c] for c in lon_candidates if c in lower_map), None)

    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not identify latitude/longitude columns. Available columns: {list(df.columns)}")

    return lat_col, lon_col


def safe_read_vector(path: str, label: str = "vector", layer: Optional[str] = None) -> gpd.GeoDataFrame:
    try:
        if layer is not None:
            gdf = gpd.read_file(path, layer=layer)
            print(f"[LOADED] {label}: {path} | layer={layer}")
        else:
            gdf = gpd.read_file(path)
            print(f"[LOADED] {label}: {path}")
        return gdf
    except Exception as exc:
        raise RuntimeError(f"Could not read {label}: {path}\n{exc}") from exc


def ensure_crs(gdf: gpd.GeoDataFrame, fallback: str = "EPSG:4326") -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(fallback)
    return gdf


def clean_geometry(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out = out[~out.geometry.isna()].copy()
    out = out[out.geometry.is_valid].copy()
    out = out[~out.geometry.is_empty].copy()
    print(f"[CLEANED] {label}: {len(out)} features retained")
    return out


def load_huc8_dataset(path: str) -> tuple[gpd.GeoDataFrame, str]:
    path_lower = path.lower()
    if path_lower.endswith(".gpkg"):
        return safe_read_vector(path, label="HUC8 watersheds", layer="WBDHU8"), "WBD"
    if path_lower.endswith(".shp") and "hybas" in path_lower and "lev08" in path_lower:
        return safe_read_vector(path, label="HydroBASINS Level 8"), "HYBAS"
    raise ValueError(f"Unsupported HUC8 dataset path: {path}")


def standardize_huc8(huc8_raw: gpd.GeoDataFrame, source_type: str) -> gpd.GeoDataFrame:
    huc8 = huc8_raw.copy()

    if source_type == "WBD":
        huc8_id_col = pick_column(huc8, ["huc8", "huc_8"], required=False)
        huc8_name_col = pick_column(huc8, ["name", "hucname", "hu_name"], required=False)

        if huc8_id_col is None:
            for col in huc8.columns:
                if str(col).upper() in {"HUC8", "HUC_8"}:
                    huc8_id_col = col
                    break
        if huc8_name_col is None:
            for col in huc8.columns:
                if str(col).upper() in {"NAME", "HUCNAME", "HU_8_NAME"}:
                    huc8_name_col = col
                    break

        if huc8_id_col is None:
            raise ValueError(f"Could not identify WBD HUC8 column. Columns: {list(huc8.columns)}")
        if huc8_name_col is None:
            huc8["HUC8_NAME_TMP"] = huc8[huc8_id_col].astype(str)
            huc8_name_col = "HUC8_NAME_TMP"

        huc8 = huc8[[huc8_id_col, huc8_name_col, "geometry"]].copy()
        huc8 = huc8.rename(columns={huc8_id_col: "HUC8", huc8_name_col: "HUC8_NAME"})
        huc8["HUC8"] = huc8["HUC8"].astype(str).str.zfill(8)
        huc8["HUC8_NAME"] = huc8["HUC8_NAME"].astype(str)
        return huc8

    if source_type == "HYBAS":
        hybas_id_col = pick_column(huc8, ["hybas_id"], required=False)
        if hybas_id_col is None:
            for col in huc8.columns:
                if str(col).upper() == "HYBAS_ID":
                    hybas_id_col = col
                    break
        if hybas_id_col is None:
            raise ValueError(f"Could not identify HYBAS_ID column. Columns: {list(huc8.columns)}")

        huc8["HUC8"] = huc8[hybas_id_col].astype(str)
        huc8["HUC8_NAME"] = huc8["HUC8"]
        return huc8[["HUC8", "HUC8_NAME", "geometry"]].copy()

    raise ValueError(f"Unsupported HUC source type: {source_type}")


def load_usgs_wateruse_csv(path: str) -> pd.DataFrame:
    header_candidates = [0, 1, 2, 3, 4, 5, 6]
    expected_any = ["FIPS", "STATEFIPS", "COUNTYFIPS", "PS-WFrTo", "DO-WFrTo", "IN-WFrTo", "IR-WFrTo"]

    for header_row in header_candidates:
        try:
            df = pd.read_csv(path, header=header_row, dtype=str)
            df.columns = [str(col).strip() for col in df.columns]
            found = sum(col in df.columns for col in expected_any)
            if found >= 3:
                print(f"[USGS LOAD SUCCESS] header={header_row}")
                return df
        except Exception:
            continue

    raise ValueError(f"Could not identify the correct header row in the USGS file: {path}")


def choose_aquifer_name_col(aquifers: gpd.GeoDataFrame) -> str:
    candidates = [
        "aq_name", "aqname", "aquifer", "aquifer_name", "name", "unit_name",
        "display_name", "principal_aq", "principal_aquifer", "fullname"
    ]
    col = pick_column(aquifers, candidates, required=False)
    if col is not None:
        return col
    for col in aquifers.columns:
        if col != "geometry" and aquifers[col].dtype == "object":
            return col
    raise ValueError("Could not identify aquifer name column.")


def choose_river_name_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    candidates = ["name", "gnis_name", "river_name", "stream_name"]
    col = pick_column(gdf, candidates, required=False)
    if col is not None:
        return col
    for candidate in gdf.columns:
        if candidate != "geometry" and gdf[candidate].dtype == "object":
            return candidate
    return None


def choose_reservoir_name_col(gdf: gpd.GeoDataFrame) -> Optional[str]:
    candidates = ["gnis_name", "name", "wb_name", "res_name", "feature_name", "waterbody_name", "lake_name", "hylak_name"]
    col = pick_column(gdf, candidates, required=False)
    if col is not None:
        return col
    for candidate in gdf.columns:
        if candidate != "geometry" and gdf[candidate].dtype == "object":
            return candidate
    return None


def km_from_m(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") / 1000.0


def add_threshold_flags(df: pd.DataFrame, source_col: str, prefix: str, thresholds_km: Iterable[int]) -> pd.DataFrame:
    for threshold in thresholds_km:
        df[f"{prefix}_within_{threshold}km"] = pd.to_numeric(df[source_col], errors="coerce") <= threshold
    return df


def nearest_join_points_to_features(
    points_gdf: gpd.GeoDataFrame,
    features_gdf: gpd.GeoDataFrame,
    feature_cols: Sequence[str],
    distance_col_m: str,
) -> gpd.GeoDataFrame:
    use_cols = [col for col in feature_cols if col in features_gdf.columns] + ["geometry"]
    feat = features_gdf[use_cols].copy()
    joined = gpd.sjoin_nearest(points_gdf, feat, how="left", distance_col=distance_col_m)
    return joined.drop(columns=["index_right"], errors="ignore")


def safe_frac(num: pd.Series, den: pd.Series) -> np.ndarray:
    return np.where(den > 0, num / den, np.nan)


def dominant_sector_row(row: pd.Series, sector_share_cols: Sequence[str]) -> Optional[str]:
    vals = row[list(sector_share_cols)]
    if vals.isna().all():
        return np.nan
    return vals.idxmax().replace("_share", "")


def mw_test(series_a: pd.Series, series_b: pd.Series) -> tuple[float, float, int, int]:
    a = pd.to_numeric(series_a, errors="coerce").dropna()
    b = pd.to_numeric(series_b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, len(a), len(b)
    stat, pvalue = mannwhitneyu(a, b, alternative="two-sided")
    return stat, pvalue, len(a), len(b)


def fdr_bh(pvals: Sequence[float]) -> np.ndarray:
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


def save_fig(fig: plt.Figure, outpath: str, dpi: int = 300) -> None:
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    print(f"[SAVED FIGURE] {outpath}")


def classify_supply_pathway(row: pd.Series) -> str:
    river_km = row.get("dist_to_major_river_km", np.nan)
    res_km = row.get("dist_to_reservoir_km", np.nan)
    ps_gw = row.get("county_PS_gw_fraction", np.nan)
    ps_sw = row.get("county_PS_sw_fraction", np.nan)

    near_river = pd.notna(river_km) and river_km <= 10
    near_res = pd.notna(res_km) and res_km <= 10
    gw_dom = pd.notna(ps_gw) and ps_gw >= 0.5
    sw_dom = pd.notna(ps_sw) and ps_sw >= 0.5

    if gw_dom and not (near_river or near_res):
        return "Groundwater-leaning"
    if sw_dom and (near_river or near_res):
        return "Surface-water-leaning"
    if gw_dom and (near_river or near_res):
        return "Mixed: groundwater + nearby surface water"
    if sw_dom and not (near_river or near_res):
        return "Surface-water county, not near mapped source"
    return "Mixed / uncertain"


def resolve_paths(data_root: str) -> ResolvedPaths:
    hints = [
        "groundwater",
        "water-project",
        "wbd_national_gpkg",
        "hydrolakes",
        "hydrorivers",
    ]

    dc_path = find_file_smart(
        data_root,
        exact_candidates=[
            os.path.join(data_root, "DC_CONUS.csv"),
            os.path.join(data_root, "DC_CONUS .csv"),
        ],
        search_patterns=["DC_CONUS.csv", "DC_CONUS .csv", "*DC*CONUS*.csv"],
        label="DC CSV",
        preferred_substrings=hints,
    )

    usgs_path = find_file_smart(
        data_root,
        exact_candidates=[
            os.path.join(data_root, "usco2015v2.0.csv"),
            os.path.join(data_root, "WaterUse2015.csv"),
        ],
        search_patterns=["usco2015v2.0.csv", "WaterUse2015.csv", "*usco2015*.csv", "*water*use*.csv"],
        label="USGS county water-use CSV",
        preferred_substrings=hints,
    )

    huc8_path = find_file_smart(
        data_root,
        exact_candidates=[
            os.path.join(data_root, "WBD_National_GPKG.gpkg"),
            os.path.join(data_root, "WBD_National_GPKG", "WBD_National_GPKG.gpkg"),
            os.path.join(data_root, "hybas_na_lev01-12_v1c", "hybas_na_lev08_v1c.shp"),
        ],
        search_patterns=["WBD_National_GPKG.gpkg", "*WBD*.gpkg", "hybas_na_lev08_v1c.shp", "*lev08*.shp"],
        label="HUC8 dataset",
        preferred_substrings=hints,
    )

    county_path = find_file_smart(
        data_root,
        exact_candidates=[
            os.path.join(data_root, "tl_2019_us_county.shp"),
            os.path.join(data_root, "tl_2023_us_county.shp"),
        ],
        search_patterns=["tl_*_us_county.shp", "cb_*_us_county_500k.shp", "*county*.shp"],
        label="County shapefile",
        preferred_substrings=hints,
    )

    aquifer_path = find_file_smart(
        data_root,
        exact_candidates=[os.path.join(data_root, "us_aquifers.shp")],
        search_patterns=["us_aquifers.shp", "*aquifer*.shp", "*Aquifer*.shp", "*principal*aquifer*.shp"],
        label="Aquifer shapefile",
        preferred_substrings=hints,
    )

    river_path = find_file_smart(
        data_root,
        exact_candidates=[os.path.join(data_root, "HydroRIVERS_v10_na.shp")],
        search_patterns=["HydroRIVERS_v10_na.shp", "*HydroRIVERS*.shp", "*hydrorivers*.shp"],
        label="River shapefile",
        preferred_substrings=hints,
    )

    reservoir_path = find_file_smart(
        data_root,
        exact_candidates=[
            os.path.join(data_root, "HydroLAKES_polys_v10_shp", "HydroLAKES_polys_v10.shp"),
            os.path.join(data_root, "HydroLAKES_polys_v10.shp"),
        ],
        search_patterns=["HydroLAKES_polys_v10.shp", "*HydroLAKES*.shp", "*hydrolakes*.shp", "*reservoir*.shp", "*lake*.shp"],
        label="Reservoir/Waterbody dataset",
        required=False,
        preferred_substrings=hints,
    )

    return ResolvedPaths(
        dc=dc_path,
        usgs=usgs_path,
        huc8=huc8_path,
        county=county_path,
        aquifer=aquifer_path,
        river=river_path,
        reservoir=reservoir_path,
    )


# ============================================================
# MAIN WORKFLOW
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrated data center water supply pathways workflow."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Root directory containing required input datasets.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Directory where outputs will be written.")
    parser.add_argument("--target-crs", default=TARGET_CRS, help="Projected CRS used for distance and area calculations.")
    args = parser.parse_args()

    outdir = ensure_dir(args.output_root)
    figdir = ensure_dir(os.path.join(outdir, "figures"))
    tabledir = ensure_dir(os.path.join(outdir, "tables"))

    paths = resolve_paths(args.data_root)

    print("\n================ PATH SUMMARY ================")
    print("DC_PATH        :", paths.dc)
    print("USGS_PATH      :", paths.usgs)
    print("HUC8_PATH      :", paths.huc8)
    print("COUNTY_PATH    :", paths.county)
    print("AQUIFER_PATH   :", paths.aquifer)
    print("RIVER_PATH     :", paths.river)
    print("RESERVOIR_PATH :", paths.reservoir)
    print("=============================================\n")

    print("Loading datasets...")
    dc_raw = pd.read_csv(paths.dc)
    usgs_raw = load_usgs_wateruse_csv(paths.usgs)
    huc8_raw, huc_source_type = load_huc8_dataset(paths.huc8)
    counties_raw = safe_read_vector(paths.county, "County boundaries")
    aquifers_raw = safe_read_vector(paths.aquifer, "Aquifers")
    rivers_raw = safe_read_vector(paths.river, "Rivers")
    reservoirs_raw = safe_read_vector(paths.reservoir, "Reservoir/Waterbody") if paths.reservoir else None

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
        dc["provider"] = "Unknown"
        provider_col = "provider"
    if facility_col is None:
        dc["facility"] = "Unknown"
        facility_col = "facility"
    if address_col is None:
        dc["address"] = ""
        address_col = "address"
    dc["dc_id"] = np.arange(1, len(dc) + 1)

    huc8 = standardize_huc8(ensure_crs(clean_geometry(huc8_raw, "HUC8")), huc_source_type)

    counties = ensure_crs(clean_geometry(counties_raw, "Counties"))
    statefp_col = pick_column(counties, ["STATEFP", "STATEFP20", "STATEFP10"], required=False)
    countyfp_col = pick_column(counties, ["COUNTYFP", "COUNTYFP20", "COUNTYFP10"], required=False)
    geoid_col = pick_column(counties, ["GEOID", "GEOID20", "GEOID10"], required=False)
    name_col = pick_column(counties, ["NAME", "NAMELSAD"], required=False)
    if geoid_col is None:
        if statefp_col is None or countyfp_col is None:
            raise ValueError("Could not identify county GEOID/FIPS columns.")
        counties["FIPS"] = counties[statefp_col].astype(str).str.zfill(2) + counties[countyfp_col].astype(str).str.zfill(3)
    else:
        counties["FIPS"] = counties[geoid_col].astype(str).str.zfill(5)
    counties["COUNTY_NAME"] = counties[name_col].astype(str) if name_col is not None else counties["FIPS"]
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
    major_rivers["RIVER_NAME"] = major_rivers[river_name_col].astype(str) if river_name_col is not None else "Unnamed river"
    major_rivers = major_rivers[["RIVER_NAME", "ORD_STRA_TMP", "geometry"]].copy()

    reservoirs = None
    if reservoirs_raw is not None:
        reservoirs = ensure_crs(clean_geometry(reservoirs_raw, "Reservoirs/Waterbodies"))
        area_col = pick_column(reservoirs, ["Lake_area", "LAKE_AREA", "lake_area"], required=False)
        if USE_HYDROLAKES_MIN_AREA_FILTER and area_col is not None:
            reservoirs[area_col] = pd.to_numeric(reservoirs[area_col], errors="coerce")
            reservoirs = reservoirs[reservoirs[area_col] >= HYDROLAKES_MIN_AREA_KM2].copy()
        res_name_col = choose_reservoir_name_col(reservoirs)
        reservoirs["RESERVOIR_NAME"] = reservoirs[res_name_col].astype(str) if res_name_col is not None else "Unnamed reservoir"
        reservoirs = reservoirs[["RESERVOIR_NAME", "geometry"]].copy()

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

    sector_map = {
        "PS_WFrTo": "PS-WFrTo",
        "DO_WFrTo": "DO-WFrTo",
        "IN_WFrTo": "IN-WFrTo",
        "IR_WFrTo": "IR-WFrTo",
    }
    optional_cols = {
        "PS_WGWFr": "PS-WGWFr",
        "PS_WSWFr": "PS-WSWFr",
        "DO_WGWFr": "DO-WGWFr",
        "DO_WSWFr": "DO-WSWFr",
        "IN_WGWFr": "IN-WGWFr",
        "IN_WSWFr": "IN-WSWFr",
        "IR_WGWFr": "IR-WGWFr",
        "IR_WSWFr": "IR-WSWFr",
        "PS_TOPop": "PS-TOPop",
        "PS_GWPop": "PS-GWPop",
        "PS_SWPop": "PS-SWPop",
    }
    missing = [value for value in sector_map.values() if value not in usgs.columns]
    if missing:
        raise ValueError(f"Missing required sector columns in USGS file: {missing}")

    needed_cols = ["FIPS"] + list(sector_map.values()) + [value for value in optional_cols.values() if value in usgs.columns]
    usgs = usgs[needed_cols].copy()
    for col in usgs.columns:
        if col != "FIPS":
            usgs[col] = pd.to_numeric(usgs[col], errors="coerce").fillna(0)
    rename_dict = {value: key for key, value in sector_map.items()}
    rename_dict.update({value: key for key, value in optional_cols.items() if value in usgs.columns})
    usgs = usgs.rename(columns=rename_dict)

    counties_usgs = counties.merge(usgs, on="FIPS", how="left")
    water_cols = [col for col in counties_usgs.columns if col not in ["FIPS", "COUNTY_NAME", "geometry"]]
    for col in water_cols:
        counties_usgs[col] = pd.to_numeric(counties_usgs[col], errors="coerce").fillna(0)

    dc_p = dc.to_crs(args.target_crs)
    huc8_p = huc8.to_crs(args.target_crs)
    counties_p = counties_usgs.to_crs(args.target_crs)
    aquifers_p = aquifers.to_crs(args.target_crs)
    major_rivers_p = major_rivers.to_crs(args.target_crs)
    reservoirs_p = reservoirs.to_crs(args.target_crs) if reservoirs is not None else None

    print("Assigning data centers to HUC8s, counties, and aquifers...")
    dc_huc8 = gpd.sjoin(dc_p, huc8_p[["HUC8", "HUC8_NAME", "geometry"]], how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
    dc_county = gpd.sjoin(dc_p, counties_p[["FIPS", "COUNTY_NAME", "geometry"]], how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
    dc_aquifer = gpd.sjoin(dc_p, aquifers_p[["AQUIFER_NAME", "geometry"]], how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
    dc_aquifer_one = dc_aquifer.sort_values(["dc_id", "AQUIFER_NAME"]).drop_duplicates(subset="dc_id", keep="first").copy()

    dc_joined = dc_p.copy()
    dc_joined = dc_joined.merge(dc_huc8.drop(columns="geometry"), left_index=True, right_index=True, how="left", suffixes=("", "_huc8"))
    dc_joined = dc_joined.merge(dc_county.drop(columns="geometry"), left_index=True, right_index=True, how="left", suffixes=("", "_county"))
    dc_joined = dc_joined.merge(dc_aquifer_one.drop(columns="geometry"), left_index=True, right_index=True, how="left", suffixes=("", "_aquifer"))
    dc_joined["provider"] = dc_joined[provider_col].astype(str)
    dc_joined["facility"] = dc_joined[facility_col].astype(str)
    dc_joined["address"] = dc_joined[address_col].astype(str)
    dc_unique = dc_joined.sort_values(["dc_id"]).drop_duplicates(subset="dc_id", keep="first").copy()

    print("Computing nearest major-river distances...")
    river_nearest = nearest_join_points_to_features(
        dc_unique[["dc_id", "geometry"]].copy(),
        major_rivers_p,
        feature_cols=["RIVER_NAME", "ORD_STRA_TMP"],
        distance_col_m="dist_to_major_river_m",
    ).rename(columns={"RIVER_NAME": "NEAREST_MAJOR_RIVER", "ORD_STRA_TMP": "NEAREST_RIVER_ORD_STRA"})
    dc_unique = dc_unique.merge(river_nearest.drop(columns="geometry"), on="dc_id", how="left")
    dc_unique["dist_to_major_river_km"] = km_from_m(dc_unique["dist_to_major_river_m"])
    dc_unique = add_threshold_flags(dc_unique, "dist_to_major_river_km", "river", DIST_BINS_KM)

    if reservoirs_p is not None and len(reservoirs_p) > 0:
        print("Computing nearest reservoir distances...")
        reservoir_nearest = nearest_join_points_to_features(
            dc_unique[["dc_id", "geometry"]].copy(),
            reservoirs_p,
            feature_cols=["RESERVOIR_NAME"],
            distance_col_m="dist_to_reservoir_m",
        ).rename(columns={"RESERVOIR_NAME": "NEAREST_RESERVOIR"})
        dc_unique = dc_unique.merge(reservoir_nearest.drop(columns="geometry"), on="dc_id", how="left")
        dc_unique["dist_to_reservoir_km"] = km_from_m(dc_unique["dist_to_reservoir_m"])
        dc_unique = add_threshold_flags(dc_unique, "dist_to_reservoir_km", "reservoir", DIST_BINS_KM)
    else:
        dc_unique["NEAREST_RESERVOIR"] = np.nan
        dc_unique["dist_to_reservoir_m"] = np.nan
        dc_unique["dist_to_reservoir_km"] = np.nan
        for threshold in DIST_BINS_KM:
            dc_unique[f"reservoir_within_{threshold}km"] = np.nan

    county_supply_cols = ["FIPS"]
    for col in [
        "PS_WFrTo", "DO_WFrTo", "IN_WFrTo", "IR_WFrTo",
        "PS_WGWFr", "PS_WSWFr", "DO_WGWFr", "DO_WSWFr",
        "IN_WGWFr", "IN_WSWFr", "IR_WGWFr", "IR_WSWFr",
        "PS_TOPop", "PS_GWPop", "PS_SWPop",
    ]:
        if col in counties_usgs.columns:
            county_supply_cols.append(col)
    county_supply = counties_usgs[county_supply_cols].drop_duplicates().copy()
    dc_unique = dc_unique.merge(county_supply, on="FIPS", how="left")

    dc_unique["county_selected_total_MGD"] = (
        dc_unique.get("PS_WFrTo", 0).fillna(0)
        + dc_unique.get("DO_WFrTo", 0).fillna(0)
        + dc_unique.get("IN_WFrTo", 0).fillna(0)
        + dc_unique.get("IR_WFrTo", 0).fillna(0)
    )

    for sector in ["PS", "DO", "IN", "IR"]:
        if f"{sector}_WFrTo" in dc_unique.columns:
            dc_unique[f"county_{sector}_share"] = safe_frac(dc_unique[f"{sector}_WFrTo"], dc_unique["county_selected_total_MGD"])
        if f"{sector}_WGWFr" in dc_unique.columns and f"{sector}_WFrTo" in dc_unique.columns:
            dc_unique[f"county_{sector}_gw_fraction"] = safe_frac(dc_unique[f"{sector}_WGWFr"], dc_unique[f"{sector}_WFrTo"])
        if f"{sector}_WSWFr" in dc_unique.columns and f"{sector}_WFrTo" in dc_unique.columns:
            dc_unique[f"county_{sector}_sw_fraction"] = safe_frac(dc_unique[f"{sector}_WSWFr"], dc_unique[f"{sector}_WFrTo"])

    if "PS_GWPop" in dc_unique.columns and "PS_TOPop" in dc_unique.columns:
        dc_unique["county_public_supply_population_gw_fraction"] = safe_frac(dc_unique["PS_GWPop"], dc_unique["PS_TOPop"])
    if "PS_SWPop" in dc_unique.columns and "PS_TOPop" in dc_unique.columns:
        dc_unique["county_public_supply_population_sw_fraction"] = safe_frac(dc_unique["PS_SWPop"], dc_unique["PS_TOPop"])

    dc_unique["combined_supply_pathway_proxy"] = dc_unique.apply(classify_supply_pathway, axis=1)

    print("Running county-to-HUC8 area-weighted overlay...")
    counties_p["county_area_m2"] = counties_p.geometry.area
    overlay_county_huc8 = gpd.overlay(counties_p, huc8_p[["HUC8", "HUC8_NAME", "geometry"]], how="intersection")
    overlay_county_huc8["intersect_area_m2"] = overlay_county_huc8.geometry.area
    overlay_county_huc8["area_weight"] = overlay_county_huc8["intersect_area_m2"] / overlay_county_huc8["county_area_m2"]

    weighted_cols = []
    for col in water_cols:
        weighted_col = f"{col}_aw"
        overlay_county_huc8[weighted_col] = overlay_county_huc8[col] * overlay_county_huc8["area_weight"]
        weighted_cols.append(weighted_col)

    huc8_water = overlay_county_huc8.groupby(["HUC8", "HUC8_NAME"])[weighted_cols].sum().reset_index()
    huc8_water = huc8_water.rename(columns={f"{col}_aw": col for col in water_cols})

    dc_count_by_huc8 = dc_unique.groupby(["HUC8", "HUC8_NAME"]).size().reset_index(name="dc_count")
    huc8_summary = huc8_p[["HUC8", "HUC8_NAME", "geometry"]].drop_duplicates().merge(huc8_water, on=["HUC8", "HUC8_NAME"], how="left").merge(dc_count_by_huc8, on=["HUC8", "HUC8_NAME"], how="left")
    for col in water_cols:
        if col in huc8_summary.columns:
            huc8_summary[col] = huc8_summary[col].fillna(0)
    huc8_summary["dc_count"] = huc8_summary["dc_count"].fillna(0).astype(int)
    huc8_summary["has_dc"] = huc8_summary["dc_count"] > 0
    huc8_summary["selected_total_MGD"] = huc8_summary.get("PS_WFrTo", 0) + huc8_summary.get("DO_WFrTo", 0) + huc8_summary.get("IN_WFrTo", 0) + huc8_summary.get("IR_WFrTo", 0)

    for sector in ["PS", "DO", "IN", "IR"]:
        huc8_summary[f"{sector}_share"] = safe_frac(huc8_summary.get(f"{sector}_WFrTo", 0), huc8_summary["selected_total_MGD"])
        gw_col = f"{sector}_WGWFr"
        sw_col = f"{sector}_WSWFr"
        tot_col = f"{sector}_WFrTo"
        if gw_col in huc8_summary.columns:
            huc8_summary[f"{sector}_gw_fraction"] = safe_frac(huc8_summary[gw_col], huc8_summary[tot_col])
        if sw_col in huc8_summary.columns:
            huc8_summary[f"{sector}_sw_fraction"] = safe_frac(huc8_summary[sw_col], huc8_summary[tot_col])

    huc8_summary["dominant_sector"] = huc8_summary.apply(lambda row: dominant_sector_row(row, ["PS_share", "DO_share", "IN_share", "IR_share"]), axis=1)

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
    for col in ["pct_dc_within_10km_river", "pct_dc_within_10km_reservoir"]:
        if col in huc8_proximity.columns:
            huc8_proximity[col] = huc8_proximity[col] * 100.0
    huc8_summary = huc8_summary.merge(huc8_proximity, on=["HUC8", "HUC8_NAME"], how="left")

    county_huc8_contrib = overlay_county_huc8.copy()
    county_huc8_contrib["PS_WFrTo_contrib"] = county_huc8_contrib.get("PS_WFrTo", 0) * county_huc8_contrib["area_weight"]
    county_huc8_contrib["selected_total_contrib"] = (
        county_huc8_contrib.get("PS_WFrTo", 0)
        + county_huc8_contrib.get("DO_WFrTo", 0)
        + county_huc8_contrib.get("IN_WFrTo", 0)
        + county_huc8_contrib.get("IR_WFrTo", 0)
    ) * county_huc8_contrib["area_weight"]
    county_huc8_table = county_huc8_contrib.groupby(["HUC8", "HUC8_NAME", "FIPS", "COUNTY_NAME"], as_index=False)[["PS_WFrTo_contrib", "selected_total_contrib"]].sum()
    dominant_county_per_huc8 = county_huc8_table.sort_values(["HUC8", "PS_WFrTo_contrib", "selected_total_contrib"], ascending=[True, False, False]).groupby("HUC8", as_index=False).first().rename(columns={
        "FIPS": "DOM_COUNTY_FIPS",
        "COUNTY_NAME": "DOM_COUNTY_NAME",
        "PS_WFrTo_contrib": "DOM_COUNTY_PS_MGD",
        "selected_total_contrib": "DOM_COUNTY_TOTAL_MGD",
    })
    huc8_summary = huc8_summary.merge(dominant_county_per_huc8[["HUC8", "DOM_COUNTY_FIPS", "DOM_COUNTY_NAME", "DOM_COUNTY_PS_MGD", "DOM_COUNTY_TOTAL_MGD"]], on="HUC8", how="left")

    print("Calculating dominant aquifer by HUC8 area overlap...")
    overlay_aq_huc8 = gpd.overlay(aquifers_p[["AQUIFER_NAME", "geometry"]], huc8_p[["HUC8", "HUC8_NAME", "geometry"]], how="intersection")
    overlay_aq_huc8["aq_area_m2"] = overlay_aq_huc8.geometry.area
    huc8_aq_area = overlay_aq_huc8.groupby(["HUC8", "HUC8_NAME", "AQUIFER_NAME"], as_index=False)["aq_area_m2"].sum()
    dominant_aquifer_per_huc8 = huc8_aq_area.sort_values(["HUC8", "aq_area_m2"], ascending=[True, False]).groupby("HUC8", as_index=False).first().rename(columns={"AQUIFER_NAME": "DOM_AQUIFER", "aq_area_m2": "DOM_AQUIFER_AREA_M2"})
    huc8_summary = huc8_summary.merge(dominant_aquifer_per_huc8[["HUC8", "DOM_AQUIFER", "DOM_AQUIFER_AREA_M2"]], on="HUC8", how="left")

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
    top_county_suppliers = county_supplier_table[county_supplier_table["dc_count"] > 0].sort_values(["dc_count", "PS_WFrTo", "selected_total_MGD"], ascending=[False, False, False])

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
        if n > 0 and 0 < p0 < 1:
            pvals.append(binomtest(k, n, p=p0, alternative="two-sided").pvalue)
        else:
            pvals.append(np.nan)
    aq_enrichment["p_value"] = pvals
    aq_enrichment["p_fdr"] = fdr_bh(pd.Series(aq_enrichment["p_value"]).fillna(1.0).values)
    aq_enrichment["significant_fdr_0.05"] = aq_enrichment["p_fdr"] < 0.05
    aq_enrichment = aq_enrichment.sort_values("enrichment_ratio", ascending=False)
    aq_enrichment_nonzero = aq_enrichment[aq_enrichment["observed_dc_count"] > 0].copy()

    master_huc8_table = huc8_summary.drop(columns="geometry").copy()
    dc_huc8_aquifer_counts = dc_unique.groupby(["HUC8", "HUC8_NAME", "AQUIFER_NAME"], as_index=False).size().rename(columns={"size": "dc_in_aquifer_count"})
    dominant_dc_aquifer_per_huc8 = dc_huc8_aquifer_counts.sort_values(["HUC8", "dc_in_aquifer_count"], ascending=[True, False]).groupby("HUC8", as_index=False).first().rename(columns={"AQUIFER_NAME": "TOP_DC_AQUIFER", "dc_in_aquifer_count": "TOP_DC_AQUIFER_COUNT"})
    master_huc8_table = master_huc8_table.merge(dominant_dc_aquifer_per_huc8[["HUC8", "TOP_DC_AQUIFER", "TOP_DC_AQUIFER_COUNT"]], on="HUC8", how="left")

    preferred_cols = [
        "HUC8", "HUC8_NAME", "dc_count", "has_dc", "selected_total_MGD",
        "PS_WFrTo", "DO_WFrTo", "IN_WFrTo", "IR_WFrTo",
        "PS_share", "DO_share", "IN_share", "IR_share", "dominant_sector",
        "PS_gw_fraction", "PS_sw_fraction", "IN_gw_fraction", "IN_sw_fraction",
        "DO_gw_fraction", "DO_sw_fraction", "IR_gw_fraction", "IR_sw_fraction",
        "mean_dist_to_major_river_km", "median_dist_to_major_river_km", "min_dist_to_major_river_km",
        "mean_dist_to_reservoir_km", "median_dist_to_reservoir_km", "min_dist_to_reservoir_km",
        "pct_dc_within_10km_river", "pct_dc_within_10km_reservoir",
        "DOM_COUNTY_FIPS", "DOM_COUNTY_NAME", "DOM_COUNTY_PS_MGD", "DOM_COUNTY_TOTAL_MGD",
        "DOM_AQUIFER", "TOP_DC_AQUIFER", "TOP_DC_AQUIFER_COUNT",
    ]
    master_huc8_table = master_huc8_table[[col for col in preferred_cols if col in master_huc8_table.columns] + [col for col in master_huc8_table.columns if col not in preferred_cols]]
    top_dc_huc8 = master_huc8_table[master_huc8_table["has_dc"]].sort_values(["dc_count", "selected_total_MGD"], ascending=[False, False]).head(25)

    dc_group = master_huc8_table[master_huc8_table["has_dc"]].copy()
    non_dc_group = master_huc8_table[~master_huc8_table["has_dc"]].copy()
    stats_vars = ["selected_total_MGD", "PS_share", "DO_share", "IN_share", "IR_share"]
    for col in ["PS_gw_fraction", "PS_sw_fraction", "IN_gw_fraction", "IN_sw_fraction", "DO_gw_fraction", "DO_sw_fraction", "IR_gw_fraction", "IR_sw_fraction"]:
        if col in master_huc8_table.columns:
            stats_vars.append(col)
    stats_results = []
    for var in stats_vars:
        stat, pvalue, n_dc, n_non_dc = mw_test(dc_group[var], non_dc_group[var])
        stats_results.append({
            "variable": var,
            "dc_median": pd.to_numeric(dc_group[var], errors="coerce").median(),
            "non_dc_median": pd.to_numeric(non_dc_group[var], errors="coerce").median(),
            "dc_mean": pd.to_numeric(dc_group[var], errors="coerce").mean(),
            "non_dc_mean": pd.to_numeric(non_dc_group[var], errors="coerce").mean(),
            "mannwhitney_u": stat,
            "p_value": pvalue,
            "n_dc": n_dc,
            "n_non_dc": n_non_dc,
        })
    stats_results = pd.DataFrame(stats_results)
    stats_results["p_fdr"] = fdr_bh(stats_results["p_value"].fillna(1.0).values)
    stats_results["significant_fdr_0.05"] = stats_results["p_fdr"] < 0.05

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
            pd.to_numeric(dc_unique["river_within_5km"], errors="coerce").mean() * 100 if "river_within_5km" in dc_unique.columns else np.nan,
            pd.to_numeric(dc_unique["river_within_10km"], errors="coerce").mean() * 100 if "river_within_10km" in dc_unique.columns else np.nan,
            pd.to_numeric(dc_unique["river_within_25km"], errors="coerce").mean() * 100 if "river_within_25km" in dc_unique.columns else np.nan,
            pd.to_numeric(dc_unique["river_within_50km"], errors="coerce").mean() * 100 if "river_within_50km" in dc_unique.columns else np.nan,
            pd.to_numeric(dc_unique["reservoir_within_5km"], errors="coerce").mean() * 100 if "reservoir_within_5km" in dc_unique.columns else np.nan,
            pd.to_numeric(dc_unique["reservoir_within_10km"], errors="coerce").mean() * 100 if "reservoir_within_10km" in dc_unique.columns else np.nan,
            pd.to_numeric(dc_unique["reservoir_within_25km"], errors="coerce").mean() * 100 if "reservoir_within_25km" in dc_unique.columns else np.nan,
            pd.to_numeric(dc_unique["reservoir_within_50km"], errors="coerce").mean() * 100 if "reservoir_within_50km" in dc_unique.columns else np.nan,
        ],
    })
    pathway_counts = dc_unique["combined_supply_pathway_proxy"].value_counts(dropna=False).rename_axis("combined_supply_pathway_proxy").reset_index(name="dc_count")

    master_csv = os.path.join(tabledir, "master_huc8_integrated_table.csv")
    top_huc8_csv = os.path.join(tabledir, "top_dc_huc8_integrated_table.csv")
    county_csv = os.path.join(tabledir, "county_supplier_proxy_table.csv")
    top_county_csv = os.path.join(tabledir, "top_dc_counties_supplier_proxy.csv")
    aq_csv = os.path.join(tabledir, "aquifer_enrichment_table.csv")
    stats_csv = os.path.join(tabledir, "dc_vs_non_dc_huc8_stats.csv")
    dc_assign_csv = os.path.join(tabledir, "data_centers_integrated_water_pathways.csv")
    dc_summary_csv = os.path.join(tabledir, "dc_proximity_summary.csv")
    pathway_csv = os.path.join(tabledir, "combined_supply_pathway_counts.csv")

    master_huc8_table.to_csv(master_csv, index=False)
    top_dc_huc8.to_csv(top_huc8_csv, index=False)
    county_supplier_table.to_csv(county_csv, index=False)
    top_county_suppliers.to_csv(top_county_csv, index=False)
    aq_enrichment_nonzero.to_csv(aq_csv, index=False)
    stats_results.to_csv(stats_csv, index=False)
    dc_proximity_summary.to_csv(dc_summary_csv, index=False)
    pathway_counts.to_csv(pathway_csv, index=False)
    dc_unique.to_crs("EPSG:4326").drop(columns="geometry").to_csv(dc_assign_csv, index=False)

    print("Making figures...")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([dc_group["selected_total_MGD"].dropna(), non_dc_group["selected_total_MGD"].dropna()], tick_labels=["DC HUC8", "Non-DC HUC8"], showfliers=False)
    ax.set_ylabel("Total water withdrawals (MGD)")
    ax.set_title("Total water withdrawals in HUC8 basins")
    save_fig(fig, os.path.join(figdir, "fig1_total_withdrawals_dc_vs_non_dc.png"))
    plt.close(fig)

    plot_df = pd.DataFrame({
        "Sector": ["Public supply", "Domestic", "Industrial", "Irrigation"],
        "DC_HUC8_mean_share": [dc_group["PS_share"].mean(), dc_group["DO_share"].mean(), dc_group["IN_share"].mean(), dc_group["IR_share"].mean()],
        "Non_DC_HUC8_mean_share": [non_dc_group["PS_share"].mean(), non_dc_group["DO_share"].mean(), non_dc_group["IN_share"].mean(), non_dc_group["IR_share"].mean()],
    })
    x = np.arange(len(plot_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, plot_df["DC_HUC8_mean_share"], width=width, label="DC HUC8")
    ax.bar(x + width / 2, plot_df["Non_DC_HUC8_mean_share"], width=width, label="Non-DC HUC8")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Sector"], rotation=20)
    ax.set_ylabel("Mean sector share")
    ax.set_title("Mean water-use sector shares: DC HUC8 vs non-DC HUC8")
    ax.legend()
    save_fig(fig, os.path.join(figdir, "fig2_sector_shares_dc_vs_non_dc.png"))
    plt.close(fig)

    if "PS_gw_fraction" in master_huc8_table.columns:
        vals = pd.DataFrame({
            "group": ["DC HUC8"] * len(dc_group) + ["Non-DC HUC8"] * len(non_dc_group),
            "value": pd.concat([dc_group["PS_gw_fraction"], non_dc_group["PS_gw_fraction"]], ignore_index=True),
        }).dropna()
        if len(vals) > 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            vals.boxplot(column="value", by="group", ax=ax)
            plt.suptitle("")
            ax.set_title("Public-supply groundwater fraction: DC vs non-DC HUC8")
            ax.set_ylabel("Groundwater fraction")
            save_fig(fig, os.path.join(figdir, "fig3_ps_groundwater_fraction_boxplot.png"))
            plt.close(fig)

    if len(aq_enrichment_nonzero) > 0:
        aq_plot = aq_enrichment_nonzero.head(15).sort_values("enrichment_ratio", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(aq_plot["AQUIFER_NAME"], aq_plot["enrichment_ratio"])
        ax.set_xlabel("Enrichment ratio (Observed / Expected by area)")
        ax.set_ylabel("Aquifer")
        ax.set_title("Aquifers with highest data-center overrepresentation")
        save_fig(fig, os.path.join(figdir, "fig4_top_aquifer_enrichment.png"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    pd.to_numeric(dc_unique["dist_to_major_river_km"], errors="coerce").dropna().hist(bins=30, ax=ax)
    ax.set_xlabel("Distance to nearest major river (km)")
    ax.set_ylabel("Number of data centers")
    ax.set_title("Distribution of data-center distance to nearest major river")
    save_fig(fig, os.path.join(figdir, "fig5_distance_to_major_river_histogram.png"))
    plt.close(fig)

    if pd.to_numeric(dc_unique["dist_to_reservoir_km"], errors="coerce").notna().sum() > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        pd.to_numeric(dc_unique["dist_to_reservoir_km"], errors="coerce").dropna().hist(bins=30, ax=ax)
        ax.set_xlabel("Distance to nearest reservoir/lake (km)")
        ax.set_ylabel("Number of data centers")
        ax.set_title("Distribution of data-center distance to nearest reservoir/lake")
        save_fig(fig, os.path.join(figdir, "fig6_distance_to_reservoir_histogram.png"))
        plt.close(fig)

    county_plot = top_county_suppliers.head(15).sort_values("dc_count", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(county_plot["COUNTY_NAME"].astype(str) + " (" + county_plot["FIPS"].astype(str) + ")", county_plot["dc_count"])
    ax.set_xlabel("Number of data centers")
    ax.set_ylabel("County")
    ax.set_title("Top counties containing data centers")
    save_fig(fig, os.path.join(figdir, "fig7_top_dc_counties.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_path = pathway_counts.sort_values("dc_count", ascending=True)
    ax.barh(plot_path["combined_supply_pathway_proxy"], plot_path["dc_count"])
    ax.set_xlabel("Number of data centers")
    ax.set_ylabel("Combined supply pathway proxy")
    ax.set_title("Combined groundwater/surface-water pathway interpretation")
    save_fig(fig, os.path.join(figdir, "fig8_combined_supply_pathway_proxy_counts.png"))
    plt.close(fig)

    sector_plot_df = master_huc8_table[master_huc8_table["has_dc"]][["HUC8_NAME", "dc_count", "PS_share", "DO_share", "IN_share", "IR_share"]].copy().sort_values(["dc_count", "PS_share"], ascending=[False, False]).head(15)
    for col in ["PS_share", "DO_share", "IN_share", "IR_share"]:
        sector_plot_df[col] = pd.to_numeric(sector_plot_df[col], errors="coerce").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(sector_plot_df))
    labels = sector_plot_df["HUC8_NAME"].astype(str).tolist()
    for col, label in [("PS_share", "Public supply"), ("DO_share", "Domestic"), ("IN_share", "Industrial"), ("IR_share", "Irrigation")]:
        ax.bar(labels, sector_plot_df[col], bottom=bottom, label=label)
        bottom += sector_plot_df[col].values
    ax.set_ylabel("Sector share")
    ax.set_title("Sector composition of top 15 HUC8s containing data centers")
    ax.legend(loc="lower left")
    plt.xticks(rotation=80, ha="right")
    save_fig(fig, os.path.join(figdir, "fig9_top15_huc8_sector_composition_stacked.png"))
    plt.close(fig)

    summary_txt = os.path.join(outdir, "summary_report.txt")
    with open(summary_txt, "w", encoding="utf-8") as handle:
        handle.write("INTEGRATED DATA CENTER WATER SUPPLY PATHWAYS ANALYSIS\n")
        handle.write("===============================================\n\n")
        handle.write(f"HUC source type used: {huc_source_type}\n")
        handle.write(f"DC file: {paths.dc}\n")
        handle.write(f"USGS file: {paths.usgs}\n")
        handle.write(f"HUC file: {paths.huc8}\n")
        handle.write(f"County file: {paths.county}\n")
        handle.write(f"Aquifer file: {paths.aquifer}\n")
        handle.write(f"River file: {paths.river}\n")
        handle.write(f"Reservoir file: {paths.reservoir}\n\n")
        handle.write(f"Total data centers: {dc_unique['dc_id'].nunique()}\n")
        handle.write(f"Rows in raw joined table: {len(dc_joined)}\n")
        handle.write(f"Rows in deduplicated table: {len(dc_unique)}\n")
        handle.write(f"HUC8s with data centers: {master_huc8_table['has_dc'].sum()}\n")
        handle.write(f"HUC8s without data centers: {(~master_huc8_table['has_dc']).sum()}\n")
        handle.write(f"Counties with data centers: {top_county_suppliers['FIPS'].nunique()}\n")
        handle.write(f"Aquifers containing data centers: {aq_enrichment_nonzero['AQUIFER_NAME'].nunique()}\n\n")
        handle.write("PROXIMITY SUMMARY\n-----------------\n")
        handle.write(dc_proximity_summary.to_string(index=False))
        handle.write("\n\nCOMBINED SUPPLY PATHWAY COUNTS\n------------------------------\n")
        handle.write(pathway_counts.to_string(index=False))
        handle.write("\n\nTOP HUC8S CONTAINING DATA CENTERS\n--------------------------------\n")
        handle.write(top_dc_huc8.head(15).to_string(index=False))
        handle.write("\n\nTOP COUNTY SUPPLIER PROXIES CONTAINING DATA CENTERS\n---------------------------------------------------\n")
        handle.write(top_county_suppliers.head(15).to_string(index=False))
        handle.write("\n\nTOP AQUIFERS BY ENRICHMENT\n--------------------------\n")
        handle.write(aq_enrichment_nonzero.head(15).to_string(index=False))
        handle.write("\n\nDC HUC8 VS NON-DC HUC8 STATISTICS\n---------------------------------\n")
        handle.write(stats_results.to_string(index=False))
        handle.write("\n")

    print("\n========================")
    print("KEY RESULTS")
    print("========================")
    print(f"HUC source type used: {huc_source_type}")
    print(f"Total data centers: {dc_unique['dc_id'].nunique()}")
    print(f"Rows in raw joined table: {len(dc_joined)}")
    print(f"Rows in deduplicated table: {len(dc_unique)}")
    print(f"HUC8s with data centers: {master_huc8_table['has_dc'].sum()}")
    print(f"HUC8s without data centers: {(~master_huc8_table['has_dc']).sum()}")
    print(f"Counties with data centers: {top_county_suppliers['FIPS'].nunique()}")
    print(f"Aquifers containing data centers: {aq_enrichment_nonzero['AQUIFER_NAME'].nunique()}")
    print("\nOutputs written to:")
    print(f"  Tables : {tabledir}")
    print(f"  Figures: {figdir}")
    print(f"  Summary: {summary_txt}")


if __name__ == "__main__":
    main()
