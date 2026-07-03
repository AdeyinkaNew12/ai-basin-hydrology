
#!/usr/bin/env python3
"""
Integrated Data Center Water Supply Pathways Analysis

This script links U.S. data center point locations to HUC8 watersheds,
counties, aquifers, major rivers, and HydroLAKES waterbodies, then combines
those spatial relationships with county-level USGS water-use attributes to
produce integrated tables, summary statistics, figures, and a plain-text
summary report.

Server/GitHub version:
- discovers files from a data root
- requires HydroRIVERS sidecars
- uses major rivers defined as ORD_STRA >= 5
- writes tables and figures to an output directory
"""

from __future__ import annotations

from pathlib import Path

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

from common_paths import DATA_FOLDERS, output_folder
from scipy.stats import mannwhitneyu


TARGET_CRS = "EPSG:5070"
DEFAULT_DATA_ROOT = DATA_FOLDERS["groundwater"]
DEFAULT_OUTPUT_ROOT = output_folder("integrated_dc_water_pathways")
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


def verify_shapefile_sidecars(shp_path: str, required_exts=(".shp", ".dbf", ".shx", ".prj"), label: str = "shapefile") -> None:
    base, _ = os.path.splitext(shp_path)
    missing = []
    for ext in required_exts:
        candidate = base + ext
        if not os.path.exists(candidate):
            missing.append(candidate)

    if missing:
        raise FileNotFoundError(
            f"Missing required sidecar files for {label}:\n" + "\n".join(missing)
        )
    print(f"[OK] {label} sidecars present for: {shp_path}")


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
        verify_shapefile_sidecars(path, label="HydroBASINS Level 8")
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
    hints = ["groundwater", "water-project", "wbd_national_gpkg", "hydrolakes", "hydrorivers"]

    dc_path = find_file_smart(
        Path(DATA_FOLDERS["water_project"]),
        exact_candidates=[os.path.join(data_root, "DC_CONUS.csv"), os.path.join(data_root, "DC_CONUS .csv")],
        search_patterns=["DC_CONUS.csv", "DC_CONUS .csv", "*DC*CONUS*.csv"],
        label="DC CSV",
        preferred_substrings=hints,
    )

    usgs_path = find_file_smart(
        data_root,
        exact_candidates=[os.path.join(data_root, "usco2015v2.0.csv"), os.path.join(data_root, "WaterUse2015.csv")],
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
        exact_candidates=[os.path.join(data_root, "tl_2019_us_county.shp"), os.path.join(data_root, "tl_2023_us_county.shp")],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated data center water supply pathways workflow.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Root directory containing required input datasets.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Directory where outputs will be written.")
    parser.add_argument("--target-crs", default=TARGET_CRS, help="Projected CRS used for distance and area calculations.")
    args = parser.parse_args()

    outdir = ensure_dir(args.output_root)
    figdir = ensure_dir(os.path.join(outdir, "figures"))
    tabledir = ensure_dir(os.path.join(outdir, "tables"))

    paths = resolve_paths(args.data_root)

    verify_shapefile_sidecars(paths.river, label="HydroRIVERS")
    verify_shapefile_sidecars(paths.county, label="County shapefile")
    verify_shapefile_sidecars(paths.aquifer, label="Aquifer shapefile")
    if paths.reservoir:
        verify_shapefile_sidecars(paths.reservoir, label="HydroLAKES")

    print("\n================ PATH SUMMARY ================")
    print("DC_PATH        :", paths.dc)
    print("USGS_PATH      :", paths.usgs)
    print("HUC8_PATH      :", paths.huc8)
    print("COUNTY_PATH    :", paths.county)
    print("AQUIFER_PATH   :", paths.aquifer)
    print("RIVER_PATH     :", paths.river)
    print("RESERVOIR_PATH :", paths.reservoir)
    print("OUTDIR         :", outdir)
    print("FIGDIR         :", figdir)
    print("TABLEDIR       :", tabledir)
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

    if ord_col is None:
        raise ValueError(
            "ORD_STRA column not found in HydroRIVERS attributes. "
            "Check that the .dbf sidecar is present and matches the .shp."
        )

    rivers["ORD_STRA_TMP"] = pd.to_numeric(rivers[ord_col], errors="coerce")
    major_rivers = rivers[rivers["ORD_STRA_TMP"] >= MAJOR_RIVER_ORD_STRA].copy()
    major_rivers["RIVER_NAME"] = rivers[river_name_col].astype(str) if river_name_col is not None else "Unnamed river"
    major_rivers = major_rivers[["RIVER_NAME", "ORD_STRA_TMP", "geometry"]].copy()
    print(f"[INFO] Using Strahler order >= {MAJOR_RIVER_ORD_STRA} for major rivers")
    print(f"Major rivers retained: {len(major_rivers)}")

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
        dc_unique["county_PS_pop_gw_fraction"] = safe_frac(dc_unique["PS_GWPop"], dc_unique["PS_TOPop"])
    if "PS_SWPop" in dc_unique.columns and "PS_TOPop" in dc_unique.columns:
        dc_unique["county_PS_pop_sw_fraction"] = safe_frac(dc_unique["PS_SWPop"], dc_unique["PS_TOPop"])

    if "county_PS_gw_fraction" not in dc_unique.columns and "county_PS_pop_gw_fraction" in dc_unique.columns:
        dc_unique["county_PS_gw_fraction"] = dc_unique["county_PS_pop_gw_fraction"]
    if "county_PS_sw_fraction" not in dc_unique.columns and "county_PS_pop_sw_fraction" in dc_unique.columns:
        dc_unique["county_PS_sw_fraction"] = dc_unique["county_PS_pop_sw_fraction"]

    sector_share_cols = [c for c in ["county_PS_share", "county_DO_share", "county_IN_share", "county_IR_share"] if c in dc_unique.columns]
    if sector_share_cols:
        dc_unique["county_dominant_sector"] = dc_unique.apply(lambda row: dominant_sector_row(row, sector_share_cols), axis=1)
    else:
        dc_unique["county_dominant_sector"] = np.nan

    dc_unique["supply_pathway_class"] = dc_unique.apply(classify_supply_pathway, axis=1)

    # ------------------------------------------------------------------
    # FINAL SAFETY CHECK
    # Spatial joins can expand rows when one point intersects multiple
    # polygons/features. Keep one record per original data-center site
    # before exporting tables or generating summaries.
    # ------------------------------------------------------------------
    before_n = len(dc_unique)
    before_unique = dc_unique["dc_id"].nunique()

    dc_unique = (
        dc_unique
        .sort_values(["dc_id"])
        .drop_duplicates(subset="dc_id", keep="first")
        .copy()
    )

    after_n = len(dc_unique)
    after_unique = dc_unique["dc_id"].nunique()

    print(f"[CHECK] Rows before final dc_id dedupe: {before_n}")
    print(f"[CHECK] Unique dc_id before final dedupe: {before_unique}")
    print(f"[CHECK] Rows after final dc_id dedupe: {after_n}")
    print(f"[CHECK] Unique dc_id after final dedupe: {after_unique}")

    if after_n != after_unique:
        raise ValueError("Duplicate dc_id values remain after final dedupe.")

    # Export corrected one-row-per-site table
    dc_export = pd.DataFrame(dc_unique.drop(columns="geometry"))
    dc_export.to_csv(os.path.join(tabledir, "dc_water_supply_pathways.csv"), index=False)

    # Export corrected Figure 10 summary
    pathway_summary = (
        dc_export["supply_pathway_class"]
        .value_counts()
        .rename_axis("supply_pathway_class")
        .reset_index(name="n_sites")
    )
    pathway_summary["percent_sites"] = (
        pathway_summary["n_sites"] / pathway_summary["n_sites"].sum() * 100
    ).round(1)

    pathway_summary.to_csv(
        os.path.join(tabledir, "figure10_supply_pathway_summary_corrected.csv"),
        index=False
    )

    print("\n[CORRECTED FIGURE 10 SUMMARY]")
    print(pathway_summary)
    print("Total sites:", pathway_summary["n_sites"].sum())

    huc8_dc_counts = dc_unique.groupby(["HUC8", "HUC8_NAME"], dropna=False).size().reset_index(name="dc_count")
    huc8_dc_counts.to_csv(os.path.join(tabledir, "huc8_dc_counts.csv"), index=False)

    aquifer_counts = (
        dc_unique.dropna(subset=["AQUIFER_NAME"])
        .groupby("AQUIFER_NAME")
        .size()
        .reset_index(name="dc_count")
        .sort_values("dc_count", ascending=False)
    )
    aquifer_counts.to_csv(os.path.join(tabledir, "aquifer_dc_counts.csv"), index=False)

    county_counts = (
        dc_unique.dropna(subset=["FIPS"])
        .groupby(["FIPS", "COUNTY_NAME"])
        .size()
        .reset_index(name="dc_count")
        .sort_values("dc_count", ascending=False)
    )
    county_counts.to_csv(os.path.join(tabledir, "county_dc_counts.csv"), index=False)

    summary = {
        "n_data_centers": int(dc_unique["dc_id"].nunique()),
        "n_huc8_with_dc": int(dc_unique["HUC8"].dropna().nunique()),
        "n_counties_with_dc": int(dc_unique["FIPS"].dropna().nunique()),
        "n_aquifers_with_dc": int(dc_unique["AQUIFER_NAME"].dropna().nunique()),
        "share_within_10km_major_river": float(pd.to_numeric(dc_unique["river_within_10km"], errors="coerce").mean()),
        "share_within_10km_reservoir": float(pd.to_numeric(dc_unique["reservoir_within_10km"], errors="coerce").mean()) if "reservoir_within_10km" in dc_unique.columns else np.nan,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(tabledir, "summary_metrics.csv"), index=False)

    if len(aquifer_counts) > 0:
        fig, ax = plt.subplots(figsize=(11, 6))
        plot_df = aquifer_counts.head(15).iloc[::-1]
        ax.barh(plot_df["AQUIFER_NAME"], plot_df["dc_count"])
        ax.set_xlabel("Number of data centers")
        ax.set_ylabel("Aquifer")
        ax.set_title("Top aquifers by data center count")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    river_dist = pd.to_numeric(dc_unique["dist_to_major_river_km"], errors="coerce").dropna()
    if len(river_dist) > 0:
        ax.hist(river_dist, bins=40)
    ax.set_xlabel("Distance to nearest major river (km)")
    ax.set_ylabel("Number of data centers")
    ax.set_title("Distance from data centers to major rivers (ORD_STRA >= 5)")
    plt.close(fig)

    if "dist_to_reservoir_km" in dc_unique.columns:
        fig, ax = plt.subplots(figsize=(9, 6))
        res_dist = pd.to_numeric(dc_unique["dist_to_reservoir_km"], errors="coerce").dropna()
        if len(res_dist) > 0:
            ax.hist(res_dist, bins=40)
        ax.set_xlabel("Distance to nearest reservoir/waterbody (km)")
        ax.set_ylabel("Number of data centers")
        ax.set_title("Distance from data centers to reservoirs/waterbodies")
        plt.close(fig)

    report_lines = [
        "Integrated Data Center Water Supply Pathways Analysis",
        "===================================================",
        f"Data centers analyzed: {summary['n_data_centers']}",
        f"HUC8 basins hosting data centers: {summary['n_huc8_with_dc']}",
        f"Counties hosting data centers: {summary['n_counties_with_dc']}",
        f"Aquifers hosting data centers: {summary['n_aquifers_with_dc']}",
        f"Share within 10 km of major rivers (ORD_STRA >= 5): {summary['share_within_10km_major_river']:.3f}" if pd.notna(summary['share_within_10km_major_river']) else "Share within 10 km of major rivers (ORD_STRA >= 5): NA",
        f"Share within 10 km of reservoirs: {summary['share_within_10km_reservoir']:.3f}" if pd.notna(summary['share_within_10km_reservoir']) else "Share within 10 km of reservoirs: NA",
        "",
        "Top aquifers by data center count:",
    ]
    if len(aquifer_counts) > 0:
        for _, row in aquifer_counts.head(10).iterrows():
            report_lines.append(f"- {row['AQUIFER_NAME']}: {int(row['dc_count'])}")
    else:
        report_lines.append("- No aquifer assignments available.")

    report_path = os.path.join(outdir, "analysis_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[SAVED REPORT] {report_path}")

    print("\nDone.")
    print("Results written to:", outdir)


if __name__ == "__main__":
    main()