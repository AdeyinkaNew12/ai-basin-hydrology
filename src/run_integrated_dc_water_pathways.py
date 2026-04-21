#!/usr/bin/env python3
"""
Integrated data center water supply pathways analysis.

This script:
- links data centers to HUC8 basins, counties, and aquifers
- computes nearest distances to major rivers and reservoirs
- builds DC-, county-, and HUC8-scale summary tables
- evaluates aquifer overrepresentation
- compares HUC8 basins with and without data centers
- writes tables, figures, and a summary report

Example:
python src/run_integrated_dc_water_pathways.py \
  --data-root /mnt/disk3/aoolaseinde/data/integrated_dc_water_pathways \
  --output-root /mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results \
  --verbose
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import mannwhitneyu, binomtest

TARGET_CRS = "EPSG:5070"
MAJOR_RIVER_ORD_STRA = 5
DIST_BINS_KM = [5, 10, 25, 50]

LOGGER = logging.getLogger("integrated_dc_water_pathways")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated DC water pathways analysis")
    parser.add_argument("--data-root", required=True, help="Directory containing input datasets")
    parser.add_argument("--output-root", required=True, help="Directory to write outputs")
    parser.add_argument("--huc-layer", default="WBDHU8", help="Layer name in HUC geopackage")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def make_dirs(output_root: Path) -> tuple[Path, Path]:
    figures = output_root / "figures"
    tables = output_root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figures, tables


def pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lookup = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def guess_lat_lon_columns(df: pd.DataFrame) -> tuple[str, str]:
    lat = pick_column(df, ["latitude", "lat", "y", "dec_lat", "dec_lat_va"])
    lon = pick_column(df, ["longitude", "lon", "long", "x", "dec_long", "dec_long_va"])
    return lat, lon


def ensure_crs(gdf: gpd.GeoDataFrame, fallback: str = "EPSG:4326") -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(fallback)
    return gdf


def clean_geometry(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[gdf.geometry.is_valid].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    LOGGER.info("%s cleaned: %s features retained", label, len(gdf))
    return gdf


def safe_read_vector(path: Path, label: str, layer: str | None = None) -> gpd.GeoDataFrame:
    LOGGER.info("Loading %s from %s", label, path)
    return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)


def load_usgs_wateruse_csv(path: Path) -> pd.DataFrame:
    expected_any = ["FIPS", "STATEFIPS", "COUNTYFIPS", "PS-WFrTo", "DO-WFrTo", "IN-WFrTo", "IR-WFrTo"]
    for header_row in range(7):
        try:
            df = pd.read_csv(path, header=header_row, dtype=str)
            df.columns = [str(c).strip() for c in df.columns]
            if sum(col in df.columns for col in expected_any) >= 3:
                LOGGER.info("Loaded USGS CSV using header row %s", header_row)
                return df
        except Exception:
            continue
    raise ValueError(f"Could not identify correct header row in {path}")


def identify_huc8_columns(huc8: gpd.GeoDataFrame) -> tuple[str, str]:
    huc8_id = pick_column(huc8, ["huc8", "huc_8", "huc8_id", "huc8code", "huc_code"], required=False)
    huc8_name = pick_column(huc8, ["name", "name_1", "hucname", "hu_name"], required=False)

    if huc8_id is None:
        for c in huc8.columns:
            if c.upper() in {"HUC8", "HUC_8"}:
                huc8_id = c
                break

    if huc8_name is None:
        for c in huc8.columns:
            if c.upper() in {"NAME", "HU_8_NAME", "HUCNAME"}:
                huc8_name = c
                break

    if huc8_id is None:
        raise ValueError("Could not identify HUC8 ID column")

    if huc8_name is None:
        huc8["huc8_name_tmp"] = huc8[huc8_id].astype(str)
        huc8_name = "huc8_name_tmp"

    return huc8_id, huc8_name


def choose_aquifer_name_col(aquifers: gpd.GeoDataFrame) -> str:
    candidates = [
        "aq_name", "aqname", "aquifer", "aquifer_name", "name", "unit_name",
        "display_name", "principal_aq", "principal_aquifer", "fullname"
    ]
    col = pick_column(aquifers, candidates, required=False)
    if col is not None:
        return col
    for c in aquifers.columns:
        if c != "geometry" and aquifers[c].dtype == "object":
            return c
    raise ValueError("Could not identify aquifer name column")


def choose_reservoir_name_col(gdf: gpd.GeoDataFrame) -> str | None:
    return pick_column(
        gdf,
        ["gnis_name", "name", "wb_name", "res_name", "feature_name", "waterbody_name", "lake_name", "HYLAK_NAME"],
        required=False,
    )


def choose_river_name_col(gdf: gpd.GeoDataFrame) -> str | None:
    return pick_column(gdf, ["name", "gnis_name", "river_name", "stream_name"], required=False)


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
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


def mw_test(series_a: pd.Series, series_b: pd.Series) -> tuple[float, float, int, int]:
    a = pd.to_numeric(series_a, errors="coerce").dropna()
    b = pd.to_numeric(series_b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, len(a), len(b)
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return stat, p, len(a), len(b)


def safe_frac(num: pd.Series, den: pd.Series) -> np.ndarray:
    return np.where(den > 0, num / den, np.nan)


def dominant_sector_row(row: pd.Series, sector_share_cols: list[str]) -> str | float:
    vals = row[sector_share_cols]
    if vals.isna().all():
        return np.nan
    return vals.idxmax().replace("_share", "")


def km_from_m(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") / 1000.0


def add_threshold_flags(df: pd.DataFrame, source_col: str, prefix: str, thresholds_km: list[int]) -> pd.DataFrame:
    for t in thresholds_km:
        df[f"{prefix}_within_{t}km"] = pd.to_numeric(df[source_col], errors="coerce") <= t
    return df


def nearest_join_points_to_features(
    points_gdf: gpd.GeoDataFrame,
    features_gdf: gpd.GeoDataFrame,
    feature_cols: list[str],
    distance_col_m: str,
) -> gpd.GeoDataFrame:
    use_cols = [c for c in feature_cols if c in features_gdf.columns] + ["geometry"]
    feat = features_gdf[use_cols].copy()
    joined = gpd.sjoin_nearest(points_gdf, feat, how="left", distance_col=distance_col_m)
    return joined.drop(columns=["index_right"], errors="ignore")


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


def save_fig(fig: plt.Figure, outpath: Path, dpi: int = 300) -> None:
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    LOGGER.info("Saved figure: %s", outpath)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    figdir, tabledir = make_dirs(output_root)

    dc_path = ensure_exists(data_root / "DC_CONUS.csv", "DC CSV")
    usgs_path = ensure_exists(data_root / "usco2015v2.0.csv", "USGS county water-use CSV")
    huc8_path = ensure_exists(data_root / "WBD_National_GPKG.gpkg", "HUC8 geopackage")
    county_path = ensure_exists(data_root / "tl_2019_us_county.shp", "County shapefile")
    aquifer_path = ensure_exists(data_root / "us_aquifers.shp", "Aquifer shapefile")
    river_path = ensure_exists(data_root / "HydroRIVERS_v10_na.shp", "River shapefile")
    reservoir_path = data_root / "HydroLAKES_polys_v10.shp"

    LOGGER.info("Loading datasets...")
    dc_raw = pd.read_csv(dc_path)
    usgs_raw = load_usgs_wateruse_csv(usgs_path)
    huc8_raw = safe_read_vector(huc8_path, "HUC8 watersheds", layer=args.huc_layer)
    counties_raw = safe_read_vector(county_path, "County boundaries")
    aquifers_raw = safe_read_vector(aquifer_path, "Aquifers")
    rivers_raw = safe_read_vector(river_path, "Rivers")
    reservoirs_raw = safe_read_vector(reservoir_path, "Reservoir/Waterbody") if reservoir_path.exists() else None

    lat_col, lon_col = guess_lat_lon_columns(dc_raw)
    dc_raw = dc_raw.copy()
    dc_raw[lat_col] = pd.to_numeric(dc_raw[lat_col], errors="coerce")
    dc_raw[lon_col] = pd.to_numeric(dc_raw[lon_col], errors="coerce")
    dc_raw = dc_raw.dropna(subset=[lat_col, lon_col]).copy()

    dc = gpd.GeoDataFrame(
        dc_raw,
        geometry=gpd.points_from_xy(dc_raw[lon_col], dc_raw[lat_col]),
        crs="EPSG:4326"
    )

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

    huc8 = clean_geometry(ensure_crs(huc8_raw), "HUC8")
    huc8_id_col, huc8_name_col = identify_huc8_columns(huc8)
    huc8 = huc8[[huc8_id_col, huc8_name_col, "geometry"]].copy()
    huc8 = huc8.rename(columns={huc8_id_col: "HUC8", huc8_name_col: "HUC8_NAME"})
    huc8["HUC8"] = huc8["HUC8"].astype(str).str.zfill(8)

    counties = clean_geometry(ensure_crs(counties_raw), "Counties")
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

    counties["COUNTY_NAME"] = counties[name_col].astype(str) if name_col else counties["FIPS"]
    counties = counties[["FIPS", "COUNTY_NAME", "geometry"]].copy()

    aquifers = clean_geometry(ensure_crs(aquifers_raw), "Aquifers")
    aq_name_col = choose_aquifer_name_col(aquifers)
    aquifers = aquifers[[aq_name_col, "geometry"]].copy()
    aquifers = aquifers.rename(columns={aq_name_col: "AQUIFER_NAME"})
    aquifers["AQUIFER_NAME"] = aquifers["AQUIFER_NAME"].astype(str).str.strip()

    rivers = clean_geometry(ensure_crs(rivers_raw), "Rivers")
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
        reservoirs = clean_geometry(ensure_crs(reservoirs_raw), "Reservoirs/Waterbodies")
        res_name_col = choose_reservoir_name_col(reservoirs)
        reservoirs["RESERVOIR_NAME"] = reservoirs[res_name_col].astype(str) if res_name_col else "Unnamed reservoir"
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

    LOGGER.info("Assigning data centers to HUC8, counties, and aquifers...")
    dc_huc8 = gpd.sjoin(dc_p, huc8_p[["HUC8", "HUC8_NAME", "geometry"]], how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
    dc_county = gpd.sjoin(dc_p, counties_p[["FIPS", "COUNTY_NAME", "geometry"]], how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
    dc_aquifer = gpd.sjoin(dc_p, aquifers_p[["AQUIFER_NAME", "geometry"]], how="left", predicate="within").drop(columns=["index_right"], errors="ignore")

    if "dc_id" not in dc_aquifer.columns:
        raise ValueError("dc_id missing from aquifer join output.")

    dc_aquifer_one = dc_aquifer.sort_values(["dc_id", "AQUIFER_NAME"]).drop_duplicates(subset="dc_id", keep="first").copy()

    dc_joined = dc_p.copy()
    dc_joined = dc_joined.join(dc_huc8[["HUC8", "HUC8_NAME"]], how="left")
    dc_joined = dc_joined.join(dc_county[["FIPS", "COUNTY_NAME"]], how="left")
    dc_joined = dc_joined.join(dc_aquifer_one[["AQUIFER_NAME"]], how="left")

    dc_joined["provider"] = dc_joined[provider_col].astype(str)
    dc_joined["facility"] = dc_joined[facility_col].astype(str)
    dc_joined["address"] = dc_joined[address_col].astype(str)

    dc_unique = dc_joined.sort_values(["dc_id"]).drop_duplicates(subset="dc_id", keep="first").copy()

    assert dc_unique["dc_id"].nunique() == len(dc_unique), "dc_unique still has duplicate dc_id values"
    assert dc["dc_id"].nunique() == dc_unique["dc_id"].nunique(), "Some original data centers were lost during deduplication"

    LOGGER.info("Rows in dc_joined: %s", len(dc_joined))
    LOGGER.info("Unique DC IDs in original points: %s", dc["dc_id"].nunique())
    LOGGER.info("Rows in dc_unique: %s", len(dc_unique))
    LOGGER.info("Unique DC IDs in dc_unique: %s", dc_unique["dc_id"].nunique())

    river_nearest = nearest_join_points_to_features(
        dc_unique[["dc_id", "geometry"]].copy(),
        major_rivers_p,
        feature_cols=["RIVER_NAME", "ORD_STRA_TMP"],
        distance_col_m="dist_to_major_river_m"
    )
    river_nearest = river_nearest.rename(columns={"RIVER_NAME": "NEAREST_MAJOR_RIVER", "ORD_STRA_TMP": "NEAREST_RIVER_ORD_STRA"})
    dc_unique = dc_unique.merge(river_nearest.drop(columns="geometry"), on="dc_id", how="left")
    dc_unique["dist_to_major_river_km"] = km_from_m(dc_unique["dist_to_major_river_m"])
    dc_unique = add_threshold_flags(dc_unique, "dist_to_major_river_km", "river", DIST_BINS_KM)

    if reservoirs_p is not None and len(reservoirs_p) > 0:
        reservoir_nearest = nearest_join_points_to_features(
            dc_unique[["dc_id", "geometry"]].copy(),
            reservoirs_p,
            feature_cols=["RESERVOIR_NAME"],
            distance_col_m="dist_to_reservoir_m"
        )
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

    county_supply_cols = ["FIPS"]
    for c in [
        "PS_WFrTo", "DO_WFrTo", "IN_WFrTo", "IR_WFrTo",
        "PS_WGWFr", "PS_WSWFr", "DO_WGWFr", "DO_WSWFr",
        "IN_WGWFr", "IN_WSWFr", "IR_WGWFr", "IR_WSWFr",
        "PS_TOPop", "PS_GWPop", "PS_SWPop"
    ]:
        if c in counties_usgs.columns:
            county_supply_cols.append(c)

    county_supply = counties_usgs[county_supply_cols].drop_duplicates().copy()
    dc_unique = dc_unique.merge(county_supply, on="FIPS", how="left")

    dc_unique["county_selected_total_MGD"] = (
        dc_unique.get("PS_WFrTo", 0).fillna(0) +
        dc_unique.get("DO_WFrTo", 0).fillna(0) +
        dc_unique.get("IN_WFrTo", 0).fillna(0) +
        dc_unique.get("IR_WFrTo", 0).fillna(0)
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

    huc8_summary["dominant_sector"] = huc8_summary.apply(lambda row: dominant_sector_row(row, ["PS_share", "DO_share", "IN_share", "IR_share"]), axis=1)

    for sector in ["PS", "DO", "IN", "IR"]:
        gw_col = f"{sector}_WGWFr"
        sw_col = f"{sector}_WSWFr"
        tot_col = f"{sector}_WFrTo"
        if gw_col in huc8_summary.columns:
            huc8_summary[f"{sector}_gw_fraction"] = safe_frac(huc8_summary[gw_col], huc8_summary[tot_col])
        if sw_col in huc8_summary.columns:
            huc8_summary[f"{sector}_sw_fraction"] = safe_frac(huc8_summary[sw_col], huc8_summary[tot_col])

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
        if c in huc8_proximity.columns:
            huc8_proximity[c] = huc8_proximity[c] * 100.0
    huc8_summary = huc8_summary.merge(huc8_proximity, on=["HUC8", "HUC8_NAME"], how="left")

    county_huc8_contrib = overlay_county_huc8.copy()
    county_huc8_contrib["PS_WFrTo_contrib"] = county_huc8_contrib.get("PS_WFrTo", 0) * county_huc8_contrib["area_weight"]
    county_huc8_contrib["selected_total_contrib"] = (
        county_huc8_contrib.get("PS_WFrTo", 0) +
        county_huc8_contrib.get("DO_WFrTo", 0) +
        county_huc8_contrib.get("IN_WFrTo", 0) +
        county_huc8_contrib.get("IR_WFrTo", 0)
    ) * county_huc8_contrib["area_weight"]

    county_huc8_table = county_huc8_contrib.groupby(["HUC8", "HUC8_NAME", "FIPS", "COUNTY_NAME"], as_index=False)[["PS_WFrTo_contrib", "selected_total_contrib"]].sum()
    dominant_county_per_huc8 = county_huc8_table.sort_values(["HUC8", "PS_WFrTo_contrib", "selected_total_contrib"], ascending=[True, False, False]).groupby("HUC8", as_index=False).first().rename(columns={
        "FIPS": "DOM_COUNTY_FIPS",
        "COUNTY_NAME": "DOM_COUNTY_NAME",
        "PS_WFrTo_contrib": "DOM_COUNTY_PS_MGD",
        "selected_total_contrib": "DOM_COUNTY_TOTAL_MGD"
    })
    huc8_summary = huc8_summary.merge(dominant_county_per_huc8[["HUC8", "DOM_COUNTY_FIPS", "DOM_COUNTY_NAME", "DOM_COUNTY_PS_MGD", "DOM_COUNTY_TOTAL_MGD"]], on="HUC8", how="left")

    overlay_aq_huc8 = gpd.overlay(aquifers_p[["AQUIFER_NAME", "geometry"]], huc8_p[["HUC8", "HUC8_NAME", "geometry"]], how="intersection")
    overlay_aq_huc8["aq_area_m2"] = overlay_aq_huc8.geometry.area
    huc8_aq_area = overlay_aq_huc8.groupby(["HUC8", "HUC8_NAME", "AQUIFER_NAME"], as_index=False)["aq_area_m2"].sum()
    dominant_aquifer_per_huc8 = huc8_aq_area.sort_values(["HUC8", "aq_area_m2"], ascending=[True, False]).groupby("HUC8", as_index=False).first().rename(columns={
        "AQUIFER_NAME": "DOM_AQUIFER",
        "aq_area_m2": "DOM_AQUIFER_AREA_M2"
    })
    huc8_summary = huc8_summary.merge(dominant_aquifer_per_huc8[["HUC8", "DOM_AQUIFER", "DOM_AQUIFER_AREA_M2"]], on="HUC8", how="left")

    county_supplier_table = counties_usgs.drop(columns="geometry").merge(
        dc_unique.groupby(["FIPS", "COUNTY_NAME"], as_index=False).size().rename(columns={"size": "dc_count"}),
        on=["FIPS", "COUNTY_NAME"], how="left"
    )
    county_supplier_table["dc_count"] = county_supplier_table["dc_count"].fillna(0).astype(int)
    county_supplier_table["has_dc"] = county_supplier_table["has_dc"] if "has_dc" in county_supplier_table.columns else county_supplier_table["dc_count"] > 0
    county_supplier_table["has_dc"] = county_supplier_table["dc_count"] > 0
    county_supplier_table["selected_total_MGD"] = county_supplier_table.get("PS_WFrTo", 0) + county_supplier_table.get("DO_WFrTo", 0) + county_supplier_table.get("IN_WFrTo", 0) + county_supplier_table.get("IR_WFrTo", 0)

    for sector in ["PS", "DO", "IN", "IR"]:
        county_supplier_table[f"{sector}_share"] = safe_frac(county_supplier_table.get(f"{sector}_WFrTo", 0), county_supplier_table["selected_total_MGD"])
        if f"{sector}_WGWFr" in county_supplier_table.columns:
            county_supplier_table[f"{sector}_gw_fraction"] = safe_frac(county_supplier_table[f"{sector}_WGWFr"], county_supplier_table[f"{sector}_WFrTo"])
        if f"{sector}_WSWFr" in county_supplier_table.columns:
            county_supplier_table[f"{sector}_sw_fraction"] = safe_frac(county_supplier_table[f"{sector}_WSWFr"], county_supplier_table[f"{sector}_WFrTo"])

    top_county_suppliers = county_supplier_table[county_supplier_table["has_dc"]].sort_values(["dc_count", "PS_WFrTo", "selected_total_MGD"], ascending=[False, False, False])

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
        if n > 0 and p0 > 0 and p0 < 1:
            pval = binomtest(k, n, p=p0, alternative="two-sided").pvalue
        else:
            pval = np.nan
        pvals.append(pval)

    aq_enrichment["p_value"] = pvals
    aq_enrichment["p_fdr"] = fdr_bh(pd.Series(aq_enrichment["p_value"]).fillna(1.0).values)
    aq_enrichment["significant_fdr_0.05"] = aq_enrichment["p_fdr"] < 0.05
    aq_enrichment = aq_enrichment.sort_values("enrichment_ratio", ascending=False)
    aq_enrichment_nonzero = aq_enrichment[aq_enrichment["observed_dc_count"] > 0].copy()

    master_huc8_table = huc8_summary.drop(columns="geometry").copy()
    dc_huc8_aquifer_counts = dc_unique.groupby(["HUC8", "HUC8_NAME", "AQUIFER_NAME"], as_index=False).size().rename(columns={"size": "dc_in_aquifer_count"})
    dominant_dc_aquifer_per_huc8 = dc_huc8_aquifer_counts.sort_values(["HUC8", "dc_in_aquifer_count"], ascending=[True, False]).groupby("HUC8", as_index=False).first().rename(columns={
        "AQUIFER_NAME": "TOP_DC_AQUIFER",
        "dc_in_aquifer_count": "TOP_DC_AQUIFER_COUNT"
    })
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
            "n_non_dc": n2
        })

    stats_results = pd.DataFrame(stats_results)
    stats_results["p_fdr"] = fdr_bh(stats_results["p_value"].fillna(1.0).values)
    stats_results["significant_fdr_0.05"] = stats_results["p_fdr"] < 0.05

    dc_proximity_summary = pd.DataFrame({
        "metric": [
            "mean_dist_to_major_river_km",
            "median_dist_to_major_river_km",
            "mean_dist_to_reservoir_km",
            "median_dist_to_reservoir_km",
            "pct_within_5km_river",
            "pct_within_10km_river",
            "pct_within_25km_river",
            "pct_within_50km_river",
            "pct_within_5km_reservoir",
            "pct_within_10km_reservoir",
            "pct_within_25km_reservoir",
            "pct_within_50km_reservoir",
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
        ]
    })

    pathway_counts = dc_unique["combined_supply_pathway_proxy"].value_counts(dropna=False).rename_axis("combined_supply_pathway_proxy").reset_index(name="dc_count")

    # ---------------------------------------------------------------------
    # MANUSCRIPT-STYLE TABLE OUTPUTS
    # ---------------------------------------------------------------------
    master_huc8_table.to_csv(
        tabledir / "Table_1_HUC8_Integrated_WaterUse_and_DataCenter_Metrics.csv",
        index=False
    )
    top_dc_huc8.to_csv(
        tabledir / "Table_S1_Top_HUC8_Basins_by_DataCenter_Concentration.csv",
        index=False
    )
    county_supplier_table.to_csv(
        tabledir / "Table_S2_CountyLevel_WaterSupply_Characteristics_for_DataCenter_Locations.csv",
        index=False
    )
    top_county_suppliers.to_csv(
        tabledir / "Table_S3_Top_DataCenter_Hosting_Counties_and_Supply_Proxies.csv",
        index=False
    )
    aq_enrichment_nonzero.to_csv(
        tabledir / "Table_2_Aquifer_Association_and_Overrepresentation_of_DataCenters.csv",
        index=False
    )
    stats_results.to_csv(
        tabledir / "Table_3_Statistical_Comparison_of_DataCenter_vs_NonHosting_Basins.csv",
        index=False
    )
    dc_proximity_summary.to_csv(
        tabledir / "Table_S4_DataCenter_Proximity_to_Major_Rivers_and_Reservoirs.csv",
        index=False
    )
    pathway_counts.to_csv(
        tabledir / "Table_S5_Integrated_Groundwater_SurfaceWater_Pathway_Classification.csv",
        index=False
    )
    dc_unique.to_crs("EPSG:4326").drop(columns="geometry").to_csv(
        tabledir / "Table_S6_DataCenter_Level_Integrated_Water_Pathways_Dataset.csv",
        index=False
    )

    # ---------------------------------------------------------------------
    # MANUSCRIPT-STYLE FIGURE OUTPUTS
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [
            dc_group["selected_total_MGD"].dropna(),
            non_dc_group["selected_total_MGD"].dropna()
        ],
        labels=["Hosting Basins", "Non-Hosting Basins"],
        showfliers=False
    )
    ax.set_ylabel("Total water withdrawals (MGD)")
    ax.set_title("Comparison of Total Water Withdrawals Between Hosting and Non-Hosting Basins")
    save_fig(fig, figdir / "Figure_1_Total_Water_Withdrawals_Hosting_vs_NonHosting_Basins.png")
    plt.close(fig)

    plot_df = pd.DataFrame({
        "Sector": ["Public supply", "Domestic", "Industrial", "Irrigation"],
        "Hosting_Basins": [
            dc_group["PS_share"].mean(),
            dc_group["DO_share"].mean(),
            dc_group["IN_share"].mean(),
            dc_group["IR_share"].mean(),
        ],
        "NonHosting_Basins": [
            non_dc_group["PS_share"].mean(),
            non_dc_group["DO_share"].mean(),
            non_dc_group["IN_share"].mean(),
            non_dc_group["IR_share"].mean(),
        ]
    })

    x = np.arange(len(plot_df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, plot_df["Hosting_Basins"], width=w, label="Hosting Basins")
    ax.bar(x + w/2, plot_df["NonHosting_Basins"], width=w, label="Non-Hosting Basins")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Sector"], rotation=20)
    ax.set_ylabel("Mean sector share")
    ax.set_title("Sectoral Water-Use Composition of Hosting and Non-Hosting Basins")
    ax.legend()
    save_fig(fig, figdir / "Figure_2_Sectoral_WaterUse_Composition_Hosting_vs_NonHosting_Basins.png")
    plt.close(fig)

    if "PS_gw_fraction" in master_huc8_table.columns:
        vals = pd.DataFrame({
            "group": ["Hosting Basins"] * len(dc_group) + ["Non-Hosting Basins"] * len(non_dc_group),
            "value": pd.concat([dc_group["PS_gw_fraction"], non_dc_group["PS_gw_fraction"]], ignore_index=True)
        }).dropna()

        if len(vals) > 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            vals.boxplot(column="value", by="group", ax=ax)
            plt.suptitle("")
            ax.set_title("Public-Supply Groundwater Fraction in Hosting and Non-Hosting Basins")
            ax.set_ylabel("Groundwater fraction")
            save_fig(fig, figdir / "Figure_3_PublicSupply_Groundwater_Fraction_Hosting_vs_NonHosting_Basins.png")
            plt.close(fig)

    if len(aq_enrichment_nonzero) > 0:
        aq_plot = aq_enrichment_nonzero.head(15).sort_values("enrichment_ratio", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(aq_plot["AQUIFER_NAME"], aq_plot["enrichment_ratio"])
        ax.set_xlabel("Enrichment ratio (Observed / Expected by area)")
        ax.set_ylabel("Aquifer")
        ax.set_title("Aquifers Showing the Highest Overrepresentation of Data Centers")
        save_fig(fig, figdir / "Figure_4_Aquifer_Overrepresentation_of_DataCenters.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    pd.to_numeric(dc_unique["dist_to_major_river_km"], errors="coerce").dropna().hist(bins=30, ax=ax)
    ax.set_xlabel("Distance to nearest major river (km)")
    ax.set_ylabel("Number of data centers")
    ax.set_title("Distribution of Data Center Distance to the Nearest Major River")
    save_fig(fig, figdir / "Figure_5_Distribution_of_DataCenter_Distance_to_Major_Rivers.png")
    plt.close(fig)

    if pd.to_numeric(dc_unique["dist_to_reservoir_km"], errors="coerce").notna().sum() > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        pd.to_numeric(dc_unique["dist_to_reservoir_km"], errors="coerce").dropna().hist(bins=30, ax=ax)
        ax.set_xlabel("Distance to nearest reservoir/lake (km)")
        ax.set_ylabel("Number of data centers")
        ax.set_title("Distribution of Data Center Distance to the Nearest Reservoir or Lake")
        save_fig(fig, figdir / "Figure_6_Distribution_of_DataCenter_Distance_to_Reservoirs_and_Lakes.png")
        plt.close(fig)

    county_plot = top_county_suppliers.head(15).sort_values("dc_count", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        county_plot["COUNTY_NAME"].astype(str) + " (" + county_plot["FIPS"].astype(str) + ")",
        county_plot["dc_count"]
    )
    ax.set_xlabel("Number of data centers")
    ax.set_ylabel("County")
    ax.set_title("Top Counties Hosting Data Centers")
    save_fig(fig, figdir / "Figure_7_Top_Counties_Hosting_DataCenters.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_path = pathway_counts.sort_values("dc_count", ascending=True)
    ax.barh(plot_path["combined_supply_pathway_proxy"], plot_path["dc_count"])
    ax.set_xlabel("Number of data centers")
    ax.set_ylabel("Integrated water pathway classification")
    ax.set_title("Integrated Groundwater and Surface-Water Pathway Classification")
    save_fig(fig, figdir / "Figure_8_Integrated_Groundwater_SurfaceWater_Pathway_Classification.png")
    plt.close(fig)

    sector_plot_df = (
        master_huc8_table[master_huc8_table["has_dc"]]
        [["HUC8_NAME", "dc_count", "PS_share", "DO_share", "IN_share", "IR_share"]]
        .copy()
        .sort_values(["dc_count", "PS_share"], ascending=[False, False])
        .head(15)
    )

    for c in ["PS_share", "DO_share", "IN_share", "IR_share"]:
        sector_plot_df[c] = pd.to_numeric(sector_plot_df[c], errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(sector_plot_df))
    labels = sector_plot_df["HUC8_NAME"].astype(str).tolist()

    sector_specs = [
        ("PS_share", "Public supply"),
        ("DO_share", "Domestic"),
        ("IN_share", "Industrial"),
        ("IR_share", "Irrigation"),
    ]

    for col, label in sector_specs:
        ax.bar(labels, sector_plot_df[col], bottom=bottom, label=label)
        bottom += sector_plot_df[col].values

    ax.set_ylabel("Sector share")
    ax.set_title("Sectoral Water-Use Composition of the Top 15 HUC8 Basins Hosting Data Centers")
    ax.legend(loc="lower left")
    plt.xticks(rotation=80, ha="right")
    save_fig(fig, figdir / "Figure_9_Sectoral_WaterUse_Composition_of_Top_Hosting_HUC8_Basins.png")
    plt.close(fig)

    # ---------------------------------------------------------------------
    # MANUSCRIPT-STYLE SUMMARY REPORT
    # ---------------------------------------------------------------------
    summary_txt = output_root / "Supplementary_Report_Integrated_DataCenter_Water_Pathways.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("INTEGRATED DATA CENTER WATER SUPPLY PATHWAYS ANALYSIS\n")
        f.write("SUPPLEMENTARY SUMMARY REPORT\n")
        f.write("===================================\n\n")
        f.write(f"Total data centers: {dc_unique['dc_id'].nunique()}\n")
        f.write(f"Rows in raw joined table: {len(dc_joined)}\n")
        f.write(f"Rows in deduplicated table: {len(dc_unique)}\n")
        f.write(f"HUC8s with data centers: {master_huc8_table['has_dc'].sum()}\n")
        f.write(f"HUC8s without data centers: {(~master_huc8_table['has_dc']).sum()}\n")
        f.write(f"Counties with data centers: {top_county_suppliers['FIPS'].nunique()}\n")
        f.write(f"Aquifers containing data centers: {aq_enrichment_nonzero['AQUIFER_NAME'].nunique()}\n\n")

        f.write("PROXIMITY SUMMARY\n")
        f.write("-----------------\n")
        f.write(dc_proximity_summary.to_string(index=False))
        f.write("\n\n")

        f.write("INTEGRATED WATER PATHWAY CLASSIFICATION COUNTS\n")
        f.write("---------------------------------------------\n")
        f.write(pathway_counts.to_string(index=False))
        f.write("\n\n")

        f.write("TOP HUC8 BASINS HOSTING DATA CENTERS\n")
        f.write("-----------------------------------\n")
        f.write(top_dc_huc8.head(15).to_string(index=False))
        f.write("\n\n")

        f.write("TOP COUNTY-LEVEL WATER SUPPLY PROXIES FOR DATA CENTER LOCATIONS\n")
        f.write("---------------------------------------------------------------\n")
        f.write(top_county_suppliers.head(15).to_string(index=False))
        f.write("\n\n")

        f.write("TOP AQUIFERS BY DATA CENTER OVERREPRESENTATION\n")
        f.write("---------------------------------------------\n")
        f.write(aq_enrichment_nonzero.head(15).to_string(index=False))
        f.write("\n\n")

        f.write("STATISTICAL COMPARISON OF HOSTING AND NON-HOSTING HUC8 BASINS\n")
        f.write("-------------------------------------------------------------\n")
        f.write(stats_results.to_string(index=False))
        f.write("\n\n")

        f.write("NOTES\n")
        f.write("-----\n")
        f.write("County boundaries are used as a proxy for water supplier/service boundaries.\n")
        f.write("HydroLAKES is used to represent lakes/reservoirs for the nearest stored-water analysis.\n")
        f.write("DC-based counts and summaries are computed from dc_unique, which keeps one row per dc_id.\n")
        f.write("One aquifer is assigned per DC using the first within-match after sorting; refine later if needed.\n\n")

        f.write("FIGURES SAVED IN:\n")
        f.write(str(figdir) + "\n\n")

        f.write("TABLES SAVED IN:\n")
        f.write(str(tabledir) + "\n\n")

    LOGGER.info("Done. Output root: %s", output_root)


if __name__ == "__main__":
    main()
