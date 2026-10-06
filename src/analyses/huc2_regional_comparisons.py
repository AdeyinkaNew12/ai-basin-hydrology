#!/usr/bin/env python3

"""
Compare AI, Power, and TRI facilities across HUC2 regions using
river proximity, hydrologic metrics, water stress, and public-supply sources.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

from scipy.stats import (
    mannwhitneyu,
    ks_2samp,
    chi2_contingency,
    fisher_exact,
    MonteCarloMethod,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/mnt/disk3/aoolaseinde/projects/ai-basin-hydrology"
)

HYDRO_MASTER = (
    ROOT
    / "final_results/hydrologic_regimes"
    / "basin_master_presence_hydro.csv"
)

HYBAS = Path(
    "/mnt/disk3/aoolaseinde/data/Groundwater/"
    "hybas_na_lev08_v1c.shp"
)

HUC = Path(
    "/mnt/disk3/aoolaseinde/data/WaterProject/"
    "WBD_National_GPKG/WBD_National_GPKG.gpkg"
)

RIVER_DIR = Path(
    "/mnt/disk3/aoolaseinde/projects/"
    "integrated_dc_water_pathways/results/"
    "water_proximity_threshold_ord5"
)

RIVER_FILES = {
    "AI":
        RIVER_DIR
        / "data_center_sector_nearest_water_BASIN_FILTER_ORD5.csv",

    "Power":
        RIVER_DIR
        / "power_plant_sector_nearest_water_BASIN_FILTER_ORD5.csv",

    "TRI":
        RIVER_DIR
        / "industry_TRI_sector_nearest_water_BASIN_FILTER_ORD5.csv",
}

STRESS_FILE = (
    ROOT
    / "final_results/water_stress_analysis"
    / "analysis4_facilities_with_stress_ALL_BASINS.csv"
)

SUPPLY_FILE = (
    ROOT
    / "results/groundwater_dc"
    / "three_sector_pathways"
    / "three_sector_pathway_assignments.csv"
)

OUT = (
    ROOT
    / "final_results/huc2_reviewer_final"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

ALPHA = 0.05

SECTORS = [
    "AI",
    "Power",
    "TRI",
]

COMPARATORS = [
    "Power",
    "TRI",
]

HYDRO_METRICS = [
    "RBI",
    "season_conc",
    "CVQ",
]

HUC_EXCLUDE = [
    "19",
    "20",
    "21",
    "22",
]


# ============================================================
# FUNCTIONS
# ============================================================

def heading(text):

    print(
        "\n"
        + "=" * 110
    )

    print(text)

    print(
        "=" * 110
    )


def bh_fdr(pvalues):
    """
    Benjamini-Hochberg FDR correction.
    """

    p = np.asarray(
        pvalues,
        dtype=float
    )

    q = np.full(
        len(p),
        np.nan
    )

    valid = np.isfinite(p)

    if valid.sum() == 0:
        return q

    pv = p[valid]

    order = np.argsort(pv)

    ranked = pv[order]

    m = len(ranked)

    adjusted = (
        ranked
        * m
        / np.arange(
            1,
            m + 1
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.minimum(
        adjusted,
        1.0
    )

    back = np.empty(m)

    back[order] = adjusted

    q[valid] = back

    return q


def cliffs_delta(x, y):

    x = (
        pd.to_numeric(
            pd.Series(x),
            errors="coerce"
        )
        .dropna()
        .to_numpy()
    )

    y = (
        pd.to_numeric(
            pd.Series(y),
            errors="coerce"
        )
        .dropna()
        .to_numpy()
    )

    if (
        len(x) == 0
        or len(y) == 0
    ):
        return np.nan

    u = mannwhitneyu(
        x,
        y,
        alternative="two-sided"
    ).statistic

    return (
        2 * u
        / (
            len(x)
            * len(y)
        )
        - 1
    )


def cramers_v(table):

    table = np.asarray(table)

    if (
        table.ndim != 2
        or table.shape[0] < 2
        or table.shape[1] < 2
        or table.sum() == 0
    ):
        return np.nan

    chi2, _, _, _ = (
        chi2_contingency(
            table,
            correction=False
        )
    )

    n = table.sum()

    r, k = table.shape

    denom = min(
        r - 1,
        k - 1
    )

    if denom <= 0:
        return np.nan

    return np.sqrt(
        chi2
        / (
            n
            * denom
        )
    )


def detect_coord_cols(df):

    possibilities = [
        (
            "Latitude",
            "Longitude"
        ),
        (
            "12. LATITUDE",
            "13. LONGITUDE"
        ),
        (
            "latitude",
            "longitude"
        ),
        (
            "lat",
            "lon"
        ),
    ]

    for lat, lon in possibilities:

        if (
            lat in df.columns
            and lon in df.columns
        ):

            lat_num = pd.to_numeric(
                df[lat],
                errors="coerce"
            )

            lon_num = pd.to_numeric(
                df[lon],
                errors="coerce"
            )

            if (
                lat_num.notna().sum() > 0
                and
                lon_num.notna().sum() > 0
            ):

                return lat, lon

    raise RuntimeError(
        "No usable coordinate "
        "columns identified."
    )


# ============================================================
# HUC2
# ============================================================

heading(
    "LOAD HUC2"
)

huc2 = gpd.read_file(
    HUC,
    layer="WBDHU2"
)

huc_col = [
    c
    for c in huc2.columns
    if c.lower() == "huc2"
][0]

name_candidates = [
    c
    for c in huc2.columns
    if c.lower()
    in [
        "name",
        "name2",
        "huc2_name",
        "huc2name",
    ]
]

name_col = (
    name_candidates[0]
    if name_candidates
    else None
)

huc2[huc_col] = (
    huc2[huc_col]
    .astype(str)
    .str.zfill(2)
)

huc2 = huc2[
    ~huc2[huc_col]
    .isin(HUC_EXCLUDE)
].copy()

huc2 = huc2.to_crs(
    "EPSG:5070"
)

if name_col:

    HUC_NAMES = dict(
        zip(
            huc2[huc_col],
            huc2[name_col]
        )
    )

else:

    HUC_NAMES = {
        h: h
        for h
        in huc2[huc_col]
    }

print(
    "HUC2 regions:",
    len(huc2)
)


# ============================================================
# FACILITY HUC2 ASSIGNMENT
# ============================================================


def assign_facility_huc2(df):
    """
    Assign facility rows to HUC2 using row-wise coordinate fallback.

    Coordinate schemas:
      AI/Power -> Latitude, Longitude
      TRI      -> 12. LATITUDE, 13. LONGITUDE

    Standard coordinates are used where available.
    TRI coordinates are used as fallback where standard
    coordinates are missing.
    """

    d = df.copy()

    # --------------------------------------------------------
    # Standard latitude
    # --------------------------------------------------------

    if "Latitude" in d.columns:
        lat_standard = pd.to_numeric(
            d["Latitude"],
            errors="coerce"
        )
    elif "latitude" in d.columns:
        lat_standard = pd.to_numeric(
            d["latitude"],
            errors="coerce"
        )
    elif "lat" in d.columns:
        lat_standard = pd.to_numeric(
            d["lat"],
            errors="coerce"
        )
    else:
        lat_standard = pd.Series(
            np.nan,
            index=d.index
        )

    # --------------------------------------------------------
    # Standard longitude
    # --------------------------------------------------------

    if "Longitude" in d.columns:
        lon_standard = pd.to_numeric(
            d["Longitude"],
            errors="coerce"
        )
    elif "longitude" in d.columns:
        lon_standard = pd.to_numeric(
            d["longitude"],
            errors="coerce"
        )
    elif "lon" in d.columns:
        lon_standard = pd.to_numeric(
            d["lon"],
            errors="coerce"
        )
    else:
        lon_standard = pd.Series(
            np.nan,
            index=d.index
        )

    # --------------------------------------------------------
    # TRI latitude
    # --------------------------------------------------------

    if "12. LATITUDE" in d.columns:
        lat_tri = pd.to_numeric(
            d["12. LATITUDE"],
            errors="coerce"
        )
    else:
        lat_tri = pd.Series(
            np.nan,
            index=d.index
        )

    # --------------------------------------------------------
    # TRI longitude
    # --------------------------------------------------------

    if "13. LONGITUDE" in d.columns:
        lon_tri = pd.to_numeric(
            d["13. LONGITUDE"],
            errors="coerce"
        )
    else:
        lon_tri = pd.Series(
            np.nan,
            index=d.index
        )

    # --------------------------------------------------------
    # ROW-WISE FALLBACK
    # --------------------------------------------------------

    d["_lat"] = lat_standard.fillna(lat_tri)
    d["_lon"] = lon_standard.fillna(lon_tri)

    d = d[
        d["_lat"].notna()
        & d["_lon"].notna()
    ].copy()

    # --------------------------------------------------------
    # POINT GEOMETRY
    # --------------------------------------------------------

    g = gpd.GeoDataFrame(
        d,
        geometry=gpd.points_from_xy(
            d["_lon"],
            d["_lat"]
        ),
        crs="EPSG:4326"
    )

    g = g.to_crs("EPSG:5070")

    # --------------------------------------------------------
    # HUC2 SPATIAL JOIN
    # --------------------------------------------------------

    joined = gpd.sjoin(
        g,
        huc2[
            [
                huc_col,
                "geometry"
            ]
        ],
        how="left",
        predicate="within"
    )

    joined["HUC2"] = joined[huc_col]

    return pd.DataFrame(
        joined.drop(
            columns=[
                "geometry",
                "index_right"
            ],
            errors="ignore"
        )
    )


# ============================================================
# 1. MAJOR RIVER PROXIMITY
# ============================================================

heading(
    "PART 1 — MAJOR-RIVER PROXIMITY"
)

river_frames = []

for sector, path in RIVER_FILES.items():

    d = pd.read_csv(
        path,
        low_memory=False
    )

    if "dist_river_km" not in d.columns:

        raise RuntimeError(
            f"{sector}: "
            "dist_river_km missing"
        )

    input_n = len(d)

    d["sector"] = sector

    d = assign_facility_huc2(d)

    print(
        f"{sector}: "
        f"{input_n:,} input -> "
        f"{d['HUC2'].notna().sum():,} "
        "HUC2 assigned"
    )

    river_frames.append(d)


river = pd.concat(
    river_frames,
    ignore_index=True
)

river["dist_river_km"] = (
    pd.to_numeric(
        river["dist_river_km"],
        errors="coerce"
    )
)


river_desc_rows = []
river_test_rows = []


for h in sorted(
    river["HUC2"]
    .dropna()
    .unique()
):

    sub = river[
        river["HUC2"] == h
    ]

    for sector in SECTORS:

        x = (
            sub.loc[
                sub["sector"] == sector,
                "dist_river_km"
            ]
            .dropna()
        )

        river_desc_rows.append({

            "HUC2": h,

            "Region":
                HUC_NAMES.get(
                    h,
                    h
                ),

            "sector":
                sector,

            "N":
                len(x),

            "median_river_km":
                x.median()
                if len(x)
                else np.nan,

            "within_10km_pct":
                100
                * (x <= 10).mean()
                if len(x)
                else np.nan,

            "within_25km_pct":
                100
                * (x <= 25).mean()
                if len(x)
                else np.nan,

            "within_50km_pct":
                100
                * (x <= 50).mean()
                if len(x)
                else np.nan,
        })


    for comparator in COMPARATORS:

        x = (
            sub.loc[
                sub["sector"] == "AI",
                "dist_river_km"
            ]
            .dropna()
            .to_numpy()
        )

        y = (
            sub.loc[
                sub["sector"] == comparator,
                "dist_river_km"
            ]
            .dropna()
            .to_numpy()
        )

        if (
            len(x) >= 2
            and
            len(y) >= 2
        ):

            mw = mannwhitneyu(
                x,
                y,
                alternative="two-sided"
            )

            ks = ks_2samp(
                x,
                y,
                alternative="two-sided"
            )

            river_test_rows.append({

                "HUC2": h,

                "Region":
                    HUC_NAMES.get(
                        h,
                        h
                    ),

                "comparison":
                    f"AI_vs_{comparator}",

                "AI_N":
                    len(x),

                "Comparator_N":
                    len(y),

                "AI_median_km":
                    np.median(x),

                "Comparator_median_km":
                    np.median(y),

                "median_difference_AI_minus_comparator_km":
                    np.median(x)
                    -
                    np.median(y),

                "MWU_p":
                    mw.pvalue,

                "KS_D":
                    ks.statistic,

                "KS_p":
                    ks.pvalue,

                "Cliffs_delta":
                    cliffs_delta(
                        x,
                        y
                    ),
            })


river_desc = pd.DataFrame(
    river_desc_rows
)

river_tests = pd.DataFrame(
    river_test_rows
)


for comparator in COMPARATORS:

    comp = (
        f"AI_vs_{comparator}"
    )

    idx = (
        river_tests["comparison"]
        == comp
    )

    river_tests.loc[
        idx,
        "MWU_q_FDR"
    ] = bh_fdr(
        river_tests.loc[
            idx,
            "MWU_p"
        ].values
    )

    river_tests.loc[
        idx,
        "KS_q_FDR"
    ] = bh_fdr(
        river_tests.loc[
            idx,
            "KS_p"
        ].values
    )


river_tests[
    "significant_FDR"
] = (
    river_tests["MWU_q_FDR"]
    < ALPHA
)


# ============================================================
# 2. HYDROLOGIC REGIME
# ============================================================

heading(
    "PART 2 — HYDROLOGIC REGIME"
)

hyd = pd.read_csv(
    HYDRO_MASTER
)

bas = gpd.read_file(
    HYBAS
)

hyd["HYBAS_ID"] = (
    pd.to_numeric(
        hyd["HYBAS_ID"],
        errors="coerce"
    )
    .astype("Int64")
)

bas["HYBAS_ID"] = (
    pd.to_numeric(
        bas["HYBAS_ID"],
        errors="coerce"
    )
    .astype("Int64")
)

bas = bas[
    bas["HYBAS_ID"]
    .isin(
        hyd["HYBAS_ID"]
    )
].copy()

bas = bas.to_crs(
    "EPSG:5070"
)

rep = bas[
    [
        "HYBAS_ID",
        "geometry"
    ]
].copy()

rep["geometry"] = (
    rep.geometry
    .representative_point()
)

hj = gpd.sjoin(
    rep,
    huc2[
        [
            huc_col,
            "geometry"
        ]
    ],
    how="left",
    predicate="within"
)

hj = (
    hj[
        [
            "HYBAS_ID",
            huc_col
        ]
    ]
    .drop_duplicates(
        "HYBAS_ID"
    )
    .rename(
        columns={
            huc_col:
                "HUC2"
        }
    )
)

hyd = hyd.merge(
    hj,
    on="HYBAS_ID",
    how="left"
)

print(
    "Hydrologic master:",
    len(hyd)
)

print(
    "HUC2 assigned:",
    hyd["HUC2"]
    .notna()
    .sum()
)

print(
    "Unassigned:",
    hyd["HUC2"]
    .isna()
    .sum()
)


# Critical QC

unassigned = hyd[
    hyd["HUC2"].isna()
]

assert (
    int(
        unassigned[
            "AI_present"
        ].sum()
    )
    == 0
)

assert (
    int(
        unassigned[
            "Power_present"
        ].sum()
    )
    == 0
)

assert (
    int(
        unassigned[
            "TRI_present"
        ].sum()
    )
    == 0
)


hyd_desc_rows = []
hyd_test_rows = []


for h in sorted(
    hyd["HUC2"]
    .dropna()
    .unique()
):

    sub = hyd[
        hyd["HUC2"] == h
    ]

    for metric in HYDRO_METRICS:

        for sector, flag in [

            (
                "AI",
                "AI_present"
            ),

            (
                "Power",
                "Power_present"
            ),

            (
                "TRI",
                "TRI_present"
            ),
        ]:

            x = (
                pd.to_numeric(
                    sub.loc[
                        sub[flag] == 1,
                        metric
                    ],
                    errors="coerce"
                )
                .dropna()
            )

            hyd_desc_rows.append({

                "HUC2":
                    h,

                "Region":
                    HUC_NAMES.get(
                        h,
                        h
                    ),

                "metric":
                    metric,

                "sector":
                    sector,

                "N":
                    len(x),

                "median":
                    x.median()
                    if len(x)
                    else np.nan,

                "mean":
                    x.mean()
                    if len(x)
                    else np.nan,
            })


        for comparator, flag in [

            (
                "Power",
                "Power_present"
            ),

            (
                "TRI",
                "TRI_present"
            ),
        ]:

            x = (
                pd.to_numeric(
                    sub.loc[
                        sub[
                            "AI_present"
                        ] == 1,
                        metric
                    ],
                    errors="coerce"
                )
                .dropna()
                .to_numpy()
            )

            y = (
                pd.to_numeric(
                    sub.loc[
                        sub[flag] == 1,
                        metric
                    ],
                    errors="coerce"
                )
                .dropna()
                .to_numpy()
            )

            if (
                len(x) >= 2
                and
                len(y) >= 2
            ):

                mw = mannwhitneyu(
                    x,
                    y,
                    alternative="two-sided"
                )

                hyd_test_rows.append({

                    "HUC2":
                        h,

                    "Region":
                        HUC_NAMES.get(
                            h,
                            h
                        ),

                    "metric":
                        metric,

                    "comparison":
                        f"AI_vs_{comparator}",

                    "AI_N":
                        len(x),

                    "Comparator_N":
                        len(y),

                    "AI_median":
                        np.median(x),

                    "Comparator_median":
                        np.median(y),

                    "median_difference_AI_minus_comparator":
                        np.median(x)
                        -
                        np.median(y),

                    "MWU_p":
                        mw.pvalue,

                    "Cliffs_delta":
                        cliffs_delta(
                            x,
                            y
                        ),
                })


hyd_desc = pd.DataFrame(
    hyd_desc_rows
)

hyd_tests = pd.DataFrame(
    hyd_test_rows
)


for metric in HYDRO_METRICS:

    for comparator in COMPARATORS:

        comp = (
            f"AI_vs_{comparator}"
        )

        idx = (
            (
                hyd_tests["metric"]
                == metric
            )
            &
            (
                hyd_tests[
                    "comparison"
                ]
                == comp
            )
        )

        hyd_tests.loc[
            idx,
            "MWU_q_FDR"
        ] = bh_fdr(
            hyd_tests.loc[
                idx,
                "MWU_p"
            ].values
        )


hyd_tests[
    "significant_FDR"
] = (
    hyd_tests["MWU_q_FDR"]
    < ALPHA
)


# ============================================================
# 3. WATER STRESS — CORRECTED
# ============================================================

heading(
    "PART 3 — WATER STRESS "
    "(CORRECTED LOWERCASE TERTILES)"
)

stress = pd.read_csv(
    STRESS_FILE,
    low_memory=False
)

assert (
    "stress_tertile"
    in stress.columns
)

assert (
    "sector"
    in stress.columns
)


# ------------------------------------------------------------
# CRITICAL FIX:
# manuscript file stores low/medium/high in lowercase
# ------------------------------------------------------------

stress[
    "stress_tertile"
] = (
    stress[
        "stress_tertile"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)


valid_stress_labels = {
    "low",
    "medium",
    "high"
}

observed_labels = set(
    stress[
        "stress_tertile"
    ].dropna().unique()
)

assert (
    observed_labels
    <= valid_stress_labels
), (
    "Unexpected stress labels: "
    f"{observed_labels}"
)


print(
    "\nRaw sector counts:"
)

print(
    stress[
        "sector"
    ].value_counts()
)

print(
    "\nStress categories:"
)

print(
    stress[
        "stress_tertile"
    ].value_counts()
)


stress = assign_facility_huc2(
    stress
)


print(
    "\nHUC2-assigned stress "
    "records by sector:"
)

print(
    stress.loc[
        stress["HUC2"].notna(),
        "sector"
    ].value_counts()
)


stress_desc_rows = []
stress_test_rows = []


for h in sorted(
    stress["HUC2"]
    .dropna()
    .unique()
):

    sub = stress[
        stress["HUC2"] == h
    ]

    for sector in SECTORS:

        s = sub[
            sub["sector"]
            == sector
        ]

        counts = (
            s[
                "stress_tertile"
            ]
            .value_counts()
        )

        n = len(s)

        low_n = int(
            counts.get(
                "low",
                0
            )
        )

        medium_n = int(
            counts.get(
                "medium",
                0
            )
        )

        high_n = int(
            counts.get(
                "high",
                0
            )
        )

        stress_desc_rows.append({

            "HUC2":
                h,

            "Region":
                HUC_NAMES.get(
                    h,
                    h
                ),

            "sector":
                sector,

            "N":
                n,

            "Low_N":
                low_n,

            "Medium_N":
                medium_n,

            "High_N":
                high_n,

            "Low_pct":
                (
                    100
                    * low_n
                    / n
                    if n
                    else np.nan
                ),

            "Medium_pct":
                (
                    100
                    * medium_n
                    / n
                    if n
                    else np.nan
                ),

            "High_pct":
                (
                    100
                    * high_n
                    / n
                    if n
                    else np.nan
                ),
        })


    for comparator in COMPARATORS:

        a = sub[
            sub["sector"] == "AI"
        ]

        b = sub[
            sub["sector"]
            == comparator
        ]

        if (
            len(a) == 0
            or
            len(b) == 0
        ):
            continue


        categories = [
            "low",
            "medium",
            "high",
        ]


        table = np.array([

            [
                int(
                    (
                        a[
                            "stress_tertile"
                        ]
                        == c
                    ).sum()
                )
                for c
                in categories
            ],

            [
                int(
                    (
                        b[
                            "stress_tertile"
                        ]
                        == c
                    ).sum()
                )
                for c
                in categories
            ],
        ])


        keep = (
            table.sum(
                axis=0
            )
            > 0
        )

        table_use = (
            table[:, keep]
        )


        full_chi2 = np.nan
        full_p = np.nan
        full_v = np.nan


        if (
            table_use.shape[1]
            >= 2
        ):

            try:

                method = MonteCarloMethod(
                    n_resamples=100000,
                    rng=np.random.default_rng(42)
                )

                result = fisher_exact(
                    table_use,
                    method=method
                )

                full_p = result.pvalue

                full_v = cramers_v(
                    table_use
                )

            except ValueError:

                pass


        # -----------------------------------------------
        # High vs not-high:
        # exact test useful when regional AI n is small
        # -----------------------------------------------

        ai_high = int(
            (
                a[
                    "stress_tertile"
                ]
                == "high"
            ).sum()
        )

        comp_high = int(
            (
                b[
                    "stress_tertile"
                ]
                == "high"
            ).sum()
        )

        ai_not_high = (
            len(a)
            - ai_high
        )

        comp_not_high = (
            len(b)
            - comp_high
        )

        table2 = np.array([

            [
                ai_high,
                ai_not_high
            ],

            [
                comp_high,
                comp_not_high
            ],
        ])


        odds_ratio, fisher_p = (
            fisher_exact(
                table2,
                alternative="two-sided"
            )
        )


        stress_test_rows.append({

            "HUC2":
                h,

            "Region":
                HUC_NAMES.get(
                    h,
                    h
                ),

            "comparison":
                f"AI_vs_{comparator}",

            "AI_N":
                len(a),

            "Comparator_N":
                len(b),

            "AI_high_N":
                ai_high,

            "AI_high_pct":
                (
                    100
                    * ai_high
                    / len(a)
                ),

            "Comparator_high_N":
                comp_high,

            "Comparator_high_pct":
                (
                    100
                    * comp_high
                    / len(b)
                ),

            "high_pct_difference_AI_minus_comparator":
                (
                    100
                    * ai_high
                    / len(a)
                )
                -
                (
                    100
                    * comp_high
                    / len(b)
                ),

            "full_distribution_chi2":
                full_chi2,

            "full_distribution_p":
                full_p,

            "full_distribution_CramersV":
                full_v,

            "high_vs_not_high_odds_ratio":
                odds_ratio,

            "high_vs_not_high_Fisher_p":
                fisher_p,
        })


stress_desc = pd.DataFrame(
    stress_desc_rows
)

stress_tests = pd.DataFrame(
    stress_test_rows
)


# Confirm BOTH comparisons exist

stress_comparisons = set(
    stress_tests[
        "comparison"
    ].unique()
)

assert (
    "AI_vs_Power"
    in stress_comparisons
)

assert (
    "AI_vs_TRI"
    in stress_comparisons
)


for comparator in COMPARATORS:

    comp = (
        f"AI_vs_{comparator}"
    )

    idx = (
        stress_tests[
            "comparison"
        ]
        == comp
    )


    stress_tests.loc[
        idx,
        "full_distribution_q_FDR"
    ] = bh_fdr(
        stress_tests.loc[
            idx,
            "full_distribution_p"
        ].values
    )


    stress_tests.loc[
        idx,
        "high_vs_not_high_q_FDR"
    ] = bh_fdr(
        stress_tests.loc[
            idx,
            "high_vs_not_high_Fisher_p"
        ].values
    )


stress_tests[
    "full_distribution_significant_FDR"
] = (
    stress_tests[
        "full_distribution_q_FDR"
    ]
    < ALPHA
)


stress_tests[
    "high_stress_significant_FDR"
] = (
    stress_tests[
        "high_vs_not_high_q_FDR"
    ]
    < ALPHA
)


# ============================================================
# 4. PUBLIC SUPPLY — CORRECTED
# ============================================================

heading(
    "PART 4 — PUBLIC-SUPPLY SOURCE COMPOSITION"
)

supply = pd.read_csv(
    SUPPLY_FILE,
    low_memory=False
)


for c in [
    "sector",
    "PS_sw_fraction",
    "PS_gw_fraction",
]:

    assert c in supply.columns


print(
    "\nRaw supply sector counts:"
)

print(
    supply[
        "sector"
    ].value_counts()
)


supply[
    "PS_sw_fraction"
] = pd.to_numeric(
    supply[
        "PS_sw_fraction"
    ],
    errors="coerce"
)

supply[
    "PS_gw_fraction"
] = pd.to_numeric(
    supply[
        "PS_gw_fraction"
    ],
    errors="coerce"
)


sw = supply[
    "PS_sw_fraction"
]

gw = supply[
    "PS_gw_fraction"
]


# ------------------------------------------------------------
# Preserve missing/equal values as Unclassified.
# ------------------------------------------------------------

supply[
    "public_supply_source"
] = "Unclassified"


valid_source = (
    sw.notna()
    &
    gw.notna()
)


supply.loc[
    valid_source
    &
    (sw > gw),
    "public_supply_source"
] = "Surface water dominated"


supply.loc[
    valid_source
    &
    (gw > sw),
    "public_supply_source"
] = "Groundwater dominated"


print(
    "\nSource classification "
    "by sector:"
)

print(
    pd.crosstab(
        supply["sector"],
        supply[
            "public_supply_source"
        ]
    )
)


supply = assign_facility_huc2(
    supply
)


print(
    "\nHUC2-assigned supply "
    "records by sector:"
)

print(
    supply.loc[
        supply["HUC2"].notna(),
        "sector"
    ].value_counts()
)


supply_desc_rows = []
supply_test_rows = []


categories = [

    "Surface water dominated",

    "Groundwater dominated",

    "Unclassified",
]


for h in sorted(
    supply["HUC2"]
    .dropna()
    .unique()
):

    sub = supply[
        supply["HUC2"] == h
    ]


    for sector in SECTORS:

        s = sub[
            sub["sector"]
            == sector
        ]

        n = len(s)

        counts = (
            s[
                "public_supply_source"
            ]
            .value_counts()
        )

        sw_n = int(
            counts.get(
                "Surface water dominated",
                0
            )
        )

        gw_n = int(
            counts.get(
                "Groundwater dominated",
                0
            )
        )

        un_n = int(
            counts.get(
                "Unclassified",
                0
            )
        )


        supply_desc_rows.append({

            "HUC2":
                h,

            "Region":
                HUC_NAMES.get(
                    h,
                    h
                ),

            "sector":
                sector,

            "N":
                n,

            "Surface_N":
                sw_n,

            "Groundwater_N":
                gw_n,

            "Unclassified_N":
                un_n,

            "Surface_pct":
                (
                    100
                    * sw_n
                    / n
                    if n
                    else np.nan
                ),

            "Groundwater_pct":
                (
                    100
                    * gw_n
                    / n
                    if n
                    else np.nan
                ),

            "Unclassified_pct":
                (
                    100
                    * un_n
                    / n
                    if n
                    else np.nan
                ),
        })


    for comparator in COMPARATORS:

        a = sub[
            sub["sector"]
            == "AI"
        ]

        b = sub[
            sub["sector"]
            == comparator
        ]

        if (
            len(a) == 0
            or
            len(b) == 0
        ):
            continue


        table = np.array([

            [
                int(
                    (
                        a[
                            "public_supply_source"
                        ]
                        == c
                    ).sum()
                )
                for c
                in categories
            ],

            [
                int(
                    (
                        b[
                            "public_supply_source"
                        ]
                        == c
                    ).sum()
                )
                for c
                in categories
            ],
        ])


        keep = (
            table.sum(
                axis=0
            )
            > 0
        )

        table_use = (
            table[:, keep]
        )


        chi2 = np.nan
        p = np.nan
        v = np.nan


        if (
            table_use.shape[1]
            >= 2
        ):

            try:

                method = MonteCarloMethod(
                    n_resamples=100000,
                    rng=np.random.default_rng(42)
                )

                result = fisher_exact(
                    table_use,
                    method=method
                )

                p = result.pvalue

                v = cramers_v(
                    table_use
                )

            except ValueError:

                pass


        ai_sw = int(
            (
                a[
                    "public_supply_source"
                ]
                ==
                "Surface water dominated"
            ).sum()
        )


        comp_sw = int(
            (
                b[
                    "public_supply_source"
                ]
                ==
                "Surface water dominated"
            ).sum()
        )


        supply_test_rows.append({

            "HUC2":
                h,

            "Region":
                HUC_NAMES.get(
                    h,
                    h
                ),

            "comparison":
                f"AI_vs_{comparator}",

            "AI_N":
                len(a),

            "Comparator_N":
                len(b),

            "AI_surface_pct":
                (
                    100
                    * ai_sw
                    / len(a)
                ),

            "Comparator_surface_pct":
                (
                    100
                    * comp_sw
                    / len(b)
                ),

            "surface_pct_difference_AI_minus_comparator":
                (
                    100
                    * ai_sw
                    / len(a)
                )
                -
                (
                    100
                    * comp_sw
                    / len(b)
                ),

            "chi2":
                chi2,

            "p":
                p,

            "Cramers_V":
                v,
        })


supply_desc = pd.DataFrame(
    supply_desc_rows
)

supply_tests = pd.DataFrame(
    supply_test_rows
)


# Confirm BOTH comparisons exist

supply_comparisons = set(
    supply_tests[
        "comparison"
    ].unique()
)

assert (
    "AI_vs_Power"
    in supply_comparisons
)

assert (
    "AI_vs_TRI"
    in supply_comparisons
)


for comparator in COMPARATORS:

    comp = (
        f"AI_vs_{comparator}"
    )

    idx = (
        supply_tests[
            "comparison"
        ]
        == comp
    )

    supply_tests.loc[
        idx,
        "q_FDR"
    ] = bh_fdr(
        supply_tests.loc[
            idx,
            "p"
        ].values
    )


supply_tests[
    "significant_FDR"
] = (
    supply_tests[
        "q_FDR"
    ]
    < ALPHA
)


# ============================================================
# SAVE COMPONENT TABLES
# ============================================================

heading(
    "SAVE COMPONENT TABLES"
)


river_desc.to_csv(
    OUT
    / "01_river_HUC2_descriptives.csv",
    index=False
)

river_tests.to_csv(
    OUT
    / "01_river_HUC2_tests.csv",
    index=False
)


hyd_desc.to_csv(
    OUT
    / "02_hydrologic_HUC2_descriptives.csv",
    index=False
)

hyd_tests.to_csv(
    OUT
    / "02_hydrologic_HUC2_tests.csv",
    index=False
)


stress_desc.to_csv(
    OUT
    / "03_stress_HUC2_descriptives.csv",
    index=False
)

stress_tests.to_csv(
    OUT
    / "03_stress_HUC2_tests.csv",
    index=False
)


supply_desc.to_csv(
    OUT
    / "04_supply_HUC2_descriptives.csv",
    index=False
)

supply_tests.to_csv(
    OUT
    / "04_supply_HUC2_tests.csv",
    index=False
)


# ============================================================
# DEFINITIVE DESCRIPTIVE MASTER TABLE
# ============================================================

heading(
    "BUILD DEFINITIVE HUC2 TABLE"
)


regions = pd.DataFrame({

    "HUC2":
        sorted(
            river[
                "HUC2"
            ]
            .dropna()
            .unique()
        )
})


regions[
    "Region"
] = regions[
    "HUC2"
].map(
    HUC_NAMES
)


master = regions.copy()


# ------------------------------------------------------------
# River
# ------------------------------------------------------------

for sector in SECTORS:

    temp = (
        river_desc[
            river_desc["sector"]
            == sector
        ][
            [
                "HUC2",
                "N",
                "median_river_km",
                "within_10km_pct",
                "within_25km_pct",
            ]
        ]
        .copy()
    )

    temp = temp.rename(
        columns={

            "N":
                f"river_N_{sector}",

            "median_river_km":
                f"river_median_km_{sector}",

            "within_10km_pct":
                f"river_within10_pct_{sector}",

            "within_25km_pct":
                f"river_within25_pct_{sector}",
        }
    )

    master = master.merge(
        temp,
        on="HUC2",
        how="left"
    )


# ------------------------------------------------------------
# Hydrologic metrics
# ------------------------------------------------------------

for metric in HYDRO_METRICS:

    for sector in SECTORS:

        temp = (
            hyd_desc[
                (
                    hyd_desc["metric"]
                    == metric
                )
                &
                (
                    hyd_desc["sector"]
                    == sector
                )
            ][
                [
                    "HUC2",
                    "N",
                    "median"
                ]
            ]
            .copy()
        )

        temp = temp.rename(
            columns={

                "N":
                    f"{metric}_N_{sector}",

                "median":
                    f"{metric}_median_{sector}",
            }
        )

        master = master.merge(
            temp,
            on="HUC2",
            how="left"
        )


# ------------------------------------------------------------
# Stress
# ------------------------------------------------------------

for sector in SECTORS:

    temp = (
        stress_desc[
            stress_desc["sector"]
            == sector
        ][
            [
                "HUC2",
                "N",
                "Low_pct",
                "Medium_pct",
                "High_pct",
            ]
        ]
        .copy()
    )

    temp = temp.rename(
        columns={

            "N":
                f"stress_N_{sector}",

            "Low_pct":
                f"stress_low_pct_{sector}",

            "Medium_pct":
                f"stress_medium_pct_{sector}",

            "High_pct":
                f"stress_high_pct_{sector}",
        }
    )

    master = master.merge(
        temp,
        on="HUC2",
        how="left"
    )


# ------------------------------------------------------------
# Supply
# ------------------------------------------------------------

for sector in SECTORS:

    temp = (
        supply_desc[
            supply_desc["sector"]
            == sector
        ][
            [
                "HUC2",
                "N",
                "Surface_pct",
                "Groundwater_pct",
                "Unclassified_pct",
            ]
        ]
        .copy()
    )

    temp = temp.rename(
        columns={

            "N":
                f"supply_N_{sector}",

            "Surface_pct":
                f"supply_surface_pct_{sector}",

            "Groundwater_pct":
                f"supply_groundwater_pct_{sector}",

            "Unclassified_pct":
                f"supply_unclassified_pct_{sector}",
        }
    )

    master = master.merge(
        temp,
        on="HUC2",
        how="left"
    )


master = master.sort_values(
    "HUC2"
)


master.to_csv(
    OUT
    / "05_DEFINITIVE_HUC2_ALL_PARAMETERS_CORRECTED.csv",
    index=False
)


# ============================================================
# ALL INFERENTIAL RESULTS
# ============================================================

heading(
    "BUILD ALL-TEST SUMMARY"
)


all_rows = []


# River

for _, r in river_tests.iterrows():

    all_rows.append({

        "parameter":
            "Major-river proximity",

        "metric":
            "dist_river_km",

        "HUC2":
            r["HUC2"],

        "Region":
            r["Region"],

        "comparison":
            r["comparison"],

        "p":
            r["MWU_p"],

        "q_FDR":
            r["MWU_q_FDR"],

        "effect_size":
            r["Cliffs_delta"],

        "effect_size_type":
            "Cliffs_delta",

        "significant_FDR":
            r["significant_FDR"],
    })


# Hydrologic

for _, r in hyd_tests.iterrows():

    all_rows.append({

        "parameter":
            "Hydrologic regime",

        "metric":
            r["metric"],

        "HUC2":
            r["HUC2"],

        "Region":
            r["Region"],

        "comparison":
            r["comparison"],

        "p":
            r["MWU_p"],

        "q_FDR":
            r["MWU_q_FDR"],

        "effect_size":
            r["Cliffs_delta"],

        "effect_size_type":
            "Cliffs_delta",

        "significant_FDR":
            r["significant_FDR"],
    })


# Stress — full three-category distribution

for _, r in stress_tests.iterrows():

    all_rows.append({

        "parameter":
            "Water stress",

        "metric":
            "Low/Medium/High distribution",

        "HUC2":
            r["HUC2"],

        "Region":
            r["Region"],

        "comparison":
            r["comparison"],

        "p":
            r["full_distribution_p"],

        "q_FDR":
            r[
                "full_distribution_q_FDR"
            ],

        "effect_size":
            r[
                "full_distribution_CramersV"
            ],

        "effect_size_type":
            "Cramers_V",

        "significant_FDR":
            r[
                "full_distribution_significant_FDR"
            ],
    })


# Stress — high vs not-high

for _, r in stress_tests.iterrows():

    all_rows.append({

        "parameter":
            "Water stress",

        "metric":
            "High vs not-high",

        "HUC2":
            r["HUC2"],

        "Region":
            r["Region"],

        "comparison":
            r["comparison"],

        "p":
            r[
                "high_vs_not_high_Fisher_p"
            ],

        "q_FDR":
            r[
                "high_vs_not_high_q_FDR"
            ],

        "effect_size":
            r[
                "high_vs_not_high_odds_ratio"
            ],

        "effect_size_type":
            "Odds_ratio",

        "significant_FDR":
            r[
                "high_stress_significant_FDR"
            ],
    })


# Supply

for _, r in supply_tests.iterrows():

    all_rows.append({

        "parameter":
            "Public-supply pathways",

        "metric":
            "Source composition",

        "HUC2":
            r["HUC2"],

        "Region":
            r["Region"],

        "comparison":
            r["comparison"],

        "p":
            r["p"],

        "q_FDR":
            r["q_FDR"],

        "effect_size":
            r["Cramers_V"],

        "effect_size_type":
            "Cramers_V",

        "significant_FDR":
            r["significant_FDR"],
    })


all_tests = pd.DataFrame(
    all_rows
)


all_tests.to_csv(
    OUT
    / "06_ALL_HUC2_STATISTICAL_TESTS_CORRECTED.csv",
    index=False
)


significant = all_tests[
    all_tests[
        "significant_FDR"
    ] == True
].copy()


significant.to_csv(
    OUT
    / "07_SIGNIFICANT_HUC2_RESULTS_FDR_CORRECTED.csv",
    index=False
)


# ============================================================
# REVIEWER SUMMARY
# ============================================================

heading(
    "REVIEWER SUMMARY"
)


summary_rows = []


for parameter in [

    "Major-river proximity",

    "Hydrologic regime",

    "Water stress",

    "Public-supply pathways",
]:

    for comparator in COMPARATORS:

        comp = (
            f"AI_vs_{comparator}"
        )

        d = all_tests[
            (
                all_tests[
                    "parameter"
                ]
                == parameter
            )
            &
            (
                all_tests[
                    "comparison"
                ]
                == comp
            )
        ]


        sig = d[
            d[
                "significant_FDR"
            ] == True
        ]


        summary_rows.append({

            "parameter":
                parameter,

            "comparison":
                comp,

            "number_tests":
                len(d),

            "number_significant_FDR":
                len(sig),

            "significant_HUC2":
                "; ".join(
                    sorted(
                        sig[
                            "HUC2"
                        ]
                        .astype(str)
                        .unique()
                    )
                ),
        })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    OUT
    / "08_REVIEWER_READY_HUC2_SUMMARY_CORRECTED.csv",
    index=False
)


# ============================================================
# HARD QC
# ============================================================

heading(
    "FINAL HARD QC"
)


print(
    "\nRiver comparisons:"
)

print(
    river_tests[
        "comparison"
    ].value_counts()
)


print(
    "\nHydrologic comparisons:"
)

print(
    hyd_tests[
        "comparison"
    ].value_counts()
)


print(
    "\nStress comparisons:"
)

print(
    stress_tests[
        "comparison"
    ].value_counts()
)


print(
    "\nSupply comparisons:"
)

print(
    supply_tests[
        "comparison"
    ].value_counts()
)


# These MUST exist.

for comp in [
    "AI_vs_Power",
    "AI_vs_TRI",
]:

    assert (
        comp
        in set(
            stress_tests[
                "comparison"
            ]
        )
    )

    assert (
        comp
        in set(
            supply_tests[
                "comparison"
            ]
        )
    )


# Expected validated national facility populations
# after HUC2/CONUS filtering for stress.

stress_counts = (
    stress.loc[
        stress["HUC2"].notna()
    ]
    .groupby(
        "sector"
    )
    .size()
)


assert (
    stress_counts.get(
        "AI",
        0
    )
    == 1383
)

assert (
    stress_counts.get(
        "Power",
        0
    )
    == 15525
)

assert (
    stress_counts.get(
        "TRI",
        0
    )
    == 21024
)


# Hydrologic QC

assert (
    len(hyd)
    == 11022
)

assert (
    hyd[
        "HUC2"
    ].notna().sum()
    == 10970
)

assert (
    hyd[
        "HUC2"
    ].isna().sum()
    == 52
)


print(
    "\n[PASS] Stress population:"
)

print(
    stress_counts
)


print(
    "\n[PASS] Hydrologic population:"
)

print(
    "Total =",
    len(hyd)
)

print(
    "HUC2 assigned =",
    hyd[
        "HUC2"
    ].notna().sum()
)

print(
    "Unassigned =",
    hyd[
        "HUC2"
    ].isna().sum()
)


# ============================================================
# FINAL RESULTS
# ============================================================

heading(
    "SIGNIFICANT REGIONAL RESULTS AFTER BH-FDR"
)


if significant.empty:

    print(
        "No significant regional "
        "comparisons after FDR."
    )

else:

    print(
        significant[
            [
                "parameter",
                "metric",
                "HUC2",
                "Region",
                "comparison",
                "q_FDR",
                "effect_size",
                "effect_size_type",
            ]
        ]
        .sort_values(
            [
                "parameter",
                "metric",
                "comparison",
                "HUC2",
            ]
        )
        .to_string(
            index=False
        )
    )


heading(
    "REVIEWER-READY SUMMARY"
)


print(
    summary.to_string(
        index=False
    )
)


heading(
    "OUTPUT FILES"
)


print(
    OUT
    / "05_DEFINITIVE_HUC2_ALL_PARAMETERS_CORRECTED.csv"
)

print(
    OUT
    / "06_ALL_HUC2_STATISTICAL_TESTS_CORRECTED.csv"
)

print(
    OUT
    / "07_SIGNIFICANT_HUC2_RESULTS_FDR_CORRECTED.csv"
)

print(
    OUT
    / "08_REVIEWER_READY_HUC2_SUMMARY_CORRECTED.csv"
)


print(
    "\n"
    + "#" * 110
)

print(
    "FINAL STATUS: CORRECTED HUC2 "
    "REVIEWER ANALYSIS COMPLETE"
)

print(
    "#" * 110
)


# =============================================================================
# MANUSCRIPT HUC2 SUMMARY
# =============================================================================

def build_manuscript_huc2_summary(all_tests):
    """
    Summarize significant HUC2 regional comparisons for the manuscript table.
    BH-FDR significance is defined as q_FDR < 0.05.
    """

    rows = []

    specifications = [
        (
            "Major-river proximity",
            "Distance to major river",
            "Major-river proximity",
            "dist_river_km",
        ),
        (
            "Hydrologic signatures",
            "CVQ",
            "Hydrologic regime",
            "CVQ",
        ),
        (
            "Hydrologic signatures",
            "Seasonal concentration",
            "Hydrologic regime",
            "season_conc",
        ),
        (
            "Hydrologic signatures",
            "RBI",
            "Hydrologic regime",
            "RBI",
        ),
        (
            "Water stress",
            "Low/medium/high distribution",
            "Water stress",
            "Low/Medium/High distribution",
        ),
        (
            "Public supply pathways",
            "Source composition",
            "Public-supply pathways",
            "Source composition",
        ),
    ]

    for parameter_label, metric_label, parameter_key, metric_key in specifications:

        row = {
            "Parameter": parameter_label,
            "Metric": metric_label,
        }

        subset = all_tests[
            (all_tests["parameter"] == parameter_key)
            & (all_tests["metric"] == metric_key)
        ].copy()

        for comparison, column in [
            ("AI_vs_Power", "AI vs Power"),
            ("AI_vs_TRI", "AI vs TRI"),
        ]:

            comp = subset[
                subset["comparison"] == comparison
            ].copy()

            n_tested = len(comp)
            n_significant = int(
                (pd.to_numeric(comp["q_FDR"], errors="coerce") < 0.05).sum()
            )

            row[column] = f"{n_significant}/{n_tested}"

        rows.append(row)

    return pd.DataFrame(rows)


manuscript_summary = build_manuscript_huc2_summary(all_tests)

MANUSCRIPT_SUMMARY_FILE = (
    Path("final_results/huc2_reviewer_final") / "10_MANUSCRIPT_HUC2_REGIONAL_SUMMARY.csv"
)

manuscript_summary.to_csv(
    MANUSCRIPT_SUMMARY_FILE,
    index=False,
)


print("\n" + "=" * 110)
print("MANUSCRIPT HUC2 REGIONAL SUMMARY")
print("=" * 110)

print(
    manuscript_summary.to_string(
        index=False
    )
)

print("\n[SAVED]", MANUSCRIPT_SUMMARY_FILE)
