from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import Point


# ============================================================
# INPUTS AND OUTPUTS
# ============================================================
AI_CSV = Path(
    "/mnt/disk3/aoolaseinde/data/WaterProject/DC_CONUS.csv"
)

RIVERS_SHP = Path(
    "/mnt/disk3/aoolaseinde/data/WaterProject/"
    "HydroRIVERS_v10_na.shp"
)

COUNTIES_SHP = Path(
    "/mnt/disk3/aoolaseinde/data/Groundwater/"
    "tl_2019_us_county.shp"
)

OUTPUT_DIR = Path(
    "/mnt/disk3/aoolaseinde/projects/"
    "integrated_dc_water_pathways/results/"
    "texas_houston_ai_water"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = (
    OUTPUT_DIR
    / "TEXAS_HOUSTON_AI_DATA_CENTERS_MAJOR_RIVERS.png"
)


# ============================================================
# CHECK INPUT FILES
# ============================================================
for path in [AI_CSV, RIVERS_SHP, COUNTIES_SHP]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


# ============================================================
# CREATE EXACT TEXAS BOUNDARY FROM CENSUS COUNTIES
# Texas state FIPS code = 48
# ============================================================
counties = gpd.read_file(COUNTIES_SHP)

if "STATEFP" not in counties.columns:
    raise RuntimeError(
        "STATEFP field was not found in the county shapefile.\n"
        f"Available columns: {list(counties.columns)}"
    )

texas_counties = counties[
    counties["STATEFP"].astype(str).str.zfill(2) == "48"
].copy()

if texas_counties.empty:
    raise RuntimeError(
        "No Texas counties were found using STATEFP = 48."
    )

texas = texas_counties.dissolve().reset_index(drop=True)
texas = texas.to_crs("EPSG:5070")
texas_geometry = texas.geometry.iloc[0]

print(
    f"[INFO] Texas counties used: {len(texas_counties):,}"
)


# ============================================================
# LOAD THE 1,383-SITE AI DATASET
# ============================================================
ai = pd.read_csv(AI_CSV)

required_columns = {"Latitude", "Longitude"}
missing = required_columns.difference(ai.columns)

if missing:
    raise RuntimeError(
        f"Missing coordinate columns: {sorted(missing)}"
    )

ai["Latitude"] = pd.to_numeric(
    ai["Latitude"], errors="coerce"
)
ai["Longitude"] = pd.to_numeric(
    ai["Longitude"], errors="coerce"
)

ai = ai.dropna(
    subset=["Latitude", "Longitude"]
).copy()

ai_gdf = gpd.GeoDataFrame(
    ai,
    geometry=gpd.points_from_xy(
        ai["Longitude"],
        ai["Latitude"],
    ),
    crs="EPSG:4326",
).to_crs("EPSG:5070")

# Exact spatial filter using the Texas polygon
ai_texas = ai_gdf[
    ai_gdf.geometry.intersects(texas_geometry)
].copy()

print(f"[INFO] National AI sites: {len(ai_gdf):,}")
print(f"[INFO] Texas AI sites: {len(ai_texas):,}")


# ============================================================
# LOAD AND CLIP MAJOR RIVERS
# ============================================================
rivers = gpd.read_file(RIVERS_SHP)

if rivers.crs is None:
    raise RuntimeError(
        "The HydroRIVERS shapefile has no CRS."
    )

if "ORD_STRA" not in rivers.columns:
    raise RuntimeError(
        "ORD_STRA was not found in HydroRIVERS.\n"
        f"Available columns: {list(rivers.columns)}"
    )

rivers["ORD_STRA"] = pd.to_numeric(
    rivers["ORD_STRA"], errors="coerce"
)

major_rivers = rivers[
    rivers["ORD_STRA"] >= 5
].copy()

major_rivers = major_rivers.to_crs("EPSG:5070")

# Spatial index first, then exact clip
major_rivers = major_rivers[
    major_rivers.intersects(texas_geometry)
].copy()

major_rivers_texas = gpd.clip(
    major_rivers,
    texas,
)

print(
    f"[INFO] Major river reaches in Texas: "
    f"{len(major_rivers_texas):,}"
)


# ============================================================
# HOUSTON LOCATION AND REGIONAL HIGHLIGHT
# ============================================================
houston = gpd.GeoSeries(
    [Point(-95.3698, 29.7604)],
    crs="EPSG:4326",
).to_crs("EPSG:5070")

houston_point = houston.iloc[0]

# 125-km regional radius around Houston
houston_region = gpd.GeoSeries(
    [houston_point.buffer(90_000)],
    crs="EPSG:5070",
)

ai_houston = ai_texas[
    ai_texas.geometry.intersects(
        houston_region.iloc[0]
    )
].copy()

print(
    f"[INFO] AI sites within 125 km of Houston: "
    f"{len(ai_houston):,}"
)


# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(15, 10))

# Texas polygon
texas.plot(
    ax=ax,
    facecolor="#f2f1ed",
    edgecolor="#252525",
    linewidth=1.8,
    zorder=1,
)

# Major rivers
major_rivers_texas.plot(
    ax=ax,
    color="#3d8fc4",
    linewidth=1.50,
    alpha=0.80,
    zorder=2,
)

# All Texas AI sites
ai_texas.plot(
    ax=ax,
    color="#e36f2d",
    markersize=55,
    edgecolor="white",
    linewidth=0.45,
    alpha=0.92,
    zorder=4,
)

# Houston regional outline
houston_region.boundary.plot(
    ax=ax,
    color="#8f1d3f",
    linewidth=2.3,
    linestyle="--",
    zorder=3,
)

# Highlight Houston-area AI sites
if not ai_houston.empty:
    ai_houston.plot(
        ax=ax,
        color="#c5163a",
        markersize=105,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )

# Houston star
ax.scatter(
    houston_point.x,
    houston_point.y,
    marker="*",
    s=625,
    color="#111111",
    edgecolor="white",
    linewidth=1.2,
    zorder=6,
)

ax.annotate(
    "Houston",
    xy=(houston_point.x, houston_point.y),
    xytext=(34, 18),
    textcoords="offset points",
    fontsize=13,
    fontweight="bold",
    color="#111111",
    zorder=7,
)


# ============================================================
# TITLE AND LEGEND
# ============================================================

legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor="#e36f2d",
        markeredgecolor="white",
        markersize=9,
        label="AI data center",
    ),
    Line2D(
        [0], [0],
        color="#3d8fc4",
        linewidth=2,
        label="Major river",
    ),
    Line2D(
        [0], [0],
        marker="*",
        linestyle="none",
        markerfacecolor="#111111",
        markeredgecolor="white",
        markersize=13,
        label="Houston",
    ),
    Line2D(
        [0], [0],
        color="#8f1d3f",
        linestyle="--",
        linewidth=2,
        label="90-km Houston region",
    ),
]

ax.legend(
    handles=legend_handles,
    loc="lower left",
    frameon=False,
    fontsize=12,
)



ax.set_axis_off()

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=350,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print(f"✅ Saved: {OUTPUT_PNG}")
