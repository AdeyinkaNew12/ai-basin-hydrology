#!/usr/bin/env python3

from PIL import Image
from pathlib import Path

BASE = Path(
"/mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/ecdf_sector_basin_matched"
)

OUT = BASE / "Figure2_WaterDistance_ECDF_FINAL.png"

files = [
    "ECDF_dist_river_km_JOURNAL.png",
    "ECDF_dist_lake_km_JOURNAL.png",
    "ECDF_dist_coast_km_JOURNAL.png",
    "ECDF_dist_any_km_JOURNAL.png",
]

imgs = []

for f in files:
    path = BASE / f
    print("Loading:", path)

    img = Image.open(path).convert("RGB")
    imgs.append(img)

# Match sizes
w = max(i.width for i in imgs)
h = max(i.height for i in imgs)

canvas = Image.new(
    "RGB",
    (2*w, 2*h),
    "white"
)

positions = [
    (0,0),
    (w,0),
    (0,h),
    (w,h)
]

for img,pos in zip(imgs,positions):
    canvas.paste(img.resize((w,h)), pos)

canvas.save(
    OUT,
    dpi=(300,300)
)

print("\nSaved:")
print(OUT)
