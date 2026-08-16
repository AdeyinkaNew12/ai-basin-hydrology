from PIL import Image
import os

indir = "/mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/ecdf_sector_basin_matched"

files = [
    "ECDF_dist_river_km_JOURNAL.png",
    "ECDF_dist_lake_km_JOURNAL.png",
    "ECDF_dist_coast_km_JOURNAL.png",
    "ECDF_dist_any_km_JOURNAL.png",
]

imgs = []

for f in files:
    path = os.path.join(indir, f)
    img = Image.open(path).convert("RGB")
    imgs.append(img)

# equal size
w = min(i.width for i in imgs)
h = min(i.height for i in imgs)

imgs = [i.resize((w,h)) for i in imgs]

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

for img, pos in zip(imgs, positions):
    canvas.paste(img, pos)

out = os.path.join(
    indir,
    "Figure2_WaterDistance_ECDF_FINAL.png"
)

canvas.save(out, dpi=(300,300))

print("Saved:", out)
