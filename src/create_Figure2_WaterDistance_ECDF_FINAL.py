from PIL import Image, ImageDraw, ImageFont
import os

indir = "/mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/ecdf_sector_basin_matched"

panels = [
    ("A. Distance to major rivers", "ECDF_dist_river_km_JOURNAL.png"),
    ("B. Distance to lakes/reservoirs", "ECDF_dist_lake_km_JOURNAL.png"),
    ("C. Distance to coastline", "ECDF_dist_coast_km_JOURNAL.png"),
    ("D. Distance to nearest surface water", "ECDF_dist_any_km_JOURNAL.png"),
]

imgs=[]

font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

for title, fname in panels:
    img = Image.open(os.path.join(indir,fname)).convert("RGB")

    canvas = Image.new("RGB",(img.width,img.height+90),"white")
    canvas.paste(img,(0,90))

    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(font_path,36)

    draw.text((40,25),title,fill="black",font=font)

    imgs.append(canvas)

w=max(i.width for i in imgs)
h=max(i.height for i in imgs)

fig=Image.new("RGB",(2*w,2*h),"white")

positions=[
    (0,0),
    (w,0),
    (0,h),
    (w,h)
]

for img,pos in zip(imgs,positions):
    fig.paste(img,pos)

out=os.path.join(
    indir,
    "Figure2_WaterDistance_ECDF_FINAL.png"
)

fig.save(out,dpi=(300,300))

print("Saved:",out)
