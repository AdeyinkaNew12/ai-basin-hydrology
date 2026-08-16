from PIL import Image, ImageDraw, ImageFont
import os

indir = "/mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/ecdf_sector_basin_matched"

files = [
    ("A. Distance to major rivers",
     "ECDF_dist_river_km_JOURNAL.png"),

    ("B. Distance to lakes/reservoirs",
     "ECDF_dist_lake_km_JOURNAL.png"),

    ("C. Distance to coastline",
     "ECDF_dist_coast_km_JOURNAL.png"),

    ("D. Distance to nearest surface water",
     "ECDF_dist_any_km_JOURNAL.png"),
]

imgs=[]

font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font=ImageFont.truetype(font_path,40)

for title,f in files:
    img=Image.open(os.path.join(indir,f)).convert("RGB")

    canvas=Image.new(
        "RGB",
        (img.width,img.height+70),
        "white"
    )

    canvas.paste(img,(0,70))

    d=ImageDraw.Draw(canvas)
    d.text((40,15),title,font=font,fill="black")

    imgs.append(canvas)

w=max(i.width for i in imgs)
h=max(i.height for i in imgs)

out=Image.new(
    "RGB",
    (2*w,2*h),
    "white"
)

positions=[
    (0,0),
    (w,0),
    (0,h),
    (w,h)
]

for im,pos in zip(imgs,positions):
    out.paste(im,pos)

outfile=os.path.join(
    indir,
    "Figure2_WaterDistance_ECDF_FINAL_CORRECT.png"
)

out.save(outfile,dpi=(300,300))

print("Saved:",outfile)
