# Integrated Data Center Water Pathways

This repository provides a reproducible workflow for linking U.S. data center locations to hydrologic systems (HUC8 basins, counties, aquifers, rivers, and reservoirs) to characterize potential water-supply pathways.

## Method Overview
The workflow:
- Assigns data centers to HUC8 basins, counties, and aquifers
- Computes nearest distances to major rivers and reservoirs
- Integrates USGS county-level water-use data (2015)
- Classifies groundwater vs surface-water supply pathways
- Evaluates aquifer overrepresentation (Observed vs Expected)
- Compares basins with and without data centers

## Inputs
All datasets should be placed in:

`/mnt/disk3/aoolaseinde/data/integrated_dc_water_pathways/`

## Run

python src/run_integrated_dc_water_pathways.py \
  --data-root /mnt/disk3/aoolaseinde/data/integrated_dc_water_pathways \
  --output-root /mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results \
  --verbose
