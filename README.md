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


## Key Results
- Data centers show strong proximity to surface water systems (rivers and reservoirs)
- The majority align with surface-water-dominated supply pathways
- Significant overrepresentation is observed in specific aquifers (non-random siting)
- Basin-scale comparisons indicate statistically significant differences between HUC8 regions with and without data centers



## Key Results
- Data centers exhibit strong spatial proximity to major surface water systems, including rivers and reservoirs
- Infrastructure is predominantly associated with surface-water-dominated supply pathways
- Aquifer-level analysis reveals significant overrepresentation, indicating non-random siting patterns
- Basin-scale comparisons (HUC8) show statistically significant differences in water use characteristics between regions with and without data centers

