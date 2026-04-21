# Integrated Data Center Water Pathways

This repository presents a reproducible analytical workflow for quantifying how U.S. data center infrastructure interacts with hydrologic systems. The framework links facility locations to watershed, groundwater, and surface-water features to characterize potential water-supply pathways across multiple spatial scales.

## Overview

The workflow integrates geospatial datasets to assess how data centers are positioned relative to:

- HUC8 watersheds  
- County-level water use  
- Principal aquifers  
- Major river networks  
- Reservoir systems  

The primary objective is to evaluate whether data center siting reflects underlying water availability and supply structures.

## Methodology

The analysis pipeline performs the following steps:

- Assigns data centers to HUC8 basins, counties, and aquifer extents  
- Computes nearest-neighbor distances to major rivers and reservoirs  
- Integrates 2015 USGS county-level water-use data  
- Classifies dominant supply pathways (groundwater vs. surface water)  
- Quantifies aquifer overrepresentation (observed vs. expected distribution)  
- Compares hydrologic and water-use characteristics of basins with and without data centers  

## Data Requirements

Input datasets should be organized under:
/mnt/disk3/aoolaseinde/data/integrated_dc_water_pathways/

Expected inputs include:

- Data center location dataset (CONUS)  
- USGS county-level water use (2015)  
- HUC8 watershed boundaries  
- County boundaries  
- Aquifer shapefile  
- HydroRIVERS dataset  
- HydroLAKES dataset (optional for reservoir analysis)  

## Usage

Run the workflow from the repository root:

```bash
python src/run_integrated_dc_water_pathways.py \
  --data-root /mnt/disk3/aoolaseinde/data/integrated_dc_water_pathways \
  --output-root /mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results \
  --verbose
