# AI Infrastructure and Basin-Scale Hydrology

## Hydrographic Analysis of Data Center Siting Patterns in the Contiguous United States

This repository contains reproducible Python workflows for evaluating how U.S. AI data-center infrastructure intersects with hydrologic systems, water-use patterns, aquifers, major rivers, reservoirs, water-stress datasets, and NWM-derived hydro-regime metrics.

## Main Workflow

Run the complete workflow on the server:

python src/run_all.py

## Included Analyses

- Integrated data-center water pathways
- Aqueduct water-stress mapping
- ECDF basin-matched comparisons
- Logistic siting analysis
- Hydrographic nearest-water analysis
- HUC2 facility-share mapping
- NWM hydro-regime analysis
- Basin-scale water-stress evaluation

## Server-Based Execution

The scripts use predefined server paths under:

/mnt/disk3/aoolaseinde/

Each analysis script contains:
- default data directories
- default output directories
- reproducible server-ready configurations

## Notebook

notebooks.ipynb is retained as a reproducible reference notebook containing the original exploratory and plotting workflows. The official production workflow is implemented through Python scripts located in src/.
