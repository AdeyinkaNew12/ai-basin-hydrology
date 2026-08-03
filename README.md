# AI Infrastructure and Basin-Scale Hydrology

[![DOI](https://zenodo.org/badge/1217127610.svg)](https://doi.org/10.5281/zenodo.21780448)

## Hydrographic Analysis of AI Data-Center Siting Patterns in the Contiguous United States

This repository contains reproducible Python workflows evaluating how AI data-center infrastructure intersects with hydrologic systems across the contiguous United States. The analyses examine relationships with rivers, lakes, coastlines, aquifers, watershed characteristics, water-stress indicators, and National Water Model hydrologic signatures.

---

## Repository Structure

```text
src/
├── common_paths.py
├── run_all.py
├── run_integrated_dc_water_pathways.py
├── run_major_river_proximity.py
├── run_water_distance_ecdf_allfeatures.py
├── run_huc2_facility_share_map.py
├── run_nwm_hydroregime_recomputed_journal.py
├── run_nwm_ecdf_matched_random.py
├── run_water_stress_siting.py
