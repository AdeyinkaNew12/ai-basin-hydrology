# Integrated Data Center Water Pathways

This repository contains reproducible Python workflows for evaluating how U.S. data-center infrastructure intersects with hydrologic systems, water-use patterns, aquifers, major rivers, reservoirs, water-stress datasets, and NWM-derived hydro-regime metrics.

## One-command run on the server

```bash
python src/run_all.py
```

## One-click run from GitHub

This repository includes a GitHub Actions workflow at:

```text
.github/workflows/run_all.yml
```

Because the scripts use server paths such as `/mnt/disk3/aoolaseinde/...`, GitHub Actions must run on a **self-hosted runner installed on the server**. Normal GitHub-hosted runners cannot access your `/mnt/disk3` folders.

## Server folder setup

Create these data folders yourself on the server and place the required files inside them:

```text
/mnt/disk3/aoolaseinde/data/Groundwater
/mnt/disk3/aoolaseinde/data/WaterProject
/mnt/disk3/aoolaseinde/data/AqueductStress
/mnt/disk3/aoolaseinde/data/NWM_HydroRegime
/mnt/disk3/aoolaseinde/data/StallingNearestWater
```

Outputs are written to:

```text
/mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/<analysis_name>
```

## Scripts

| Script | Purpose | Default data folder | Default output folder |
|---|---|---|---|
| `src/run_integrated_dc_water_pathways.py` | Integrated groundwater/surface-water pathway analysis | `data/Groundwater` | `results` |
| `src/run_ecdf_sector_basin_matched.py` | ECDF sector vs basin-matched random analysis | `data/WaterProject` | `results/ecdf_sector_basin_matched` |
| `src/run_aqueduct_stress_join_map.py` | Aqueduct stress spatial-join verification map | `data/AqueductStress` | `results/aqueduct_stress_join_map` |
| `src/run_logistic_ai_power_tri.py` | Logistic regression vs matched random major-river proximity | `data/WaterProject` | `results/logistic_ai_power_tri` |
| `src/run_stalling_nearest_water.py` | Stallings Center nearest-water analysis | `data/StallingNearestWater` | `results/stalling_nearest_water` |
| `src/run_huc2_facility_share_map.py` | HUC2 facility-share choropleth maps | `data/WaterProject` | `results/huc2_facility_share_map` |
| `src/run_nwm_hydroregime.py` | NWM hydro-regime pipeline | `data/NWM_HydroRegime` | `results/nwm_hydroregime` |
| `src/run_nwm_ecdf_matched_random.py` | NWM ECDF and matched-random comparisons | `data/NWM_HydroRegime` | `results/nwm_ecdf_matched_random` |
| `src/run_nwm_tables.py` | NWM final table builder | `data/NWM_HydroRegime` | `results/nwm_tables` |
| `src/run_water_stress_siting.py` | Water-stress siting analysis | `data/AqueductStress` | `results/water_stress_siting` |
```
