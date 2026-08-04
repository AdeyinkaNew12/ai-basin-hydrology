# AI Infrastructure and Basin-Scale Hydrology

[![DOI](https://zenodo.org/badge/1217127610.svg)](https://doi.org/10.5281/zenodo.21781070)

## Hydrographic Analysis of AI Data-Center Siting Patterns in the Contiguous United States

This repository contains reproducible Python workflows for analyzing hydrographic and hydrologic characteristics of AI data-center siting across the contiguous United States. The workflow integrates AI facility locations with HydroBASINS, HydroRIVERS, HydroLAKES, Aqueduct 4.0 water-stress indicators, aquifer datasets, and NOAA National Water Model hydrologic metrics to evaluate relationships between digital infrastructure and freshwater systems.

The analyses quantify infrastructure proximity to rivers, lakes, coastlines, and groundwater resources; evaluate basin-scale water-stress conditions; characterize water-supply pathways; and compare AI data centers with thermoelectric power plants and Toxic Release Inventory (TRI) facilities.

---

## Repository Structure

```text
src/
├── common_paths.py
├── run_all.py
├── run_ecdf_all_water_features.py
├── run_ecdf_sector_basin_matched.py
├── run_groundwater_dc_analysis.py
├── run_huc2_facility_share_map.py
├── run_nwm_hydroregime_recomputed_journal.py
├── run_nwm_ecdf_matched_random.py
├── run_water_stress_siting.py
```

---

## Installation

Clone the repository and create a Python environment.

```bash
git clone https://github.com/AdeyinkaNew12/ai-basin-hydrology.git
cd ai-basin-hydrology

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Path Configuration

Input and output directories are controlled through:

```text
src/common_paths.py
```

Configure the required data locations before running analyses.

---

## Running the Workflow

Execute the complete workflow:

```bash
python src/run_all.py
```

The workflow executes all production analyses sequentially.

---

## Analyses Included

### Integrated Data-Center Water Pathways

Evaluates relationships among AI data centers, aquifers, counties, HUC8 watersheds, reservoirs, and water-supply pathways.

### Surface-Water Proximity Analysis

Evaluates distances to rivers, lakes, coastlines, and nearest water features using empirical cumulative distribution functions (ECDFs), statistical comparisons, and matched baseline analyses.

### HUC2 Facility Share Mapping

Generates HUC2-scale facility distribution maps comparing AI data centers, thermoelectric power plants, and TRI facilities across basin-level water-stress categories.

### National Water Model Hydro-Regime Analysis

Calculates basin-scale hydrologic signatures from NOAA National Water Model streamflow simulations, including runoff stability (RBI), seasonal concentration, and coefficient of variation of discharge (CVQ). Infrastructure-associated basins are compared with matched baseline watersheds using bootstrap-derived reference distributions.

### NWM ECDF Matched-Random Comparisons

Produces ECDF comparisons between infrastructure sectors and matched-random reference locations, including statistical comparisons of hydrologic signatures.

### Water-Stress Siting Analysis

Evaluates whether AI data centers, thermoelectric power plants, and TRI facilities exhibit different distributions across basin-level water-stress categories using Aqueduct 4.0 indicators and statistical comparisons.

---

## Production Figures

The workflow generates five production figures used for manuscript preparation:

1. **Figure 1 — Infrastructure distribution across basin-level water-stress categories**  
   Script: `run_huc2_facility_share_map.py`

2. **Figure 2 — ECDF comparisons of water-feature distances**  
   Script: `run_ecdf_all_water_features.py`

3. **Figure 3 — Matched-baseline hydrologic signature effect sizes**  
   Script: `run_nwm_hydroregime_recomputed_journal.py`

4. **Figure 4 — Basin-scale water-stress distribution comparisons**  
   Script: `run_water_stress_siting.py`

5. **Figure 5 — Water-source pathway classification**
   Script: `run_groundwater_dc_analysis.py`
  

---

## Output Structure

```text
results/
├── integrated_dc_water_pathways/
├── water_distance_ecdf_allfeatures/
├── huc2_facility_share_map/
├── nwm_hydroregime/
├── nwm_ecdf_matched_random/
└── water_stress_siting/
```

---

## Reproducibility

All production analyses are implemented through Python scripts contained in `src/`.

This repository provides the computational workflow required to reproduce hydrographic, hydrologic, and water-stress analyses of AI infrastructure siting across the contiguous United States.

Archived scripts are retained for documentation purposes and are not required for the production workflow.
