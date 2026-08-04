# Hydrographic and Hydrologic Analysis of AI Data-Center Siting Patterns in the Contiguous United States

This repository contains the reproducible Python workflows supporting the analysis of hydrographic, hydrologic, and water-stress characteristics associated with artificial intelligence (AI) data-center siting across the contiguous United States (CONUS).

The archived software release is available through Zenodo:

https://doi.org/10.5281/zenodo.21795410

The workflow integrates AI data-center locations with hydrographic datasets, hydrologic simulations, water-stress indicators, aquifer information, and industrial infrastructure datasets to evaluate relationships between digital infrastructure and freshwater systems.

The analyses include:

- surface-water proximity analysis (rivers, lakes, coastlines, and nearest surface-water features);
- watershed hydrologic characterization using NOAA National Water Model metrics;
- water-supply pathway classification;
- basin-level water-stress analysis; and
- comparisons among AI data centers, thermoelectric power plants, and Toxic Release Inventory (TRI) facilities.

---

## Repository Structure

```
src/
├── common_paths.py
├── run_all.py
├── run_water_distance_ecdf_allfeatures.py
├── run_ecdf_sector_basin_matched.py
├── run_groundwater_dc_analysis.py
├── run_huc2_facility_share_map.py
├── run_nwm_hydroregime_recomputed_journal.py
├── run_nwm_ecdf_matched_random.py
└── run_water_stress_siting.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AdeyinkaNew12/ai-basin-hydrology.git
cd ai-basin-hydrology
```

Create an environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Data Configuration

Input and output paths are configured in:

```
src/common_paths.py
```

Original external datasets are not included because of data licensing restrictions and file-size limitations. Users should obtain the required datasets from their respective providers.

Required datasets include:

- HydroBASINS, HydroRIVERS, and HydroLAKES
- WRI Aqueduct 4.0 water-stress indicators
- NOAA National Water Model retrospective streamflow simulations
- National aquifer datasets
- EPA Toxic Release Inventory (TRI)
- EIA thermoelectric power plant data

---

## Running the Workflow

Run all production analyses:

```bash
python src/run_all.py
```

Individual analyses can also be executed using scripts in the `src/` directory.

---

## Analysis Workflows

### Surface-Water Proximity Analysis

Script:

```
run_water_distance_ecdf_allfeatures.py
```

Generates ECDF comparisons of infrastructure proximity to rivers, lakes, coastlines, and nearest surface-water features using basin-matched random baselines.

---

### Water-Supply Pathway Classification

Script:

```
run_groundwater_dc_analysis.py
```

Classifies AI data-center locations according to potential surface-water, groundwater, and mixed water-supply pathways.

---

### Hydrologic Regime Analysis

Scripts:

```
run_nwm_hydroregime_recomputed_journal.py
run_nwm_ecdf_matched_random.py
```

Calculates watershed-scale hydrologic characteristics from NOAA National Water Model streamflow metrics.

---

### Water-Stress Analysis

Script:

```
run_water_stress_siting.py
```

Evaluates infrastructure distributions across basin-level water-stress categories.

---

## Production Figures

| Figure | Description | Script |
|---|---|---|
| Figure 1 | Basin-level infrastructure distribution across water-stress categories | `run_huc2_facility_share_map.py` |
| Figure 2 | ECDF comparisons of water-feature distances | `run_water_distance_ecdf_allfeatures.py` |
| Figure 3 | Hydrologic regime comparison | `run_nwm_hydroregime_recomputed_journal.py` |
| Figure 4 | Water-stress distribution comparison | `run_water_stress_siting.py` |
| Figure 5 | Water-supply pathway classification | `run_groundwater_dc_analysis.py` |

---

## Output Structure

```
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

All production workflows are implemented as Python scripts in the `src/` directory. This repository provides the computational framework required to reproduce the hydrographic, hydrologic, and water-stress analyses of AI infrastructure siting across CONUS.

