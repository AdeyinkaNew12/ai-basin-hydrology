# Hydrographic and Hydrologic Analysis of AI Data-Center Siting Patterns in the Contiguous United States

This repository contains reproducible Python workflows for evaluating hydrographic, hydrologic, and water-stress characteristics associated with artificial intelligence (AI) data-center locations across the contiguous United States (CONUS).

The workflow integrates AI data-center locations with national hydrographic datasets, hydrologic simulations, water-stress indicators, aquifer information, and industrial infrastructure datasets to characterize relationships between digital infrastructure and freshwater systems.

The analyses evaluate:

- infrastructure proximity to rivers, lakes, coastlines, and surface-water features;
- watershed-scale hydrologic characteristics using NOAA National Water Model metrics;
- water-supply pathway classifications;
- basin-level water-stress conditions; and
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

Create and activate a Python environment:

```bash
python3 -m venv venv

source venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Data Configuration

Input and output directories are controlled through:

```
src/common_paths.py
```

Update the directory paths to match local data locations before running analyses.

Due to dataset licensing restrictions and file sizes, original external datasets are not included in this repository. Users must obtain required datasets from their respective providers.

Datasets used in this workflow include:

- HydroBASINS and HydroRIVERS hydrographic datasets
- HydroLAKES surface-water dataset
- World Resources Institute (WRI) Aqueduct 4.0 water-stress indicators
- NOAA National Water Model retrospective streamflow simulations
- National aquifer datasets
- EPA Toxic Release Inventory (TRI) data
- EIA thermoelectric power plant data

---

## Running the Workflow

Execute the complete production workflow:

```bash
python src/run_all.py
```

Individual analyses can also be executed separately using scripts contained in the `src/` directory.

---

# Analyses Included

## Surface-Water Proximity Analysis

**Script**

```
run_water_distance_ecdf_allfeatures.py
```

Evaluates infrastructure proximity to rivers, lakes, coastlines, and nearest surface-water features using empirical cumulative distribution functions (ECDFs), statistical comparisons, and basin-matched random baselines.

---

## Water-Supply Pathway Classification

**Script**

```
run_groundwater_dc_analysis.py
```

Characterizes potential water-supply pathways associated with AI data centers using surface-water, groundwater, and mixed pathway classifications.

---

## Basin-Level Facility Distribution Mapping

**Script**

```
run_huc2_facility_share_map.py
```

Generates basin-scale comparisons of AI data centers, thermoelectric power plants, and TRI facilities across water-stress categories.

---

## National Water Model Hydrologic Analysis

**Scripts**

```
run_nwm_hydroregime_recomputed_journal.py

run_nwm_ecdf_matched_random.py
```

Quantifies watershed hydrologic characteristics using NOAA National Water Model streamflow metrics, including:

- runoff stability (RBI);
- seasonal concentration; and
- coefficient of variation of discharge (CVQ).

Infrastructure-associated watersheds are compared with matched baseline watersheds.

---

## Water-Stress Siting Analysis

**Script**

```
run_water_stress_siting.py
```

Evaluates whether AI data centers, thermoelectric power plants, and TRI facilities exhibit different distributions across basin-level water-stress categories using Aqueduct 4.0 indicators and statistical comparisons.

---

# Production Figures

The workflow generates the production figures used for manuscript preparation.

| Figure | Description | Script |
|---|---|---|
| Figure 1 | Infrastructure distribution across basin-level water-stress categories | `run_huc2_facility_share_map.py` |
| Figure 2 | ECDF comparisons of infrastructure proximity to water features | `run_water_distance_ecdf_allfeatures.py` |
| Figure 3 | Hydrologic regime comparison using matched-baseline effect sizes | `run_nwm_hydroregime_recomputed_journal.py` |
| Figure 4 | Basin-scale water-stress distribution comparisons | `run_water_stress_siting.py` |
| Figure 5 | Water-supply pathway classification | `run_groundwater_dc_analysis.py` |

---

# Output Structure

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

# Reproducibility

All production analyses are implemented through Python scripts contained in the `src/` directory.

The workflow provides the computational framework required to reproduce hydrographic, hydrologic, and water-stress analyses of AI infrastructure siting across the contiguous United States.

Archived scripts are retained for documentation purposes and are not required for the production workflow.
