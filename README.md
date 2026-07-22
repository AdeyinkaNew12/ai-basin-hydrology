# AI Infrastructure and Basin-Scale Hydrology

## Hydrographic Analysis of AI Data-Center Siting Patterns in the Contiguous United States

This repository contains reproducible Python workflows for evaluating how AI data-center infrastructure intersects with hydrologic systems, major rivers, lakes, coastlines, aquifers, water-stress indicators, watershed characteristics, and National Water Model (NWM) hydro-regime metrics across the contiguous United States.

---

## Repository Structure

```text
src/
├── common_paths.py
├── run_all.py
├── run_integrated_dc_water_pathways.py
├── run_major_river_proximity.py
├── archive/
│   └── run_threshold_water_proximity_ord5.py  # archived; not part of active workflow
├── run_water_distance_ecdf_allfeatures.py
├── run_huc2_facility_share_map.py
├── run_nwm_hydroregime.py
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

All input and output locations are managed through:

```text
src/common_paths.py
```

Users should configure data and output directories in a single location before running analyses.

No individual script requires path editing when `common_paths.py` is configured correctly.

---

## Running All Analyses

Execute the full workflow:

```bash
python src/run_all.py
```

This runs all production analyses in sequence.

---

## Analyses Included

### Integrated Data-Center Water Pathways
Evaluates relationships among data centers, aquifers, counties, HUC8 watersheds, major rivers, reservoirs, and water-use patterns.

### Major River Proximity
Quantifies basin-matched proximity of AI facilities, power plants, and TRI facilities to major rivers (ORD_STRA ≥ 5).

### Water Proximity Threshold Analysis
Computes nearest-water relationships and odds-ratio statistics for rivers, lakes, and coastlines using basin-matched random baselines.

### Water Distance ECDF Analysis
Produces ECDF comparisons for:

- Rivers
- Lakes
- Coastlines
- Nearest water feature

for AI facilities, power plants, and TRI facilities.

### HUC2 Facility Share Mapping
Generates HUC2-scale facility concentration maps.

### NWM Hydro-Regime Analysis
Evaluates hydrologic characteristics of host basins using National Water Model streamflow data.

### NWM ECDF Matched-Random Comparisons
Produces:

- Sector-vs-sector ECDFs
- Sector-vs-matched-random ECDFs
- Effect-size summaries
- Statistical comparison tables

### Water-Stress Siting Analysis
Evaluates siting relationships between infrastructure facilities and Aqueduct water-stress indicators.

---

## Output Structure

Results are written automatically to analysis-specific output folders.

Example:

```text
results/
├── integrated_dc_water_pathways/
├── major_river_proximity/
├── water_proximity_threshold_ord5/
├── water_distance_ecdf_allfeatures/
├── huc2_facility_share_map/
├── nwm_hydroregime/
├── nwm_ecdf_matched_random/
└── water_stress_siting/
```

---

## Reproducibility

The production workflow is implemented entirely through Python scripts contained in `src/`.

Any notebooks included in the repository are retained for exploratory analysis and visualization but are not required to reproduce the published results.
