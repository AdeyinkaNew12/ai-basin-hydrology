#!/usr/bin/env python3
"""
Shared server paths for the Integrated Data Center Water Pathways repository.

Update only these base folders if your server folder structure changes.
Each analysis script creates its own output subfolder under DEFAULT_BASE_OUTPUT_ROOT.
"""

import os

DEFAULT_BASE_DATA_ROOT = "/mnt/disk3/aoolaseinde/data"
DEFAULT_BASE_OUTPUT_ROOT = "/mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results"

# Analysis-specific default data folders.
# Create these folders on the server and place the required files inside them.
DATA_FOLDERS = {
    "groundwater": os.path.join(DEFAULT_BASE_DATA_ROOT, "Groundwater"),
    "water_project": os.path.join(DEFAULT_BASE_DATA_ROOT, "WaterProject"),
    "aqueduct_stress": os.path.join(DEFAULT_BASE_DATA_ROOT, "AqueductStress"),
    "nwm_hydroregime": os.path.join(DEFAULT_BASE_DATA_ROOT, "NWM_HydroRegime"),
    "stalling_nearest_water": os.path.join(DEFAULT_BASE_DATA_ROOT, "StallingNearestWater"),
}

def output_folder(name: str) -> str:
    return os.path.join(DEFAULT_BASE_OUTPUT_ROOT, name)
