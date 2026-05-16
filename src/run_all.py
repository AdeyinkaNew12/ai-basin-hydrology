#!/usr/bin/env python3
"""
Run All Integrated Data Center Water Pathways Analyses

This script runs all repository analysis scripts in sequence. It is designed for
one-command execution on the server or one-click execution through GitHub Actions
when the repository is connected to a self-hosted runner that can access
/mnt/disk3/aoolaseinde/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "run_integrated_dc_water_pathways.py",
    "run_ecdf_sector_basin_matched.py",
    "run_aqueduct_stress_join_map.py",
    "run_logistic_ai_power_tri.py",
    "run_stalling_nearest_water.py",
    "run_huc2_facility_share_map.py",
    "run_nwm_hydroregime.py",
    "run_nwm_ecdf_matched_random.py",
    "run_nwm_tables.py",
    "run_water_stress_siting.py",
]

SRC_DIR = Path(__file__).resolve().parent


def main() -> None:
    for script in SCRIPTS:
        script_path = SRC_DIR / script
        print("
" + "=" * 72)
        print(f"RUNNING: {script}")
        print("=" * 72)
        if not script_path.exists():
            raise FileNotFoundError(f"Missing script: {script_path}")
        subprocess.run([sys.executable, str(script_path)], check=True)

    print("
All analyses completed successfully.")


if __name__ == "__main__":
    main()
