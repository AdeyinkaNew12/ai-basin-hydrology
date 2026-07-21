#!/usr/bin/env python3
"""
Run All AI Basin Hydrology Analyses

This script runs the active final workflow in sequence.
It uses central paths defined in src/common_paths.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "run_groundwater_dc_analysis.py",
    # ARCHIVED FIGURE 3 — removed from active run-all workflow:
    "run_threshold_water_proximity_ord5.py",
    "run_water_distance_ecdf_allfeatures.py",
    "run_huc2_facility_share_map.py",
    "run_nwm_hydroregime_recomputed_journal.py",
    "run_water_stress_siting.py",
]

SRC_DIR = Path(__file__).resolve().parent


def main() -> None:
    for script in SCRIPTS:
        script_path = SRC_DIR / script

        print("\n" + "=" * 72)
        print(f"RUNNING: {script}")
        print("=" * 72)

        if not script_path.exists():
            raise FileNotFoundError(f"Missing script: {script_path}")

        subprocess.run([sys.executable, str(script_path)], check=True)

    print("\nAll analyses completed successfully.")


if __name__ == "__main__":
    main()
