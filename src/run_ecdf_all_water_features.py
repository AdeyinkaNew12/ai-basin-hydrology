#!/usr/bin/env python3

from pathlib import Path
import os
import subprocess
import sys

SCRIPT = Path(__file__).with_name(
    "run_ecdf_sector_basin_matched.py"
)

DISTANCE_COLUMNS = [
    "dist_river_km",
    "dist_lake_km",
    "dist_coast_km",
    "dist_any_km",
]

for column in DISTANCE_COLUMNS:
    print("\n" + "=" * 80, flush=True)
    print(f"Running ECDF analysis for: {column}", flush=True)
    print("=" * 80, flush=True)

    env = os.environ.copy()
    env["ECDF_DISTANCE_COL"] = column

    subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=True,
        env=env,
    )

print("\nAll four ECDF water-feature analyses completed successfully.")
