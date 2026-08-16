#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


SRC_DIR = Path(__file__).parent
ANALYSIS_DIR = SRC_DIR / "analyses"


ANALYSES = {
    "huc2_facility_distribution":
        "huc2_facility_distribution.py",

    "water_distance_ecdf":
        "water_distance_ecdf.py",

    "water_supply_pathways":
        "water_supply_pathways.py",

    "water_stress_analysis":
        "water_stress_analysis.py",

    "hydrologic_regimes":
        "hydrologic_regimes.py",
}


def run_analysis(name):

    script = ANALYSIS_DIR / ANALYSES[name]

    if not script.exists():
        raise FileNotFoundError(
            f"Missing analysis script: {script}"
        )

    print("\n" + "=" * 70)
    print(f"Running analysis: {name}")
    print("=" * 70)

    subprocess.run(
        [sys.executable, str(script)],
        check=True
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible hydrologic and infrastructure "
            "analyses"
        )
    )

    parser.add_argument(
        "--analysis",
        choices=ANALYSES.keys(),
        help=(
            "Run a single analysis. "
            "If omitted, all analyses are executed."
        )
    )

    args = parser.parse_args()


    if args.analysis:

        run_analysis(args.analysis)

    else:

        for name in ANALYSES:
            run_analysis(name)

        print("\nAll analyses completed successfully.")


if __name__ == "__main__":
    main()
