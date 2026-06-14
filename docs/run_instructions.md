# Run Instructions

## Activate Environment

```bash
cd /mnt/disk3/aoolaseinde/projects/ai-basin-hydrology
source venv/bin/activate
```

## Run Complete Workflow

```bash
python3 src/run_all.py
```

This command executes all production analyses and writes outputs to the central results directory.

## Output Location

```text
/mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/
```

Each analysis creates its own subdirectory within the results folder.

## Run Individual Analyses

To run a single analysis:

```bash
python3 src/<script_name>.py
```

Example:

```bash
python3 src/run_nwm_hydroregime.py
```



