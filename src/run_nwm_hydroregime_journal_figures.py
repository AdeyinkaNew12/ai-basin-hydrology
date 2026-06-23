#!/usr/bin/env python3
"""
Final NWM hydro-regime journal figure workflow.

Uses the validated cached NWM hydro-regime outputs and exports only the
approved journal figures and supporting tables to the project results folder.
"""

from pathlib import Path
import shutil
import pandas as pd

from common_paths import DATA_FOLDERS, output_folder

DATA_DIR = (
    Path(DATA_FOLDERS["nwm_hydroregime"])
    / "NWM_HydroRegime_FULL_RECOMPUTED_JOURNAL"
)

OUTDIR = Path(output_folder("nwm_hydroregime_journal_figures"))
OUTDIR.mkdir(parents=True, exist_ok=True)

MASTER = DATA_DIR / "basin_master_presence_hydro_RECOMPUTED.csv"
EFFECTS = DATA_DIR / "NWM_matched_baseline_effect_sizes_RECOMPUTED.csv"

FIGURES = [
    "FIG_1_MatchedBaseline_ECDF_3x3_FINAL_JOURNAL.png",
    "FIG_2_MatchedBaseline_EffectSizes_FINAL_JOURNAL.png",
    "FIG_3_CombinedSector_ECDF_3panel_FINAL_JOURNAL.png",
]

TABLES = [
    "basin_master_presence_hydro_RECOMPUTED.csv",
    "basin_hydrologic_metrics_RECOMPUTED.csv",
    "NWM_matched_baseline_effect_sizes_RECOMPUTED.csv",
]

print("[INFO] Final NWM hydro-regime journal figure workflow")
print("[INFO] Input:", DATA_DIR)
print("[INFO] Output:", OUTDIR)

if not MASTER.exists():
    raise FileNotFoundError(f"Missing master table: {MASTER}")

if not EFFECTS.exists():
    raise FileNotFoundError(f"Missing effect-size table: {EFFECTS}")

df = pd.read_csv(MASTER)
eff = pd.read_csv(EFFECTS)

# Standardize group labels for reproducible verification.
df["group_excl"] = df["group_excl"].fillna("None")
df.loc[df["any_infra"] == 0, "group_excl"] = "None"
df.loc[df["any_infra"] == 1, "group_excl"] = "Infrastructure"

# Save corrected master back to the validated cache.
df.to_csv(MASTER, index=False)

required_counts = {
    "valid_basins": 11022,
    "none_basins": 6417,
    "infrastructure_basins": 4605,
    "AI_basins": 369,
    "Power_basins": 3546,
    "TRI_basins": 3278,
}

actual_counts = {
    "valid_basins": len(df),
    "none_basins": int((df["group_excl"] == "None").sum()),
    "infrastructure_basins": int((df["group_excl"] == "Infrastructure").sum()),
    "AI_basins": int(df["AI_present"].sum()),
    "Power_basins": int(df["Power_present"].sum()),
    "TRI_basins": int(df["TRI_present"].sum()),
}

print("[INFO] Verification counts:")
for k, v in actual_counts.items():
    print(f"  {k}: {v:,}")

for k, expected in required_counts.items():
    observed = actual_counts[k]
    if observed != expected:
        raise ValueError(f"Count check failed for {k}: observed={observed}, expected={expected}")

for metric in ["RBI", "season_conc", "CVQ"]:
    missing = int(df[metric].isna().sum())
    print(f"  missing_{metric}: {missing}")
    if missing != 0:
        raise ValueError(f"Missing hydrologic metric values found for {metric}: {missing}")

none_df = df[df["group_excl"] == "None"]

baseline_medians = {
    "RBI": none_df["RBI"].median(),
    "season_conc": none_df["season_conc"].median(),
    "CVQ": none_df["CVQ"].median(),
}

print("[INFO] Baseline medians:")
for k, v in baseline_medians.items():
    print(f"  {k}: {v}")

for _, row in eff.iterrows():
    metric = row["Metric"]
    expected = baseline_medians[metric]
    observed = row["Median_matched_baseline"]
    if abs(observed - expected) > 1e-10:
        raise ValueError(
            f"Baseline median mismatch for {row['Sector']} {metric}: "
            f"table={observed}, none_pool={expected}"
        )

for fig in FIGURES:
    src = DATA_DIR / fig
    dst = OUTDIR / fig
    if not src.exists():
        raise FileNotFoundError(f"Missing approved figure: {src}")
    shutil.copy2(src, dst)
    print("[SAVED FIGURE]", dst)

for table in TABLES:
    src = DATA_DIR / table
    dst = OUTDIR / table
    if src.exists():
        shutil.copy2(src, dst)
        print("[SAVED TABLE]", dst)

verification = pd.DataFrame([
    {"check": k, "value": v}
    for k, v in actual_counts.items()
] + [
    {"check": f"baseline_median_{k}", "value": v}
    for k, v in baseline_medians.items()
])

verification.to_csv(OUTDIR / "NWM_journal_verification.csv", index=False)
print("[SAVED TABLE]", OUTDIR / "NWM_journal_verification.csv")

print("\nDONE")
print("Final journal figures saved in:", OUTDIR)
