import pandas as pd
from scipy.stats import chi2_contingency
from pathlib import Path

# ------------------------------------------------------------
# FILES
# ------------------------------------------------------------

INPUT = Path(
    "/mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/"
    "results/water_stress_siting/"
    "analysis4_facilities_with_stress_ALL_BASINS.csv"
)

OUTPUT_DIR = Path(
    "/mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/"
    "results/water_stress_siting/"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_csv(INPUT, low_memory=False)

BASIN = "pfaf_id_lev08"
SECTOR = "sector"
STRESS = "stress_tertile"

# ------------------------------------------------------------
# FACILITY-LEVEL
# ------------------------------------------------------------

facility_high = (
    df[STRESS].astype(str).str.lower().str.strip() == "high"
)

df["high_stress"] = facility_high

facility_table = pd.crosstab(
    df[SECTOR],
    df["high_stress"]
)

chi2_fac, p_fac, dof_fac, _ = chi2_contingency(
    facility_table
)

# ------------------------------------------------------------
# ONE OBSERVATION PER SECTOR-HUC8
# ------------------------------------------------------------

basin_df = (
    df
    .drop_duplicates(subset=[SECTOR, BASIN])
    .copy()
)

basin_df["high_stress"] = (
    basin_df[STRESS]
    .astype(str)
    .str.lower()
    .str.strip()
    .eq("high")
)

basin_table = pd.crosstab(
    basin_df[SECTOR],
    basin_df["high_stress"]
)

chi2_basin, p_basin, dof_basin, _ = chi2_contingency(
    basin_table
)

# ------------------------------------------------------------
# AI HIGH-STRESS PERCENTAGES
# ------------------------------------------------------------

ai_fac = df[
    df[SECTOR].astype(str).str.upper() == "AI"
]

ai_basin = basin_df[
    basin_df[SECTOR].astype(str).str.upper() == "AI"
]

ai_fac_high = int(ai_fac["high_stress"].sum())
ai_fac_n = len(ai_fac)
ai_fac_pct = 100 * ai_fac_high / ai_fac_n

ai_basin_high = int(ai_basin["high_stress"].sum())
ai_basin_n = len(ai_basin)
ai_basin_pct = 100 * ai_basin_high / ai_basin_n

# ------------------------------------------------------------
# CLUSTERING
# ------------------------------------------------------------

ai_counts = ai_fac.groupby(BASIN).size()

multi_basins = ai_counts[ai_counts > 1]

multi_facilities = int(multi_basins.sum())
multi_pct = 100 * multi_facilities / ai_fac_n

# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------

summary = pd.DataFrame({
    "metric": [
        "AI facilities",
        "AI HUC8 basins",
        "AI facilities in multi-facility HUC8 basins",
        "Percent AI facilities in multi-facility HUC8 basins",
        "Facility-level AI high stress (%)",
        "One-observation-per-HUC8 AI high stress (%)",
        "Facility-level high-vs-not-high chi-square",
        "Facility-level high-vs-not-high p-value",
        "Basin-level high-vs-not-high chi-square",
        "Basin-level high-vs-not-high p-value",
    ],
    "value": [
        ai_fac_n,
        ai_basin_n,
        multi_facilities,
        multi_pct,
        ai_fac_pct,
        ai_basin_pct,
        chi2_fac,
        p_fac,
        chi2_basin,
        p_basin,
    ]
})

csv_out = OUTPUT_DIR / "water_stress_basin_robustness_summary.csv"

summary.to_csv(csv_out, index=False)

# ------------------------------------------------------------
# SAVE DETAILED TEXT REPORT
# ------------------------------------------------------------

txt_out = OUTPUT_DIR / "water_stress_basin_robustness_report.txt"

with open(txt_out, "w") as f:

    f.write("WATER-STRESS BASIN-DEPENDENCE ROBUSTNESS CHECK\n")
    f.write("=" * 70 + "\n\n")

    f.write("Input dataset:\n")
    f.write(str(INPUT) + "\n\n")

    f.write("PURPOSE\n")
    f.write("-" * 70 + "\n")
    f.write(
        "Assess whether the sectoral water-stress association is "
        "sensitive to multiple facilities occurring within the same "
        "HUC8 basin.\n\n"
    )

    f.write("AI FACILITY CLUSTERING\n")
    f.write("-" * 70 + "\n")
    f.write(f"AI facilities: {ai_fac_n:,}\n")
    f.write(f"AI HUC8 basins: {ai_basin_n:,}\n")
    f.write(
        f"AI facilities in multi-facility HUC8 basins: "
        f"{multi_facilities:,}\n"
    )
    f.write(
        f"Percentage in multi-facility HUC8 basins: "
        f"{multi_pct:.2f}%\n\n"
    )

    f.write("AI HIGH-STRESS PROPORTION\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Facility level: "
        f"{ai_fac_high:,}/{ai_fac_n:,} = {ai_fac_pct:.2f}%\n"
    )
    f.write(
        f"One observation per HUC8: "
        f"{ai_basin_high:,}/{ai_basin_n:,} = {ai_basin_pct:.2f}%\n\n"
    )

    f.write("HIGH VS NOT-HIGH SECTORAL TEST\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Facility-level chi-square: {chi2_fac:.4f}\n"
    )
    f.write(
        f"Facility-level p-value: {p_fac:.6g}\n"
    )
    f.write(
        f"Basin-level chi-square: {chi2_basin:.4f}\n"
    )
    f.write(
        f"Basin-level p-value: {p_basin:.6g}\n\n"
    )

    f.write("FACILITY-LEVEL CONTINGENCY TABLE\n")
    f.write("-" * 70 + "\n")
    f.write(facility_table.to_string())
    f.write("\n\n")

    f.write("ONE-OBSERVATION-PER-HUC8 CONTINGENCY TABLE\n")
    f.write("-" * 70 + "\n")
    f.write(basin_table.to_string())
    f.write("\n\n")

    f.write("INTERPRETATION\n")
    f.write("-" * 70 + "\n")

    if p_basin < 0.05:
        f.write(
            "The high-vs-not-high sectoral association remains "
            "statistically significant after reducing each "
            "sector-HUC8 combination to one observation.\n"
        )
    else:
        f.write(
            "The high-vs-not-high sectoral association does not "
            "remain statistically significant after the basin-level "
            "robustness check.\n"
        )

    f.write("\n")
    f.write(
        "This is a one-observation-per-HUC8 robustness check. "
        "It is NOT a basin-block bootstrap.\n"
    )

print("\nSaved:")
print(csv_out)
print(txt_out)
