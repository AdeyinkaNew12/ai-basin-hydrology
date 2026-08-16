import pandas as pd
from scipy.stats import chi2_contingency

# ============================================================
# SETTINGS
# ============================================================

WATER_STRESS_FILE = "results/water_stress_siting/water_stress_facilities.csv"

BASIN_COL = "huc8"
STRESS_COL = "stress_tertile"
SECTOR_COL = "sector"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(WATER_STRESS_FILE)

print("\n" + "=" * 70)
print("COLUMNS IN DATASET")
print("=" * 70)
print(df.columns.tolist())

print("\nDataset shape:", df.shape)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = [BASIN_COL, STRESS_COL, SECTOR_COL]

missing = [c for c in required if c not in df.columns]

if missing:
    print("\nERROR: The following required columns were not found:")
    print(missing)
    print("\nAvailable columns are:")
    print(df.columns.tolist())
    raise SystemExit


# ============================================================
# 1. FACILITIES PER BASIN
# ============================================================

print("\n" + "=" * 70)
print("1. FACILITIES PER LEVEL-8 BASIN")
print("=" * 70)

basin_counts = df.groupby(BASIN_COL).size()

print(f"Total facility observations: {len(df):,}")
print(f"Unique Level-8 basins: {len(basin_counts):,}")
print(f"Mean facilities per basin: {basin_counts.mean():.2f}")
print(f"Median facilities per basin: {basin_counts.median():.2f}")
print(f"Maximum facilities in one basin: {basin_counts.max():,}")

single = (basin_counts == 1).sum()
multiple = (basin_counts > 1).sum()

facilities_clustered = basin_counts[basin_counts > 1].sum()

print(f"\nBasins with exactly one facility: {single:,}")
print(f"Basins with multiple facilities: {multiple:,}")
print(
    f"Facilities located in multi-facility basins: "
    f"{facilities_clustered:,} "
    f"({facilities_clustered / len(df) * 100:.2f}%)"
)


# ============================================================
# 2. CURRENT FACILITY-LEVEL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("2. CURRENT FACILITY-LEVEL ANALYSIS")
print("=" * 70)

facility_table = pd.crosstab(
    df[SECTOR_COL],
    df[STRESS_COL]
)

print("\nFacility-level stress table:")
print(facility_table)

chi2, p, dof, expected = chi2_contingency(facility_table)

print(f"\nChi-square = {chi2:.4f}")
print(f"Degrees of freedom = {dof}")
print(f"P-value = {p:.6g}")


# ============================================================
# 3. ONE OBSERVATION PER LEVEL-8 BASIN
# ============================================================

print("\n" + "=" * 70)
print("3. ONE OBSERVATION PER LEVEL-8 BASIN")
print("=" * 70)

# Retain one facility from each Level-8 basin.
# This is a robustness check, NOT a bootstrap.

basin_df = (
    df.sort_values(BASIN_COL)
      .drop_duplicates(subset=[BASIN_COL])
      .copy()
)

print(f"Facility observations: {len(df):,}")
print(f"Basin-level observations: {len(basin_df):,}")


basin_table = pd.crosstab(
    basin_df[SECTOR_COL],
    basin_df[STRESS_COL]
)

print("\nBasin-level stress table:")
print(basin_table)

chi2_b, p_b, dof_b, expected_b = chi2_contingency(basin_table)

print(f"\nBasin-level chi-square = {chi2_b:.4f}")
print(f"Degrees of freedom = {dof_b}")
print(f"Basin-level P-value = {p_b:.6g}")


# ============================================================
# 4. HIGH VS NOT-HIGH STRESS
# ============================================================

print("\n" + "=" * 70)
print("4. HIGH VS NOT-HIGH STRESS")
print("=" * 70)

# Make a robust high-stress indicator.
# Adjust the condition if your category is coded differently.

df["high_stress"] = (
    df[STRESS_COL]
    .astype(str)
    .str.lower()
    .str.strip()
    .eq("high")
)

basin_df["high_stress"] = (
    basin_df[STRESS_COL]
    .astype(str)
    .str.lower()
    .str.strip()
    .eq("high")
)


facility_high_table = pd.crosstab(
    df[SECTOR_COL],
    df["high_stress"]
)

basin_high_table = pd.crosstab(
    basin_df[SECTOR_COL],
    basin_df["high_stress"]
)

print("\nFacility-level HIGH vs NOT-HIGH:")
print(facility_high_table)

print("\nBasin-level HIGH vs NOT-HIGH:")
print(basin_high_table)


chi2_f, p_f, dof_f, _ = chi2_contingency(
    facility_high_table
)

chi2_bh, p_bh, dof_bh, _ = chi2_contingency(
    basin_high_table
)

print("\nFacility-level:")
print(f"  Chi-square = {chi2_f:.4f}")
print(f"  P-value   = {p_f:.6g}")

print("\nBasin-level:")
print(f"  Chi-square = {chi2_bh:.4f}")
print(f"  P-value   = {p_bh:.6g}")


# ============================================================
# 5. AI HIGH-STRESS PROPORTION
# ============================================================

print("\n" + "=" * 70)
print("5. AI HIGH-STRESS PROPORTION")
print("=" * 70)

ai_facility = df[
    df[SECTOR_COL].astype(str).str.lower().str.strip() == "ai"
]

ai_basin = basin_df[
    basin_df[SECTOR_COL].astype(str).str.lower().str.strip() == "ai"
]

facility_ai_high = ai_facility["high_stress"].sum()
facility_ai_total = len(ai_facility)

basin_ai_high = ai_basin["high_stress"].sum()
basin_ai_total = len(ai_basin)

if facility_ai_total > 0:
    facility_pct = (
        facility_ai_high / facility_ai_total * 100
    )
else:
    facility_pct = float("nan")

if basin_ai_total > 0:
    basin_pct = (
        basin_ai_high / basin_ai_total * 100
    )
else:
    basin_pct = float("nan")

print(
    f"Facility-level AI high stress: "
    f"{facility_ai_high}/{facility_ai_total} "
    f"= {facility_pct:.2f}%"
)

print(
    f"Basin-level AI high stress: "
    f"{basin_ai_high}/{basin_ai_total} "
    f"= {basin_pct:.2f}%"
)


# ============================================================
# 6. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("6. ROBUSTNESS-CHECK SUMMARY")
print("=" * 70)

print(
    f"""
                 FACILITY LEVEL       BASIN LEVEL
---------------------------------------------------------
Observations      {len(df):>10,}       {len(basin_df):>10,}
AI high stress    {facility_pct:>10.2f}%       {basin_pct:>10.2f}%
Chi-square p      {p_f:>10.6g}       {p_bh:>10.6g}
"""
)

print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

if p_f < 0.05 and p_bh < 0.05:
    print(
        "The high-vs-not-high sectoral association remains "
        "statistically significant when using one observation "
        "per Level-8 basin."
    )
elif p_f < 0.05 and p_bh >= 0.05:
    print(
        "IMPORTANT: The facility-level association is significant, "
        "but the basin-level robustness check is not significant."
    )
else:
    print(
        "The facility-level high-vs-not-high association is not "
        "statistically significant at p < 0.05."
    )

print("\nThis analysis is a basin-level robustness check.")
print("It is NOT a basin-block bootstrap.")
print("=" * 70)
