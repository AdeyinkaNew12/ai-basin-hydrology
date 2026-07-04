#!/usr/bin/env python3
"""
NWM hydro-regime baseline comparison analysis.

This script evaluates whether AI, power-plant, and TRI-hosting basins
differ from non-infrastructure basins in NWM-derived hydrologic regime.

Method:
- Uses the recomputed basin-level NWM hydro-regime table.
- Retains only basins with valid hydrologic metrics.
- Allows non-exclusive sector membership.
- Defines the baseline pool as basins with no AI, Power, or TRI facilities.
- For each sector and metric, repeatedly samples a sector-size baseline
  from the non-infrastructure pool without replacement.
- Produces ECDF figures, effect-size summaries, and verification tables.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common_paths import DATA_FOLDERS, output_folder


# =============================================================================
# Configuration
# =============================================================================

NWM_ROOT = Path(DATA_FOLDERS["nwm_hydroregime"])
INPUT = (
    NWM_ROOT
    / "NWM_HydroRegime_FULL_RECOMPUTED_JOURNAL"
    / "basin_master_presence_hydro_RECOMPUTED.csv"
)

OUTDIR = Path(output_folder("nwm_hydroregime_baseline_comparison"))
OUTDIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["AI", "Power", "TRI"]
METRICS = ["RBI", "season_conc", "CVQ"]

NBOOT = 800
SEED = 7
GRID_N = 250

COLORS = {
    "AI": "#1f77b4",
    "Power": "#ff7f0e",
    "TRI": "#2ca02c",
    "Baseline": "0.20",
    "None": "0.60",
    "Infrastructure": "black",
}

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 400,
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# =============================================================================
# Helper functions
# =============================================================================

def to_numeric_array(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return values.to_numpy()


def ecdf_xy(values: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None, None

    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)

    return x, y


def ecdf_on_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = np.sort(values)

    if len(values) == 0:
        return np.full_like(grid, np.nan, dtype=float)

    return np.searchsorted(values, grid, side="right") / len(values)


def validate_input_table(df: pd.DataFrame) -> None:
    required = [
        "HYBAS_ID",
        "AI_present",
        "Power_present",
        "TRI_present",
        "RBI",
        "season_conc",
        "CVQ",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")


def prepare_analysis_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["AI_present", "Power_present", "TRI_present"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = df.dropna(subset=METRICS).copy()

    df["None_present"] = (
        (df["AI_present"] == 0)
        & (df["Power_present"] == 0)
        & (df["TRI_present"] == 0)
    ).astype(int)

    df["Infrastructure_present"] = (
        (df["AI_present"] == 1)
        | (df["Power_present"] == 1)
        | (df["TRI_present"] == 1)
    ).astype(int)

    df["Infrastructure_class"] = np.where(
        df["Infrastructure_present"] == 1,
        "Infrastructure",
        "None",
    )

    return df


def matched_random_without_replacement(
    df: pd.DataFrame,
    sector: str,
    metric: str,
    nboot: int = NBOOT,
    seed: int = SEED,
    grid_n: int = GRID_N,
) -> dict:
    rng = np.random.default_rng(seed)

    sector_values = to_numeric_array(df.loc[df[f"{sector}_present"] == 1, metric])
    baseline_pool = to_numeric_array(df.loc[df["None_present"] == 1, metric])

    n_sector = len(sector_values)

    if n_sector == 0:
        raise ValueError(f"No valid values for sector={sector}, metric={metric}")

    if n_sector > len(baseline_pool):
        raise ValueError(
            f"Cannot sample without replacement for sector={sector}, metric={metric}: "
            f"N_sector={n_sector}, N_baseline_pool={len(baseline_pool)}"
        )

    grid = np.linspace(
        np.nanmin(np.concatenate([sector_values, baseline_pool])),
        np.nanmax(np.concatenate([sector_values, baseline_pool])),
        grid_n,
    )

    baseline_ecdfs = []
    baseline_medians = []
    median_differences = []

    for _ in range(nboot):
        baseline_sample = rng.choice(
            baseline_pool,
            size=n_sector,
            replace=False,
        )

        baseline_ecdfs.append(ecdf_on_grid(baseline_sample, grid))
        baseline_median = np.median(baseline_sample)
        baseline_medians.append(baseline_median)
        median_differences.append(np.median(sector_values) - baseline_median)

    baseline_ecdfs = np.asarray(baseline_ecdfs)
    baseline_medians = np.asarray(baseline_medians)
    median_differences = np.asarray(median_differences)

    return {
        "Sector": sector,
        "Metric": metric,
        "N_sector": n_sector,
        "N_none_pool": len(baseline_pool),
        "N_matched_baseline_each_boot": n_sector,
        "Median_sector": np.median(sector_values),
        "Median_none_pool": np.median(baseline_pool),
        "Median_matched_baseline": np.median(baseline_medians),
        "MedianDiff_sector_minus_matched": np.median(median_differences),
        "CI95_low": np.percentile(median_differences, 2.5),
        "CI95_high": np.percentile(median_differences, 97.5),
        "grid": grid,
        "ecdf_sector": ecdf_on_grid(sector_values, grid),
        "ecdf_baseline_median": np.nanmedian(baseline_ecdfs, axis=0),
        "ecdf_baseline_lo": np.nanpercentile(baseline_ecdfs, 2.5, axis=0),
        "ecdf_baseline_hi": np.nanpercentile(baseline_ecdfs, 97.5, axis=0),
    }


def save_effect_table(results: dict[tuple[str, str], dict]) -> pd.DataFrame:
    rows = []

    for sector in SECTORS:
        for metric in METRICS:
            result = results[(sector, metric)]
            rows.append({
                key: result[key]
                for key in [
                    "Sector",
                    "Metric",
                    "N_sector",
                    "N_none_pool",
                    "N_matched_baseline_each_boot",
                    "Median_sector",
                    "Median_none_pool",
                    "Median_matched_baseline",
                    "MedianDiff_sector_minus_matched",
                    "CI95_low",
                    "CI95_high",
                ]
            })

    effects = pd.DataFrame(rows)
    out = OUTDIR / "NWM_hydroregime_baseline_effect_sizes.csv"
    effects.to_csv(out, index=False)
    print(f"[SAVED] {out}")

    return effects


def save_verification_table(df: pd.DataFrame, effects: pd.DataFrame) -> None:
    rows = [
        {"check": "valid_hydrologic_basins", "value": len(df)},
        {"check": "none_basins", "value": int(df["None_present"].sum())},
        {"check": "any_infrastructure_basins", "value": int(df["Infrastructure_present"].sum())},
        {"check": "AI_basins", "value": int(df["AI_present"].sum())},
        {"check": "Power_basins", "value": int(df["Power_present"].sum())},
        {"check": "TRI_basins", "value": int(df["TRI_present"].sum())},
        {"check": "missing_RBI", "value": int(df["RBI"].isna().sum())},
        {"check": "missing_season_conc", "value": int(df["season_conc"].isna().sum())},
        {"check": "missing_CVQ", "value": int(df["CVQ"].isna().sum())},
        {
            "check": "baseline_N_equals_sector_N_all_rows",
            "value": bool((effects["N_sector"] == effects["N_matched_baseline_each_boot"]).all()),
        },
        {
            "check": "none_pool_consistent_all_rows",
            "value": bool((effects["N_none_pool"] == df["None_present"].sum()).all()),
        },
    ]

    verification = pd.DataFrame(rows)
    out = OUTDIR / "NWM_hydroregime_baseline_verification.csv"
    verification.to_csv(out, index=False)
    print(f"[SAVED] {out}")


def plot_matched_ecdf(results: dict[tuple[str, str], dict]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharey=True)

    for i, metric in enumerate(METRICS):
        for j, sector in enumerate(SECTORS):
            ax = axes[i, j]
            result = results[(sector, metric)]
            grid = result["grid"]

            ax.fill_between(
                grid,
                result["ecdf_baseline_lo"],
                result["ecdf_baseline_hi"],
                color="0.80",
                alpha=0.55,
                linewidth=0,
            )

            ax.plot(
                grid,
                result["ecdf_baseline_median"],
                color=COLORS["Baseline"],
                lw=2.0,
                ls="--",
            )

            ax.plot(
                grid,
                result["ecdf_sector"],
                color=COLORS[sector],
                lw=2.4,
            )

            if i == 0:
                ax.set_title(sector, fontweight="bold")

            if j == 0:
                ax.set_ylabel(f"{metric}\nECDF")

            if i == 2:
                ax.set_xlabel(metric)

            ax.grid(alpha=0.20)

    legend_items = [
        Line2D([0], [0], color="black", lw=2.4, label="Observed sector"),
        Line2D([0], [0], color=COLORS["Baseline"], lw=2.0, ls="--", label="Matched random baseline"),
    ]

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out = OUTDIR / "FIG_1_NWM_HydroRegime_Baseline_ECDF_3x3.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


def plot_effect_sizes(effects: pd.DataFrame) -> None:
    plot_df = effects.copy()
    plot_df["label"] = plot_df["Sector"] + " - " + plot_df["Metric"]

    y = np.arange(len(plot_df))
    x = plot_df["MedianDiff_sector_minus_matched"].to_numpy()
    lo = plot_df["CI95_low"].to_numpy()
    hi = plot_df["CI95_high"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    ax.errorbar(
        x,
        y,
        xerr=[x - lo, hi - x],
        fmt="o",
        capsize=3,
        color="black",
        ecolor="0.35",
    )

    ax.axvline(0, color="0.25", lw=1.3, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Median difference: sector minus matched random baseline")
    ax.grid(axis="x", alpha=0.20)

    plt.tight_layout()

    out = OUTDIR / "FIG_2_NWM_HydroRegime_Baseline_EffectSizes.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


def plot_combined_infrastructure_ecdf(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)

    for ax, metric in zip(axes, METRICS):
        none_values = to_numeric_array(df.loc[df["Infrastructure_class"] == "None", metric])
        infra_values = to_numeric_array(df.loc[df["Infrastructure_class"] == "Infrastructure", metric])

        x_none, y_none = ecdf_xy(none_values)
        x_infra, y_infra = ecdf_xy(infra_values)

        ax.plot(x_none, y_none, color=COLORS["None"], lw=2.3, ls="--", label="None")
        ax.plot(x_infra, y_infra, color=COLORS["Infrastructure"], lw=2.5, label="Infrastructure")

        ax.set_xlabel(metric)
        ax.grid(alpha=0.20)

    axes[0].set_ylabel("ECDF")
    axes[-1].legend(frameon=False)

    plt.tight_layout()

    out = OUTDIR / "FIG_3_NWM_HydroRegime_CombinedInfrastructure_ECDF.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    print("[INFO] NWM hydro-regime baseline comparison analysis")
    print(f"[INFO] Input: {INPUT}")
    print(f"[INFO] Output: {OUTDIR}")

    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    raw = pd.read_csv(INPUT)
    validate_input_table(raw)

    df = prepare_analysis_table(raw)

    print(f"Valid basins: {len(df):,}")
    print(f"None: {int(df['None_present'].sum()):,}")
    print(f"Any infrastructure: {int(df['Infrastructure_present'].sum()):,}")
    print(f"AI: {int(df['AI_present'].sum()):,}")
    print(f"Power: {int(df['Power_present'].sum()):,}")
    print(f"TRI: {int(df['TRI_present'].sum()):,}")

    analysis_table = OUTDIR / "basin_hydrologic_metrics_valid.csv"
    df.to_csv(analysis_table, index=False)
    print(f"[SAVED] {analysis_table}")

    results = {}

    for sector in SECTORS:
        for metric in METRICS:
            results[(sector, metric)] = matched_random_without_replacement(
                df=df,
                sector=sector,
                metric=metric,
            )

    effects = save_effect_table(results)
    save_verification_table(df, effects)

    plot_matched_ecdf(results)
    plot_effect_sizes(effects)
    plot_combined_infrastructure_ecdf(df)

    print("\nDONE")
    print(f"Outputs saved in: {OUTDIR}")


if __name__ == "__main__":
    main()
