#!/usr/bin/env python3
"""
NWM Hydro-Regime ECDF and Matched Random Comparisons

This script reads the basin-level hydro-regime master table and generates ECDF plots plus matched-random comparison statistics for NWM-derived hydrologic metrics.

Input files used:
- basin_master_presence_hydro_manual.csv

Server/GitHub version:
- reads input files from DEFAULT_DATA_ROOT
- default input folder: /mnt/disk3/aoolaseinde/data/NWM_HydroRegime
- writes outputs to DEFAULT_OUTPUT_ROOT
- default output folder: /mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/nwm_ecdf_matched_random
- can be run independently or through src/run_all.py
"""

from __future__ import annotations

import os
from common_paths import DATA_FOLDERS, output_folder

DEFAULT_DATA_ROOT = DATA_FOLDERS["nwm_hydroregime"]
DEFAULT_OUTPUT_ROOT = output_folder("nwm_ecdf_matched_random")
os.makedirs(DEFAULT_OUTPUT_ROOT, exist_ok=True)

# Project-style variables retained for compatibility with notebook-derived code.
PROJECT_DIR = DEFAULT_DATA_ROOT

print("[INFO] Analysis: NWM Hydro-Regime ECDF and Matched Random Comparisons")
print("[INFO] DEFAULT_DATA_ROOT =", DEFAULT_DATA_ROOT)
print("[INFO] DEFAULT_OUTPUT_ROOT =", DEFAULT_OUTPUT_ROOT)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUTDIR = DEFAULT_OUTPUT_ROOT
os.makedirs(OUTDIR, exist_ok=True)

INFILE = os.path.join(DEFAULT_DATA_ROOT, 'basin_master_presence_hydro_manual.csv')

master = pd.read_csv(INFILE)
valid = master.dropna(subset=["meanQ"]).copy()

def label_exclusive(row):
    if row.get("AI_present", 0) == 1: return "AI"
    if row.get("Power_present", 0) == 1: return "Power"
    if row.get("TRI_present", 0) == 1: return "TRI"
    return "None"

valid["group_excl"] = valid.apply(label_exclusive, axis=1)

SECTORS = ["AI", "Power", "TRI"]
BASELINE = "None"
METRICS = ["RBI", "season_conc", "CVQ"]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.4
})

COLOR = {"AI": "#1f77b4", "Power": "#ff7f0e", "TRI": "#2ca02c", "None": "0.70"}

def to_num(x):
    return pd.to_numeric(pd.Series(x), errors="coerce").dropna().values

def ecdf_step(values):
    v = np.sort(to_num(values))
    if v.size == 0:
        return None, None
    y = np.arange(1, v.size + 1) / v.size
    return v, y

def median_iqr(values):
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    return float(v.quantile(0.5)), float(v.quantile(0.25)), float(v.quantile(0.75))

def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return out
    p = pvals[ok]
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out_ok = np.empty_like(p)
    out_ok[order] = adj
    out[ok] = out_ok
    return out

def xlim_quantiles(df, metric, qlo=0.02, qhi=0.98):
    pooled = to_num(df[metric])
    if pooled.size == 0:
        return None
    lo, hi = np.quantile(pooled, [qlo, qhi])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return None
    return (lo, hi)

def matched_random_bootstrap(df, metric, sector, baseline="None", nboot=800, seed=7, grid_n=250):
    rng = np.random.default_rng(seed)
    g = to_num(df.loc[df["group_excl"] == sector, metric])
    bpool = to_num(df.loc[df["group_excl"] == baseline, metric])
    n = g.size
    if n < 30 or bpool.size < n:
        return None
    allv = np.concatenate([g, bpool])
    x_grid = np.quantile(allv, np.linspace(0.01, 0.99, grid_n))
    g_sorted = np.sort(g)
    g_ecdf = np.searchsorted(g_sorted, x_grid, side="right") / g_sorted.size
    b_ecdfs = np.zeros((nboot, x_grid.size), dtype=float)
    diffs = np.zeros(nboot, dtype=float)
    for i in range(nboot):
        b = rng.choice(bpool, size=n, replace=False)
        b_sorted = np.sort(b)
        b_ecdfs[i] = np.searchsorted(b_sorted, x_grid, side="right") / b_sorted.size
        diffs[i] = np.median(g) - np.median(b)
    b_med = np.median(b_ecdfs, axis=0)
    b_lo  = np.quantile(b_ecdfs, 0.025, axis=0)
    b_hi  = np.quantile(b_ecdfs, 0.975, axis=0)
    d_med = float(np.median(diffs))
    d_lo  = float(np.quantile(diffs, 0.025))
    d_hi  = float(np.quantile(diffs, 0.975))
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_two = float(min(1.0, p_two))
    return {
        "n": int(n),
        "x_grid": x_grid,
        "g_ecdf": g_ecdf,
        "b_med": b_med,
        "b_lo": b_lo,
        "b_hi": b_hi,
        "d_med": d_med,
        "d_lo": d_lo,
        "d_hi": d_hi,
        "p_two": p_two
    }

fig1, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig1.subplots_adjust(top=0.78, wspace=0.22)

legend_handles = []
legend_labels = []

for ax, metric in zip(axes, METRICS):
    lim = xlim_quantiles(valid, metric, 0.02, 0.98)
    if lim:
        ax.set_xlim(*lim)

    x0, y0 = ecdf_step(valid.loc[valid["group_excl"] == "None", metric])
    if x0 is not None:
        h0 = ax.plot(x0, y0, drawstyle="steps-post", color=COLOR["None"], lw=2.2)[0]
        if metric == METRICS[0]:
            legend_handles.append(h0); legend_labels.append("None")

    for s in SECTORS:
        xs, ys = ecdf_step(valid.loc[valid["group_excl"] == s, metric])
        if xs is None:
            continue
        hs = ax.plot(xs, ys, drawstyle="steps-post", color=COLOR[s], lw=3.0)[0]
        if metric == METRICS[0]:
            legend_handles.append(hs); legend_labels.append(s)

    ax.set_title(metric, pad=10)
    ax.grid(True, alpha=0.18)
    ax.set_xlabel("Value")

axes[0].set_ylabel("Cumulative frequency")

fig1.legend(legend_handles, legend_labels, loc="upper center",
            bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)

FIG1_OUT = os.path.join(OUTDIR, "FIG1_ECDF_Sector_vs_Sector.png")
fig1.savefig(FIG1_OUT, dpi=600, bbox_inches="tight")
plt.show()

fig2, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=False, sharey=True)
fig2.subplots_adjust(top=0.90, wspace=0.14, hspace=0.18)

style_handles = [
    Line2D([0],[0], color="k", lw=2.8, linestyle="-", label="Sector"),
    Line2D([0],[0], color="k", lw=2.2, linestyle="--", label="Matched random"),
    Patch(facecolor="0.7", edgecolor="none", alpha=0.18, label="95% CI")
]
fig2.legend(handles=style_handles, loc="upper center",
            bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False)

effect_rows = []

for i, sector in enumerate(SECTORS):
    for j, metric in enumerate(METRICS):
        ax = axes[i, j]
        out = matched_random_bootstrap(valid, metric, sector, baseline=BASELINE)
        if out is None:
            ax.set_visible(False)
            continue
        xg = out["x_grid"]
        ax.plot(xg, out["g_ecdf"], color=COLOR[sector], lw=3.0)
        ax.plot(xg, out["b_med"], color=COLOR[sector], lw=2.4, linestyle="--", alpha=0.95)
        ax.fill_between(xg, out["b_lo"], out["b_hi"], color=COLOR[sector], alpha=0.14, lw=0)

        lim = xlim_quantiles(valid, metric, 0.02, 0.98)
        if lim:
            ax.set_xlim(*lim)

        if i == 0:
            ax.set_title(metric, pad=8)

        if j == 0:
            ax.set_ylabel(sector)
        else:
            ax.set_ylabel("")

        ax.set_xlabel("")
        ax.grid(True, alpha=0.15)

        effect_rows.append({
            "Sector": sector,
            "Metric": metric,
            "N_sector": out["n"],
            "MedianDiff_sector_minus_matched": out["d_med"],
            "CI95_low": out["d_lo"],
            "CI95_high": out["d_hi"],
            "p_two_bootstrap": out["p_two"]
        })

FIG2_OUT = os.path.join(OUTDIR, "FIG2_Sector_vs_MatchedRandom_ECDF_3x3.png")
fig2.savefig(FIG2_OUT, dpi=600, bbox_inches="tight")
plt.show()

eff = pd.DataFrame(effect_rows)
EFF_OUT = os.path.join(OUTDIR, "TABLE_EffectSizes_SectorMinusMatched.csv")
eff.to_csv(EFF_OUT, index=False)

fig3, ax = plt.subplots(1, 1, figsize=(11, 5))
fig3.subplots_adjust(top=0.82)

x = np.arange(len(METRICS))
offset = {"AI": -0.18, "Power": 0.00, "TRI": 0.18}

for sector in SECTORS:
    sub = eff[eff["Sector"] == sector].set_index("Metric").loc[METRICS].reset_index()
    y = sub["MedianDiff_sector_minus_matched"].values
    lo = sub["CI95_low"].values
    hi = sub["CI95_high"].values
    yerr = np.vstack([y - lo, hi - y])

    ax.errorbar(
        x + offset[sector], y, yerr=yerr,
        fmt="o", capsize=4, markersize=7,
        color=COLOR[sector], label=sector
    )

ax.axhline(0, color="0.25", linestyle="--", lw=1.8)
ax.set_xticks(x)
ax.set_xticklabels(METRICS)
ax.set_ylabel("Median Δ (sector − matched)")
ax.grid(True, axis="y", alpha=0.18)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))

FIG3_OUT = os.path.join(OUTDIR, "FIG3_EffectSizes_AllSectors_Combined.png")
fig3.savefig(FIG3_OUT, dpi=600, bbox_inches="tight")
plt.show()

desc_rows = []
for metric in METRICS:
    for g in ["None"] + SECTORS:
        vals = valid.loc[valid["group_excl"] == g, metric]
        v = to_num(vals)
        med, q1, q3 = median_iqr(vals)
        desc_rows.append({
            "Metric": metric,
            "Group": g,
            "N": int(v.size),
            "Median": med,
            "Q1": q1,
            "Q3": q3
        })

desc = pd.DataFrame(desc_rows)
DESC_OUT = os.path.join(OUTDIR, "TABLE_Descriptives_Median_IQR_N.csv")
desc.to_csv(DESC_OUT, index=False)

test_rows = []
for metric in METRICS:
    base = to_num(valid.loc[valid["group_excl"] == "None", metric])
    for sector in SECTORS:
        a = to_num(valid.loc[valid["group_excl"] == sector, metric])
        if a.size < 30 or base.size < 30:
            continue

        ks = ks_2samp(a, base)
        mw = mannwhitneyu(a, base, alternative="two-sided")

        rng = np.random.default_rng(7)
        nboot = 2000
        diffs = np.empty(nboot, dtype=float)
        for i in range(nboot):
            aa = rng.choice(a, size=a.size, replace=True)
            bb = rng.choice(base, size=base.size, replace=True)
            diffs[i] = np.median(aa) - np.median(bb)

        d = float(np.median(a) - np.median(base))
        lo = float(np.quantile(diffs, 0.025))
        hi = float(np.quantile(diffs, 0.975))

        test_rows.append({
            "Metric": metric,
            "Sector": sector,
            "CompareTo": "None",
            "N_sector": int(a.size),
            "N_none": int(base.size),
            "MedianDiff_sector_minus_none": d,
            "CI95_low": lo,
            "CI95_high": hi,
            "KS_D": float(ks.statistic),
            "KS_p": float(ks.pvalue),
            "MWU_p": float(mw.pvalue)
        })

tests = pd.DataFrame(test_rows)
tests["KS_pFDR"] = bh_fdr(tests["KS_p"].values)
tests["MWU_pFDR"] = bh_fdr(tests["MWU_p"].values)

TESTS_OUT = os.path.join(OUTDIR, "TABLE_Tests_Sector_vs_None_KS_MWU_FDR.csv")
tests.to_csv(TESTS_OUT, index=False)

print("Figure 1:", FIG1_OUT)
print("Figure 2:", FIG2_OUT)
print("Figure 3:", FIG3_OUT)
print("Table A:", DESC_OUT)
print("Table B:", EFF_OUT)
print("Table C:", TESTS_OUT)
