#!/usr/bin/env python3
"""
ECDF Sector vs Basin-Matched Random Analysis

This script generates ECDF plots and statistical comparison tables for AI, Power, and TRI facility distances against basin-matched random locations. It produces journal-ready ECDF textfigures, descriptive statistics, KS/Mann-Whitney tests, Cliff’s delta effect sizes, and FDR-adjusted summary tables.

Input files used:
- DC_CONUS.csv
- Power.xlsx
- TRI_2024.csv
- hybas_na_lev08_v1c.shp
- HydroRIVERS_v10_na.shp

Server/GitHub version:
- reads input files from DEFAULT_DATA_ROOT
- default input folder: /mnt/disk3/aoolaseinde/data/WaterProject
- writes outputs to DEFAULT_OUTPUT_ROOT
- default output folder: /mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/ecdf_sector_basin_matched
- can be run independently or through src/run_all.py
"""

from __future__ import annotations

import os
from common_paths import DATA_FOLDERS, output_folder

DEFAULT_DATA_ROOT = DATA_FOLDERS["water_project"]
DEFAULT_OUTPUT_ROOT = output_folder("ecdf_sector_basin_matched")
os.makedirs(DEFAULT_OUTPUT_ROOT, exist_ok=True)

# Project-style variables retained for compatibility with notebook-derived code.
PROJECT_DIR = DEFAULT_DATA_ROOT

print("[INFO] Analysis: ECDF Sector vs Basin-Matched Random Analysis")
print("[INFO] DEFAULT_DATA_ROOT =", DEFAULT_DATA_ROOT)
print("[INFO] DEFAULT_OUTPUT_ROOT =", DEFAULT_OUTPUT_ROOT)



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, ks_2samp

PROJECT_DIR = DEFAULT_DATA_ROOT
OUT_DIR = DEFAULT_OUTPUT_ROOT
os.makedirs(OUT_DIR, exist_ok=True)

COL = 'dist_river_km'  # dist_lake_km / dist_coast_km / dist_any_km
SECTORS = ['AI', 'Power', 'TRI']

SEC_COLORS = {'AI': '#1f77b4', 'Power': '#2ca02c', 'TRI': '#9467bd'}
FIG_OUT = os.path.join(OUT_DIR, f'ECDF_{COL}_JOURNAL.png')

ECDF_XMAX_Q = 0.995
ECDF_LOG_XMIN = 0.05
SEC_LW = 3.0
RND_LW = 2.0
RND_LS = (0, (1, 2))
RND_A  = 0.95
FIGSIZE = (10.8, 6.4)
DPI_SAVE = 450

def ecdf(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    a = a[a > 0]
    a = np.sort(a)
    if a.size == 0:
        return np.array([np.nan]), np.array([np.nan])
    y = np.arange(1, len(a) + 1) / len(a)
    return a, y

def clean_vec(s):
    v = pd.to_numeric(pd.Series(s), errors='coerce').to_numpy(float)
    v = v[np.isfinite(v)]
    v = v[v > 0]
    return v

def load_dist(sec, grp):
    f = os.path.join(OUT_DIR, f'{sec}_{grp}_distances.csv')
    if not os.path.exists(f):
        raise FileNotFoundError(f'Missing: {f}')
    df = pd.read_csv(f, low_memory=False)
    if COL not in df.columns:
        raise KeyError(f"{f} missing '{COL}'. Has: {list(df.columns)[:30]}")
    return clean_vec(df[COL])

def cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    n, m = len(x), len(y)
    gt = lt = 0
    chunk = max(1, 2_000_000 // max(1, m))
    for i in range(0, n, chunk):
        xx = x[i:i+chunk]
        diff = xx[:, None] - y[None, :]
        gt += np.sum(diff > 0)
        lt += np.sum(diff < 0)
    return float((gt - lt) / (n*m))

def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    idx = np.where(np.isfinite(pvals))[0]
    if len(idx) == 0:
        return out
    pv = pvals[idx]
    order = np.argsort(pv)
    pv_sorted = pv[order]
    n = len(pv_sorted)
    q = pv_sorted * n / (np.arange(1, n+1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[idx[order]] = np.clip(q, 0, 1)
    return out

def qnt(a, p):
    return float(np.quantile(a, p)) if len(a) else np.nan

def fmt_num(x, nd=2):
    if pd.isna(x):
        return ''
    try:
        x = float(x)
        if abs(x) >= 1000:
            return f'{x:,.0f}'
        return f'{x:.{nd}f}'
    except:
        return str(x)

def fmt_p(x):
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if s.startswith('<') or 'e-' in s.lower():
        return s
    try:
        v = float(x)
        if v < 1e-4: return '<1e-4'
        if v < 0.001: return '<0.001'
        return f'{v:.3f}'
    except:
        return s

# ------------------ Load data
data = {}
all_vals = []
for sec in SECTORS:
    x = load_dist(sec, 'sector')
    y = load_dist(sec, 'random')
    data[sec] = {'sector': x, 'random': y}
    all_vals.append(x); all_vals.append(y)
    print(f"{sec}: N_sector={len(x):,} | N_random={len(y):,}")

all_vals = np.concatenate(all_vals) if len(all_vals) else np.array([])
xmax = float(np.quantile(all_vals, ECDF_XMAX_Q)) if all_vals.size else 1.0

# ------------------ ECDF plot
plt.style.use('default')
plt.rcParams.update({
    'figure.dpi': 160,
    'savefig.dpi': DPI_SAVE,
    'font.size': 12,
    'axes.linewidth': 1.1,
    'xtick.major.width': 1.1,
    'ytick.major.width': 1.1,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

fig, ax = plt.subplots(figsize=FIGSIZE)
for sec in SECTORS:
    xs, ys = ecdf(data[sec]['sector'])
    xr, yr = ecdf(data[sec]['random'])
    c = SEC_COLORS[sec]
    ax.plot(xs, ys, color=c, lw=SEC_LW, ls='-', alpha=1.0)
    ax.plot(xr, yr, color=c, lw=RND_LW, ls=RND_LS, alpha=RND_A)

ax.set_xscale('log')
ax.set_xlim(ECDF_LOG_XMIN, max(1.0, xmax))
ax.set_ylim(0, 1.02)

title_main = 'ECDF Distance to Major Rivers — Sector vs Basin-matched Random'
ax.set_title(title_main, pad=18, fontweight='semibold')

ax.set_xlabel('Distance to major rivers (km, log scale) — HydroRIVERS ORD_STRA ≥ 5')
ax.set_ylabel('ECDF')

n_text = 'N matched (in basins):\n' + '\n'.join([f"{sec}: {len(data[sec]['sector']):,}" for sec in SECTORS])
ax.text(0.02, 0.98, n_text, transform=ax.transAxes, ha='left', va='top', fontsize=10.5,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='black', linewidth=1.0, alpha=0.95))

style_handles = [
    Line2D([0],[0], color='black', lw=SEC_LW, ls='-', label='Sector'),
    Line2D([0],[0], color='black', lw=RND_LW, ls=RND_LS, label='Random (N-matched)'),
]
color_handles = [
    Line2D([0],[0], color=SEC_COLORS['AI'], lw=3.2, label='AI'),
    Line2D([0],[0], color=SEC_COLORS['Power'], lw=3.2, label='Power'),
    Line2D([0],[0], color=SEC_COLORS['TRI'], lw=3.2, label='TRI'),
]
fig.legend(handles=style_handles + color_handles, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=11,
           handlelength=3.0, columnspacing=1.8)

ax.grid(True, which='major', linewidth=0.6, alpha=0.18)
ax.grid(False, which='minor')
ax.set_axisbelow(True)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(FIG_OUT, bbox_inches='tight')
plt.show()
print('✅ Saved:', FIG_OUT)

# ------------------ Stats tables
test_rows = []
desc_rows = []

for sec in SECTORS:
    x = data[sec]['sector']
    y = data[sec]['random']

    desc_rows.append({'Distance_Column': COL, 'Sector': sec, 'Group': 'Sector',
                      'N': int(len(x)), 'Median': qnt(x, 0.50),
                      'IQR_25': qnt(x, 0.25), 'IQR_75': qnt(x, 0.75),
                      'Mean': float(np.mean(x)) if len(x) else np.nan})
    desc_rows.append({'Distance_Column': COL, 'Sector': sec, 'Group': 'Random',
                      'N': int(len(y)), 'Median': qnt(y, 0.50),
                      'IQR_25': qnt(y, 0.25), 'IQR_75': qnt(y, 0.75),
                      'Mean': float(np.mean(y)) if len(y) else np.nan})

    if len(x) and len(y):
        ksD, ksP = ks_2samp(x, y, alternative='two-sided', mode='auto')
        u, p_u = mannwhitneyu(x, y, alternative='two-sided')
        cd = cliffs_delta(x, y)
        med_diff = qnt(x, 0.50) - qnt(y, 0.50)
    else:
        ksD = ksP = u = p_u = cd = med_diff = np.nan

    test_rows.append({
        'Distance_Column': COL, 'Sector': sec,
        'N_sector': int(len(x)), 'N_random': int(len(y)),
        'KS_D': float(ksD), 'KS_p': float(ksP),
        'MWU_U': float(u), 'MWU_p': float(p_u),
        'Cliffs_delta': float(cd),
        'Median_diff_sector_minus_random': float(med_diff),
    })

tests = pd.DataFrame(test_rows)
descs = pd.DataFrame(desc_rows)

tests['KS_q_FDR'] = bh_fdr(tests['KS_p'].values)
tests['MWU_q_FDR'] = bh_fdr(tests['MWU_p'].values)

OUT_TESTS = os.path.join(OUT_DIR, f'TABLE_ECDF_tests_{COL}.csv')
OUT_DESCS = os.path.join(OUT_DIR, f'TABLE_ECDF_descriptives_{COL}.csv')
tests.to_csv(OUT_TESTS, index=False)
descs.to_csv(OUT_DESCS, index=False)
print('✅ Saved:', OUT_TESTS)
print('✅ Saved:', OUT_DESCS)

# ------------------ Final wide journal table
sector_desc = descs[descs['Group'] == 'Sector'].rename(columns={
    'N': 'N_sector', 'Median': 'Median_sector', 'IQR_25': 'IQR25_sector',
    'IQR_75': 'IQR75_sector', 'Mean': 'Mean_sector'
}).drop(columns=['Group'])

random_desc = descs[descs['Group'] == 'Random'].rename(columns={
    'N': 'N_random', 'Median': 'Median_random', 'IQR_25': 'IQR25_random',
    'IQR_75': 'IQR75_random', 'Mean': 'Mean_random'
}).drop(columns=['Group'])

final = (sector_desc
         .merge(random_desc, on=['Distance_Column', 'Sector'], how='left', validate='one_to_one')
         .merge(tests, on=['Distance_Column', 'Sector', 'N_sector', 'N_random'], how='left', validate='one_to_one'))

final_disp = final.copy()
final_disp['Median (sector)'] = final_disp['Median_sector'].apply(lambda x: fmt_num(x, 2))
final_disp['IQR (sector)'] = final_disp.apply(lambda r: f"{fmt_num(r['IQR25_sector'],2)}–{fmt_num(r['IQR75_sector'],2)}", axis=1)
final_disp['Median (random)'] = final_disp['Median_random'].apply(lambda x: fmt_num(x, 2))
final_disp['IQR (random)'] = final_disp.apply(lambda r: f"{fmt_num(r['IQR25_random'],2)}–{fmt_num(r['IQR75_random'],2)}", axis=1)

final_disp['KS D'] = final_disp['KS_D'].apply(lambda x: fmt_num(x, 3))
final_disp['KS p'] = final_disp['KS_p'].apply(fmt_p)
final_disp['KS q(FDR)'] = final_disp['KS_q_FDR'].apply(fmt_p)
final_disp['MWU p'] = final_disp['MWU_p'].apply(fmt_p) 
final_disp['MWU q(FDR)'] = final_disp['MWU_q_FDR'].apply(fmt_p)
final_disp["Cliff's δ"] = final_disp['Cliffs_delta'].apply(lambda x: fmt_num(x, 3))
final_disp['Δ median (sector-random)'] = final_disp['Median_diff_sector_minus_random'].apply(lambda x: fmt_num(x, 2))

final_table = final_disp[[
    'Sector',
    'N_sector', 'Median (sector)', 'IQR (sector)',
    'N_random', 'Median (random)', 'IQR (random)',
    'KS D', 'KS p', 'KS q(FDR)',
    'MWU p', 'MWU q(FDR)',
    "Cliff's δ", 'Δ median (sector-random)'
]].copy()

OUT_FINAL = os.path.join(OUT_DIR, f'TABLE_ECDF_FINAL_JOURNAL_{COL}.csv')
final_table.to_csv(OUT_FINAL, index=False)
print('✅ Saved:', OUT_FINAL)
print(final_table)

