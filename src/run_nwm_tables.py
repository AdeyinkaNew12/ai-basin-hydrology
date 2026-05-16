#!/usr/bin/env python3
"""
NWM Hydro-Regime Table Builder

This script compiles and exports final hydro-regime summary tables from the NWM analysis, including descriptives, effect sizes, statistical tests, and odds-ratio tables.

Input files used:
- TABLE_Descriptives_Median_IQR_N.csv
- TABLE_EffectSizes_SectorMinusMatched.csv
- TABLE_Tests_SectorVsNone_KS_MWU_FDR.csv
- TABLE_OddsRatios_Logistic.csv

Server/GitHub version:
- reads input files from DEFAULT_DATA_ROOT
- default input folder: /mnt/disk3/aoolaseinde/data/NWM_HydroRegime
- writes outputs to DEFAULT_OUTPUT_ROOT
- default output folder: /mnt/disk3/aoolaseinde/projects/integrated_dc_water_pathways/results/nwm_tables
- can be run independently or through src/run_all.py
"""

from __future__ import annotations

import os
from common_paths import DATA_FOLDERS, output_folder

DEFAULT_DATA_ROOT = DATA_FOLDERS["nwm_hydroregime"]
DEFAULT_OUTPUT_ROOT = output_folder("nwm_tables")
os.makedirs(DEFAULT_OUTPUT_ROOT, exist_ok=True)

# Project-style variables retained for compatibility with notebook-derived code.
PROJECT_DIR = DEFAULT_DATA_ROOT

print("[INFO] Analysis: NWM Hydro-Regime Table Builder")
print("[INFO] DEFAULT_DATA_ROOT =", DEFAULT_DATA_ROOT)
print("[INFO] DEFAULT_OUTPUT_ROOT =", DEFAULT_OUTPUT_ROOT)

import os
import pandas as pd
import matplotlib.pyplot as plt

NB_NAME = os.path.basename(os.getcwd())

OUTDIR = DEFAULT_OUTPUT_ROOT
os.makedirs(OUTDIR, exist_ok=True)

TABLE_SPECS = [
    ("Table A. Descriptives (Median / IQR / N)", "TABLE_Descriptives_Median_IQR_N.csv", "Metric"),
    ("Table B. Sector − Matched Random (Effect sizes)", "TABLE_EffectSizes_SectorMinusMatched.csv", "Sector"),
    ("Table C. Sector vs None (KS / MWU / FDR)", "TABLE_Tests_Sector_vs_None_KS_MWU_FDR.csv", "Metric"),
]

def nice_num(x, nd=2):
    if pd.isna(x):
        return ""
    try:
        x = float(x)
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)

def insert_group_headers_and_blank_groupcol(df, group_col):
    df = df.copy()
    df[group_col] = df[group_col].astype(str)

    rows = []
    group_header_rows = []
    current = None

    for _, r in df.iterrows():
        g = r[group_col]
        if g != current:
            hdr = {c: "" for c in df.columns}
            hdr[df.columns[0]] = g
            rows.append(hdr)
            group_header_rows.append(len(rows) - 1)
            current = g

        rr = r.to_dict()
        rr[group_col] = ""
        rows.append(rr)

    out = pd.DataFrame(rows, columns=df.columns)
    return out, group_header_rows

def maybe_shorten_columns(df):
    rename = {
        "MedianDiff_sector_minus_matched": "MedianΔ",
        "MedianDiff_sector_minus_none": "MedianΔ",
        "p_two_bootstrap": "p (2-sided)",
        "CI95_low": "CI95 low",
        "CI95_high": "CI95 high",
        "BaselinePool": "Baseline",
        "N_sector": "N",
        "CompareTo": "Compare",
        "KS_D": "KS D",
        "KS_p": "KS p",
        "KS_pFDR": "KS p(FDR)",
        "MWU_p": "MWU p",
        "MWU_pFDR": "MWU p(FDR)",
    }
    cols = {c: rename[c] for c in df.columns if c in rename}
    return df.rename(columns=cols)

def render_table_png(
    df,
    title,
    out_png,
    group_col=None,
    ndigits=2,
    font_size=14,
    title_size=20,
    row_height=0.060,
    header_height=0.070,
    pad_inches=0.40,
    header_gray="0.90",
    group_gray="0.97",
):
    df = df.copy()
    df = maybe_shorten_columns(df)

    for c in df.columns:
        df[c] = df[c].apply(lambda v: nice_num(v, nd=ndigits))

    group_rows = []
    if group_col and group_col in df.columns:
        df, group_rows = insert_group_headers_and_blank_groupcol(df, group_col)

    nrows, ncols = df.shape
    fig_w = 1.6 + 1.35 * ncols
    fig_h = 1.8 + row_height * (nrows + 2) * 10

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=title_size, fontweight="bold", pad=14)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="upper center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    try:
        table.auto_set_column_width(col=list(range(ncols)))
    except Exception:
        pass

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("0.25")
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor(header_gray)
            cell.set_text_props(weight="bold")
            cell.set_linewidth(1.5)
            cell.set_edgecolor("0.0")
            cell.set_height(header_height)
        else:
            cell.set_height(row_height)

    for gr in group_rows:
        rr = gr + 1
        for c in range(ncols):
            cell = table[(rr, c)]
            cell.set_facecolor(group_gray)
            cell.set_linewidth(0.0)
            if c == 0:
                cell.set_text_props(weight="bold", ha="left")
            else:
                cell.get_text().set_text("")

    last_r = nrows
    for c in range(ncols):
        table[(last_r, c)].set_linewidth(2.2)
        table[(last_r, c)].set_edgecolor("0.0")

    plt.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print("Saved:", out_png)

print("Notebook:", NB_NAME)
print("Working directory:", os.getcwd())
print("OUTDIR:", OUTDIR)

for title, fname, group_col in TABLE_SPECS:
    path = os.path.join(OUTDIR, fname)
    if not os.path.exists(path):
        print("Missing:", path)
        continue

    df = pd.read_csv(path)

    if "Sector" in df.columns:
        cols = ["Sector"] + [c for c in df.columns if c != "Sector"]
        df = df[cols]
    if "Metric" in df.columns:
        cols = ["Metric"] + [c for c in df.columns if c != "Metric"]
        df = df[cols]

    out_png = os.path.join(OUTDIR, fname.replace(".csv", f"_SLIDE_{NB_NAME.replace(' ', '_')}.png"))
    render_table_png(
        df=df,
        title=title,
        out_png=out_png,
        group_col=group_col,
        ndigits=2,
        font_size=14,
        title_size=22,
        row_height=0.060,
        header_height=0.075,
        pad_inches=0.45
    )

print("DONE")
