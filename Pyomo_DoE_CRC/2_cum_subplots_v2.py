#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# -----------------------------
# FILES
# -----------------------------
CENTRAL_XLSX  = "doe_benchmark_central.xlsx"
SYMBOLIC_XLSX = "doe_benchmark_symbolic.xlsx"

OUTDIR = Path("results_plots/iteration_work_metrics_poster")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# CASE MAPPING (A/B/C/D)
# -----------------------------
CASES = [
    ("A", "case2_2paramsin"),   # two-parameter sine
    ("B", "case1_reactor"),     # reactor
    ("C", "case4_6param"),      # six-parameter ODE
    ("D", "case3_scalarPDE"),   # PDE
]

# -----------------------------
# METRICS (ORDERED, FIXED)
# -----------------------------
METRICS = [
    "ipopt_iterations",        # 1
    "ipopt_obj_eval",          # 2
    "ipopt_grad_eval",         # 3
    "ipopt_eq_con_eval",       # 4
    "ipopt_eq_jac_eval",       # 5
    "ipopt_hess_eval",         # 6
    "ipopt_cpu_no_eval_s",     # 7
    "ipopt_cpu_nlp_eval_s",    # 8
    "solve_time_s",            # 9
    "build_time_s",            # 10
    "init_time_s",             # 11
    "wall_time_s",             # 12
]

# index sets you requested
IDX_1_TO_6 = list(range(0, 6))      # metrics 1–6
IDX_7_TO_12 = list(range(6, 12))    # metrics 7–12

# -----------------------------
# LOAD DATA
# -----------------------------
central_wb  = pd.read_excel(CENTRAL_XLSX, sheet_name=None)
symbolic_wb = pd.read_excel(SYMBOLIC_XLSX, sheet_name=None)

def success_only(df):
    return df[df["returncode"] == 0] if "returncode" in df else df

def mean_vector(df, metric_names):
    return np.array([pd.to_numeric(df[m], errors="coerce").mean() for m in metric_names], dtype=float)

# -----------------------------
# STYLE (as requested)
# -----------------------------
CENTRAL_EDGE_COLOR = "black"
CENTRAL_LINEWIDTH  = 2.5
CENTRAL_ALPHA      = 1.0      # ensures border is black

SYMBOLIC_COLOR = "0.4"
SYMBOLIC_ALPHA = 0.7

BAR_W = 0.78

legend_handles = [
    Patch(facecolor="white",
          edgecolor="black",
          linewidth=CENTRAL_LINEWIDTH,
          label="Central finite difference"),
    Patch(facecolor=SYMBOLIC_COLOR,
          edgecolor="none",
          alpha=SYMBOLIC_ALPHA,
          label="Symbolic derivatives"),
]

# -----------------------------
# PLOTTING FUNCTION
# -----------------------------
def make_plot_per_specs(metric_indices, x_tick_numbers, out_stem):
    """
    metric_indices: list of indices into METRICS (0-based)
    x_tick_numbers: the actual numbers to show on x-axis (e.g., [1..6] or [7..12])
    out_stem: filename stem
    """
    metric_names = [METRICS[i] for i in metric_indices]

    
    
    for case_letter, sheet in CASES:
        plt. figure(figsize=(3.3,2.4))
        ax = plt.gca()

        x = np.array(x_tick_numbers, dtype=float)
        # Read from the excel sheet

        c_means = mean_vector(central_wb[sheet], metric_names)
        s_means = mean_vector(symbolic_wb[sheet], metric_names)

        # CENTRAL FIRST (black border guaranteed)
        ax.bar(
            x,
            c_means,
            width=BAR_W,
            facecolor="white",
            edgecolor=CENTRAL_EDGE_COLOR,
            linewidth=CENTRAL_LINEWIDTH,
            alpha=CENTRAL_ALPHA,
            zorder=2,
        )

        # SYMBOLIC OVERLAY
        ax.bar(
            x,
            s_means,
            width=BAR_W,
            color=SYMBOLIC_COLOR,
            edgecolor="none",
            alpha=SYMBOLIC_ALPHA,
            zorder=3,
        )

        # X ticks are the numbers you want (do not renumber)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x_tick_numbers], fontsize=7)

        # Left label
        # ax.set_ylabel("Count")

        # Grid + full box
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)



        outbase = OUTDIR / f"{out_stem}_case_{case_letter}"
        fig = ax.figure
        fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(outbase.with_suffix(".eps"), bbox_inches="tight")
        fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    
        print("Saved:")
        print(outbase.with_suffix(".pdf"))
        print(outbase.with_suffix(".eps"))
        print(outbase.with_suffix(".png"))

# -----------------------------
# FIGURE 1: metrics 1–6 (A–D)
# -----------------------------
make_plot_per_specs(
    metric_indices=IDX_1_TO_6,
    x_tick_numbers=list(range(1, 7)),          # 1–6
    out_stem="cases_1_6"
)

# -----------------------------
# FIGURE 2: metrics 7–12 (A–D)
# -----------------------------
make_plot_per_specs(
    metric_indices=IDX_7_TO_12,
    x_tick_numbers=list(range(7, 13)),         # 7–12
    out_stem="cases_7to12"
)
