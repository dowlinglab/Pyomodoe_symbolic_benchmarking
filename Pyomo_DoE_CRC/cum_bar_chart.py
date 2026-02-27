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

OUTDIR = Path("results_plots/iteration_work_metrics")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# CASES (A–D mapping)
# -----------------------------
CASES = [
    ("A", "case2_2paramsin"),   # two-parameter sine
    ("B", "case1_reactor"),     # reactor
    ("C", "case4_6param"),      # six-parameter ODE
    ("D", "case3_scalarPDE"),   # PDE system
]

# -----------------------------
# METRICS (ORDER)
# -----------------------------
METRICS = [
    "ipopt_iterations",
    "ipopt_obj_eval",
    "ipopt_grad_eval",
    "ipopt_eq_con_eval",
    "ipopt_eq_jac_eval",
    "ipopt_hess_eval",
    "ipopt_cpu_no_eval_s",
    "ipopt_cpu_nlp_eval_s",
    "solve_time_s",
    "build_time_s",
    "init_time_s",
    "wall_time_s",
]

# -----------------------------
# LOAD DATA
# -----------------------------
central_wb  = pd.read_excel(CENTRAL_XLSX, sheet_name=None)
symbolic_wb = pd.read_excel(SYMBOLIC_XLSX, sheet_name=None)

def success_only(df):
    return df[df["returncode"] == 0] if "returncode" in df else df

# -----------------------------
# FIGURE SETUP (ESCAPE)
# -----------------------------
FIG_W = 6.69
FIG_H = 4.95

fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H))
axes = axes.ravel()

x = np.arange(1, len(METRICS) + 1)
bar_w = 0.75

# -----------------------------
# STYLE (ABSOLUTE)
# -----------------------------
CENTRAL_EDGE_COLOR = "black"
CENTRAL_LINEWIDTH  = 2.5
CENTRAL_ALPHA      = 1.0      # 🔴 CRITICAL FIX

SYMBOLIC_COLOR = "0.4"
SYMBOLIC_ALPHA = 0.7

# -----------------------------
# PLOTTING
# -----------------------------
for ax, (case_label, sheet) in zip(axes, CASES):
    cdf = success_only(central_wb[sheet])
    sdf = success_only(symbolic_wb[sheet])

    central_means  = [cdf[m].mean() for m in METRICS]
    symbolic_means = [sdf[m].mean() for m in METRICS]

    # ---- CENTRAL FD FIRST (SOLID BLACK BORDERS) ----
    ax.bar(
        x,
        central_means,
        width=bar_w,
        facecolor="white",
        edgecolor=CENTRAL_EDGE_COLOR,
        linewidth=CENTRAL_LINEWIDTH,
        alpha=CENTRAL_ALPHA,
        zorder=2,
    )

    # ---- SYMBOLIC OVERLAY ----
    ax.bar(
        x,
        symbolic_means,
        width=bar_w,
        color=SYMBOLIC_COLOR,
        edgecolor="none",
        alpha=SYMBOLIC_ALPHA,
        zorder=3,
    )

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=7)

    # Y-axis (left)
    ax.set_ylabel("Count")

    # Right-side case label (A/B/C/D)
    ax_r = ax.twinx()
    ax_r.set_ylabel(case_label, rotation=0, labelpad=10, va="center")
    ax_r.set_yticks([])
    for spine in ax_r.spines.values():
        spine.set_visible(False)

    # Grid + full box
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

# -----------------------------
# LEGEND (GLOBAL)
# -----------------------------
legend_handles = [
    Patch(
        facecolor="white",
        edgecolor="black",
        linewidth=CENTRAL_LINEWIDTH,
        label="Central finite difference",
    ),
    Patch(
        facecolor=SYMBOLIC_COLOR,
        edgecolor="none",
        alpha=SYMBOLIC_ALPHA,
        label="Symbolic derivatives",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=2,
    frameon=True,
    edgecolor="black",
    fontsize=8,
    bbox_to_anchor=(0.5, 1.02),
)

fig.tight_layout(rect=[0, 0, 1, 0.92])

# -----------------------------
# SAVE
# -----------------------------
outbase = OUTDIR / "AtoD_all_metrics_overlaid_BLACKBORDERS"
fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(outbase.with_suffix(".eps"), bbox_inches="tight")
fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")

plt.close(fig)

print("Saved:")
print(outbase.with_suffix(".pdf"))
print(outbase.with_suffix(".eps"))
print(outbase.with_suffix(".png"))
