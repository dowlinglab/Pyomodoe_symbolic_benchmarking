#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# FILES
# -----------------------------
CENTRAL_XLSX = "doe_benchmark_central.xlsx"
SYMBOLIC_XLSX = "doe_benchmark_symbolic.xlsx"

CASE_NAME = "case4_6param"   # ONLY 6-parameter system
OUTDIR = Path("results_plots/iteration_work_metrics")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# METRICS TO COMPARE
# -----------------------------
METRICS = [
    ("ipopt_iterations", "Iterations"),
    ("ipopt_obj_eval", "Objective evals"),
    ("ipopt_grad_eval", "Gradient evals"),
    ("ipopt_eq_jac_eval", "Jacobian evals"),
    ("ipopt_hess_eval", "Hessian evals"),
]

# -----------------------------
# LOAD DATA
# -----------------------------
central = pd.read_excel(CENTRAL_XLSX, sheet_name=CASE_NAME)
symbolic = pd.read_excel(SYMBOLIC_XLSX, sheet_name=CASE_NAME)

central_means = [central[m].mean() for m, _ in METRICS]
symbolic_means = [symbolic[m].mean() for m, _ in METRICS]

labels = [lbl for _, lbl in METRICS]
x = np.arange(len(labels))
width = 0.65

# -----------------------------
# FIGURE SETUP (ESCAPE FORMAT)
# -----------------------------
FIG_W = 6.5   # inches
FIG_H = 2.5

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

# -----------------------------
# CENTRAL FD (BACKGROUND)
# -----------------------------
ax.bar(
    x,
    central_means,
    width=width,
    facecolor="white",
    edgecolor="black",
    linewidth=2.0,
    alpha=0.9,
    label="Central finite difference",
    zorder=2,
)

# -----------------------------
# SYMBOLIC (OVERLAY)
# -----------------------------
ax.bar(
    x,
    symbolic_means,
    width=width,
    facecolor="gray",
    edgecolor="none",
    alpha=0.6,
    label="Symbolic derivatives",
    zorder=3,
)

# -----------------------------
# AXIS FORMATTING
# -----------------------------
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0, ha="center")

ax.set_ylabel("")
ax.set_title("")
fig.suptitle("")

ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

# -----------------------------
# FULL BOX AROUND FIGURE
# -----------------------------
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

# -----------------------------
# LEGEND (TOP-RIGHT, 2 ROWS)
# -----------------------------
ax.legend(
    frameon=False,
    ncol=1,
    loc="upper right",
    handlelength=1.8,
)

# -----------------------------
# SAVE
# -----------------------------
fig.tight_layout()
fig.savefig(OUTDIR / "iteration_work_metrics_case4.pdf", dpi=300)
plt.close(fig)

print("Saved:", OUTDIR / "iteration_work_metrics_case4.pdf")
