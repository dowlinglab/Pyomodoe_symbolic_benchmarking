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
# FIGURE SETUP (ESCAPE-SAFE)
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
# FORMATTING
# -----------------------------
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0, ha="center")

ax.set_ylabel("")
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

# ---- FULL BOX (all spines on) ----
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

# ---- LEGEND: top-right, two rows ----
ax.legend(
    frameon=True,
    loc="upper right",
    ncol=1,              # forces two rows (one entry per row)
    fontsize=9,
)

# -----------------------------
# LAYOUT
# -----------------------------
fig.tight_layout()

# -----------------------------
# SAVE (PDF + EPS + PNG)
# -----------------------------
outbase = OUTDIR / "iteration_work_metrics_case4"

fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(outbase.with_suffix(".eps"), bbox_inches="tight")
fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")

plt.close(fig)

print("Saved:")
print(outbase.with_suffix(".pdf"))
print(outbase.with_suffix(".eps"))
print(outbase.with_suffix(".png"))
