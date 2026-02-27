#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Author: Shilpa

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

# -----------------------------
# FILES / SETTINGS
# -----------------------------
CENTRAL_XLSX  = "doe_benchmark_central.xlsx"
SYMBOLIC_XLSX = "doe_benchmark_symbolic.xlsx"
CASE_NAME = "case4_6param"

OUTDIR = Path("results_plots") / "performance_time"
OUTDIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "build_time_s",
    "solve_time_s",
    "init_time_s",
    "wall_time_s",
]

SUBFIG_LABELS = ["(a)", "(b)", "(c)", "(d)"]

# -----------------------------
# ESCAPE two-column width (17.0 cm) -> 6.69 in
# -----------------------------
FIG_W = 6.69   # inches
FIG_H = 4.80   # inches

mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 1.0,
})

# -----------------------------
# LOAD + MERGE (pair by run_id)
# -----------------------------
central  = pd.read_excel(CENTRAL_XLSX,  sheet_name=CASE_NAME)
symbolic = pd.read_excel(SYMBOLIC_XLSX, sheet_name=CASE_NAME)

# keep successful runs only (if present)
if "returncode" in central.columns:
    central = central[central["returncode"] == 0].copy()
if "returncode" in symbolic.columns:
    symbolic = symbolic[symbolic["returncode"] == 0].copy()

if "run_id" not in central.columns or "run_id" not in symbolic.columns:
    raise ValueError("Expected 'run_id' column in both sheets.")

df = pd.merge(
    central,
    symbolic,
    on="run_id",
    how="inner",
    suffixes=("_central", "_symbolic"),
).sort_values("run_id").reset_index(drop=True)

if df.empty:
    raise RuntimeError("No matching run_id rows after merging central and symbolic sheets.")

# -----------------------------
# PLOT: 2x2 parity subplot panel
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H))
axes = axes.ravel()

scatter_handle = None
parity_handle = None

for i, (ax, metric) in enumerate(zip(axes, METRICS)):
    x = pd.to_numeric(df[f"{metric}_central"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[f"{metric}_symbolic"], errors="coerce").to_numpy(dtype=float)

   
    if len(x) == 0:
        continue

    # ---- TRUE PARITY LIMITS (shared x/y) ----
    data_min = min(x.min(), y.min())
    data_max = max(x.max(), y.max())
    lo = data_min - 0.0001
    hi = data_max + 0.0001

    sc = ax.scatter(
        x, y,
        s=50,
        marker="o",
        color="0.3",
        alpha=1,
        label="All runs",
        zorder=3
    )

    ln, = ax.plot(
        [lo, hi], [lo, hi],
        linestyle="--",
        color="0.5",
        linewidth=1.0,
        label="Parity",
        zorder=2
    )

    if scatter_handle is None:
        scatter_handle = sc
    if parity_handle is None:
        parity_handle = ln

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)

    # Boxed axes
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    # Subfigure label: bottom center outside each axis
    ax.text(
        0.5, -0.32,
        SUBFIG_LABELS[i],
        transform=ax.transAxes,
        ha="center",
        va="top",
    )

# Axis labels only on outer edges
axes[2].set_xlabel("Central finite difference (s)")
axes[3].set_xlabel("Central finite difference (s)")
axes[0].set_ylabel("Symbolic derivatives (s)")
axes[2].set_ylabel("Symbolic derivatives (s)")

# Legend in top-right subplot only (axes[1])
axes[1].legend(
    handles=[scatter_handle, parity_handle],
    labels=["All runs", "Parity"],
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="black",
    ncol=1,
)

fig.tight_layout()

# -----------------------------
# SAVE (PDF + EPS + PNG)
# -----------------------------
outbase = OUTDIR / "case4_6param_performance_time_parity_2x2"
fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(outbase.with_suffix(".eps"), bbox_inches="tight")
fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(outbase.with_suffix(".pdf"))
print(outbase.with_suffix(".eps"))
print(outbase.with_suffix(".png"))




# import pandas as pd
# import numpy as np

# CENTRAL_XLSX  = "doe_benchmark_central.xlsx"
# SYMBOLIC_XLSX = "doe_benchmark_symbolic.xlsx"
# CASE_NAME = "case4_6param"
# metric = "build_time_s"

# cdf = pd.read_excel(CENTRAL_XLSX, sheet_name=CASE_NAME)
# sdf = pd.read_excel(SYMBOLIC_XLSX, sheet_name=CASE_NAME)

# # if present, keep only successful runs
# if "returncode" in cdf.columns:
#     cdf = cdf[cdf["returncode"] == 0].copy()
# if "returncode" in sdf.columns:
#     sdf = sdf[sdf["returncode"] == 0].copy()

# m = pd.merge(cdf[["run_id", metric]], sdf[["run_id", metric]],
#              on="run_id", suffixes=("_central","_symbolic")).sort_values("run_id")

# print(m)

# x = m[f"{metric}_central"].to_numpy()
# y = m[f"{metric}_symbolic"].to_numpy()

# print("x min/max:", x.min(), x.max())
# print("y min/max:", y.min(), y.max())

# # If these asserts pass, there is no (0.1,0.1) point in the case4 build-time data.
# assert x.min() > 0.25, "Central build_time has unexpectedly small values (<<0.3)."
# assert y.min() > 0.10, "Symbolic build_time has unexpectedly small values (<<0.12)."
