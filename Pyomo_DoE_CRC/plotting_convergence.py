#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# -----------------------------
# INPUTS
# -----------------------------
CENTRAL_XLSX  = Path("doe_benchmark_central.xlsx")
SYMBOLIC_XLSX = Path("doe_benchmark_symbolic.xlsx")

# -----------------------------
# OUTPUTS
# -----------------------------
BASE_OUTDIR = Path("results_plots")
OUTDIR = BASE_OUTDIR / "convergence_properties"
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# ESCAPE one-column figure sizing (8.5 cm wide)
# -----------------------------
FIG_W = 3.35  # inches
FIG_H = 2.40  # inches

mpl.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.0,
})

# -----------------------------
# Metrics (unscaled IPOPT end-of-solve diagnostics)
# -----------------------------
PLOTS = [
    ("ipopt_final_obj_unscaled",        "D-optimality objective (unscaled)", "bars_d_opt_unscaled"),
    ("ipopt_dual_inf_unscaled",         "Dual infeasibility (unscaled)",     "bars_dual_infeas_unscaled"),
    ("ipopt_constr_viol_unscaled",      "Constraint violation (unscaled)",   "bars_constr_viol_unscaled"),
    ("ipopt_compl_unscaled",            "Complementarity (unscaled)",        "bars_complementarity_unscaled"),
    ("ipopt_overall_nlp_err_unscaled",  "Overall NLP error (unscaled)",      "bars_overall_nlp_error_unscaled"),
]

def load_workbook_sheets(path: Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    for k in sheets:
        sheets[k].columns = [str(c).strip() for c in sheets[k].columns]
    return sheets

def unique_value(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return np.nan
    v = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
    if len(v) == 0:
        return np.nan
    u = np.unique(v)
    return float(u[0]) if len(u) == 1 else float(np.mean(v))  # should be constant; mean is safe fallback

def overlay_bar_chart(metric_col: str, stem: str,
                      central_sheets: dict[str, pd.DataFrame],
                      symbolic_sheets: dict[str, pd.DataFrame]):

    cases = sorted(set(central_sheets.keys()) & set(symbolic_sheets.keys()))
    if not cases:
        return

    central_vals = []
    symbolic_vals = []
    case_labels = []

    for case in cases:
        c = unique_value(central_sheets[case], metric_col)
        s = unique_value(symbolic_sheets[case], metric_col)
        if np.isfinite(c) and np.isfinite(s):
            case_labels.append(case)
            central_vals.append(c)
            symbolic_vals.append(s)

    if len(case_labels) == 0:
        return

    x = np.arange(len(case_labels))
    width = 0.75

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Plot symbolic first (transparent grey), then central (white with thick border) so border stays visible
    ax.bar(
        x,
        symbolic_vals,
        width=width,
        facecolor="0.6",
        edgecolor="none",
        alpha=0.35,
        label="Symbolic",
        zorder=1,
    )

    ax.bar(
        x,
        central_vals,
        width=width,
        facecolor="white",
        edgecolor="black",
        linewidth=2.0,
        alpha=0.90,
        label="Central FD",
        zorder=2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, rotation=0, ha="center")
    ax.set_ylabel("")     # no y-label
    ax.set_title("")      # no title
    fig.suptitle("")      # extra safety: no title

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)

    # Boxed axes
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    # Legend: top-right, multiple rows (ncol=1 => 2 entries stacked)
    handles, labels = ax.get_legend_handles_labels()
    # Put Central first, then Symbolic
    ax.legend(
        [handles[1], handles[0]],
        [labels[1], labels[0]],
        loc="upper right",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        borderpad=0.3,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    fig.tight_layout()

    outbase = OUTDIR / stem
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".eps"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", outbase.with_suffix(".pdf"))
    print("Saved:", outbase.with_suffix(".eps"))
    print("Saved:", outbase.with_suffix(".png"))

def main():
    central_sheets = load_workbook_sheets(CENTRAL_XLSX)
    symbolic_sheets = load_workbook_sheets(SYMBOLIC_XLSX)

    for metric_col, _metric_label, stem in PLOTS:
        overlay_bar_chart(metric_col, stem, central_sheets, symbolic_sheets)

if __name__ == "__main__":
    main()
