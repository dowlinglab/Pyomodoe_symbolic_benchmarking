#!/usr/bin/env python3
"""
Plot-only generator for FOCAPO/CPC central-vs-symbolic benchmark figure.

Reads precomputed summary data from Excel/CSV and writes the final PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EQUATION_TEXT = {
    "4st_6pmt": (
        r"$\dot{x}(t)=\theta_1 x(t)+\theta_2 u_1(t)+\theta_3 u_1^2(t)$" "\n"
        r"$\quad\ +\theta_4 u_2(t)+\theta_5 u_2^2(t)+\theta_6 e^{-x(t)}$"
    ),
    "PDE_diffusion": r"$\frac{\partial T}{\partial t}=\theta\,\frac{\partial^2 T}{\partial x^2}$",
    "4_state_reactor": (
        r"$\dot{C}_A(t)=-k_1 C_A(t)$" "\n"
        r"$\dot{C}_B(t)=k_1 C_A(t)-k_2 C_B(t)$" "\n"
        r"$C_A(0)=C_A(t)+C_B(t)+C_C(t)$" "\n"
        r"$k_i=A_i e^{-E_i/(RT)},\ i\in\{1,2\}$"
    ),
    "two_param_sin": r"$\dot{x}(t)=\theta_1 x(t)+\theta_2\sin(w\,x(t))+u(t)$",
}


def normalize_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize accepted summary input into canonical plot columns."""
    # Already canonical (summary_results / summary_results_full)
    if "case_key" in df.columns and "avg_ipopt_iterations" in df.columns:
        return df.copy()

    # Compact benchmark_summary format
    if "case" in df.columns and "avg_iterations" in df.columns:
        case_map = {
            "six_parameter": "4st_6pmt",
            "four_state_reactor": "4_state_reactor",
            "pde": "PDE_diffusion",
            "two_param_sin": "two_param_sin",
        }
        out = pd.DataFrame()
        out["case_key"] = df["case"].map(case_map)
        out["case_label"] = df["case"]
        out["method"] = df["method"]
        out["avg_solve_time_s"] = df["avg_solve_time"]
        out["avg_build_time_s"] = df["avg_build_time"]
        out["avg_initialization_time_s"] = df["avg_init_time"]
        out["avg_wall_time_s"] = df["avg_wall_time"]
        out["avg_ipopt_iterations"] = df["avg_iterations"]
        out["avg_objective_value"] = df["avg_objective_value"]
        out["avg_fim_condition_number"] = df["avg_FIM_condition_number"]
        # metrics not present in compact summary cannot be plotted in this mode
        for col in [
            "avg_obj_fun_evals",
            "avg_obj_grad_evals",
            "avg_eq_constr_evals",
            "avg_eq_jac_evals",
            "avg_lag_hess_evals",
            "avg_cpu_time_s",
            "avg_nlp_eval_time_s",
        ]:
            out[col] = np.nan
        return out

    raise RuntimeError("Unrecognized summary schema. Provide summary_results_full.xlsx or summary_results.csv.")


def plot_final_reference_style(summary_df: pd.DataFrame, out_png: Path) -> None:
    case_order = ["two_param_sin", "4_state_reactor", "PDE_diffusion", "4st_6pmt"]
    case_titles = {
        "two_param_sin": "1 parameter ODE system",
        "4_state_reactor": "4 parameter DAE system",
        "PDE_diffusion": "1 parameter PDE system",
        "4st_6pmt": "6 parameter ODE system",
    }
    methods = ["central", "symbolic"]
    colors = {"central": "#ffffff", "symbolic": "#a6a6a6"}
    edge = "#000000"

    counts = [
        ("avg_ipopt_iterations", "Iterations"),
        ("avg_obj_fun_evals", "Obj. fun."),
        ("avg_obj_grad_evals", "Obj. grad."),
        ("avg_eq_constr_evals", "Eq. constr."),
        ("avg_eq_jac_evals", "Eq. Jacobian"),
        ("avg_lag_hess_evals", "Lagr. Hess."),
    ]
    times = [
        ("avg_cpu_time_s", "CPU"),
        ("avg_nlp_eval_time_s", "NLP eval."),
        ("avg_solve_time_s", "Solve"),
        ("avg_build_time_s", "Build"),
        ("avg_initialization_time_s", "Initialization"),
        ("avg_wall_time_s", "Wall-clock"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(7.8, 14.2), constrained_layout=False)
    plt.subplots_adjust(top=0.885, bottom=0.048, left=0.075, right=0.985, hspace=0.42, wspace=0.30)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#ffffff", edgecolor=edge, linewidth=2),
        plt.Rectangle((0, 0), 1, 1, facecolor="#a6a6a6", edgecolor="#a6a6a6", linewidth=2),
    ]
    fig.legend(
        legend_handles,
        ["Central finite difference", "Symbolic derivatives"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        fontsize=12,
        handlelength=1.6,
        handleheight=1.6,
    )
    fig.text(0.24, 0.945, "Opt. Performance (counts)", ha="center", va="center", fontsize=14, fontweight="bold")
    fig.text(0.74, 0.945, "Comp. Time (seconds)", ha="center", va="center", fontsize=14, fontweight="bold")

    central_w = 0.70
    symbolic_w = 0.45
    count_cols = [c for c, _ in counts]
    time_cols = [c for c, _ in times]
    y_limits = {
        ("two_param_sin", 0): (0.0, 36.0),
        ("two_param_sin", 1): (0.0, 0.365),
        ("4_state_reactor", 0): (0.0, 24.0),
        ("4_state_reactor", 1): (0.0, 2.45),
        ("PDE_diffusion", 0): (0.0, 18.8),
        ("PDE_diffusion", 1): (0.0, 0.13),
        ("4st_6pmt", 0): (0.0, 430.0),
        ("4st_6pmt", 1): (0.0, 5.5),
    }

    for ridx, case_key in enumerate(case_order):
        sub = summary_df[summary_df["case_key"] == case_key].set_index("method")
        if sub.empty:
            raise RuntimeError(f"Missing summary rows for case '{case_key}'.")
        for method in methods:
            if method not in sub.index:
                raise RuntimeError(f"Missing method '{method}' rows for case '{case_key}'.")

        needed_cols = count_cols + time_cols
        missing_cols = [c for c in needed_cols if c not in sub.columns]
        if missing_cols:
            raise RuntimeError(f"Missing required metric columns for plotting: {missing_cols}")
        for method in methods:
            vals = sub.loc[method, needed_cols].to_numpy(dtype=float)
            if vals.size != len(needed_cols):
                raise RuntimeError(f"Metric length mismatch for case={case_key}, method={method}.")
            if not np.all(np.isfinite(vals)):
                raise RuntimeError(f"Non-finite metric values for case={case_key}, method={method}.")

        for cidx, metric_pairs in enumerate([counts, times]):
            ax = axes[ridx, cidx]
            x = np.arange(len(metric_pairs), dtype=float)
            central_vals = np.array([sub.at["central", col] for col, _ in metric_pairs], dtype=float)
            symbolic_vals = np.array([sub.at["symbolic", col] for col, _ in metric_pairs], dtype=float)

            ax.bar(x, central_vals, width=central_w, facecolor=colors["central"], edgecolor=edge, linewidth=1.8, zorder=2)
            ax.bar(x, symbolic_vals, width=symbolic_w, facecolor=colors["symbolic"], edgecolor=edge, linewidth=1.1, zorder=3)
            ax.set_xticks(x)
            # Use numeric tick labels like the reference screenshot
            if cidx == 0:
                labels = [str(i) for i in range(1, 7)]
            else:
                labels = [str(i) for i in range(7, 13)]
            ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
            ax.tick_params(axis="x", labeltop=False, top=False, labelbottom=True, bottom=True, pad=2)
            ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
            ax.tick_params(axis="y", labelsize=9)
            ylim = y_limits.get((case_key, cidx))
            if ylim is not None:
                ax.set_ylim(*ylim)
            for spine in ax.spines.values():
                spine.set_color("#000000")

        left_ax = axes[ridx, 0]
        right_ax = axes[ridx, 1]
        y_top = max(left_ax.get_position().y1, right_ax.get_position().y1)
        x_left = left_ax.get_position().x0
        y_bot = min(left_ax.get_position().y0, right_ax.get_position().y0)
        fig.text(
            x_left + 0.01,
            y_bot - 0.01,
            case_titles[case_key],
            ha="left",
            va="top",
            fontsize=12,
            bbox=dict(facecolor="#b9dbff", edgecolor="none", pad=4),
        )
        eq = EQUATION_TEXT.get(case_key, "")
        fig.text(
            right_ax.get_position().x0 + 0.012,
            y_bot - 0.050,
            eq,
            ha="left",
            va="top",
            fontsize=9.8,
            bbox=dict(facecolor="#b9dbff", edgecolor="none", pad=2.0),
        )

    # Top-row metric labels (alphabetic), shown once like the reference screenshot.
    top_left_ax = axes[0, 0]
    top_right_ax = axes[0, 1]
    left_labels = ["Iterations", "Obj. fun.", "Obj. grad.", "Eq. constr.", "Eq. Jacobian", "Lagr. Hess."]
    right_labels = ["CPU", "NLP eval.", "Solve", "Build", "Initialization", "Wall-clock"]
    for i, label in enumerate(left_labels):
        xf, yf = fig.transFigure.inverted().transform(
            top_left_ax.transData.transform((i, 1.02))
        )
        fig.text(xf, yf + 0.006, label, rotation=60, ha="left", va="bottom", fontsize=9, color="black")
    for i, label in enumerate(right_labels):
        xf, yf = fig.transFigure.inverted().transform(
            top_right_ax.transData.transform((i, 1.02))
        )
        fig.text(xf, yf + 0.006, label, rotation=60, ha="left", va="bottom", fontsize=9, color="black")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate final benchmark plot from summary data file.")
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("results/summary_results_full.xlsx"),
        help="Input summary file (.xlsx or .csv). Recommended: summary_results_full.xlsx",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/plots/central_vs_symbolic_benchmark.png"),
        help="Output PNG path",
    )
    return parser


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise RuntimeError(f"Unsupported summary file extension: {path.suffix}")
    return normalize_summary_columns(df)


def main() -> None:
    args = build_parser().parse_args()
    summary_df = load_summary(args.summary_file)
    plot_final_reference_style(summary_df, args.output_png)
    print(f"Wrote: {args.output_png}")


if __name__ == "__main__":
    main()
