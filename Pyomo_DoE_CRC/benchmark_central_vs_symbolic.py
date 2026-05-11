#!/usr/bin/env python3
"""
Benchmark runner for FOCAPO/CPC Pyomo.DoE central-vs-symbolic comparisons.

Author: Shilpa Narasimhan
Support: Codex
QA/testing: Shilpa Narasimhan
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


METRIC_COLUMNS = [
    "solve_time_s",
    "build_time_s",
    "initialization_time_s",
    "wall_time_s",
    "ipopt_iterations",
    "objective_value",
    "fim_condition_number",
    "solver_status",
    "termination_condition",
    "termination_message",
    "obj_fun_evals",
    "obj_grad_evals",
    "eq_constr_evals",
    "eq_jac_evals",
    "lag_hess_evals",
    "cpu_time_s",
    "nlp_eval_time_s",
]


CASE_LABELS = {
    "4st_6pmt": "six_parameter",
    "4_state_reactor": "four_state_reactor",
    "PDE_diffusion": "pde",
    "two_param_sin": "two_param_sin",
}


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


@dataclass
class CasePair:
    """Container for central/symbolic script pair metadata."""

    case_key: str
    central_script: Path
    symbolic_script: Path


def discover_case_pairs(root: Path) -> Dict[str, CasePair]:
    """Discover all *_central.py / *_sym.py pairs under root."""
    central_files = {p.stem.replace("_central", ""): p for p in root.glob("*_central.py")}
    sym_files = {p.stem.replace("_sym", ""): p for p in root.glob("*_sym.py")}
    keys = sorted(set(central_files).intersection(sym_files))
    pairs = {
        k: CasePair(case_key=k, central_script=central_files[k], symbolic_script=sym_files[k])
        for k in keys
    }
    return pairs


def select_required_cases(pairs: Dict[str, CasePair]) -> Dict[str, CasePair]:
    """Return required benchmark systems for the final combined figure."""
    required = ["two_param_sin", "PDE_diffusion", "4_state_reactor", "4st_6pmt"]
    missing = [k for k in required if k not in pairs]
    if missing:
        raise RuntimeError(
            f"Missing required case(s): {missing}. "
            "Expected two_param_sin, PDE_diffusion, 4_state_reactor, and 4st_6pmt."
        )
    return {k: pairs[k] for k in required}


def regex_float(pattern: str, text: str) -> float:
    """Extract first regex float group or NaN."""
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return np.nan
    try:
        return float(m.group(1))
    except Exception:
        return np.nan


def regex_float_last(pattern: str, text: str) -> float:
    """Extract last regex float group or NaN."""
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return np.nan
    last = matches[-1]
    if isinstance(last, tuple):
        last = last[0]
    try:
        return float(last)
    except Exception:
        return np.nan


def regex_text(pattern: str, text: str) -> Optional[str]:
    """Extract first regex text group or None."""
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip()


def regex_text_last(pattern: str, text: str) -> Optional[str]:
    """Extract last regex text group or None."""
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    last = matches[-1]
    if isinstance(last, tuple):
        last = last[0]
    return str(last).strip()


def parse_fim_block(stdout: str) -> Optional[np.ndarray]:
    """Parse printed FIM block into a square numpy array when possible."""
    block_match = re.search(
        r"FIM at optimal design:\s*(?:\\n|\n)?\s*(\[\[.*?\]\])",
        stdout,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not block_match:
        return None
    block = block_match.group(1)
    nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", block)
    if not nums:
        return None
    vals = np.array([float(v) for v in nums], dtype=float)
    n = int(round(math.sqrt(vals.size)))
    if n * n != vals.size:
        return None
    return vals.reshape((n, n))


def parse_metrics(stdout: str, stderr: str) -> Dict[str, object]:
    """Parse benchmark metrics from stdout/stderr text."""
    text = f"{stdout}\n{stderr}"
    metrics: Dict[str, object] = {
        "solve_time_s": regex_float_last(
            r"Solve time \(s\):\s*([0-9eE+\-\.]+)", text
        ),
        "build_time_s": regex_float_last(
            r"Build time \(s\):\s*([0-9eE+\-\.]+)", text
        ),
        "initialization_time_s": regex_float_last(
            r"Initialization time \(s\):\s*([0-9eE+\-\.]+)", text
        ),
        "wall_time_s": regex_float_last(
            r"Total wall time \(s\):\s*([0-9eE+\-\.]+)", text
        ),
        "ipopt_iterations": regex_float_last(
            r"Number of Iterations.*:\s*(\d+)", text
        ),
        "objective_value": regex_float_last(
            r"Objective value at optimal design:\s*([0-9eE+\-\.]+)", text
        ),
        "fim_condition_number": regex_float_last(
            r"FIM Condition Number['\"]?\]?\)?[:=\s]+([0-9eE+\-\.]+)", text
        ),
        "solver_status": regex_text_last(
            r"Solver Status[:=\s]+([A-Za-z_]+)", text
        ),
        "termination_condition": regex_text_last(
            r"termination condition:\s*([A-Za-z_]+)", text
        )
        or regex_text_last(r"Termination Condition[:=\s]+([A-Za-z_]+)", text),
        "termination_message": regex_text_last(
            r"Termination Message[:=\s]+(.+)", text
        )
        or regex_text_last(r"EXIT:\s*(.+)", text),
        "obj_fun_evals": regex_float_last(
            r"Number of objective function evaluations\s*=\s*(\d+)", text
        ),
        "obj_grad_evals": regex_float_last(
            r"Number of objective gradient evaluations\s*=\s*(\d+)", text
        ),
        "eq_constr_evals": regex_float_last(
            r"Number of equality constraint evaluations\s*=\s*(\d+)", text
        ),
        "eq_jac_evals": regex_float_last(
            r"Number of equality constraint Jacobian evaluations\s*=\s*(\d+)", text
        ),
        "lag_hess_evals": regex_float_last(
            r"Number of Lagrangian Hessian evaluations\s*=\s*(\d+)", text
        ),
        "cpu_time_s": regex_float_last(
            r"Total CPU secs in IPOPT \(w/o function evaluations\)\s*=\s*([0-9eE+\-\.]+)",
            text,
        ),
        "nlp_eval_time_s": regex_float_last(
            r"Total CPU secs in NLP function evaluations\s*=\s*([0-9eE+\-\.]+)",
            text,
        ),
    }

    if np.isnan(metrics["fim_condition_number"]):
        fim = parse_fim_block(stdout)
        if fim is not None:
            try:
                metrics["fim_condition_number"] = float(np.linalg.cond(fim))
            except Exception:
                metrics["fim_condition_number"] = np.nan

    return metrics


def run_script(script_path: Path, workdir: Path, timeout_s: int = 1800) -> Tuple[int, str, str]:
    """Execute a benchmark script with subprocess and capture stdout/stderr."""
    cmd = [sys.executable, str(script_path.resolve())]
    proc = subprocess.run(
        cmd,
        cwd=str(script_path.resolve().parent),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_benchmarks(
    pairs: Dict[str, CasePair],
    runs: int,
    selected_case: Optional[str],
    logs_dir: Path,
) -> pd.DataFrame:
    """Run central and symbolic scripts for all selected cases and collect raw rows."""
    rows: List[Dict[str, object]] = []
    logs_dir.mkdir(parents=True, exist_ok=True)
    case_keys = [selected_case] if selected_case else list(pairs.keys())

    for case_key in case_keys:
        pair = pairs[case_key]
        for method, script in [("central", pair.central_script), ("symbolic", pair.symbolic_script)]:
            for run_idx in range(1, runs + 1):
                return_code, stdout, stderr = run_script(script, SCRIPT_DIR)
                parsed = parse_metrics(stdout, stderr)

                failure_message = None
                if return_code != 0:
                    failure_message = f"Non-zero return code: {return_code}"
                elif re.search(r"Traceback \(most recent call last\):", stdout + "\n" + stderr):
                    failure_message = "Python traceback detected"

                run_tag = f"{case_key}__{method}__run{run_idx:02d}"
                stdout_path = logs_dir / f"{run_tag}.stdout.log"
                stderr_path = logs_dir / f"{run_tag}.stderr.log"
                stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
                stderr_path.write_text(stderr, encoding="utf-8", errors="replace")

                row = {
                    "case_key": case_key,
                    "case_label": CASE_LABELS.get(case_key, case_key),
                    "script": script.name,
                    "method": method,
                    "run_index": run_idx,
                    "return_code": return_code,
                    "success": failure_message is None,
                    "failure_message": failure_message,
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                }
                row.update(parsed)
                rows.append(row)

    return pd.DataFrame(rows)


def aggregate_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate run-level records into per-case/per-method averages."""
    numeric_cols = [
        "solve_time_s",
        "build_time_s",
        "initialization_time_s",
        "wall_time_s",
        "ipopt_iterations",
        "objective_value",
        "fim_condition_number",
        "obj_fun_evals",
        "obj_grad_evals",
        "eq_constr_evals",
        "eq_jac_evals",
        "lag_hess_evals",
        "cpu_time_s",
        "nlp_eval_time_s",
    ]
    agg = (
        raw_df.groupby(["case_key", "case_label", "method"], dropna=False)
        .agg(
            runs=("run_index", "count"),
            successes=("success", "sum"),
            failures=("success", lambda s: int((~s).sum())),
            **{f"avg_{c}": (c, "mean") for c in numeric_cols},
            **{f"std_{c}": (c, "std") for c in numeric_cols},
        )
        .reset_index()
    )
    return agg


def plot_case(summary_df: pd.DataFrame, case_key: str, out_png: Path) -> None:
    """Generate one high-resolution PNG plot for a single benchmark case."""
    sub = summary_df[summary_df["case_key"] == case_key].copy()
    if sub.empty:
        return
    sub = sub.set_index("method")
    methods = ["central", "symbolic"]

    counts_metrics = [
        ("avg_ipopt_iterations", "Iterations"),
        ("avg_obj_fun_evals", "Obj. fun."),
        ("avg_obj_grad_evals", "Obj. grad."),
        ("avg_eq_constr_evals", "Eq. constr."),
        ("avg_eq_jac_evals", "Eq. Jacob."),
        ("avg_lag_hess_evals", "Lagr. Hess."),
    ]
    time_metrics = [
        ("avg_solve_time_s", "Solve"),
        ("avg_build_time_s", "Build"),
        ("avg_initialization_time_s", "Initialization"),
        ("avg_wall_time_s", "Wall-clock"),
    ]
    quality_metrics = [
        ("avg_objective_value", "Objective"),
        ("avg_fim_condition_number", "FIM cond."),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [
        (axes[0], counts_metrics, "Opt. Performance (counts)"),
        (axes[1], time_metrics, "Comp. Time (seconds)"),
        (axes[2], quality_metrics, "Quality Metrics"),
    ]

    colors = {"central": "#ffffff", "symbolic": "#9e9e9e"}
    edge = "#000000"
    central_w = 0.70
    symbolic_w = 0.45

    for ax, metric_pairs, title in panels:
        labels = [lab for _, lab in metric_pairs]
        x = np.arange(len(labels), dtype=float)
        for i, method in enumerate(methods):
            vals = []
            for col, _lab in metric_pairs:
                v = sub.at[method, col] if method in sub.index and col in sub.columns else np.nan
                vals.append(v)
            vals = np.array(vals, dtype=float)
            ax.bar(
                x + (i - 0.5) * bar_w,
                vals,
                width=bar_w,
                label="Central finite difference" if method == "central" else "Symbolic derivatives",
                facecolor=colors[method],
                edgecolor=edge,
                linewidth=1.2,
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=9)

    eq_txt = EQUATION_TEXT.get(case_key, "")
    if eq_txt:
        fig.text(
            0.5,
            0.99,
            eq_txt,
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="#d8ecff", edgecolor="none", pad=3),
        )
    fig.suptitle(f"{CASE_LABELS.get(case_key, case_key)}", y=1.07, fontsize=13, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.17))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined(summary_df: pd.DataFrame, out_png: Path) -> None:
    """Generate optional combined summary plot across cases."""
    cases = list(summary_df["case_key"].drop_duplicates())
    if not cases:
        return
    metrics = ["avg_wall_time_s", "avg_ipopt_iterations", "avg_objective_value", "avg_fim_condition_number"]
    metric_labels = ["Wall time", "Iterations", "Objective", "FIM cond."]
    methods = ["central", "symbolic"]

    fig, axes = plt.subplots(len(cases), 1, figsize=(12, 3.1 * len(cases)), squeeze=False)
    for r, case_key in enumerate(cases):
        ax = axes[r, 0]
        sub = summary_df[summary_df["case_key"] == case_key].set_index("method")
        x = np.arange(len(metrics), dtype=float)
        for i, method in enumerate(methods):
            vals = [sub.at[method, m] if method in sub.index else np.nan for m in metrics]
            ax.bar(x + (i - 0.5) * 0.36, vals, width=0.36, label=method.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=20, ha="right")
        ax.set_title(CASE_LABELS.get(case_key, case_key), fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle=":")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_reference_style(summary_df: pd.DataFrame, out_png: Path) -> None:
    """Create one publication-style 3x2 figure matching the provided reference layout."""
    case_order = ["two_param_sin", "4_state_reactor", "PDE_diffusion", "4st_6pmt"]
    case_titles = {
        "two_param_sin": "1 parameter ODE system",
        "4_state_reactor": "4 parameter DAE system$^{[2]}$",
        "PDE_diffusion": "1 parameter PDE system$^{[3]}$",
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

    # Compact 4x2 structure to match reference aspect and spacing
    fig, axes = plt.subplots(4, 2, figsize=(7.8, 14.2), constrained_layout=False)
    plt.subplots_adjust(top=0.885, bottom=0.048, left=0.075, right=0.985, hspace=0.42, wspace=0.30)

    # Legend and headers
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
    # Fixed metric order for the two required columns
    count_cols = [c for c, _ in counts]
    time_cols = [c for c, _ in times]
    # Reference-like y-axis limits by row and panel
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
        # Validate all required displayed metrics are present and finite before plotting
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
            # Keep finite-value validation only; zero values can be legitimate for
            # fast solves in some IPOPT timing counters.

        for cidx, metric_pairs in enumerate([counts, times]):
            ax = axes[ridx, cidx]
            x = np.arange(len(metric_pairs), dtype=float)
            central_vals = []
            symbolic_vals = []
            for col, _lab in metric_pairs:
                central_vals.append(sub.at["central", col] if ("central" in sub.index and col in sub.columns) else np.nan)
                symbolic_vals.append(sub.at["symbolic", col] if ("symbolic" in sub.index and col in sub.columns) else np.nan)
            central_vals = np.array(central_vals, dtype=float)
            symbolic_vals = np.array(symbolic_vals, dtype=float)

            # Overlay style: central (wide) behind symbolic (narrow), same x-position
            ax.bar(
                x,
                central_vals,
                width=central_w,
                facecolor=colors["central"],
                edgecolor=edge,
                linewidth=1.8,
                zorder=2,
            )
            ax.bar(
                x,
                symbolic_vals,
                width=symbolic_w,
                facecolor=colors["symbolic"],
                edgecolor=edge,
                linewidth=1.1,
                zorder=3,
            )
            ax.set_xticks(x)
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

        # Section title in light-blue box
        left_ax = axes[ridx, 0]
        right_ax = axes[ridx, 1]
        y_top = max(left_ax.get_position().y1, right_ax.get_position().y1)
        y_bot = min(left_ax.get_position().y0, right_ax.get_position().y0)
        x_left = left_ax.get_position().x0
        title_artist = fig.text(
            x_left + 0.01,
            y_bot - 0.055,
            case_titles[case_key],
            ha="left",
            va="top",
            fontsize=12,
            bbox=dict(facecolor="#b9dbff", edgecolor="none", pad=4),
        )
        # Equation box OUTSIDE axes, in whitespace above right subplot
        eq = EQUATION_TEXT.get(case_key, "")
        eq_x = right_ax.get_position().x0 + 0.012
        eq_y = y_bot - 0.050
        fig.text(
            eq_x,
            eq_y,
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

        # No horizontal divider lines between systems (per requested style)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(description="Benchmark central vs symbolic Pyomo.DoE scripts.")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per script (default: 10).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for CSV files and plots.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        choices=["six_parameter", "pde", "two_param_sin", "four_state_reactor"],
        help="Run a single case only.",
    )
    return parser


def case_choice_to_key(case_choice: Optional[str]) -> Optional[str]:
    """Map CLI case labels to internal case key."""
    if case_choice is None:
        return None
    reverse = {v: k for k, v in CASE_LABELS.items()}
    return reverse[case_choice]


def main() -> None:
    """Program entry point."""
    args = build_parser().parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")

    pairs = discover_case_pairs(SCRIPT_DIR)
    pairs = select_required_cases(pairs)
    selected_key = case_choice_to_key(args.case)

    out_dir = args.output_dir
    plots_dir = out_dir / "plots"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    raw_df = run_benchmarks(
        pairs=pairs,
        runs=args.runs,
        selected_case=selected_key,
        logs_dir=logs_dir,
    )
    summary_df = aggregate_summary(raw_df)

    raw_df.to_csv(out_dir / "raw_results.csv", index=False)
    summary_df.to_csv(out_dir / "summary_results.csv", index=False)
    excel_df = summary_df.copy()
    excel_df["case"] = excel_df["case_label"]
    excel_df["avg_solve_time"] = excel_df["avg_solve_time_s"]
    excel_df["avg_build_time"] = excel_df["avg_build_time_s"]
    excel_df["avg_init_time"] = excel_df["avg_initialization_time_s"]
    excel_df["avg_wall_time"] = excel_df["avg_wall_time_s"]
    excel_df["avg_iterations"] = excel_df["avg_ipopt_iterations"]
    excel_df["avg_objective_value"] = excel_df["avg_objective_value"]
    excel_df["avg_FIM_condition_number"] = excel_df["avg_fim_condition_number"]
    excel_df = excel_df[
        [
            "case",
            "method",
            "avg_solve_time",
            "avg_build_time",
            "avg_init_time",
            "avg_wall_time",
            "avg_iterations",
            "avg_objective_value",
            "avg_FIM_condition_number",
        ]
    ]
    excel_df.to_excel(out_dir / "benchmark_summary.xlsx", index=False)

    # Per request: single final figure only (no per-system PNGs).
    if selected_key is not None:
        raise RuntimeError("--case mode is incompatible with final combined 3-system figure.")
    plot_final_reference_style(summary_df, plots_dir / "central_vs_symbolic_benchmark.png")

    print(f"Wrote: {out_dir / 'raw_results.csv'}")
    print(f"Wrote: {out_dir / 'summary_results.csv'}")
    print(f"Wrote: {out_dir / 'benchmark_summary.xlsx'}")
    print(f"Wrote logs: {logs_dir}")
    print(f"Wrote: {plots_dir / 'central_vs_symbolic_benchmark.png'}")


if __name__ == "__main__":
    main()
