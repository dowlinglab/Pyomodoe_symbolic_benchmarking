#!/usr/bin/env python3
"""Build determinant/trace 2x2 composite ratio-12metrics figures for 4 systems."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS_CSV = Path("data_greybox_benchmarking/runs_all_12metrics_from_logs.csv")
BASELINE_CSV = Path("data_greybox_benchmarking/case2_speedup_large_all_cases_det_trace_matched_selected_central_cases.csv")
OUTDIR = Path("data_greybox_benchmarking")
OUT_DET = OUTDIR / "figure_det_2x2_ratio12metrics.png"
OUT_TRACE = OUTDIR / "figure_trace_2x2_ratio12metrics.png"

PROBLEM_ORDER: List[Tuple[str, str]] = [
    ("two_param_sin", "(1)"),
    ("PDE_diffusion", "(2)"),
    ("4st_6pmt", "(3)"),
    ("4_state_reactor", "(4)"),
]
OBJECTIVES = ["determinant", "trace"]

METRICS: List[Tuple[str, str]] = [
    ("build_time_s", "Build Time"),
    ("init_time_s", "Initialization Time"),
    ("solve_time_s", "Solve Time"),
    ("wall_time_s", "Wall Time"),
    ("solve_time_per_iter", "Solve Time per Iteration"),
    ("ipopt_iterations", "Iteration Count"),
    ("ipopt_cpu_no_eval_s", "CPU Time (without feval)"),
    ("ipopt_cpu_nlp_eval_s", "CPU Time (feval)"),
    ("ipopt_obj_eval", "Objective Evaluations"),
    ("ipopt_grad_eval", "Gradient Evaluations"),
    ("ipopt_eq_con_eval", "Constraint Evaluations"),
    ("ipopt_eq_jac_eval", "Jacobian Evaluations"),
    ("ipopt_hess_eval", "Hessian Evaluations"),
]


def safe_div(num: float, den: float) -> float:
    """Return num/den or NaN if invalid."""
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return np.nan
    return float(num / den)


def make_lookup(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Build stdout-log-path keyed lookup for quick row retrieval."""
    return {str(r.stdout_log_file): r for r in df.itertuples(index=False)}


def metric_value(row: pd.Series, key: str) -> float:
    """Fetch metric value from row, including derived solve_time_per_iter."""
    if key == "solve_time_per_iter":
        return safe_div(float(row.solve_time_s), float(row.ipopt_iterations))
    value = getattr(row, key)
    try:
        return float(value)
    except Exception:
        return np.nan


def compute_problem_objective_stats(
    runs_lookup: Dict[str, pd.Series],
    baseline_sub: pd.DataFrame,
) -> Tuple[List[float], List[float], List[int], List[int], bool, List[str]]:
    """Compute mean/std ratios per metric for one problem/objective baseline set."""
    central_indices = sorted(int(v) for v in baseline_sub["run_index"].tolist())
    dropped: List[int] = []
    pair_rows: List[Tuple[pd.Series, pd.Series, int]] = []

    for row in baseline_sub.itertuples(index=False):
        c = runs_lookup.get(str(row.central_log))
        g = runs_lookup.get(str(row.greybox_log))
        idx = int(row.run_index)
        if c is None or g is None:
            dropped.append(idx)
            continue
        pair_rows.append((c, g, idx))

    means: List[float] = []
    stds: List[float] = []
    any_invalid = False
    unavailable_metrics: List[str] = []

    for metric_key, _ in METRICS:
        vals: List[float] = []
        for c_row, g_row, _ in pair_rows:
            c_val = metric_value(c_row, metric_key)
            g_val = metric_value(g_row, metric_key)
            ratio = safe_div(g_val, c_val)
            if np.isfinite(ratio):
                vals.append(ratio)

        if not vals:
            means.append(np.nan)
            stds.append(np.nan)
            unavailable_metrics.append(metric_key)
            continue

        arr = np.array(vals, dtype=float)
        if not np.isfinite(arr).all():
            any_invalid = True
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan)

    return means, stds, central_indices, sorted(set(dropped)), any_invalid, unavailable_metrics


def plot_objective_2x2(
    objective: str,
    baseline_df: pd.DataFrame,
    runs_lookup: Dict[str, pd.Series],
    out_path: Path,
) -> None:
    """Create 2x2 figure for one objective with 4 active subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(METRICS)))
    active_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    all_valid = True
    for idx, (problem, code) in enumerate(PROBLEM_ORDER):
        ax = active_axes[idx]
        sub = baseline_df[(baseline_df["problem"] == problem) & (baseline_df["objective_option"] == objective)].copy()
        sub = sub.sort_values("run_index")

        means, stds, central_indices, dropped, invalid, unavailable = compute_problem_objective_stats(runs_lookup, sub)
        all_valid = all_valid and (not invalid)

        n_used = len(central_indices) - len(dropped)
        print(f"[{objective}] {problem} central indices: {central_indices}")
        print(f"[{objective}] {problem} dropped indices: {dropped}")
        print(f"[{objective}] {problem} N used: {n_used}")
        print(f"[{objective}] {problem} ratios finite: {not invalid}")
        if unavailable:
            print(f"[{objective}] {problem} unavailable metrics: {unavailable}")

        y = np.arange(len(METRICS))
        ax.barh(
            y,
            means,
            xerr=stds,
            alpha=0.7,
            color=colors,
            error_kw={"elinewidth": 3, "capsize": 6, "capthick": 3, "ecolor": "black"},
        )
        ax.axvline(1.0, color="black", linewidth=3, linestyle="-")
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6)
        ax.invert_yaxis()
        ax.set_yticks(y)
        ax.set_yticklabels([label for _, label in METRICS], fontsize=10)
        ax.tick_params(axis="x", labelsize=10)
        if idx in (2, 3):
            ax.set_xlabel(r"$\mathrm{Ratio}\left(\frac{\mathrm{Greybox}}{\mathrm{Central}}\right)$", fontsize=12)
        else:
            ax.set_xlabel("")

        ax.text(0.5, -0.17, code, transform=ax.transAxes, ha="center", va="top", fontsize=12)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[{objective}] no NaN/inf across subplot metric ratios: {all_valid}")
    print(f"[{objective}] saved figure: {out_path}")


def main() -> None:
    """Generate determinant and trace composite 4x4 figures."""
    if not RUNS_CSV.exists():
        raise SystemExit(f"Missing runs CSV: {RUNS_CSV}")
    if not BASELINE_CSV.exists():
        raise SystemExit(f"Missing baseline selection CSV: {BASELINE_CSV}")

    runs = pd.read_csv(RUNS_CSV)
    baseline = pd.read_csv(BASELINE_CSV)
    runs_lookup = make_lookup(runs)

    for objective, out_path in [("determinant", OUT_DET), ("trace", OUT_TRACE)]:
        plot_objective_2x2(
            objective=objective,
            baseline_df=baseline,
            runs_lookup=runs_lookup,
            out_path=out_path,
        )


if __name__ == "__main__":
    main()
