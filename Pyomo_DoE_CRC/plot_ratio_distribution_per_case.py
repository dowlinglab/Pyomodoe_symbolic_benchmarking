#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CENTRAL_XLSX = Path("doe_benchmark_central.xlsx")
SYMBOLIC_XLSX = Path("doe_benchmark_symbolic.xlsx")
OUTDIR = Path("results_plots") / "ratio_distribution"

METRICS = [
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


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def _build_merged_case(central: pd.DataFrame, symbolic: pd.DataFrame) -> pd.DataFrame:
    c = central.copy()
    s = symbolic.copy()

    c["solve_time_per_iter"] = _safe_div(c["solve_time_s"], c["ipopt_iterations"])
    s["solve_time_per_iter"] = _safe_div(s["solve_time_s"], s["ipopt_iterations"])

    merged = pd.merge(
        s[["run_id"] + [m for m, _ in METRICS]],
        c[["run_id"] + [m for m, _ in METRICS]],
        on="run_id",
        how="inner",
        suffixes=("_symbolic", "_central"),
    ).sort_values("run_id", kind="stable")

    return merged


def _compute_stats(merged: pd.DataFrame) -> tuple[list[float], list[float], list[str]]:
    means: list[float] = []
    stds: list[float] = []
    labels: list[str] = []

    for metric, label in METRICS:
        ratios = _safe_div(
            merged[f"{metric}_symbolic"],
            merged[f"{metric}_central"],
        )
        ratios = ratios.clip(upper=1.0)
        valid = ratios[np.isfinite(ratios)]
        if valid.empty:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(float(valid.mean()))
            stds.append(float(valid.std(ddof=0)))
        labels.append(label)

    return means, stds, labels


def _plot_case(means: list[float], stds: list[float], labels: list[str], output_path: Path) -> None:
    fig = plt.figure(figsize=(14, 8))
    y_pos = np.arange(len(labels))
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(labels)))

    plt.barh(
        y_pos,
        means,
        xerr=stds,
        alpha=0.7,
        color=colors,
        error_kw=dict(
            elinewidth=3,
            capsize=6,
            capthick=3,
            ecolor="black",
        ),
    )

    plt.axvline(1.0, color="black", linewidth=3, linestyle="--")
    plt.axvline(0.5, color="black", linewidth=3)

    plt.yticks(y_pos, labels, fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel(
        r"$\mathrm{Ratio}\left(\frac{\mathrm{symbolic}}{\mathrm{central}}\right)$",
        fontsize=18,
    )
    plt.ylabel("")
    plt.grid(True, axis="x", linestyle="--", linewidth=0.6)

    plt.gca().invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    if not CENTRAL_XLSX.exists() or not SYMBOLIC_XLSX.exists():
        raise SystemExit("Input Excel files not found in current directory.")

    central_xl = pd.ExcelFile(CENTRAL_XLSX)
    symbolic_xl = pd.ExcelFile(SYMBOLIC_XLSX)
    common_sheets = [s for s in central_xl.sheet_names if s in symbolic_xl.sheet_names]
    sheets_to_plot = [s for idx, s in enumerate(common_sheets) if idx != 2]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y-%m-%d")

    for sheet in sheets_to_plot:
        cdf = pd.read_excel(CENTRAL_XLSX, sheet_name=sheet)
        sdf = pd.read_excel(SYMBOLIC_XLSX, sheet_name=sheet)
        merged = _build_merged_case(cdf, sdf)
        means, stds, labels = _compute_stats(merged)
        output_path = OUTDIR / f"case_{sheet}_ratio_distribution_{date_tag}.png"
        _plot_case(means, stds, labels, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
