#!/usr/bin/env python3
"""
Generate the final symbolic/central ratio figure from an existing aggregated table.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


METRIC_SPECS: Sequence[Tuple[str, str]] = [
    ("doe_build_time", "Build Time"),
    ("doe_init_time", "Initialization Time"),
    ("doe_solve_time", "Solve Time"),
    ("doe_wall_time", "Wall Time"),
    ("solve_time_per_iter", "Solve Time per Iteration"),
    ("ipopt_iters", "Iteration Count"),
    ("ipopt_cpu_wo_feval", "CPU Time (without feval)"),
    ("ipopt_cpu_nlp_feval", "CPU Time (feval)"),
    ("obj_eval", "Objective Evaluations"),
    ("grad_eval", "Gradient Evaluations"),
    ("eq_con_eval", "Constraint Evaluations"),
    ("eq_jac_eval", "Jacobian Evaluations"),
    ("hess_eval", "Hessian Evaluations"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help="Path to aggregated pde_scaling_table.csv (default: newest one under out/).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("out"),
        help="Directory for output figure (default: out/).",
    )
    return parser.parse_args()


def find_latest_table(root: Path) -> Path:
    candidates = list(root.glob("**/pde_scaling_table.csv"))
    if not candidates:
        raise SystemExit(f"ERROR: no pde_scaling_table.csv found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def normalize_methods(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "method" not in out.columns:
        raise SystemExit("ERROR: input table must contain 'method' column")
    out["method"] = out["method"].astype(str).str.strip().str.lower()
    return out


def ensure_solve_time_per_iter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "solve_time_per_iter" in out.columns:
        return out
    needed = {"doe_solve_time", "ipopt_iters"}
    if not needed.issubset(out.columns):
        return out
    solve = pd.to_numeric(out["doe_solve_time"], errors="coerce")
    iters = pd.to_numeric(out["ipopt_iters"], errors="coerce")
    out["solve_time_per_iter"] = np.where(iters > 0, solve / iters, np.nan)
    return out


def discretization_sort_key(row: pd.Series) -> Tuple[float, float]:
    return (float(row["nfe_x"]), float(row["nfe_t"]))


def build_ratio_table(df: pd.DataFrame, metric_names: List[str]) -> pd.DataFrame:
    required = {"nfe_x", "nfe_t", "method"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: input table missing required columns: {sorted(missing)}")

    rows = []
    grouped = df.groupby(["nfe_x", "nfe_t"], sort=False)
    for (nfe_x, nfe_t), group in grouped:
        central = group[group["method"] == "central"]
        symbolic = group[group["method"] == "symbolic"]
        if central.empty or symbolic.empty:
            continue
        c = central.iloc[0]
        s = symbolic.iloc[0]
        row = {"nfe_x": nfe_x, "nfe_t": nfe_t}
        for metric in metric_names:
            c_val = pd.to_numeric(c.get(metric), errors="coerce")
            s_val = pd.to_numeric(s.get(metric), errors="coerce")
            if pd.isna(c_val) or pd.isna(s_val) or c_val <= 0 or s_val <= 0:
                row[metric] = np.nan
            else:
                row[metric] = float(s_val) / float(c_val)
        rows.append(row)

    ratio_df = pd.DataFrame(rows)
    if ratio_df.empty:
        raise SystemExit("ERROR: no valid central/symbolic pairs found to compute ratios")
    ratio_df = ratio_df.sort_values(by=["nfe_x", "nfe_t"], kind="stable").reset_index(drop=True)
    return ratio_df


def plot_grouped_horizontal_ratio(
    ratio_df: pd.DataFrame,
    metric_specs: Sequence[Tuple[str, str]],
    output_path: Path,
) -> None:
    metrics = [m for m, _ in metric_specs]
    labels = [label for _, label in metric_specs]
    n_groups = len(ratio_df)
    n_metrics = len(metrics)
    if n_groups == 0 or n_metrics == 0:
        raise SystemExit("ERROR: nothing to plot")

    fig, ax = plt.subplots(figsize=(14, 8))
    base_y = np.arange(n_groups, dtype=float)
    group_height = 0.82
    bar_h = group_height / max(n_metrics, 1)
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2.0) * bar_h

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = pd.to_numeric(ratio_df[metric], errors="coerce").to_numpy(dtype=float)
        y = base_y + offsets[i]
        valid = np.isfinite(vals) & (vals > 0)
        if np.any(valid):
            ax.barh(y[valid], vals[valid], height=bar_h * 0.95, label=label)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_minor_locator(LogLocator(base=2.0, subs=np.arange(1.1, 2.0, 0.1)))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _pos: f"{int(x)}" if np.isfinite(x) and x >= 1 and float(x).is_integer() else "")
    )
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.7)
    ax.grid(True, which="minor", axis="x", linestyle="--", alpha=0.35)
    ax.grid(False, axis="y")
    ax.axvline(1.0, color="black", linewidth=3)
    ax.text(1.02, 0.98, "ratio = 1", transform=ax.get_xaxis_transform(), ha="left", va="top")

    ylabels = [f"({int(r.nfe_x)},{int(r.nfe_t)})" for _, r in ratio_df.iterrows()]
    ax.set_yticks(base_y)
    ax.set_yticklabels(ylabels, fontsize=13)

    ax.set_xlabel(
        r"$\mathrm{Ratio}\left(\frac{\mathrm{symbolic}}{\mathrm{central}}\right)$",
        fontsize=18,
    )
    ax.set_ylabel(r"$\mathrm{Discretization}\;(n_{x},\,n_{t})$", fontsize=16)
    ax.tick_params(axis="x", labelsize=13)

    legend = ax.legend(
        loc="lower right",
        ncol=2,
        frameon=True,
        framealpha=1.0,
        fontsize=12,
    )
    legend.get_frame().set_alpha(1.0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table_path = args.table if args.table else find_latest_table(Path("out"))
    if not table_path.exists():
        raise SystemExit(f"ERROR: table not found: {table_path}")

    df = pd.read_csv(table_path)
    df = normalize_methods(df)
    df = ensure_solve_time_per_iter(df)

    available_specs = [(m, label) for (m, label) in METRIC_SPECS if m in df.columns]
    if not available_specs:
        raise SystemExit("ERROR: none of the target metrics are present in the table")

    ratio_df = build_ratio_table(df, [m for m, _ in available_specs])
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = args.outdir / f"all_metrics_ratio_final_{today}.png"
    plot_grouped_horizontal_ratio(ratio_df, available_specs, out_path)

    print(f"Input table: {table_path}")
    print(f"Output figure: {out_path}")
    print(f"Metrics plotted ({len(available_specs)}): {[m for m, _ in available_specs]}")


if __name__ == "__main__":
    main()
