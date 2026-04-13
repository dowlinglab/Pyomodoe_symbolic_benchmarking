#!/usr/bin/env python3
"""Build Case-2 speedup plot for all 4 benchmark problems (determinant + trace)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_CSV = Path("data_greybox_benchmarking/runs_all_12metrics_from_logs.csv")
OUTDIR = Path("data_greybox_benchmarking")
OUT_PNG = OUTDIR / "case2_speedup_large_all_cases_det_trace.png"
OUT_STATS = OUTDIR / "case2_speedup_large_all_cases_det_trace_stats.csv"

CASE_ORDER: List[Tuple[str, str]] = [
    ("two_param_sin", "1"),
    ("PDE_diffusion", "2"),
    ("4st_6pmt", "3"),
    ("4_state_reactor", "4"),
]

DET_COLOR = "#AEC7E8"
TRACE_COLOR = "#98DF8A"


def infer_problem(script_name: str) -> str:
    """Infer problem name from script filename when the column is missing."""
    name = str(script_name)
    if "two_param_sin" in name:
        return "two_param_sin"
    if "PDE_diffusion" in name:
        return "PDE_diffusion"
    if "4st_6pmt" in name:
        return "4st_6pmt"
    if "4_state_reactor" in name:
        return "4_state_reactor"
    return ""


def normalize_mode(df: pd.DataFrame) -> pd.Series:
    """Normalize mode labels to existing/greybox."""
    if "mode_case2" in df.columns:
        mode_series = df["mode_case2"].astype(str).str.lower().replace({"nan": ""})
        use_mode_case2 = mode_series.isin(["existing", "greybox"])
    else:
        mode_series = pd.Series([""] * len(df), index=df.index)
        use_mode_case2 = pd.Series([False] * len(df), index=df.index)

    raw_mode = df["mode"].astype(str).str.lower().replace({"central": "existing"})
    out = raw_mode.where(~use_mode_case2, mode_series)
    return out


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing context fields and keep only rows needed for Case-2 objectives."""
    out = df.copy()
    out["problem"] = out.get("problem", pd.Series([np.nan] * len(out), index=out.index))
    out["problem"] = out["problem"].where(out["problem"].notna(), out["script_name"].map(infer_problem))
    out["instance"] = out.get("instance", pd.Series([np.nan] * len(out), index=out.index))
    out["instance"] = out["instance"].where(out["instance"].notna(), "large")
    out["objective_option"] = out["objective_option"].astype(str).str.lower()
    out["mode_norm"] = normalize_mode(out)
    out["status"] = out["status"].astype(str).str.lower()
    out["run_index"] = pd.to_numeric(out["run_index"], errors="coerce")
    out["wall_time_s"] = pd.to_numeric(out["wall_time_s"], errors="coerce")

    filt = (
        out["problem"].isin([c[0] for c in CASE_ORDER])
        & out["instance"].eq("large")
        & out["objective_option"].isin(["determinant", "trace"])
        & out["status"].eq("ok")
        & out["mode_norm"].isin(["existing", "greybox"])
    )
    return out[filt].copy()


def pair_case_runs(df: pd.DataFrame, problem: str, objective: str) -> Tuple[pd.DataFrame, List[int]]:
    """Pair central/greybox rows by run index for one problem/objective and return missing indices."""
    sub = df[(df["problem"] == problem) & (df["objective_option"] == objective)].copy()
    # Keep latest record per run index/mode to avoid duplicate historical batches.
    sub = sub.sort_values("timestamp_utc", kind="stable").drop_duplicates(
        subset=["problem", "objective_option", "mode_norm", "run_index"], keep="last"
    )
    central = sub[sub["mode_norm"] == "existing"][["run_index", "wall_time_s"]].rename(
        columns={"wall_time_s": "wall_time_central"}
    )
    greybox = sub[sub["mode_norm"] == "greybox"][["run_index", "wall_time_s"]].rename(
        columns={"wall_time_s": "wall_time_greybox"}
    )
    paired = pd.merge(central, greybox, on="run_index", how="inner").sort_values("run_index")

    seen = set(int(x) for x in paired["run_index"].dropna().astype(int).tolist())
    expected = set(range(1, 11))
    missing = sorted(expected - seen)
    return paired, missing


def compute_case_stats(paired: pd.DataFrame) -> Tuple[int, float, float, bool, bool]:
    """Compute speedup stats and validation flags for one paired dataframe."""
    denom_ok = bool((paired["wall_time_greybox"] > 0).all()) if not paired.empty else True
    ratios = paired["wall_time_central"] / paired["wall_time_greybox"]
    no_nan = bool(np.isfinite(ratios).all()) if not ratios.empty else True
    n = int(ratios.shape[0])
    mean = float(ratios.mean()) if n > 0 else np.nan
    std = float(ratios.std(ddof=1)) if n > 1 else np.nan
    return n, mean, std, denom_ok, no_nan


def plot_case2(stats: pd.DataFrame) -> None:
    """Plot horizontal bars for determinant/trace speedup across all 4 cases."""
    fig, ax = plt.subplots(figsize=(12, 6.75))
    problem_rows = stats.drop_duplicates(subset=["problem", "y_label"]).sort_values("y_label")
    y = np.arange(len(problem_rows), dtype=float)
    bar_h = 0.34

    for objective, color, offset, label in [
        ("determinant", DET_COLOR, -bar_h / 2, "Determinant"),
        ("trace", TRACE_COLOR, bar_h / 2, "Trace"),
    ]:
        sub = stats[stats["objective_option"] == objective].sort_values("y_label")
        ax.barh(
            y + offset,
            sub["mean_ratio"].to_numpy(dtype=float),
            xerr=sub["std_ratio"].to_numpy(dtype=float),
            color=color,
            alpha=0.85,
            error_kw={"ecolor": "black", "elinewidth": 2.2, "capsize": 6, "capthick": 2.2},
            height=bar_h * 0.92,
            label=label,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(problem_rows["y_label"].tolist(), fontsize=15)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel(
        r"$\mathrm{Speedup\ Ratio}\left[\frac{t_{\mathrm{Central}}}{t_{\mathrm{Greybox}}}\right]$",
        fontsize=17,
    )
    ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.35, color="gray")
    ax.grid(False, axis="y")
    ax.legend(frameon=False, fontsize=12, loc="best")
    fig.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)


def main() -> None:
    """Build all-case determinant speedup stats and figure with validations."""
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    data = prepare_dataframe(df)

    rows: List[Dict[str, object]] = []
    all_denoms_ok = True
    all_no_nan = True
    all_n10 = True

    for problem, y_label in CASE_ORDER:
        for objective in ["determinant", "trace"]:
            paired, missing = pair_case_runs(data, problem, objective)
            n, mean, std, denom_ok, no_nan = compute_case_stats(paired)
            all_denoms_ok = all_denoms_ok and denom_ok
            all_no_nan = all_no_nan and no_nan
            all_n10 = all_n10 and (n == 10)
            print(f"{problem} [{objective}]: paired={n}, missing_indices={missing}")
            print(f"{problem} [{objective}]: N={n}, mean={mean}, std={std}")
            rows.append(
                {
                    "problem": problem,
                    "objective_option": objective,
                    "N": n,
                    "mean_ratio": mean,
                    "std_ratio": std,
                    "y_label": y_label,
                }
            )

    stats = pd.DataFrame(rows)
    stats[["problem", "objective_option", "N", "mean_ratio", "std_ratio"]].to_csv(OUT_STATS, index=False)
    plot_case2(stats)

    print(f"No NaNs: {all_no_nan}")
    print(f"No division by zero (greybox wall_time_s > 0): {all_denoms_ok}")
    print(f"All N = 10: {all_n10}")
    print(f"Saved plot: {OUT_PNG}")
    print(f"Saved stats: {OUT_STATS}")


if __name__ == "__main__":
    main()
