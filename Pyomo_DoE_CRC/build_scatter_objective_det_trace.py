#!/usr/bin/env python3
"""Build determinant/trace objective scatter plots (2x2) for 4 benchmark systems."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_CSV = Path("data_greybox_benchmarking/runs_all_12metrics_from_logs.csv")
OUTDIR = Path("data_greybox_benchmarking")
OUT_DET = OUTDIR / "scatter_determinant_objective.png"
OUT_TRACE = OUTDIR / "scatter_trace_objective.png"

PROBLEMS: List[Tuple[str, str]] = [
    ("two_param_sin", "(1)"),
    ("PDE_diffusion", "(2)"),
    ("4st_6pmt", "(3)"),
    ("4_state_reactor", "(4)"),
]

OBJ_VAL_RE = re.compile(r"Objective value at optimal design:\s*([-+0-9.eE]+)")


def infer_problem(script_name: str) -> str:
    """Infer problem from script name if column value is missing."""
    s = str(script_name)
    if "two_param_sin" in s:
        return "two_param_sin"
    if "PDE_diffusion" in s:
        return "PDE_diffusion"
    if "4st_6pmt" in s:
        return "4st_6pmt"
    if "4_state_reactor" in s:
        return "4_state_reactor"
    return ""


def normalize_mode(df: pd.DataFrame) -> pd.Series:
    """Normalize mode to existing/greybox."""
    base = df["mode"].astype(str).str.lower().replace({"central": "existing"})
    if "mode_case2" in df.columns:
        alt = df["mode_case2"].astype(str).str.lower().replace({"nan": ""})
        use_alt = alt.isin(["existing", "greybox"])
        return base.where(~use_alt, alt)
    return base


def parse_objective_from_log(path_str: str) -> float:
    """Parse objective value from a run stdout log."""
    path = Path(str(path_str))
    if not path.exists():
        return np.nan
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = OBJ_VAL_RE.findall(text)
    if not matches:
        return np.nan
    try:
        return float(matches[-1])
    except Exception:
        return np.nan


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and filter to valid rows for scatter plotting."""
    out = df.copy()
    out["problem"] = out.get("problem", pd.Series([np.nan] * len(out), index=out.index))
    out["problem"] = out["problem"].where(out["problem"].notna(), out["script_name"].map(infer_problem))
    out["instance"] = out.get("instance", pd.Series([np.nan] * len(out), index=out.index))
    out["instance"] = out["instance"].where(out["instance"].notna(), "large")
    out["mode_norm"] = normalize_mode(out)
    out["objective_option"] = out["objective_option"].astype(str).str.lower()
    out["status"] = out["status"].astype(str).str.lower()
    out["run_index"] = pd.to_numeric(out["run_index"], errors="coerce").astype("Int64")
    out["objective_value_from_log"] = out["stdout_log_file"].map(parse_objective_from_log)

    filt = (
        out["problem"].isin([p for p, _ in PROBLEMS])
        & out["instance"].eq("large")
        & out["status"].eq("ok")
        & out["mode_norm"].isin(["existing", "greybox"])
        & out["run_index"].notna()
    )
    return out[filt].copy()


def latest_by_key(df: pd.DataFrame) -> pd.DataFrame:
    """Keep latest row for repeated runs by key."""
    if "timestamp_utc" in df.columns:
        return df.sort_values("timestamp_utc", kind="stable").drop_duplicates(
            subset=["problem", "objective_option", "mode_norm", "run_index"], keep="last"
        )
    return df.drop_duplicates(subset=["problem", "objective_option", "mode_norm", "run_index"], keep="last")


def pair_problem_objective(df: pd.DataFrame, problem: str, objective: str) -> pd.DataFrame:
    """Pair central and greybox rows by run_index for one problem/objective."""
    sub = latest_by_key(df[(df["problem"] == problem) & (df["objective_option"] == objective)].copy())
    c = sub[sub["mode_norm"] == "existing"][["run_index", "objective_value_from_log"]].rename(
        columns={"objective_value_from_log": "x_central"}
    )
    g = sub[sub["mode_norm"] == "greybox"][["run_index", "objective_value_from_log"]].rename(
        columns={"objective_value_from_log": "y_greybox"}
    )
    paired = pd.merge(c, g, on="run_index", how="inner").sort_values("run_index")
    paired = paired[np.isfinite(paired["x_central"]) & np.isfinite(paired["y_greybox"])].copy()
    return paired


def scatter_figure(df: pd.DataFrame, objective: str, out_path: Path) -> None:
    """Create 2x2 scatter figure for one objective."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    any_data = False
    all_finite = True

    for idx, ((problem, code), ax) in enumerate(zip(PROBLEMS, axes_list)):
        paired = pair_problem_objective(df, problem, objective)
        run_ids = paired["run_index"].astype(int).tolist()
        n = len(paired)
        if n == 0:
            ax.axis("off")
            print(f"[{objective}] WARNING: {problem} has no paired runs; subplot skipped.")
            continue

        x = paired["x_central"].to_numpy(dtype=float)
        y = paired["y_greybox"].to_numpy(dtype=float)
        any_data = True

        finite_ok = bool(np.isfinite(x).all() and np.isfinite(y).all())
        all_finite = all_finite and finite_ok

        abs_diff = np.abs(y - x)
        mean_abs_diff = float(abs_diff.mean())
        max_abs_diff = float(abs_diff.max())
        if n > 1 and np.std(x) > 0 and np.std(y) > 0:
            corr = float(np.corrcoef(x, y)[0, 1])
        else:
            corr = np.nan

        lo = float(np.min(np.concatenate([x, y])))
        hi = float(np.max(np.concatenate([x, y])))
        span = max(hi - lo, 1e-12)
        pad = 0.05 * span
        xmin, xmax = lo - pad, hi + pad
        ax.scatter(x, y, s=35, color="#AEC7E8", alpha=0.8, edgecolors="none")
        ax.plot([xmin, xmax], [xmin, xmax], color="black", linewidth=2)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        ax.set_aspect("equal", "box")
        ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.35, color="gray")

        r, c = divmod(idx, 2)
        if r == 1:
            ax.set_xlabel("Central")
        else:
            ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel("Greybox")
        else:
            ax.set_ylabel("")

        ax.text(0.5, -0.16, code, transform=ax.transAxes, ha="center", va="top", fontsize=13, alpha=0.8)

        label = "D_opt_report" if objective == "determinant" else "A_opt_report"
        print(f"[{objective}] {problem}: run_index used={run_ids}")
        print(f"[{objective}] {problem}: N paired runs={n}")
        print(f"[{objective}] {problem}: mean absolute difference ({label})={mean_abs_diff}")
        print(f"[{objective}] {problem}: max absolute difference ({label})={max_abs_diff}")
        print(f"[{objective}] {problem}: correlation coefficient={corr}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[{objective}] No NaNs/Infs in plotted points: {all_finite}")
    print(f"[{objective}] Saved: {out_path}")
    if not any_data:
        print(f"[{objective}] WARNING: figure contains no active subplot data.")


def main() -> None:
    """Generate determinant and trace scatter figures."""
    if not INPUT_CSV.exists():
        raise SystemExit(f"Missing input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    data = prepare(df)

    det_df = data[data["objective_option"] == "determinant"].copy()
    tr_df = data[data["objective_option"] == "trace"].copy()

    scatter_figure(det_df, "determinant", OUT_DET)
    scatter_figure(tr_df, "trace", OUT_TRACE)


if __name__ == "__main__":
    main()
