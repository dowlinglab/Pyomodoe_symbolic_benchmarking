#!/usr/bin/env python3
"""Rebuild Case-2 speedup plot from summary_by_script.csv.

Detects whether the input contains raw runs or aggregated rows, then computes
speedup ratio `t_central / t_greybox` for 4st_6pmt-large determinant/trace.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_CSV = Path("data_greybox_benchmarking/summary_by_script.csv")
OUTDIR = Path("data_greybox_benchmarking")
OUT_PNG = OUTDIR / "case2_speedup_4st_6pmt_large.png"
OUT_STATS = OUTDIR / "case2_speedup_4st_6pmt_large_stats.csv"

DET_COLOR = "#AEC7E8"
TRACE_COLOR = "#98DF8A"
OBJECTIVES = ["determinant", "trace"]


def infer_problem(script_name: str) -> str:
    """Infer problem key from script filename."""
    name = Path(str(script_name)).name.lower()
    if "4st_6pmt" in name:
        return "4st_6pmt"
    if "4_state_reactor" in name:
        return "4_state_reactor"
    if "pde_diffusion" in name:
        return "PDE_diffusion"
    if "two_param_sin" in name:
        return "two_param_sin"
    return ""


def infer_mode(mode_val: str) -> str:
    """Normalize mode labels into existing/greybox."""
    v = str(mode_val).strip().lower()
    if v in {"existing", "central"}:
        return "existing"
    if v == "greybox":
        return "greybox"
    return v


def detect_structure(df: pd.DataFrame) -> str:
    """Return 'raw' or 'aggregated' based on available columns."""
    if {"run_index", "wall_time_s"}.issubset(df.columns):
        return "raw"
    if {"mean_wall_time", "std_wall_time"}.issubset(df.columns):
        return "aggregated"
    raise SystemExit(
        "Unsupported summary_by_script.csv structure: expected raw columns "
        "(run_index, wall_time_s) or aggregated columns (mean_wall_time, std_wall_time)."
    )


def add_missing_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add/normalize expected fields so filtering logic is consistent."""
    out = df.copy()
    if "mode" not in out.columns:
        raise SystemExit("Missing required column: mode")
    if "script_name" not in out.columns:
        raise SystemExit("Missing required column: script_name")

    out["mode"] = out["mode"].map(infer_mode)
    if "problem" not in out.columns:
        out["problem"] = out["script_name"].map(infer_problem)
    if "instance" not in out.columns:
        out["instance"] = "large"
    if "objective_option" not in out.columns:
        # For current harness summaries, objective is not persisted; infer as determinant default.
        out["objective_option"] = np.where(
            out["script_name"].astype(str).str.contains("4st_6pmt", na=False),
            "determinant",
            "",
        )
    out["objective_option"] = out["objective_option"].astype(str).str.strip().str.lower()
    return out


def filter_case2(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to 4st_6pmt-large, existing/greybox, determinant/trace."""
    filt = (
        (df["problem"] == "4st_6pmt")
        & (df["instance"] == "large")
        & (df["mode"].isin(["existing", "greybox"]))
        & (df["objective_option"].isin(OBJECTIVES))
    )
    if "status" in df.columns:
        filt &= df["status"].astype(str).str.lower().eq("ok")
    return df[filt].copy()


def raw_stats(df: pd.DataFrame, objective: str) -> tuple[int, float, float]:
    """Compute N/mean/std from raw paired runs."""
    sub = df[df["objective_option"] == objective].copy()
    c = sub[sub["mode"] == "existing"][["run_index", "wall_time_s"]].copy()
    g = sub[sub["mode"] == "greybox"][["run_index", "wall_time_s"]].copy()
    c["run_index"] = pd.to_numeric(c["run_index"], errors="coerce")
    g["run_index"] = pd.to_numeric(g["run_index"], errors="coerce")
    c["wall_time_s"] = pd.to_numeric(c["wall_time_s"], errors="coerce")
    g["wall_time_s"] = pd.to_numeric(g["wall_time_s"], errors="coerce")
    merged = pd.merge(c, g, on="run_index", how="inner", suffixes=("_central", "_greybox"))
    merged = merged[(merged["wall_time_s_greybox"] > 0) & np.isfinite(merged["wall_time_s_central"])]
    ratios = merged["wall_time_s_central"] / merged["wall_time_s_greybox"]
    ratios = ratios[np.isfinite(ratios)]
    n = int(ratios.shape[0])
    mean = float(ratios.mean()) if n > 0 else np.nan
    std = float(ratios.std(ddof=1)) if n > 1 else np.nan
    return n, mean, std


def aggregated_stats(df: pd.DataFrame, objective: str) -> tuple[int, float, float]:
    """Compute N/mean/std from aggregated mean/std rows using error propagation."""
    sub = df[df["objective_option"] == objective].copy()
    c = sub[sub["mode"] == "existing"]
    g = sub[sub["mode"] == "greybox"]
    if c.empty or g.empty:
        return 0, np.nan, np.nan

    c0 = c.iloc[0]
    g0 = g.iloc[0]
    mu_c = float(pd.to_numeric(c0.get("mean_wall_time"), errors="coerce"))
    mu_g = float(pd.to_numeric(g0.get("mean_wall_time"), errors="coerce"))
    sd_c = float(pd.to_numeric(c0.get("std_wall_time"), errors="coerce"))
    sd_g = float(pd.to_numeric(g0.get("std_wall_time"), errors="coerce"))
    n_c = int(pd.to_numeric(c0.get("runs"), errors="coerce")) if "runs" in c0.index else 0
    n_g = int(pd.to_numeric(g0.get("runs"), errors="coerce")) if "runs" in g0.index else 0
    n = int(min(n_c, n_g)) if n_c and n_g else 0

    if not np.isfinite(mu_c) or not np.isfinite(mu_g) or mu_g <= 0:
        return n, np.nan, np.nan

    r = mu_c / mu_g
    # sigma_r ≈ r * sqrt((sigma_C/mu_C)^2 + (sigma_G/mu_G)^2)
    if np.isfinite(sd_c) and np.isfinite(sd_g) and mu_c > 0 and mu_g > 0:
        sr = r * np.sqrt((sd_c / mu_c) ** 2 + (sd_g / mu_g) ** 2)
    else:
        sr = np.nan
    return n, float(r), float(sr)


def plot_case2(stats: pd.DataFrame) -> None:
    """Render the approved Case-2 horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6.75))
    y_center = 0.0
    bar_h = 0.34

    spec = [
        ("determinant", "Determinant", DET_COLOR, -bar_h / 2),
        ("trace", "Trace", TRACE_COLOR, bar_h / 2),
    ]
    for objective, label, color, offset in spec:
        row = stats[stats["objective_option"] == objective]
        mean = row["mean_ratio"].iloc[0] if not row.empty else np.nan
        std = row["std_ratio"].iloc[0] if not row.empty else np.nan
        if np.isfinite(mean):
            ax.barh(
                [y_center + offset],
                [mean],
                xerr=[std] if np.isfinite(std) else None,
                height=bar_h * 0.92,
                color=color,
                alpha=0.85,
                error_kw={"ecolor": "black", "elinewidth": 2.2, "capsize": 6, "capthick": 2.2},
                label=label,
            )

    ax.set_yticks([y_center])
    ax.set_yticklabels(["3"], fontsize=16)
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
    """Build stats/plot and print requested validation summary."""
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    structure = detect_structure(df)
    prepared = add_missing_context(df)
    filtered = filter_case2(prepared)

    rows = []
    for obj in OBJECTIVES:
        if structure == "raw":
            n, mean, std = raw_stats(filtered, obj)
        else:
            n, mean, std = aggregated_stats(filtered, obj)
        rows.append({"objective_option": obj, "N": n, "mean_ratio": mean, "std_ratio": std})
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT_STATS, index=False)
    plot_case2(stats)

    no_nans = bool(np.isfinite(stats["mean_ratio"].dropna()).all() and np.isfinite(stats["std_ratio"].dropna()).all())
    total_runs = int(filtered["runs"].sum()) if structure == "aggregated" and "runs" in filtered.columns else int(len(filtered))

    print(f"Detected structure: {structure}")
    print(f"Number of runs detected: {total_runs}")
    for r in stats.itertuples(index=False):
        print(f"{r.objective_option}: N={int(r.N)}, mean={r.mean_ratio}, std={r.std_ratio}")
    print(f"No NaNs in reported finite stats: {no_nans}")
    print(f"Saved plot: {OUT_PNG}")
    print(f"Saved stats: {OUT_STATS}")


if __name__ == "__main__":
    main()
