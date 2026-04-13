#!/usr/bin/env python3
"""Build Case-3 solver-iterations plot for all 4 cases (determinant, large)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_CSV = Path("data_greybox_benchmarking/runs_all_12metrics_from_logs.csv")
OUTDIR = Path("data_greybox_benchmarking")
OUT_PNG = OUTDIR / "case3_iterations_large_determinant_all_cases.png"
OUT_STATS = OUTDIR / "case3_iterations_large_determinant_all_cases_stats.csv"

CASE_ORDER: List[Tuple[str, str]] = [
    ("two_param_sin", "1"),
    ("PDE_diffusion", "2"),
    ("4st_6pmt", "3"),
    ("4_state_reactor", "4"),
]

CENTRAL_COLOR = "#AEC7E8"
GREYBOX_COLOR = "#98DF8A"


def infer_problem(script_name: str) -> str:
    """Infer problem label from script filename when the column is missing."""
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
    """Normalize mode labels to existing/greybox."""
    base = df["mode"].astype(str).str.lower().replace({"central": "existing"})
    if "mode_case2" in df.columns:
        alt = df["mode_case2"].astype(str).str.lower().replace({"nan": ""})
        use_alt = alt.isin(["existing", "greybox"])
        return base.where(~use_alt, alt)
    return base


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and filter rows for Case-3 determinant iteration analysis."""
    out = df.copy()
    out["problem"] = out.get("problem", pd.Series([np.nan] * len(out), index=out.index))
    out["problem"] = out["problem"].where(out["problem"].notna(), out["script_name"].map(infer_problem))
    out["instance"] = out.get("instance", pd.Series([np.nan] * len(out), index=out.index))
    out["instance"] = out["instance"].where(out["instance"].notna(), "large")
    out["objective_option"] = out["objective_option"].astype(str).str.lower()
    out["status"] = out["status"].astype(str).str.lower()
    out["mode_norm"] = normalize_mode(out)
    out["run_index"] = pd.to_numeric(out["run_index"], errors="coerce").astype("Int64")
    out["ipopt_iterations"] = pd.to_numeric(out["ipopt_iterations"], errors="coerce")

    filt = (
        out["problem"].isin([p for p, _ in CASE_ORDER])
        & out["instance"].eq("large")
        & out["objective_option"].eq("determinant")
        & out["status"].eq("ok")
        & out["mode_norm"].isin(["existing", "greybox"])
        & out["run_index"].notna()
    )
    return out[filt].copy()


def latest_unique_by_key(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate repeated batches by keeping latest row per key."""
    if "timestamp_utc" in df.columns:
        return df.sort_values("timestamp_utc", kind="stable").drop_duplicates(
            subset=["problem", "objective_option", "mode_norm", "run_index"], keep="last"
        )
    return df.drop_duplicates(subset=["problem", "objective_option", "mode_norm", "run_index"], keep="last")


def pair_case(df: pd.DataFrame, problem: str) -> Tuple[pd.DataFrame, List[int]]:
    """Pair central and greybox rows by run_index for one problem."""
    sub = latest_unique_by_key(df[df["problem"] == problem].copy())
    c = sub[sub["mode_norm"] == "existing"][["run_index", "ipopt_iterations"]].rename(
        columns={"ipopt_iterations": "iters_c"}
    )
    g = sub[sub["mode_norm"] == "greybox"][["run_index", "ipopt_iterations"]].rename(
        columns={"ipopt_iterations": "iters_g"}
    )
    paired = pd.merge(c, g, on="run_index", how="inner").sort_values("run_index")
    seen = set(int(v) for v in paired["run_index"].dropna().astype(int).tolist())
    missing = sorted(set(range(1, 11)) - seen)
    return paired, missing


def compute_stats(paired: pd.DataFrame) -> Tuple[int, float, float, float, float, bool, bool]:
    """Compute central/greybox mean/std and validation flags."""
    c = paired["iters_c"]
    g = paired["iters_g"]
    valid_c = bool(np.isfinite(c).all()) and bool((c > 0).all())
    valid_g = bool(np.isfinite(g).all()) and bool((g > 0).all())
    n = int(len(paired))
    mean_c = float(c.mean()) if n else np.nan
    std_c = float(c.std(ddof=1)) if n > 1 else np.nan
    mean_g = float(g.mean()) if n else np.nan
    std_g = float(g.std(ddof=1)) if n > 1 else np.nan
    return n, mean_c, std_c, mean_g, std_g, valid_c, valid_g


def plot(stats: pd.DataFrame) -> None:
    """Render Case-3 horizontal side-by-side bar plot."""
    fig, ax = plt.subplots(figsize=(12, 6.75))
    y = np.arange(len(CASE_ORDER), dtype=float)
    bar_h = 0.34

    ordered = stats.sort_values("y_label")
    ax.barh(
        y - bar_h / 2,
        ordered["mean_iters_C"].to_numpy(dtype=float),
        xerr=ordered["std_iters_C"].to_numpy(dtype=float),
        color=CENTRAL_COLOR,
        alpha=0.85,
        error_kw={"ecolor": "black", "elinewidth": 2.2, "capsize": 6, "capthick": 2.2},
        height=bar_h * 0.92,
        label="Central",
    )
    ax.barh(
        y + bar_h / 2,
        ordered["mean_iters_G"].to_numpy(dtype=float),
        xerr=ordered["std_iters_G"].to_numpy(dtype=float),
        color=GREYBOX_COLOR,
        alpha=0.85,
        error_kw={"ecolor": "black", "elinewidth": 2.2, "capsize": 6, "capthick": 2.2},
        height=bar_h * 0.92,
        label="Greybox",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in CASE_ORDER], fontsize=15)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel("Solver Iterations", fontsize=17)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.35, color="gray")
    ax.grid(False, axis="y")
    ax.legend(frameon=False, fontsize=12, loc="best")
    fig.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)


def main() -> None:
    """Run Case-3 end-to-end and print required validation outputs."""
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    data = prepare(df)

    rows: List[Dict[str, object]] = []
    all_n10 = True
    no_nans = True
    no_zero_iters = True

    for problem, y_label in CASE_ORDER:
        paired, missing = pair_case(data, problem)
        n, mean_c, std_c, mean_g, std_g, valid_c, valid_g = compute_stats(paired)

        print(f"{problem}: paired={n}, missing_indices={missing}")
        print(f"  Central: {mean_c} ± {std_c}")
        print(f"  Greybox: {mean_g} ± {std_g}")

        all_n10 = all_n10 and (n == 10)
        no_nans = no_nans and valid_c and valid_g
        no_zero_iters = no_zero_iters and valid_c and valid_g

        rows.append(
            {
                "problem": problem,
                "N": n,
                "mean_iters_C": mean_c,
                "std_iters_C": std_c,
                "mean_iters_G": mean_g,
                "std_iters_G": std_g,
                "y_label": y_label,
            }
        )

    stats = pd.DataFrame(rows)
    stats[["problem", "N", "mean_iters_C", "std_iters_C", "mean_iters_G", "std_iters_G"]].to_csv(
        OUT_STATS, index=False
    )
    plot(stats)

    print(f"All N = 10: {all_n10}")
    print(f"No NaNs: {no_nans}")
    print(f"No zero iterations: {no_zero_iters}")
    print(f"Saved plot: {OUT_PNG}")
    print(f"Saved stats: {OUT_STATS}")


if __name__ == "__main__":
    main()
