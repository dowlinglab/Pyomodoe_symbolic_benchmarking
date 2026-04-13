#!/usr/bin/env python3
"""Build Case-2 speedup plot (determinant + trace) using matched central run indices."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_CSV = Path("data_greybox_benchmarking/runs_all_12metrics_from_logs.csv")
OUTDIR = Path("data_greybox_benchmarking")
OUT_PNG = OUTDIR / "case2_speedup_large_all_cases_det_trace_matched.png"
OUT_STATS_CSV = OUTDIR / "case2_speedup_large_all_cases_det_trace_matched_stats.csv"
OUT_SELECTED_CSV = OUTDIR / "case2_speedup_large_all_cases_det_trace_matched_selected_central_cases.csv"

CASE_ORDER: List[Tuple[str, str]] = [
    ("two_param_sin", "1"),
    ("PDE_diffusion", "2"),
    ("4st_6pmt", "3"),
    ("4_state_reactor", "4"),
]
OBJECTIVES = ["determinant", "trace"]

DET_COLOR = "#AEC7E8"
TRACE_COLOR = "#98DF8A"
RNG = random.Random(20260227)


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
    """Normalize mode labels to existing/greybox."""
    base = df["mode"].astype(str).str.lower().replace({"central": "existing"})
    if "mode_case2" in df.columns:
        alt = df["mode_case2"].astype(str).str.lower().replace({"nan": ""})
        use_alt = alt.isin(["existing", "greybox"])
        return base.where(~use_alt, alt)
    return base


def source_tag(stdout_log_file: str) -> str:
    """Tag row source batch from log path."""
    path = str(stdout_log_file)
    if "/logs_trace_all/" in path:
        return "new"
    if "/logs_trace/" in path:
        return "old"
    if "/logs/" in path:
        return "old"
    return "unknown"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize fields and keep eligible rows."""
    out = df.copy()
    out["problem"] = out.get("problem", pd.Series([np.nan] * len(out), index=out.index))
    out["problem"] = out["problem"].where(out["problem"].notna(), out["script_name"].map(infer_problem))
    out["instance"] = out.get("instance", pd.Series([np.nan] * len(out), index=out.index))
    out["instance"] = out["instance"].where(out["instance"].notna(), "large")
    out["mode_norm"] = normalize_mode(out)
    out["objective_option"] = out["objective_option"].astype(str).str.lower()
    out["status"] = out["status"].astype(str).str.lower()
    out["run_index"] = pd.to_numeric(out["run_index"], errors="coerce").astype("Int64")
    out["wall_time_s"] = pd.to_numeric(out["wall_time_s"], errors="coerce")
    out["source_batch"] = out["stdout_log_file"].map(source_tag)

    filt = (
        out["problem"].isin([p for p, _ in CASE_ORDER])
        & out["instance"].eq("large")
        & out["objective_option"].isin(OBJECTIVES)
        & out["status"].eq("ok")
        & out["mode_norm"].isin(["existing", "greybox"])
        & out["run_index"].notna()
    )
    return out[filt].copy()


def choose_pair_for_index(sub: pd.DataFrame, run_index: int) -> Dict[str, object] | None:
    """Choose one central/greybox pair for a run index, random over available source batches."""
    idx_rows = sub[sub["run_index"] == run_index].copy()
    c = idx_rows[idx_rows["mode_norm"] == "existing"]
    g = idx_rows[idx_rows["mode_norm"] == "greybox"]
    if c.empty or g.empty:
        return None

    common_sources = sorted(set(c["source_batch"]) & set(g["source_batch"]))
    if common_sources:
        chosen_source = RNG.choice(common_sources)
        c_pick = c[c["source_batch"] == chosen_source].sample(n=1, random_state=RNG.randint(1, 10**9)).iloc[0]
        g_pick = g[g["source_batch"] == chosen_source].sample(n=1, random_state=RNG.randint(1, 10**9)).iloc[0]
    else:
        c_pick = c.sample(n=1, random_state=RNG.randint(1, 10**9)).iloc[0]
        g_pick = g.sample(n=1, random_state=RNG.randint(1, 10**9)).iloc[0]
        chosen_source = f"{c_pick['source_batch']}|{g_pick['source_batch']}"

    if not np.isfinite(c_pick["wall_time_s"]) or not np.isfinite(g_pick["wall_time_s"]) or g_pick["wall_time_s"] <= 0:
        return None

    ratio = float(c_pick["wall_time_s"] / g_pick["wall_time_s"])
    return {
        "run_index": int(run_index),
        "source_batch": chosen_source,
        "central_log": c_pick["stdout_log_file"],
        "greybox_log": g_pick["stdout_log_file"],
        "central_wall_time_s": float(c_pick["wall_time_s"]),
        "greybox_wall_time_s": float(g_pick["wall_time_s"]),
        "ratio": ratio,
    }


def build_selected_pairs(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build selected pairs and aggregate stats using same run indices per problem."""
    selected_rows: List[Dict[str, object]] = []
    stats_rows: List[Dict[str, object]] = []

    for problem, y_label in CASE_ORDER:
        det_idx = set(
            data[(data["problem"] == problem) & (data["objective_option"] == "determinant")]["run_index"]
            .dropna()
            .astype(int)
            .tolist()
        )
        tr_idx = set(
            data[(data["problem"] == problem) & (data["objective_option"] == "trace")]["run_index"]
            .dropna()
            .astype(int)
            .tolist()
        )
        common_idx = sorted(det_idx & tr_idx)
        if len(common_idx) < 10:
            raise SystemExit(f"{problem}: fewer than 10 common run indices between determinant and trace: {common_idx}")
        chosen_indices = common_idx[:10]

        for objective in OBJECTIVES:
            sub = data[(data["problem"] == problem) & (data["objective_option"] == objective)].copy()
            ratios: List[float] = []
            missing: List[int] = []
            for idx in chosen_indices:
                pick = choose_pair_for_index(sub, idx)
                if pick is None:
                    missing.append(idx)
                    continue
                selected_rows.append(
                    {
                        "problem": problem,
                        "objective_option": objective,
                        "y_label": y_label,
                        **pick,
                    }
                )
                ratios.append(pick["ratio"])

            n = len(ratios)
            mean = float(np.mean(ratios)) if ratios else np.nan
            std = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else np.nan
            print(f"{problem} [{objective}]: N={n}, mean={mean}, std={std}, missing_indices={missing}")
            stats_rows.append(
                {
                    "problem": problem,
                    "objective_option": objective,
                    "N": n,
                    "mean_ratio": mean,
                    "std_ratio": std,
                    "y_label": y_label,
                    "selected_run_indices": ",".join(str(i) for i in chosen_indices),
                }
            )

    return pd.DataFrame(selected_rows), pd.DataFrame(stats_rows)


def plot(stats: pd.DataFrame) -> None:
    """Render determinant+trace side-by-side speedup bars for all cases."""
    fig, ax = plt.subplots(figsize=(12, 6.75))
    y = np.arange(len(CASE_ORDER), dtype=float)
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
    ax.set_yticklabels([lab for _, lab in CASE_ORDER], fontsize=15)
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
    """Execute selection, stats, plotting, and export selected-case companion CSV."""
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    data = prepare(df)

    selected, stats = build_selected_pairs(data)
    no_nan = bool(np.isfinite(selected["ratio"]).all()) if not selected.empty else True
    no_div0 = bool((selected["greybox_wall_time_s"] > 0).all()) if not selected.empty else True
    all_n10 = bool((stats["N"] == 10).all()) if not stats.empty else False

    stats[["problem", "objective_option", "N", "mean_ratio", "std_ratio", "selected_run_indices"]].to_csv(
        OUT_STATS_CSV, index=False
    )
    plot(stats)

    selected.to_csv(OUT_SELECTED_CSV, index=False)

    print(f"No NaNs: {no_nan}")
    print(f"No division by zero: {no_div0}")
    print(f"All N = 10: {all_n10}")
    print(f"Saved plot: {OUT_PNG}")
    print(f"Saved stats CSV: {OUT_STATS_CSV}")
    print(f"Saved selected cases CSV: {OUT_SELECTED_CSV}")


if __name__ == "__main__":
    main()
