#!/usr/bin/env python3
"""
Benchmark data runner for FOCAPO/CPC central-vs-symbolic comparisons.

Writes raw and summary data files, but does not generate plots.
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

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

CASE_LABELS = {
    "4st_6pmt": "six_parameter",
    "4_state_reactor": "four_state_reactor",
    "PDE_diffusion": "pde",
    "two_param_sin": "two_param_sin",
}


@dataclass
class CasePair:
    case_key: str
    central_script: Path
    symbolic_script: Path


def discover_case_pairs(root: Path) -> Dict[str, CasePair]:
    central_files = {p.stem.replace("_central", ""): p for p in root.glob("*_central.py")}
    sym_files = {p.stem.replace("_sym", ""): p for p in root.glob("*_sym.py")}
    keys = sorted(set(central_files).intersection(sym_files))
    return {
        k: CasePair(case_key=k, central_script=central_files[k], symbolic_script=sym_files[k])
        for k in keys
    }


def select_required_cases(pairs: Dict[str, CasePair]) -> Dict[str, CasePair]:
    required = ["two_param_sin", "PDE_diffusion", "4_state_reactor", "4st_6pmt"]
    missing = [k for k in required if k not in pairs]
    if missing:
        raise RuntimeError(f"Missing required case(s): {missing}")
    return {k: pairs[k] for k in required}


def regex_float_last(pattern: str, text: str) -> float:
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


def regex_text_last(pattern: str, text: str) -> Optional[str]:
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    last = matches[-1]
    if isinstance(last, tuple):
        last = last[0]
    return str(last).strip()


def parse_fim_block(stdout: str) -> Optional[np.ndarray]:
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
    text = f"{stdout}\n{stderr}"
    metrics: Dict[str, object] = {
        "solve_time_s": regex_float_last(r"Solve time \(s\):\s*([0-9eE+\-\.]+)", text),
        "build_time_s": regex_float_last(r"Build time \(s\):\s*([0-9eE+\-\.]+)", text),
        "initialization_time_s": regex_float_last(r"Initialization time \(s\):\s*([0-9eE+\-\.]+)", text),
        "wall_time_s": regex_float_last(r"Total wall time \(s\):\s*([0-9eE+\-\.]+)", text),
        "ipopt_iterations": regex_float_last(r"Number of Iterations.*:\s*(\d+)", text),
        "objective_value": regex_float_last(r"Objective value at optimal design:\s*([0-9eE+\-\.]+)", text),
        "fim_condition_number": regex_float_last(r"FIM Condition Number['\"]?\]?\)?[:=\s]+([0-9eE+\-\.]+)", text),
        "solver_status": regex_text_last(r"Solver Status[:=\s]+([A-Za-z_]+)", text),
        "termination_condition": regex_text_last(r"termination condition:\s*([A-Za-z_]+)", text)
        or regex_text_last(r"Termination Condition[:=\s]+([A-Za-z_]+)", text),
        "termination_message": regex_text_last(r"Termination Message[:=\s]+(.+)", text)
        or regex_text_last(r"EXIT:\s*(.+)", text),
        "obj_fun_evals": regex_float_last(r"Number of objective function evaluations\s*=\s*(\d+)", text),
        "obj_grad_evals": regex_float_last(r"Number of objective gradient evaluations\s*=\s*(\d+)", text),
        "eq_constr_evals": regex_float_last(r"Number of equality constraint evaluations\s*=\s*(\d+)", text),
        "eq_jac_evals": regex_float_last(r"Number of equality constraint Jacobian evaluations\s*=\s*(\d+)", text),
        "lag_hess_evals": regex_float_last(r"Number of Lagrangian Hessian evaluations\s*=\s*(\d+)", text),
        "cpu_time_s": regex_float_last(r"Total CPU secs in IPOPT \(w/o function evaluations\)\s*=\s*([0-9eE+\-\.]+)", text),
        "nlp_eval_time_s": regex_float_last(r"Total CPU secs in NLP function evaluations\s*=\s*([0-9eE+\-\.]+)", text),
    }

    if np.isnan(metrics["fim_condition_number"]):
        fim = parse_fim_block(stdout)
        if fim is not None:
            try:
                metrics["fim_condition_number"] = float(np.linalg.cond(fim))
            except Exception:
                metrics["fim_condition_number"] = np.nan
    return metrics


def run_script(script_path: Path, timeout_s: int = 1800) -> Tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(script_path.resolve())],
        cwd=str(script_path.resolve().parent),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_benchmarks(
    pairs: Dict[str, CasePair], runs: int, selected_case: Optional[str], logs_dir: Path
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    logs_dir.mkdir(parents=True, exist_ok=True)
    case_keys = [selected_case] if selected_case else list(pairs.keys())

    for case_key in case_keys:
        pair = pairs[case_key]
        for method, script in [("central", pair.central_script), ("symbolic", pair.symbolic_script)]:
            for run_idx in range(1, runs + 1):
                return_code, stdout, stderr = run_script(script)
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
    return (
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write benchmark data for central vs symbolic Pyomo.DoE scripts.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--overwrite-excel",
        action="store_true",
        help="Allow overwriting existing Excel outputs.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        choices=["six_parameter", "pde", "two_param_sin", "four_state_reactor"],
    )
    return parser


def case_choice_to_key(case_choice: Optional[str]) -> Optional[str]:
    if case_choice is None:
        return None
    reverse = {v: k for k, v in CASE_LABELS.items()}
    return reverse[case_choice]


def main() -> None:
    args = build_parser().parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")

    pairs = select_required_cases(discover_case_pairs(SCRIPT_DIR))
    selected_key = case_choice_to_key(args.case)

    out_dir = args.output_dir
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    raw_df = run_benchmarks(pairs=pairs, runs=args.runs, selected_case=selected_key, logs_dir=logs_dir)
    summary_df = aggregate_summary(raw_df)

    raw_df.to_csv(out_dir / "raw_results.csv", index=False)
    summary_df.to_csv(out_dir / "summary_results.csv", index=False)

    # full summary used by plot-only script
    summary_full_xlsx = out_dir / "summary_results_full.xlsx"
    benchmark_xlsx = out_dir / "benchmark_summary.xlsx"
    if not args.overwrite_excel:
        if summary_full_xlsx.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {summary_full_xlsx}. "
                "Use --overwrite-excel to allow overwrite."
            )
        if benchmark_xlsx.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {benchmark_xlsx}. "
                "Use --overwrite-excel to allow overwrite."
            )

    summary_df.to_excel(summary_full_xlsx, index=False)

    # compact summary requested by user
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
    excel_df.to_excel(benchmark_xlsx, index=False)

    print(f"Wrote: {out_dir / 'raw_results.csv'}")
    print(f"Wrote: {out_dir / 'summary_results.csv'}")
    print(f"Wrote: {summary_full_xlsx}")
    print(f"Wrote: {benchmark_xlsx}")
    print(f"Wrote logs: {logs_dir}")


if __name__ == "__main__":
    main()
