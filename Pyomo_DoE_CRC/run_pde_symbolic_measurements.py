#!/usr/bin/env python
"""
Run bounded PDE diffusion measurements for 5 existing discretizations.

This script executes one baseline run for both central and symbolic methods,
collects structural/timing/derivative metrics, and writes a summary table.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ROOT_DIR = Path("/Users/snarasi2/projects/Pyomo_DoE_CRC")
OUT_DIR = ROOT_DIR / "out"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUT_DIR / f"symbolic_measurements_{TIMESTAMP}"

DISCRETIZATIONS: List[Tuple[int, int]] = [
    (2, 2),
    (5, 10),
    (10, 20),
    (25, 40),
    (50, 75),
]


def run_command(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def run_one(method: str, env_name: str, nfe_x: int, nfe_t: int) -> Dict[str, object]:
    method_dir = RUN_DIR / method / f"nfe_x{nfe_x}_nfe_t{nfe_t}"
    method_dir.mkdir(parents=True, exist_ok=True)

    out_json = method_dir / f"{method}_nfe_x{nfe_x}_nfe_t{nfe_t}.json"
    ipopt_out = method_dir / f"{method}_nfe_x{nfe_x}_nfe_t{nfe_t}.ipopt.out"

    cmd = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        str(ROOT_DIR / "pde_doe_execution_helper.py"),
        "--method",
        method,
        "--mode",
        "baseline",
        "--nfe_x",
        str(nfe_x),
        "--nfe_t",
        str(nfe_t),
        "--out_json",
        str(out_json),
        "--run_dir",
        str(method_dir),
        "--ipopt_out",
        str(ipopt_out),
    ]
    run_command(cmd)

    with out_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_summary_rows(
    central_records: Dict[Tuple[int, int], Dict[str, object]],
    symbolic_records: Dict[Tuple[int, int], Dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for nfe_x, nfe_t in DISCRETIZATIONS:
        c = central_records[(nfe_x, nfe_t)]
        s = symbolic_records[(nfe_x, nfe_t)]
        rows.append(
            {
                "nfe_x": nfe_x,
                "nfe_t": nfe_t,
                "n_cons": s.get("n_constraints"),
                "n_vars": s.get("n_variables"),
                "n_cons_x_n_vars": s.get("problem_size_metric"),
                "build_time": s.get("total_build_time"),
                "diff_time": s.get("differentiation_time"),
                "total_deriv_entries": s.get("total_derivative_entries_stored"),
                "nonzeros": s.get("number_of_nonzero_derivatives"),
                "density": s.get("derivative_density"),
                "solve_time": s.get("doe_solve_time"),
                "ipopt_iters": s.get("ipopt_iters"),
                "solve_time_central": c.get("doe_solve_time"),
                "ipopt_iters_central": c.get("ipopt_iters"),
                "ipopt_cpu_wo_feval_symbolic": s.get("ipopt_cpu_wo_feval"),
                "ipopt_cpu_wo_feval_central": c.get("ipopt_cpu_wo_feval"),
                "ma57_linear_time_symbolic": s.get("ma57_linear_solver_time"),
                "ma57_linear_time_central": c.get("ma57_linear_solver_time"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    central_records: Dict[Tuple[int, int], Dict[str, object]] = {}
    symbolic_records: Dict[Tuple[int, int], Dict[str, object]] = {}

    for nfe_x, nfe_t in DISCRETIZATIONS:
        central_records[(nfe_x, nfe_t)] = run_one("central", "pyomo", nfe_x, nfe_t)
        symbolic_records[(nfe_x, nfe_t)] = run_one("symbolic", "pyomosym", nfe_x, nfe_t)

    summary = to_summary_rows(central_records, symbolic_records)
    csv_path = RUN_DIR / "symbolic_scaling_summary.csv"
    md_path = RUN_DIR / "symbolic_scaling_summary.md"
    summary.to_csv(csv_path, index=False)

    cols = [
        "nfe_x",
        "nfe_t",
        "n_cons",
        "n_vars",
        "n_cons_x_n_vars",
        "build_time",
        "diff_time",
        "total_deriv_entries",
        "nonzeros",
        "solve_time",
        "ipopt_iters",
    ]
    md = summary[cols].to_markdown(index=False)
    md_path.write_text(md + "\n", encoding="utf-8")

    print(f"Wrote summary CSV: {csv_path}")
    print(f"Wrote summary markdown: {md_path}")


if __name__ == "__main__":
    main()
