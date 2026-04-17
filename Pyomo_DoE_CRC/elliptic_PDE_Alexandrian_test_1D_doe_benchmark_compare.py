#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark copy of the Alexandrian 1D DoE case.

This script compares central finite differences vs symbolic derivatives for
both objective choices used in the 1D case:

- determinant
- trace

For each (objective, method) pair, the script runs the model multiple times,
parses IPOPT solver metrics and DoE timing metrics, averages successful runs,
and generates a screenshot-style comparison figure.

Plot style:
- central finite difference: white bars with black borders
- symbolic derivatives: light blue overlay bars
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from matplotlib.patches import Patch
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.dae import ContinuousSet, DerivativeVar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--nfe-x", type=int, default=20)
    parser.add_argument("--nfe-t", type=int, default=40)
    parser.add_argument("--outdir", type=str, default="results_plots/alexandrian_1d_benchmark")
    parser.add_argument("--smoke", action="store_true", help="Run one quick smoke pass and still produce outputs.")
    return parser.parse_args()


class PDEAlexandrian1D(Experiment):
    def __init__(self, data, nfe_x, nfe_t):
        self.data = data
        self.nfe_t = nfe_t
        self.nfe_x = nfe_x
        self.model = None

    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_experiment()
        return self.model

    def create_model(self):
        m = self.model = pyo.ConcreteModel()

        m.x = ContinuousSet(bounds=(0, 1))
        m.t = ContinuousSet(bounds=(0, 1))

        m.u = pyo.Var(m.t, m.x)
        m.dudt = DerivativeVar(m.u, wrt=m.t)
        m.dudx = DerivativeVar(m.u, wrt=m.x)
        m.d2udx2 = DerivativeVar(m.dudx, wrt=m.x)

        m.theta1 = pyo.Var(within=pyo.Reals)
        m.theta2 = pyo.Var(within=pyo.Reals)

        m.kappa = pyo.Param(initialize=0.025, mutable=True)
        m.v = pyo.Param(initialize=0.1, mutable=True)

        m.w1 = pyo.Var(bounds=(0, 1), initialize=1.0)
        m.w2 = pyo.Var(bounds=(0, 1), initialize=1.0)
        m.w3 = pyo.Var(bounds=(0, 1), initialize=1.0)
        m.w4 = pyo.Var(bounds=(0, 1), initialize=1.0)

        @m.Constraint(m.t, m.x)
        def pde(m, t, x):
            if x == m.x.first() or x == m.x.last():
                return pyo.Constraint.Skip
            return m.dudt[t, x] == m.kappa * m.d2udx2[t, x] - m.v * m.dudx[t, x]

    def finalize_model(self):
        m = self.model

        m.theta1.fix(self.data["theta1"])
        m.theta2.fix(self.data["theta2"])
        m.kappa.set_value(self.data["kappa"])
        m.v.set_value(self.data["v"])

        def ic_rule(m, x):
            if x < 0.5:
                return m.u[m.t.first(), x] == m.theta1
            return m.u[m.t.first(), x] == m.theta2

        m.ic = pyo.Constraint(m.x, rule=ic_rule)

        m.bc1 = pyo.Constraint(
            m.t, rule=lambda m, t: m.dudx[t, m.x.last()] == 0.0 if t != m.t.first() else pyo.Constraint.Skip
        )
        m.bc2 = pyo.Constraint(
            m.t, rule=lambda m, t: m.dudx[t, m.x.first()] == 0.0 if t != m.t.first() else pyo.Constraint.Skip
        )
        m.bc1_dummy = pyo.Constraint(
            m.t, rule=lambda m, t: m.d2udx2[t, m.x.last()] == 0.0 if t != m.t.first() else pyo.Constraint.Skip
        )
        m.bc2_dummy = pyo.Constraint(
            m.t, rule=lambda m, t: m.d2udx2[t, m.x.first()] == 0.0 if t != m.t.first() else pyo.Constraint.Skip
        )
        m.ic_dummy = pyo.Constraint(expr=m.dudx[m.t.first(), m.x.first()] == 0.0)

        disc = pyo.TransformationFactory("dae.finite_difference")
        disc.apply_to(m, wrt=m.x, nfe=self.nfe_x, scheme="BACKWARD")
        disc.apply_to(m, wrt=m.t, nfe=self.nfe_t, scheme="BACKWARD")

    def label_experiment(self):
        m = self.model

        m.y1 = pyo.Expression(expr=m.w1 * m.u[0.1, 0.2])
        m.y2 = pyo.Expression(expr=m.w2 * m.u[0.1, 0.9])
        m.y3 = pyo.Expression(expr=m.w3 * m.u[0.125, 0.2])
        m.y4 = pyo.Expression(expr=m.w4 * m.u[0.125, 0.9])

        outputs = [m.y1, m.y2, m.y3, m.y4]

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update((k, 0.0) for k in outputs)

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update((k, 1e-2) for k in outputs)

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.theta1, m.theta2])

        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update((k, None) for k in [m.w1, m.w2, m.w3, m.w4])


DATA_EX = {
    "theta1": 1.0,
    "theta2": 0.2,
    "kappa": 0.025,
    "v": 0.1,
}

SOLVER_METRICS = [
    "ipopt_iterations",
    "ipopt_obj_eval",
    "ipopt_grad_eval",
    "ipopt_eq_con_eval",
    "ipopt_eq_jac_eval",
    "ipopt_hess_eval",
]

TIME_METRICS = [
    "ipopt_cpu_no_eval_s",
    "ipopt_cpu_nlp_eval_s",
    "solve_time_s",
    "build_time_s",
    "init_time_s",
    "wall_time_s",
]

METRIC_LABELS = {
    "ipopt_iterations": "1",
    "ipopt_obj_eval": "2",
    "ipopt_grad_eval": "3",
    "ipopt_eq_con_eval": "4",
    "ipopt_eq_jac_eval": "5",
    "ipopt_hess_eval": "6",
    "ipopt_cpu_no_eval_s": "7",
    "ipopt_cpu_nlp_eval_s": "8",
    "solve_time_s": "9",
    "build_time_s": "10",
    "init_time_s": "11",
    "wall_time_s": "12",
}


def make_ipopt(output_file: Path):
    solver = pyo.SolverFactory("ipopt")
    solver.options["output_file"] = str(output_file)
    solver.options["file_print_level"] = 12
    return solver


def grab_float(text: str, pat: str):
    ms = re.findall(pat, text)
    return float(ms[-1]) if ms else None


def grab_int(text: str, pat: str):
    ms = re.findall(pat, text)
    return int(ms[-1]) if ms else None


def parse_ipopt_metrics(ipopt_out: Path) -> dict:
    text = ipopt_out.read_text(encoding="utf-8", errors="replace") if ipopt_out.exists() else ""
    return {
        "ipopt_iterations": grab_int(text, r"Number of Iterations\.*:\s*(\d+)"),
        "ipopt_obj_eval": grab_int(text, r"Number of objective function evaluations\s*=\s*(\d+)"),
        "ipopt_grad_eval": grab_int(text, r"Number of objective gradient evaluations\s*=\s*(\d+)"),
        "ipopt_eq_con_eval": grab_int(text, r"Number of equality constraint evaluations\s*=\s*(\d+)"),
        "ipopt_eq_jac_eval": grab_int(text, r"Number of equality constraint Jacobian evaluations\s*=\s*(\d+)"),
        "ipopt_hess_eval": grab_int(text, r"Number of Lagrangian Hessian evaluations\s*=\s*(\d+)"),
        "ipopt_cpu_no_eval_s": grab_float(text, r"Total CPU secs in IPOPT \(w/o function evaluations\)\s*=\s*([0-9.eE+-]+)"),
        "ipopt_cpu_nlp_eval_s": grab_float(text, r"Total CPU secs in NLP function evaluations\s*=\s*([0-9.eE+-]+)"),
    }


def run_one_case(method: str, objective: str, nfe_x: int, nfe_t: int, outdir: Path, run_id: int) -> dict:
    case_dir = outdir / objective / method
    case_dir.mkdir(parents=True, exist_ok=True)
    ipopt_out = case_dir / f"run_{run_id:02d}.ipopt.out"
    if ipopt_out.exists():
        ipopt_out.unlink()

    experiment = PDEAlexandrian1D(data=DATA_EX, nfe_x=nfe_x, nfe_t=nfe_t)
    solver = make_ipopt(ipopt_out)

    doe_kwargs = {
        "fd_formula": "central",
        "step": 1e-3,
        "objective_option": objective,
        "scale_constant_value": 1,
        "scale_nominal_param_value": True,
        "prior_FIM": None,
        "jac_initial": None,
        "fim_initial": None,
        "L_diagonal_lower_bound": 1e-7,
        "solver": solver,
        "tee": False,
        "get_labeled_model_args": None,
        "_Cholesky_option": True,
        "_only_compute_fim_lower": True,
    }
    if method == "symbolic":
        doe_kwargs["gradient_method"] = "pynumero"

    t0 = time.perf_counter()
    doe_obj = DesignOfExperiments(experiment, **doe_kwargs)
    build_done = time.perf_counter()
    doe_obj.run_doe()
    wall_done = time.perf_counter()

    results = doe_obj.results
    metrics = parse_ipopt_metrics(ipopt_out)
    metrics.update(
        {
            "objective": objective,
            "method": method,
            "run_id": run_id,
            "build_time_s": results.get("Build Time", build_done - t0),
            "init_time_s": results.get("Initialization Time"),
            "solve_time_s": results.get("Solve Time"),
            "wall_time_s": results.get("Wall-clock Time", wall_done - t0),
            "returncode": 0,
        }
    )
    return metrics


def average_successful(rows: list[dict]) -> dict:
    out = {}
    for key in rows[0].keys():
        if key in {"objective", "method"}:
            out[key] = rows[0][key]
            continue
        if key == "run_id":
            continue
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and r.get(key) is not None]
        out[key] = mean(vals) if vals else np.nan
    return out


def plot_comparison(summary_df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    central_edge_color = "black"
    central_linewidth = 2.5
    symbolic_color = "#B7D7F0"
    symbolic_alpha = 0.9
    bar_w = 0.75

    panel_specs = [
        ("determinant", SOLVER_METRICS, "A", "Determinant objective"),
        ("determinant", TIME_METRICS, "B", "Determinant objective"),
        ("trace", SOLVER_METRICS, "C", "Trace objective"),
        ("trace", TIME_METRICS, "D", "Trace objective"),
    ]

    for ax, (objective, metrics, case_label, panel_title) in zip(axes, panel_specs):
        x = np.arange(1, len(metrics) + 1)
        c_row = summary_df[(summary_df["objective"] == objective) & (summary_df["method"] == "central")].iloc[0]
        s_row = summary_df[(summary_df["objective"] == objective) & (summary_df["method"] == "symbolic")].iloc[0]
        central_vals = [c_row[m] for m in metrics]
        symbolic_vals = [s_row[m] for m in metrics]

        ax.bar(
            x,
            central_vals,
            width=bar_w,
            facecolor="white",
            edgecolor=central_edge_color,
            linewidth=central_linewidth,
            alpha=1.0,
            zorder=2,
        )
        ax.bar(
            x,
            symbolic_vals,
            width=bar_w,
            color=symbolic_color,
            edgecolor="none",
            alpha=symbolic_alpha,
            zorder=3,
        )

        label_nums = [METRIC_LABELS[m] for m in metrics]
        ax.set_xticks(x)
        ax.set_xticklabels(label_nums, fontsize=8)
        ax.set_ylabel("Average")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_title(panel_title, fontsize=10)

        ax_r = ax.twinx()
        ax_r.set_ylabel(case_label, rotation=0, labelpad=10, va="center")
        ax_r.set_yticks([])
        for spine in ax_r.spines.values():
            spine.set_visible(False)

        legend_handles = [
            Patch(facecolor="white", edgecolor="black", linewidth=central_linewidth, label="Central finite difference"),
            Patch(facecolor=symbolic_color, edgecolor="none", alpha=symbolic_alpha, label="Symbolic derivatives"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=2,
            frameon=True,
            edgecolor="black",
            fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    n_runs = 1 if args.smoke else args.n_runs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for objective in ["determinant", "trace"]:
        for method in ["central", "symbolic"]:
            for run_id in range(1, n_runs + 1):
                all_rows.append(run_one_case(method, objective, args.nfe_x, args.nfe_t, outdir, run_id))

    raw_df = pd.DataFrame(all_rows)
    raw_csv = outdir / "alexandrian_1d_benchmark_raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    summary_rows = []
    for objective in ["determinant", "trace"]:
        for method in ["central", "symbolic"]:
            rows = raw_df[(raw_df["objective"] == objective) & (raw_df["method"] == method)].to_dict(orient="records")
            summary_rows.append(average_successful(rows))

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = outdir / "alexandrian_1d_benchmark_summary.csv"
    summary_json = outdir / "alexandrian_1d_benchmark_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(summary_df.to_json(orient="records", indent=2), encoding="utf-8")

    plot_comparison(summary_df, outdir / "alexandrian_1d_symbolic_vs_central")
    metric_map = (
        "1 IPOPT iterations, 2 objective evaluations, 3 gradient evaluations, "
        "4 equality constraint evaluations, 5 Jacobian evaluations, 6 Hessian evaluations, "
        "7 IPOPT CPU time, 8 NLP CPU time, 9 solve time, 10 build time, 11 initialization time, 12 wall-clock time.\n"
    )
    (outdir / "metric_map.txt").write_text(metric_map, encoding="utf-8")


if __name__ == "__main__":
    main()
