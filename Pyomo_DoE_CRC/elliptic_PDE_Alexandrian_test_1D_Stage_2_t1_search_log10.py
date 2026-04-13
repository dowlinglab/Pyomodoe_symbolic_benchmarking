#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 (time selection) for the Alexandrian 1D toy:

Fix the SECOND sampling time t2, and sweep the FIRST sampling time t1.

This is the time-analogue of the Stage-1 x2 sweep and Stage-2 t2 sweep.
For each candidate t1, we compute the FIM and plot metrics vs t1:

  - log10 E-opt = log10(lambda_min(FIM)) (maximize)
  - cond(FIM) = lambda_max/lambda_min (minimize)
  - log10 D-opt = log10(det(FIM)) (maximize)
  - log10 A-opt = log10(trace(FIM)) (maximize; Pyomo “trace” objective)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe import DesignOfExperiments


def nice_curve(
    results,
    xvar,
    metric,
    out_base,
    nticks=6,
    best="max",
    line_color="#111111",
    star_color="#0B2A6F",
    star_size=220,
):
    df = pd.DataFrame(results).sort_values(by=xvar)
    x = df[xvar].to_numpy(dtype=float)
    y = df[metric].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    ax.plot(x, y, color=line_color, linewidth=2)

    if best == "max":
        i_best = int(np.nanargmax(y))
    elif best == "min":
        i_best = int(np.nanargmin(y))
    elif best is None:
        i_best = None
    else:
        raise ValueError("best must be 'max', 'min', or None")

    if i_best is not None:
        ax.scatter(
            [x[i_best]],
            [y[i_best]],
            marker="*",
            s=star_size,
            c=star_color,
            edgecolors="white",
            linewidths=0.9,
            zorder=5,
        )
        ax.text(
            0.02,
            0.98,
            f"best: {best} @ {xvar}={x[i_best]:.3f}\n{metric}={y[i_best]:.3g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                alpha=0.75,
                boxstyle="round,pad=0.25",
            ),
        )
        print(f"[{metric}] best={best} at {xvar}={x[i_best]:.6g}, value={y[i_best]:.6g}")

    xt = np.linspace(0, len(x) - 1, nticks).round().astype(int)
    ax.set_xticks(x[xt])
    ax.set_xticklabels([f"{x[i]:.2f}" for i in xt])

    ax.set_xlabel(xvar)
    ax.set_ylabel(metric)
    ax.set_title(metric)
    fig.tight_layout()

    out_base = Path(out_base)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".eps"))
    plt.close(fig)


class PDEAlexandrian1D_T1Search(Experiment):
    def __init__(self, data, nfe_x, nfe_t, x1, x2, t1, t2):
        self.data = data
        self.nfe_t = int(nfe_t)
        self.nfe_x = int(nfe_x)
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.t1 = float(t1)
        self.t2 = float(t2)
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

        # No design variables: t1 is searched externally via a Python loop.
        # DoE requires at least one experiment_input label, so we add a dummy input.
        m.dummy_input = pyo.Var(initialize=0.0)

        @m.Constraint(m.t, m.x)
        def pde(m, t, x):
            if x == m.x.first() or x == m.x.last() or t == m.t.first():
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
            m.t,
            rule=lambda m, t: (
                pyo.Constraint.Skip
                if t == m.t.first()
                else m.dudx[t, m.x.last()] == 0.0
            ),
        )
        m.bc2 = pyo.Constraint(
            m.t,
            rule=lambda m, t: (
                pyo.Constraint.Skip
                if t == m.t.first()
                else m.dudx[t, m.x.first()] == 0.0
            ),
        )

        m.bc1_dummy = pyo.Constraint(
            m.t,
            rule=lambda m, t: (
                pyo.Constraint.Skip
                if t == m.t.first()
                else m.d2udx2[t, m.x.last()] == 0.0
            ),
        )
        m.bc2_dummy = pyo.Constraint(
            m.t,
            rule=lambda m, t: (
                pyo.Constraint.Skip
                if t == m.t.first()
                else m.d2udx2[t, m.x.first()] == 0.0
            ),
        )
        m.ic_dummy = pyo.Constraint(expr=m.dudx[m.t.first(), m.x.first()] == 0.0)

        disc = pyo.TransformationFactory("dae.finite_difference")
        disc.apply_to(m, wrt=m.x, nfe=self.nfe_x, scheme="BACKWARD")
        disc.apply_to(m, wrt=m.t, nfe=self.nfe_t, scheme="BACKWARD")

        m.dummy_input.fix(0.0)

    def label_experiment(self):
        m = self.model
        x1, x2 = self.x1, self.x2
        t1, t2 = self.t1, self.t2

        # 2 sensors x 2 times
        m.y1 = pyo.Expression(expr=m.u[t1, x1])
        m.y2 = pyo.Expression(expr=m.u[t1, x2])
        m.y3 = pyo.Expression(expr=m.u[t2, x1])
        m.y4 = pyo.Expression(expr=m.u[t2, x2])

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update((k, 0.0) for k in [m.y1, m.y2, m.y3, m.y4])

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update((k, 1e-2) for k in [m.y1, m.y2, m.y3, m.y4])

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.theta1, m.theta2])

        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update([(m.dummy_input, None)])


def compute_metrics_for_t1(data_ex, nfe_x, nfe_t, x1, x2, t1, t2):
    exp = PDEAlexandrian1D_T1Search(
        data=data_ex, nfe_x=nfe_x, nfe_t=nfe_t, x1=x1, x2=x2, t1=t1, t2=t2
    )
    doe = DesignOfExperiments(
        exp,
        fd_formula="central",
        step=1e-3,
        objective_option="determinant",
        scale_constant_value=1,
        scale_nominal_param_value=True,
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_diagonal_lower_bound=1e-7,
        solver=pyo.SolverFactory("ipopt"),
        tee=False,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )
    F = np.array(doe.compute_FIM(), dtype=float)
    evals = np.linalg.eigvals(F).real
    lmin = float(evals.min())
    lmax = float(evals.max())
    detF = float(np.linalg.det(F))
    trF = float(np.trace(F))
    return {
        "t1": float(t1),
        "log10 D-opt": float(np.log10(detF)) if detF > 0 else float("nan"),
        "log10 A-opt": float(np.log10(trF)) if trF > 0 else float("nan"),
        "log10 E-opt": float(np.log10(lmin)) if lmin > 0 else float("nan"),
        "cond": float(lmax / lmin) if lmin > 0 else float("inf"),
        "lambda_min": lmin,
        "lambda_max": lmax,
    }


if __name__ == "__main__":
    data_ex = {"theta1": 1.0, "theta2": 0.2, "kappa": 0.025, "v": 0.1}

    # Defaults consistent with our current workflow
    NFE_X = 20
    NFE_T = 40
    X1 = 0.2
    X2 = 0.9

    # Fix t2 and sweep t1 over all grid-aligned interior points from 0.1 to 1-dt,
    # excluding t1=t2.
    T2 = 0.1
    dt = 1.0 / NFE_T

    t1_candidates = np.arange(T2, 1.0, dt)
    t1_candidates = [
        round(float(v), 3)
        for v in t1_candidates
        if (v < 1.0 - 1e-12) and (abs(v - T2) > 1e-12)
    ]

    rows = []
    for t1 in t1_candidates:
        print(f"Computing metrics for t1={t1} (t2 fixed at {T2}) ...")
        rows.append(compute_metrics_for_t1(data_ex, NFE_X, NFE_T, X1, X2, t1, T2))

    df = pd.DataFrame(rows).sort_values("t1")
    print("\nTop t1 by log10 E-opt (maximize):")
    print(df.sort_values("log10 E-opt", ascending=False).head(10)[["t1", "log10 E-opt", "cond", "lambda_min"]])

    OUTDIR = Path("/Users/snarasi2/projects/Pyomo_DoE_CRC/results/heatmaps")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    res_plot = {k: df[k].tolist() for k in df.columns}
    nice_curve(res_plot, "t1", "log10 E-opt", OUTDIR / "t1_search_log10_E_opt", best="max")
    nice_curve(res_plot, "t1", "cond", OUTDIR / "t1_search_cond", best="min")
    nice_curve(res_plot, "t1", "log10 D-opt", OUTDIR / "t1_search_log10_D_opt", best="max")
    nice_curve(res_plot, "t1", "log10 A-opt", OUTDIR / "t1_search_log10_A_opt", best="max")

    print(f"\nSaved t1 search plots to: {OUTDIR}")

