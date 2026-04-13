#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 (time selection) for the Alexandrian 1D toy.

We assume Stage 1 has selected a sensor location x_sensor (space fixed).
Now we choose *when* to sample by assigning weights to the two sampling times:

  wt_02 in [0,1]
  wt_05 = 1 - wt_02

Outputs (time-weighted, at fixed x_sensor):
  y1 = wt_02 * u(t=0.2, x_sensor)
  y2 = wt_05 * u(t=0.5, x_sensor)

We then do a 1D factorial sweep over wt_02 and plot A/D/E/ME metrics vs wt_02.
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
        print(
            f"[{metric}] best={best} at {xvar}={x[i_best]:.6g}, value={y[i_best]:.6g}"
        )

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


class PDEAlexandrian1D_Time(Experiment):
    def __init__(self, data, nfe_x, nfe_t, x_sensor):
        self.data = data
        self.nfe_t = nfe_t
        self.nfe_x = nfe_x
        self.x_sensor = float(x_sensor)
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

        # Stage 2 design vars: time weights (space fixed to x_sensor)
        m.wt_02 = pyo.Var(bounds=(0, 1), initialize=0.5)
        m.wt_05 = pyo.Expression(expr=1.0 - m.wt_02)

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

        m.obj = pyo.Objective(expr=0.0)

    def label_experiment(self):
        m = self.model
        x = self.x_sensor

        m.y1 = pyo.Expression(expr=m.wt_02 * m.u[0.2, x])
        m.y2 = pyo.Expression(expr=m.wt_05 * m.u[0.5, x])

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update((k, 0.0) for k in [m.y1, m.y2])

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update((k, 1e-2) for k in [m.y1, m.y2])

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.theta1, m.theta2])

        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update([(m.wt_02, None)])


if __name__ == "__main__":
    # Stage 2 assumes a chosen sensor location from Stage 1.
    # Default: use x=0.2 (as in our "binary pick" from D-opt if w_05 > 0.5).
    X_SENSOR = 0.2

    data_ex = {"theta1": 1.0, "theta2": 0.2, "kappa": 0.025, "v": 0.1}

    experiment = PDEAlexandrian1D_Time(data=data_ex, nfe_t=40, nfe_x=20, x_sensor=X_SENSOR)

    doe_time = DesignOfExperiments(
        experiment,
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

    OUTDIR = Path("/Users/snarasi2/projects/Pyomo_DoE_CRC/results/heatmaps")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    res2 = doe_time.compute_FIM_full_factorial(
        design_ranges={"wt_02": [1e-3, 1.0 - 1e-3, 50]},
    )

    nice_curve(res2, "wt_02", "log10 D-opt", OUTDIR / "time_D_opt_1D", best="max")
    nice_curve(res2, "wt_02", "log10 A-opt", OUTDIR / "time_A_opt_1D", best="max")
    nice_curve(res2, "wt_02", "log10 E-opt", OUTDIR / "time_E_opt_1D", best="max")
    nice_curve(res2, "wt_02", "log10 ME-opt", OUTDIR / "time_ME_opt_1D", best="min")

    print(f"Saved time-sweep plots to: {OUTDIR}")

