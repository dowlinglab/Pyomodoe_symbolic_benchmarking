#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:30:29 2026

@author: snarasi2, assisted by Codex
"""
"""
This code reproduces the result from the Alexandrian et al (2014) paper
The original MBDoE formulation solves an elliptic PDE problem to find the
optimal initial initial conditions for sensor placement. The PDE is:
    ut - k(uxx + uyy + uzz) + v.(ux + uy + uz) = 0
    Initial condition: u(.,0) = m
    Boundary condition: k(ux + uy + uz).n =0
    
A contaminant has already been released, but you do not know its initial spatial distribution. 
Measurement starts after the release, and the goal is to reconstruct where contaminant
was initially exposre can be understood, event can be traced, or effective response can be planned.
 
The original infinite dimensional Bayesian inversion problem where the design
variable m(x) is infinite dimensional. Here, the correct discretization for m(x)
to solve the problem within a frequentist approach is chosen by analyzing 
the observability and identifiability of the parameters. Various basis functions (phi(x)) 
are chosen. First 1D version is solved:
    ut -kuxx + v ux = 0
x \in [0,1] and t \in [0,T]
IC: u(x,0) = m(x)
BC: kux = 0; x = 0,1


 """       
# import os
# import argparse
# import json
# import time
# from pathlib import Path
# "Modifications to avoid the IPOPT Error on CRC"

# # import shutil

# # IPOPT_BIN = shutil.which("ipopt")
# # IPOPT_LINEAR_SOLVER_DEFAULT = "ma57"
# ## Parameterization of m(x): This is for inference

"""
Unit basis functions:
    m(x) = theta1 phi1(x) + theta2 phi2(x)
    phi{1,2}(x) = 1
    splitting the x (sensor placement) domain into two: x\in [0,0.5) (corresponds to theta1) and x \in[0.5,1] (corresponds to theta2)
"""

# Discretization of the PDE: This is for solving the problem
"""
x-grid: 20 pts
t-grid: 40 pts

Objective: Find the optimal sensor locations that give max info
Sampling times: t = 0.2 and t = 0.5
k = 0.025

Measurement vector: y(theta) =[u(0.25,0.2),u(0.25,0.5),u(0.75,0.2), u(0.75,0.5)]^T

This will be used within the frequentist framework for estimating the parameters theta1 and theta2 and hence m(x)
"""

"""
Beginning code
"""

## Importing packages

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
import numpy as np
from pyomo.contrib.parmest.experiment import Experiment
from pathlib import Path

# We use our own heatmap plotting so we can control tick density/formatting.
import pandas as pd
import matplotlib.pyplot as plt


def nice_heatmap(
    results,
    xvar,
    yvar,
    metric,
    out_base,
    nticks=6,
    cmap="hot_r",
    best="max",
    star_color="#0B2A6F",
    star_size=220,
):
    """
    results: dict returned by DesignOfExperiments.compute_FIM_full_factorial
    xvar/yvar: design variable names (strings)
    metric: e.g. "log10 D-opt", "log10 A-opt", "log10 E-opt", "log10 ME-opt"
    out_base: Path or str, file path *without* extension (we save .png and .eps)
    """
    df = pd.DataFrame(results)
    M = df.pivot(index=yvar, columns=xvar, values=metric).sort_index().sort_index(axis=1)
    xvals = M.columns.to_numpy(dtype=float)
    yvals = M.index.to_numpy(dtype=float)
    Z = M.to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap)

    # Mark the best point with a star
    # A/D/E are maximized; ME-opt is minimized (since it is log10(cond(FIM))).
    if best not in ("max", "min", None):
        raise ValueError("best must be 'max', 'min', or None")
    if best is not None:
        if best == "max":
            flat = np.nanargmax(Z)
        else:
            flat = np.nanargmin(Z)
        i_best, j_best = np.unravel_index(flat, Z.shape)  # row, col
        ax.scatter(
            [j_best],
            [i_best],
            marker="*",
            s=star_size,
            c=star_color,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
        best_x = float(xvals[j_best])
        best_y = float(yvals[i_best])
        best_val = float(Z[i_best, j_best])
        # Small annotation in the plot corner + print for easy copy/paste
        ax.text(
            0.02,
            0.98,
            f"best: {best} @ ({xvar},{yvar})=({best_x:.3f},{best_y:.3f})\\n{metric}={best_val:.3g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.75, boxstyle="round,pad=0.25"),
        )
        print(f"[{metric}] best={best} at {xvar}={best_x:.6g}, {yvar}={best_y:.6g}, value={best_val:.6g}")

    xt = np.linspace(0, len(xvals) - 1, nticks).round().astype(int)
    yt = np.linspace(0, len(yvals) - 1, nticks).round().astype(int)

    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{xvals[i]:.2f}" for i in xt])
    ax.set_yticklabels([f"{yvals[i]:.2f}" for i in yt])

    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_title(metric)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    out_base = Path(out_base)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".eps"))
    plt.close(fig)


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
    """
    1D alternative to nice_heatmap when a budget is enforced via an expression,
    e.g. w_09 = 1 - w_05 (only one free design variable).
    """
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
            f"best: {best} @ {xvar}={x[i_best]:.3f}\\n{metric}={y[i_best]:.3g}",
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

    # show only a few ticks, rounded
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

class PDEAlexandrian1D(Experiment):
    def __init__(self,data,nfe_x,nfe_t):
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
        
        # Discretize space grid
        
        m.x = ContinuousSet(bounds = (0,1))
        m.t = ContinuousSet(bounds =(0,1))

        m.u = pyo.Var(m.t,m.x)

        m.dudt = DerivativeVar(m.u, wrt = m.t)

        m.dudx = DerivativeVar(m.u, wrt = m.x)

        m.d2udx2 = DerivativeVar(m.dudx, wrt = m.x)

        m.theta1 = pyo.Var(within = pyo.Reals)
        
        m.theta2 = pyo.Var(within = pyo.Reals)
        
        m.kappa = pyo.Param(initialize = 0.025, mutable = True)
        
        m.v = pyo.Param(initialize = 0.1, mutable = True)
        
        # Stage 1 design variables: spatial weights (times are fixed in outputs).
        # Enforce a one-sensor "budget" by parameterizing with ONE free weight:
        #   w_05 in [0,1] and w_09 = 1 - w_05
        # Candidate sensor locations for this run: x=0.20 and x=0.90
        m.w_05 = pyo.Var(bounds=(0, 1), initialize=0.5)
        m.w_09 = pyo.Expression(expr=1.0 - m.w_05)
        
        @m.Constraint(m.t,m.x)

        def pde(m,t,x):
            if x == m.x.first() or x == m.x.last() :
                return pyo.Constraint.Skip
            return m.dudt[t,x] == m.kappa*m.d2udx2[t,x] - m.v*m.dudx[t,x]
        
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

        ## Boundary conditions

        m.bc1 = pyo.Constraint(m.t, rule = lambda m, t: m.dudx[t,m.x.last()] == 0.0 if t!= m.t.first() else pyo.Constraint.Skip)

        m.bc2 = pyo.Constraint(m.t, rule = lambda m, t: m.dudx[t,m.x.first()] == 0.0 if t!= m.t.first() else pyo.Constraint.Skip)
        
        m.bc1_dummy = pyo.Constraint(m.t, rule = lambda m, t: m.d2udx2[t,m.x.last()] == 0.0 if t!= m.t.first() else pyo.Constraint.Skip)

        m.bc2_dummy = pyo.Constraint(m.t, rule = lambda m, t: m.d2udx2[t,m.x.first()] == 0.0 if t!= m.t.first() else pyo.Constraint.Skip)
        
        m.ic_dummy = pyo.Constraint(expr=m.dudx[m.t.first(), m.x.first()] == 0.0)
        

        
        disc = pyo.TransformationFactory("dae.finite_difference")
        disc.apply_to(m, wrt = m.x, nfe = self.nfe_x, scheme = "BACKWARD")
        disc.apply_to(m, wrt = m.t, nfe = self.nfe_t, scheme = "BACKWARD")
    
    def label_experiment(self):
        m = self.model
        
        # Keep sampling instances the same as before:
        #   times: 0.2, 0.5
        # Candidate sensor locations for this run: x=0.20, 0.90
        # Weight by spatial location (wx) so we "select space" first.
        m.y1 = pyo.Expression(expr=m.w_05 * m.u[0.2, 0.2])
        m.y2 = pyo.Expression(expr=m.w_09 * m.u[0.2, 0.9])
        m.y3 = pyo.Expression(expr=m.w_05 * m.u[0.5, 0.2])
        m.y4 = pyo.Expression(expr=m.w_09 * m.u[0.5, 0.9])
        
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update((k, 0.0) for k in [m.y1, m.y2, m.y3, m.y4])
        
        m.measurement_error = pyo.Suffix(direction = pyo.Suffix.LOCAL)
        m.measurement_error.update((k, 1e-2) for k in [m.y1, m.y2, m.y3, m.y4])
        
        m.unknown_parameters = pyo.Suffix(direction = pyo.Suffix.LOCAL)
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.theta1, m.theta2])
        
        m.experiment_inputs = pyo.Suffix(direction= pyo.Suffix.LOCAL)
        # Only the free weight is a design variable; w_09 is derived.
        m.experiment_inputs.update([(m.w_05, None)])
        
        
data_ex = {
    "theta1": 1.0,
    "theta2": 0.2,
    "kappa": 0.025,
    "v": 0.1,
}

experiment = PDEAlexandrian1D(
    data=data_ex,
    nfe_t=40,
    nfe_x=20,
)

from pyomo.contrib.doe import DesignOfExperiments

doe_space = DesignOfExperiments(
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

res1 = doe_space.compute_FIM_full_factorial(
    design_ranges={"w_05": [1e-3, 1.0 - 1e-3, 50]},
)

# Save "pretty" 1D curves under the one-sensor budget (writes both .png and .eps)
nice_curve(res1, "w_05", "log10 D-opt", OUTDIR / "space_D_opt_1D", best="max")
nice_curve(res1, "w_05", "log10 A-opt", OUTDIR / "space_A_opt_1D", best="max")
nice_curve(res1, "w_05", "log10 E-opt", OUTDIR / "space_E_opt_1D", best="max")
nice_curve(res1, "w_05", "log10 ME-opt", OUTDIR / "space_ME_opt_1D", best="min")

print(f"Saved heatmaps to: {OUTDIR}")
