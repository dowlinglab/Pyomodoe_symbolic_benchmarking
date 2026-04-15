#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alexandrian-style 3D advection-diffusion written in the SAME Pyomo.DAE style as the 1D model:

  - x, y, z, t are ContinuousSet
  - u(t,x,y,z) is a Var indexed by (t,x,y,z)
  - spatial derivatives are DerivativeVar wrt x/y/z
  - time derivative is DerivativeVar wrt t
  - then we discretize all 4 ContinuousSets with dae.finite_difference

This is conceptually the closest to your 1D code, but it can get very large in 3D.
So this file is a *coarse* "base" example to compute an FIM once (not a full sensor-placement solve).

Forward PDE (paper form):
  u_t - kappa*(u_xx + u_yy + u_zz) + vx*u_x + vy*u_y + vz*u_z = 0
Rearranged:
  u_t = kappa*(u_xx + u_yy + u_zz) - (vx*u_x + vy*u_y + vz*u_z)

Boundary conditions (paper): zero-flux / Neumann:
  u_x = 0 on x=0 and x=1 (for all y,z,t)
  u_y = 0 on y=0 and y=1 (for all x,z,t)
  u_z = 0 on z=0 and z=1 (for all x,y,t)

Unknown initial condition m(x,y,z) parameterized by 8 Neumann-friendly cosine modes:
  m(x,y,z) = sum_{a,b,c in {0,1}} theta[a,b,c] * cos(a*pi*x)cos(b*pi*y)cos(c*pi*z)

Measurements: 4 sensors x 2 times = 8 outputs (enough for 8 parameters).

Authorship / responsibilities:
- Code author: Shilpa Narasimhan.
- Supporting role: Codex (AI coding assistant) helped with code structure and explanations.
- Testing/validation responsibility: Shilpa Narasimhan.
"""

from itertools import product
from math import pi
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe import DesignOfExperiments


class Alexandrian3D_DAE_Experiment(Experiment):
    def __init__(
        self,
        data: dict,
        nfe_t: int = 10,
        nfe_x: int = 4,
        nfe_y: int = 4,
        nfe_z: int = 4,
        t_bounds=(0.0, 1.0),
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
        z_bounds=(0.0, 1.0),
        # IMPORTANT: choose times that lie ON the time grid after discretization.
        # With bounds (0,1) and nfe_t=10 (BACKWARD), time points are multiples of 0.1.
        t_samples=(0.1, 0.8),
        # IMPORTANT: choose sensor coords ON the space grids after discretization.
        # With nfe_x=nfe_y=nfe_z=4, grid points are {0,0.25,0.5,0.75,1}.
        sensors=((0.25, 0.25, 0.25), (0.75, 0.25, 0.25), (0.25, 0.75, 0.25), (0.25, 0.25, 0.75)),
    ):
        self.data = dict(data)
        self.nfe_t = int(nfe_t)
        self.nfe_x = int(nfe_x)
        self.nfe_y = int(nfe_y)
        self.nfe_z = int(nfe_z)
        self.t_bounds = tuple(float(v) for v in t_bounds)
        self.x_bounds = tuple(float(v) for v in x_bounds)
        self.y_bounds = tuple(float(v) for v in y_bounds)
        self.z_bounds = tuple(float(v) for v in z_bounds)
        self.t_samples = tuple(float(v) for v in t_samples)
        self.sensors = tuple(tuple(float(v) for v in s) for s in sensors)
        if len(self.sensors) != 4:
            raise ValueError("This base script assumes exactly 4 sensors.")
        if len(self.t_samples) != 2:
            raise ValueError("This base script assumes exactly 2 sampling times.")
        self.model = None

    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_experiment()
        return self.model

    def create_model(self):
        m = self.model = pyo.ConcreteModel()

        # 1) Continuous domains (THIS mirrors your 1D DAE style)
        m.t = ContinuousSet(bounds=self.t_bounds)
        m.x = ContinuousSet(bounds=self.x_bounds)
        m.y = ContinuousSet(bounds=self.y_bounds)
        m.z = ContinuousSet(bounds=self.z_bounds)

        # 2) State u(t,x,y,z)
        m.u = pyo.Var(m.t, m.x, m.y, m.z, initialize=0.0)

        # 3) Derivatives in time and space
        m.dudt = DerivativeVar(m.u, wrt=m.t)

        m.dudx = DerivativeVar(m.u, wrt=m.x)
        m.dudy = DerivativeVar(m.u, wrt=m.y)
        m.dudz = DerivativeVar(m.u, wrt=m.z)

        m.d2udx2 = DerivativeVar(m.dudx, wrt=m.x)
        m.d2udy2 = DerivativeVar(m.dudy, wrt=m.y)
        m.d2udz2 = DerivativeVar(m.dudz, wrt=m.z)

        # 4) PDE coefficients
        m.kappa = pyo.Param(initialize=float(self.data.get("kappa", 0.025)), mutable=True)
        m.vx = pyo.Param(initialize=float(self.data.get("vx", 0.1)), mutable=True)
        m.vy = pyo.Param(initialize=float(self.data.get("vy", 0.0)), mutable=True)
        m.vz = pyo.Param(initialize=float(self.data.get("vz", 0.0)), mutable=True)

        # 5) Unknown initial condition parameters (8 params: a,b,c in {0,1})
        m.ThetaI = pyo.Set(initialize=[0, 1])
        m.theta = pyo.Var(m.ThetaI, m.ThetaI, m.ThetaI, within=pyo.Reals, initialize=1e-3)

        # 6) DoE needs experiment_inputs; if we are not optimizing design variables yet,
        # we add a dummy input variable and fix it.
        m.dummy_input = pyo.Var(initialize=0.0)

        # 7) PDE constraint:
        # We skip t=0 because IC defines u there.
        # We skip boundary points in x/y/z because Neumann BC defines derivatives there.
        @m.Constraint(m.t, m.x, m.y, m.z)
        def pde(m, t, x, y, z):
            if t == m.t.first():
                return pyo.Constraint.Skip
            if x == m.x.first() or x == m.x.last():
                return pyo.Constraint.Skip
            if y == m.y.first() or y == m.y.last():
                return pyo.Constraint.Skip
            if z == m.z.first() or z == m.z.last():
                return pyo.Constraint.Skip
            return m.dudt[t, x, y, z] == m.kappa * (
                m.d2udx2[t, x, y, z] + m.d2udy2[t, x, y, z] + m.d2udz2[t, x, y, z]
            ) - (
                m.vx * m.dudx[t, x, y, z] + m.vy * m.dudy[t, x, y, z] + m.vz * m.dudz[t, x, y, z]
            )

        # 8) Initial condition u(0,x,y,z) = m(x,y,z)
        def ic_expr(m, x, y, z):
            # 8-parameter cosine basis that is smooth and consistent with Neumann BC.
            expr = 0.0
            for a, b, c in product([0, 1], [0, 1], [0, 1]):
                expr = expr + m.theta[a, b, c] * pyo.cos(a * pi * x) * pyo.cos(b * pi * y) * pyo.cos(c * pi * z)
            return expr

        @m.Constraint(m.x, m.y, m.z)
        def ic(m, x, y, z):
            return m.u[m.t.first(), x, y, z] == ic_expr(m, x, y, z)

        # 9) Neumann boundary conditions (zero flux): u_x=0 on x=0,1; u_y=0 on y=0,1; u_z=0 on z=0,1
        # These are constraints indexed over the remaining free directions.
        @m.Constraint(m.t, m.y, m.z)
        def bc_x0(m, t, y, z):
            if t == m.t.first():
                return pyo.Constraint.Skip
            return m.dudx[t, m.x.first(), y, z] == 0.0

        @m.Constraint(m.t, m.y, m.z)
        def bc_x1(m, t, y, z):
            if t == m.t.first():
                return pyo.Constraint.Skip
            return m.dudx[t, m.x.last(), y, z] == 0.0

        @m.Constraint(m.t, m.x, m.z)
        def bc_y0(m, t, x, z):
            if t == m.t.first():
                return pyo.Constraint.Skip
            return m.dudy[t, x, m.y.first(), z] == 0.0

        @m.Constraint(m.t, m.x, m.z)
        def bc_y1(m, t, x, z):
            if t == m.t.first():
                return pyo.Constraint.Skip
            return m.dudy[t, x, m.y.last(), z] == 0.0

        @m.Constraint(m.t, m.x, m.y)
        def bc_z0(m, t, x, y):
            if t == m.t.first():
                return pyo.Constraint.Skip
            return m.dudz[t, x, y, m.z.first()] == 0.0

        @m.Constraint(m.t, m.x, m.y)
        def bc_z1(m, t, x, y):
            if t == m.t.first():
                return pyo.Constraint.Skip
            return m.dudz[t, x, y, m.z.last()] == 0.0

        # Dummy input fixed
        m.dummy_input.fix(0.0)

        # Objective placeholder
        m.obj = pyo.Objective(expr=0.0)

    def finalize_model(self):
        m = self.model

        # IMPORTANT for Pyomo.DoE sequential central-FD:
        # If a nominal parameter value is exactly 0, the perturbation step can become 0
        # and the FD Jacobian/FIM computation can divide by 0.
        #
        # Therefore, set a small nonzero default for any "unused" theta modes.
        for a, b, c in product([0, 1], [0, 1], [0, 1]):
            m.theta[a, b, c].fix(float(self.data.get("theta_nominal", {}).get((a, b, c), 1e-3)))

        # Give the IC a meaningful nonzero shape by overriding a couple modes
        # (you can change these as you like)
        if (0, 0, 0) not in self.data.get("theta_nominal", {}):
            m.theta[0, 0, 0].set_value(0.6)
        if (1, 0, 0) not in self.data.get("theta_nominal", {}):
            m.theta[1, 0, 0].set_value(0.2)

        # Discretize t, x, y, z using finite differences.
        # BACKWARD is robust and mirrors what we did in 1D.
        disc = pyo.TransformationFactory("dae.finite_difference")
        disc.apply_to(m, wrt=m.t, nfe=self.nfe_t, scheme="BACKWARD")
        disc.apply_to(m, wrt=m.x, nfe=self.nfe_x, scheme="BACKWARD")
        disc.apply_to(m, wrt=m.y, nfe=self.nfe_y, scheme="BACKWARD")
        disc.apply_to(m, wrt=m.z, nfe=self.nfe_z, scheme="BACKWARD")

    def label_experiment(self):
        m = self.model

        # Create 8 outputs: 4 sensors at each of the 2 times.
        t1, t2 = self.t_samples

        outs = []
        for tt in (t1, t2):
            for (xs, ys, zs) in self.sensors:
                outs.append(pyo.Expression(expr=m.u[tt, xs, ys, zs]))

        # Store outputs on a dedicated block name (avoid clobbering m.y which is the y-domain)
        m.obs = pyo.Block()
        for idx, expr in enumerate(outs, start=1):
            setattr(m.obs, f"y{idx}", expr)

        out_list = [getattr(m.obs, f"y{i}") for i in range(1, 9)]

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update((k, 0.0) for k in out_list)

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update((k, 1e-2) for k in out_list)

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((m.theta[a, b, c], pyo.value(m.theta[a, b, c])) for a, b, c in product([0, 1], [0, 1], [0, 1]))

        # No real design variables yet => dummy input satisfies DoE interface
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update([(m.dummy_input, None)])


if __name__ == "__main__":
    # Default coefficients
    data_ex = {"kappa": 0.025, "vx": 0.1, "vy": 0.0, "vz": 0.0}

    exp = Alexandrian3D_DAE_Experiment(
        data=data_ex,
        nfe_t=10,
        nfe_x=4,
        nfe_y=4,
        nfe_z=4,
        t_samples=(0.1, 0.8),
        sensors=((0.25, 0.25, 0.25), (0.75, 0.25, 0.25), (0.25, 0.75, 0.25), (0.25, 0.25, 0.75)),
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
    vals = np.linalg.eigvals(F).real
    print("FIM shape:", F.shape)
    print("lambda_min:", float(vals.min()), "lambda_max:", float(vals.max()), "cond:", float(vals.max() / vals.min()))
