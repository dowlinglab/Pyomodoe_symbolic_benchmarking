#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alexandrian-style 3D advection-diffusion (method-of-lines in space) + FIM compute.

We keep the forward model faithful to the paper form:
  u_t - kappa * (u_xx + u_yy + u_zz) + v · (u_x + u_y + u_z) = 0
on a 3D box with zero-flux (Neumann) boundary conditions.

Unknown initial condition m(x,y,z) is parameterized with 8 Neumann-friendly
tensor-product cosine modes (i,j,k in {0,1}):

  m(x,y,z) = sum_{i,j,k in {0,1}} theta[i,j,k] * cos(i*pi*x)*cos(j*pi*y)*cos(k*pi*z)

Measurements: 4 sensors x 2 sampling times = 8 outputs (enough for 8 parameters).

This script focuses on building a solvable Pyomo model and computing the FIM once.
It uses pyomo.dae only in time; space is fully discretized with finite differences.

Why method-of-lines (MOL) here:
- Doing `ContinuousSet` in x,y,z,t is far too large in Pyomo.
- MOL keeps only `t` continuous, and treats space as finite index sets.

Why 4 sensors x 2 times:
- With 8 unknown parameters, you need at least 8 outputs for a full-rank FIM.
- This is the *minimum* configuration; it can still be nearly singular if the two
  times are too close (the “time blocks” of sensitivities are redundant).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import pi
from typing import Iterable, Tuple

import numpy as np
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe import DesignOfExperiments


@dataclass(frozen=True)
class Grid3D:
    nx: int = 5
    ny: int = 5
    nz: int = 5

    @property
    def hx(self) -> float:
        return 1.0 / (self.nx - 1)

    @property
    def hy(self) -> float:
        return 1.0 / (self.ny - 1)

    @property
    def hz(self) -> float:
        return 1.0 / (self.nz - 1)

    def xcoord(self, i: int) -> float:
        return i * self.hx

    def ycoord(self, j: int) -> float:
        return j * self.hy

    def zcoord(self, k: int) -> float:
        return k * self.hz


def _idx_minus(i: int) -> int:
    """
    For central finite differences near boundaries we need "ghost" values.

    Neumann BC (zero normal derivative) can be enforced with a mirror ghost:
      u[-1] = u[+1]
    which makes (u[+1] - u[-1])/(2h) = 0 at the boundary.
    """
    return 1 if i == 0 else i - 1


def _idx_plus(i: int, n: int) -> int:
    # Mirror boundary (Neumann): u[n] = u[n-2]
    return n - 2 if i == n - 1 else i + 1


class Alexandrian3DExperiment(Experiment):
    def __init__(
        self,
        data: dict,
        grid: Grid3D,
        nfe_t: int = 20,
        t_bounds: Tuple[float, float] = (0.0, 1.0),
        t_samples: Tuple[float, float] = (0.1, 0.125),
        sensors: Tuple[Tuple[float, float, float], ...] = (
            (0.25, 0.25, 0.25),
            (0.75, 0.25, 0.25),
            (0.25, 0.75, 0.25),
            (0.25, 0.25, 0.75),
        ),
    ):
        self.data = dict(data)
        self.grid = grid
        self.nfe_t = int(nfe_t)
        self.t_bounds = tuple(float(v) for v in t_bounds)
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
        g = self.grid

        # Time only as ContinuousSet (method-of-lines)
        m.t = ContinuousSet(bounds=self.t_bounds)

        # Spatial index sets (already discretized, not continuous)
        m.I = pyo.RangeSet(0, g.nx - 1)
        m.J = pyo.RangeSet(0, g.ny - 1)
        m.K = pyo.RangeSet(0, g.nz - 1)

        # State u(t, i, j, k) and time derivative du/dt
        m.u = pyo.Var(m.t, m.I, m.J, m.K, initialize=0.0)
        m.dudt = DerivativeVar(m.u, wrt=m.t)

        # PDE coefficients (diffusion + constant advection velocity)
        m.kappa = pyo.Param(initialize=float(self.data.get("kappa", 0.025)), mutable=True)
        # Velocity vector (vx, vy, vz)
        m.vx = pyo.Param(initialize=float(self.data.get("vx", 0.1)), mutable=True)
        m.vy = pyo.Param(initialize=float(self.data.get("vy", 0.0)), mutable=True)
        m.vz = pyo.Param(initialize=float(self.data.get("vz", 0.0)), mutable=True)

        # 8 IC parameters theta[a,b,c] with (a,b,c) in {0,1}^3
        m.ThetaI = pyo.Set(initialize=[0, 1])
        m.theta = pyo.Var(m.ThetaI, m.ThetaI, m.ThetaI, within=pyo.Reals, initialize=0.0)

        # Pyomo.DoE requires an `experiment_inputs` suffix.
        # When we are not optimizing any design variables yet, we add a dummy var and fix it.
        m.dummy_input = pyo.Var(initialize=0.0)

        hx, hy, hz = g.hx, g.hy, g.hz

        def laplacian_expr(t, i, j, k):
            # 3D Laplacian using central differences with mirror-ghost indices for Neumann BC.
            im, ip = _idx_minus(i), _idx_plus(i, g.nx)
            jm, jp = _idx_minus(j), _idx_plus(j, g.ny)
            km, kp = _idx_minus(k), _idx_plus(k, g.nz)
            u = m.u
            d2x = (u[t, ip, j, k] - 2 * u[t, i, j, k] + u[t, im, j, k]) / (hx * hx)
            d2y = (u[t, i, jp, k] - 2 * u[t, i, j, k] + u[t, i, jm, k]) / (hy * hy)
            d2z = (u[t, i, j, kp] - 2 * u[t, i, j, k] + u[t, i, j, km]) / (hz * hz)
            return d2x + d2y + d2z

        def gradx_expr(t, i, j, k):
            # du/dx with central differences + mirror ghosts at boundary.
            im, ip = _idx_minus(i), _idx_plus(i, g.nx)
            return (m.u[t, ip, j, k] - m.u[t, im, j, k]) / (2 * hx)

        def grady_expr(t, i, j, k):
            jm, jp = _idx_minus(j), _idx_plus(j, g.ny)
            return (m.u[t, i, jp, k] - m.u[t, i, jm, k]) / (2 * hy)

        def gradz_expr(t, i, j, k):
            km, kp = _idx_minus(k), _idx_plus(k, g.nz)
            return (m.u[t, i, j, kp] - m.u[t, i, j, km]) / (2 * hz)

        @m.Constraint(m.t, m.I, m.J, m.K)
        def pde(m, t, i, j, k):
            if t == m.t.first():
                return pyo.Constraint.Skip
            # Matches paper sign convention:
            #   u_t - kappa*Delta u + v·grad u = 0
            # rearranged to:
            #   u_t = kappa*Delta u - v·grad u
            return (
                m.dudt[t, i, j, k]
                == m.kappa * laplacian_expr(t, i, j, k)
                - (m.vx * gradx_expr(t, i, j, k) + m.vy * grady_expr(t, i, j, k) + m.vz * gradz_expr(t, i, j, k))
            )

        # Initial condition: u(t0,i,j,k) = m(x,y,z) via 8 cosine modes.
        # This is "physics-friendly" for Neumann BC because cos(k*pi*x) has zero derivative at 0 and 1.
        def ic_rule(m, i, j, k):
            x = g.xcoord(i)
            y = g.ycoord(j)
            z = g.zcoord(k)
            expr = 0.0
            for a, b, c in product([0, 1], [0, 1], [0, 1]):
                expr = expr + m.theta[a, b, c] * pyo.cos(a * pi * x) * pyo.cos(b * pi * y) * pyo.cos(c * pi * z)
            return m.u[m.t.first(), i, j, k] == expr

        m.ic = pyo.Constraint(m.I, m.J, m.K, rule=ic_rule)

        # Time discretization (implicit backward differences)
        disc = pyo.TransformationFactory("dae.finite_difference")
        disc.apply_to(m, wrt=m.t, nfe=self.nfe_t, scheme="BACKWARD")

        # Dummy input fixed
        m.dummy_input.fix(0.0)

        # Zero objective (we will call compute_FIM / run_doe)
        m.obj = pyo.Objective(expr=0.0)

    def finalize_model(self):
        m = self.model

        # Set nominal theta values (optional)
        theta0 = self.data.get("theta_nominal", None)
        if theta0 is not None:
            # expects dict-like {(a,b,c): val} or list length 8 in lex order
            if isinstance(theta0, dict):
                for (a, b, c), val in theta0.items():
                    m.theta[a, b, c].fix(float(val))
            else:
                vals = list(theta0)
                if len(vals) != 8:
                    raise ValueError("theta_nominal list must have length 8")
                for idx, (a, b, c) in enumerate(product([0, 1], [0, 1], [0, 1])):
                    m.theta[a, b, c].fix(float(vals[idx]))
        else:
            # Default: mean + a single x-mode so the IC is not identically zero.
            # IMPORTANT for Pyomo.DoE central-FD:
            # - The sequential FD method scales perturbations by the nominal parameter value.
            # - If a nominal value is exactly 0, the computed step can become 0, causing divide-by-zero.
            m.theta[0, 0, 0].fix(0.6)
            m.theta[1, 0, 0].fix(0.2)
            for a, b, c in product([0, 1], [0, 1], [0, 1]):
                if (a, b, c) not in [(0, 0, 0), (1, 0, 0)]:
                    # IMPORTANT: Pyomo.DoE's sequential central-FD step is scaled by
                    # the nominal parameter value. If a nominal value is exactly 0,
                    # the perturbation step becomes 0 and compute_FIM can divide by 0.
                    # Use a tiny nonzero nominal value to keep step sizes well-defined.
                    m.theta[a, b, c].fix(1e-3)

        # Provide a mild initialization for u at all times (helps IPOPT).
        # Start with the IC value everywhere in time.
        t0 = m.t.first()
        for t in list(m.t):
            for i in m.I:
                for j in m.J:
                    for k in m.K:
                        m.u[t, i, j, k].set_value(pyo.value(m.u[t0, i, j, k]))

    def label_experiment(self):
        m = self.model
        g = self.grid

        # Map sensor coordinates to nearest grid indices (grid-aligned assumed)
        def find_idx(val: float, n: int) -> int:
            # nearest grid index in [0, n-1]
            idx = int(round(val * (n - 1)))
            return max(0, min(n - 1, idx))

        sensor_indices = []
        for (xs, ys, zs) in self.sensors:
            ii = find_idx(xs, g.nx)
            jj = find_idx(ys, g.ny)
            kk = find_idx(zs, g.nz)
            sensor_indices.append((ii, jj, kk))

        t1, t2 = self.t_samples
        # 8 outputs: (time 1, 4 sensors) then (time 2, 4 sensors)
        y_exprs = []
        for tt in (t1, t2):
            for (ii, jj, kk) in sensor_indices:
                y_exprs.append(pyo.Expression(expr=m.u[tt, ii, jj, kk]))

        # Attach to model for easier debugging
        m.y = pyo.Block()
        for idx, expr in enumerate(y_exprs, start=1):
            setattr(m.y, f"y{idx}", expr)

        outs = [getattr(m.y, f"y{i}") for i in range(1, 9)]

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update((k, 0.0) for k in outs)

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update((k, 1e-2) for k in outs)

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((m.theta[a, b, c], pyo.value(m.theta[a, b, c])) for a, b, c in product([0, 1], [0, 1], [0, 1]))

        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update([(m.dummy_input, None)])


if __name__ == "__main__":
    # Minimal "base" run: compute FIM once for a fixed sensor/time design.
    grid = Grid3D(nx=5, ny=5, nz=5)

    data_ex = {
        "kappa": 0.025,
        "vx": 0.1,
        "vy": 0.0,
        "vz": 0.0,
        # optional: "theta_nominal": dict or list of 8 values
    }

    exp = Alexandrian3DExperiment(
        data=data_ex,
        grid=grid,
        nfe_t=40,
        t_samples=(0.1, 0.125),
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
