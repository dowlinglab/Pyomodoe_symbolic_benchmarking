#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential (two-stage) DoE for the Alexandrian 1D toy:

Stage 1: spatial selection (choose x-locations) with design vars wx_025, wx_075.
Stage 2: temporal selection (choose sampling times) at the chosen x with design vars wt_02, wt_05.

Sampling instances are kept the same as before:
  x in {0.25, 0.75}
  t in {0.2, 0.5}
"""

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe import DesignOfExperiments


class _Alexandrian1DBase(Experiment):
    def __init__(self, data, nfe_x, nfe_t):
        self.data = data
        self.nfe_x = nfe_x
        self.nfe_t = nfe_t
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

        # Unknowns we estimate (parameterized IC)
        m.theta1 = pyo.Var(within=pyo.Reals)
        m.theta2 = pyo.Var(within=pyo.Reals)

        # Known PDE coefficients
        m.kappa = pyo.Param(initialize=0.025, mutable=True)
        m.v = pyo.Param(initialize=0.1, mutable=True)

        @m.Constraint(m.t, m.x)
        def pde(m, t, x):
            if x == m.x.first() or x == m.x.last() or t == m.t.first():
                return pyo.Constraint.Skip
            return m.dudt[t, x] == m.kappa * m.d2udx2[t, x] - m.v * m.dudx[t, x]

    def finalize_model(self):
        m = self.model

        # Nominal parameter values (DoE linearizes around these)
        m.theta1.fix(self.data["theta1"])
        m.theta2.fix(self.data["theta2"])
        m.kappa.set_value(self.data["kappa"])
        m.v.set_value(self.data["v"])

        def ic_rule(m, x):
            if x < 0.5:
                return m.u[m.t.first(), x] == m.theta1
            return m.u[m.t.first(), x] == m.theta2

        m.ic = pyo.Constraint(m.x, rule=ic_rule)

        # Neumann BCs (skip t=0 because dudx is not discretized there yet)
        m.bc_left = pyo.Constraint(
            m.t,
            rule=lambda m, t: (
                pyo.Constraint.Skip
                if t == m.t.first()
                else m.dudx[t, m.x.first()] == 0.0
            ),
        )
        m.bc_right = pyo.Constraint(
            m.t,
            rule=lambda m, t: (
                pyo.Constraint.Skip
                if t == m.t.first()
                else m.dudx[t, m.x.last()] == 0.0
            ),
        )

        # Dummy constraints to square up the FD discretization at the boundaries + initial line
        m.bc_left_dummy = pyo.Constraint(
            m.t,
            rule=lambda m, t: (
                pyo.Constraint.Skip
                if t == m.t.first()
                else m.d2udx2[t, m.x.first()] == 0.0
            ),
        )
        m.bc_right_dummy = pyo.Constraint(
            m.t,
            rule=lambda m, t: (
                pyo.Constraint.Skip
                if t == m.t.first()
                else m.d2udx2[t, m.x.last()] == 0.0
            ),
        )
        m.ic_dummy = pyo.Constraint(expr=m.dudx[m.t.first(), m.x.first()] == 0.0)

        disc = pyo.TransformationFactory("dae.finite_difference")
        disc.apply_to(m, wrt=m.x, nfe=self.nfe_x, scheme="BACKWARD")
        disc.apply_to(m, wrt=m.t, nfe=self.nfe_t, scheme="BACKWARD")

        # Feasibility solve (DoE will add its own objective later)
        m.obj = pyo.Objective(expr=0.0)

    def _add_common_suffixes(self, outputs, inputs):
        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update((k, 0.0) for k in outputs)

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update((k, 1e-2) for k in outputs)

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.theta1, m.theta2])

        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update((k, None) for k in inputs)


class Alexandrian1D_Space(_Alexandrian1DBase):
    """Stage 1: choose x (spatial weights)."""

    def create_model(self):
        super().create_model()
        m = self.model
        m.wx_025 = pyo.Var(bounds=(0, 1), initialize=0.5)
        m.wx_075 = pyo.Var(bounds=(0, 1), initialize=0.5)

        # Budget: total sensor "activation" limited to 1 unit
        m.spatial_budget = pyo.Constraint(expr=m.wx_025 + m.wx_075 <= 1.0)

    def finalize_model(self):
        super().finalize_model()
        m = self.model
        # Fixed for initial feasibility solve (DoE will unfix)
        m.wx_025.fix(0.5)
        m.wx_075.fix(0.5)

    def label_experiment(self):
        m = self.model
        # 4 measurements, but weights are per x-location
        m.y1 = pyo.Expression(expr=m.wx_025 * m.u[0.2, 0.25])
        m.y2 = pyo.Expression(expr=m.wx_075 * m.u[0.2, 0.75])
        m.y3 = pyo.Expression(expr=m.wx_025 * m.u[0.5, 0.25])
        m.y4 = pyo.Expression(expr=m.wx_075 * m.u[0.5, 0.75])
        self._add_common_suffixes(
            outputs=[m.y1, m.y2, m.y3, m.y4], inputs=[m.wx_025, m.wx_075]
        )


class Alexandrian1D_Time(_Alexandrian1DBase):
    """Stage 2: choose t (temporal weights) at a fixed x-location."""

    def __init__(self, data, nfe_x, nfe_t, x_selected):
        self.x_selected = float(x_selected)
        super().__init__(data=data, nfe_x=nfe_x, nfe_t=nfe_t)

    def create_model(self):
        super().create_model()
        m = self.model
        m.wt_02 = pyo.Var(bounds=(0, 1), initialize=0.5)
        m.wt_05 = pyo.Var(bounds=(0, 1), initialize=0.5)
        m.time_budget = pyo.Constraint(expr=m.wt_02 + m.wt_05 <= 1.0)

    def finalize_model(self):
        super().finalize_model()
        m = self.model
        m.wt_02.fix(0.5)
        m.wt_05.fix(0.5)

    def label_experiment(self):
        m = self.model
        x = self.x_selected
        m.y1 = pyo.Expression(expr=m.wt_02 * m.u[0.2, x])
        m.y2 = pyo.Expression(expr=m.wt_05 * m.u[0.5, x])
        self._add_common_suffixes(outputs=[m.y1, m.y2], inputs=[m.wt_02, m.wt_05])


def run_doe(experiment, objective_option="determinant", fd_formula="central", step_size=1e-3):
    doe = DesignOfExperiments(
        experiment,
        fd_formula=fd_formula,
        step=step_size,
        objective_option=objective_option,  # "determinant" or "trace" (Cholesky path)
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
    doe.run_doe()
    return doe


if __name__ == "__main__":
    data_ex = {"theta1": 1.0, "theta2": 0.2, "kappa": 0.025, "v": 0.1}

    # Stage 1: space
    exp_space = Alexandrian1D_Space(data=data_ex, nfe_x=20, nfe_t=40)
    doe_space = run_doe(exp_space, objective_option="determinant")
    names = doe_space.results["Experiment Design Names"]
    vals = doe_space.results["Experiment Design"]
    print("Stage 1 (space) design:", list(zip(names, vals)))

    wx = dict(zip(names, vals))
    x_sel = 0.25 if wx.get("wx_025", 0.0) >= wx.get("wx_075", 0.0) else 0.75
    print("Selected x:", x_sel)

    # Stage 2: time at selected x
    exp_time = Alexandrian1D_Time(data=data_ex, nfe_x=20, nfe_t=40, x_selected=x_sel)
    doe_time = run_doe(exp_time, objective_option="determinant")
    t_names = doe_time.results["Experiment Design Names"]
    t_vals = doe_time.results["Experiment Design"]
    print("Stage 2 (time) design:", list(zip(t_names, t_vals)))
