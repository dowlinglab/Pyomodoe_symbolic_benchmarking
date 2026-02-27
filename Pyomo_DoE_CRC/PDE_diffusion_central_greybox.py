#!/usr/bin/env python
# coding: utf-8


def _print_model_sizes_from_ipopt(path="ipopt.out"):
    import re
    try:
        txt = open(path, "r", encoding="utf-8", errors="replace").read()
    except Exception:
        return

    def grab_int(pat):
        m = re.search(pat, txt)
        return int(m.group(1)) if m else None

    n_cons = grab_int(r"Number of constraints\s*:\s*(\d+)")
    nnz_jac = grab_int(r"Number of nonzeros in Jacobian\s*:\s*(\d+)")
    nnz_hess = grab_int(r"Number of nonzeros in Hessian\s*:\s*(\d+)")
    print(
        f"Model sizes (from IPOPT): n_cons={n_cons}, nnz_jac={nnz_jac}, nnz_hess={nnz_hess}"
    )


# This code builds on the transient heat conduction example from the Pyomo tutorial examples by Jeff Kantor. The model simulates the following equation:
#
# $\frac{\partial T}{\partial t} = \alpha \frac{ \partial ^2 T}{\partial x^2}$
#
# $\alpha = 1$
#
# $T(0,x) = 0$ for all $0 \leq x \leq 1$
#
# $T(t,1) = u(t)$ for all $t >0$
#
# $\frac{dT}{dx}(t,0) = 0$ for all $t \geq 0$
#


import os
import argparse
import json
import time
from pathlib import Path

"Modifications to avoid the IPOPT Error on CRC"

import shutil
import pyomo.environ as pyo

IPOPT_BIN = shutil.which("ipopt")


def make_ipopt(output_file):
    set = pyo.SolverFactory("ipopt", executable=IPOPT_BIN)
    set.options["linear_solver"] = "ma57"
    # # ---- IPOPT logging ----
    set.options["output_file"] = output_file
    set.options["file_print_level"] = 12
    return set


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D  ## Needed to create 3D plots

import shutil  ## Shell utilities for python, allows to copy/move/rename files or delete directories
import sys  ## interact with python interpreter itself
import os.path  ## path manipulation utilities

# from pyomo.environ import *
# from pyomo.dae import *
from idaes.core.util import DiagnosticsToolbox

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
import numpy as np
from pyomo.contrib.parmest.experiment import Experiment


## Defining experiment class
class PDE_diffusion(Experiment):
    def __init__(self, data, nfe_t, nfe_x, ncp_x, ncp_t):
        self.data = data
        self.nfe_t = nfe_t
        self.nfe_x = nfe_x
        self.ncp_t = ncp_t
        self.ncp_x = ncp_x
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
        m.t = ContinuousSet(bounds=(0, 2))

        m.u = pyo.Var(m.t, within=pyo.Reals, initialize=0, bounds=(0, 1))

        m.T = Var(m.t, m.x)

        m.dTdt = DerivativeVar(m.T, wrt=m.t)

        m.dTdx = DerivativeVar(m.T, wrt=m.x)

        m.d2Tdx2 = DerivativeVar(m.dTdx, wrt=m.x)

        m.alpha = Var(within=pyo.Reals)

        @m.Constraint(m.t, m.x)
        def pde(m, t, x):
            if x == 0 or t == 0:
                return Constraint.Skip
            else:
                return m.dTdt[t, x] == m.alpha * m.d2Tdx2[t, x]

    def finalize_model(self):
        m = self.model

        m.alpha.fix(self.data["alpha"])

        m.u.fix()

        m.ic = Constraint(
            m.x,
            rule=lambda m, x: m.T[0, x] == 0 if x < m.x.last() else Constraint.Skip,
        )

        ## Boundary conditions

        m.bc1 = Constraint(m.t, rule=lambda m, t: m.T[t, m.x.last()] == 1)

        m.bc2 = Constraint(m.t, rule=lambda m, t: m.dTdx[t, m.x.first()] == m.u[t])

        TransformationFactory("dae.finite_difference").apply_to(
            m, nfe=self.nfe_x, wrt=m.x
        )
        TransformationFactory("dae.finite_difference").apply_to(
            m, nfe=self.nfe_t, wrt=m.t
        )

        # TransformationFactory("dae.collocation").apply_to(m,nfe = 20,ncp = 3, wrt = m.x)
        # TransformationFactory("dae.collocation").apply_to(m,nfe = 40,ncp = 3, wrt = m.t)

    def label_experiment(self):
        m = self.model

        # Set measurement labels
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        m.experiment_outputs.update((m.T[t, x], None) for t in m.t for x in m.x)

        # Adding error for measurement values (assuming no covariance and constant error for all measurements)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        meas_error = 1e-2  # Error in state measurement

        m.measurement_error.update((m.T[t, x], meas_error) for t in m.t for x in m.x)

        # Identify design variables (experiment inputs) for the model
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add experimental input label for control input
        m.experiment_inputs.update((m.u[t], None) for t in m.t)

        # Add unknown parameter labels
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add labels to all unknown parameters with nominal value as the value
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.alpha])


import idaes
from idaes.core.util.model_diagnostics import degrees_of_freedom
# print("Degrees of freedom:", degrees_of_freedom(m))

from pyomo.common.dependencies import numpy as np, pathlib

from pyomo.contrib.doe.examples.reactor_experiment import ReactorExperiment
from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo
from pyomo.environ import (
    ConcreteModel,
    Var,
    Param,
    Constraint,
    TransformationFactory,
    SolverFactory,
    Objective,
    minimize,
    value as pyovalue,
    Suffix,
    Expression,
    sin,
    NonNegativeReals,
)

flat_u = {
    0.0: 0.5,
    0.125: 0.5,
    0.25: 0.5,
    0.375: 0.5,
    0.5: 0,
    0.625: 0.5,
    0.75: 0.2,
    0.875: 0.5,
    1.0: 0.5,
}

data_ex = {"alpha": 0.8, "control_points": flat_u}

# Use a central difference, with step size 1e-3
fd_formula = "central"
step_size = 1e-3

# Use the determinant objective with scaled sensitivity matrix
objective_option = os.environ.get("BENCH_OBJECTIVE_OPTION", "determinant")
print(f"Objective option: {objective_option}")
scale_nominal_param_value = True


# "Calling the Doe object 1000 times, saving to an excel file"
## Importing required packages

from pathlib import Path
from openpyxl import Workbook, load_workbook


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfe_t", type=int, default=75)
    parser.add_argument("--nfe_x", type=int, default=50)
    parser.add_argument("--ncp_t", type=int, default=2)
    parser.add_argument("--ncp_x", type=int, default=2)
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--ipopt_out", type=str, default="")
    return parser.parse_args()


def parse_ipopt_out(ipopt_out_path):
    if not ipopt_out_path or not os.path.exists(ipopt_out_path):
        return {}
    import re

    with open(ipopt_out_path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()

    def grab_int(pat):
        m = re.search(pat, txt)
        return int(m.group(1)) if m else None

    def grab_float(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else None

    stats = {}
    stats["iterations"] = grab_int(r"Number of Iterations.*:\s+(\d+)")

    m = re.search(
        r"Objective\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+).*?"
        r"Dual infeasibility\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+).*?"
        r"Constraint violation\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+).*?"
        r"Complementarity\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+).*?"
        r"Overall NLP error\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)",
        txt,
        re.DOTALL,
    )
    if m:
        stats["objective_scaled"] = float(m.group(1))
        stats["objective_unscaled"] = float(m.group(2))
        stats["dual_infeasibility_scaled"] = float(m.group(3))
        stats["dual_infeasibility_unscaled"] = float(m.group(4))
        stats["constraint_violation_scaled"] = float(m.group(5))
        stats["constraint_violation_unscaled"] = float(m.group(6))
        stats["complementarity_scaled"] = float(m.group(7))
        stats["complementarity_unscaled"] = float(m.group(8))
        stats["overall_nlp_error_scaled"] = float(m.group(9))
        stats["overall_nlp_error_unscaled"] = float(m.group(10))

    stats["obj_eval_count"] = grab_int(
        r"Number of objective function evaluations\s*=\s*(\d+)"
    )
    stats["grad_eval_count"] = grab_int(
        r"Number of objective gradient evaluations\s*=\s*(\d+)"
    )
    stats["eq_con_eval_count"] = grab_int(
        r"Number of equality constraint evaluations\s*=\s*(\d+)"
    )
    stats["ineq_con_eval_count"] = grab_int(
        r"Number of inequality constraint evaluations\s*=\s*(\d+)"
    )
    stats["eq_con_jac_eval_count"] = grab_int(
        r"Number of equality constraint Jacobian evaluations\s*=\s*(\d+)"
    )
    stats["ineq_con_jac_eval_count"] = grab_int(
        r"Number of inequality constraint Jacobian evaluations\s*=\s*(\d+)"
    )
    stats["hess_eval_count"] = grab_int(
        r"Number of Lagrangian Hessian evaluations\s*=\s*(\d+)"
    )

    stats["ipopt_cpu_secs_no_eval"] = grab_float(
        r"Total CPU secs in IPOPT \(w/o function evaluations\)\s*=\s*([0-9.]+)"
    )
    stats["nlp_cpu_secs_eval"] = grab_float(
        r"Total CPU secs in NLP function evaluations\s*=\s*([0-9.]+)"
    )

    m = re.search(r"EXIT:\s*(.*)", txt)
    stats["exit"] = m.group(1).strip() if m else None
    return stats


def main():
    args = parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else Path(".")
    run_dir.mkdir(parents=True, exist_ok=True)

    ipopt_out_path = args.ipopt_out or str(run_dir / "ipopt.out")
    if os.path.exists(ipopt_out_path):
        os.remove(ipopt_out_path)

    # Create an Experiment object; data and discretization information are part
    # of the constructor of this object
    experiment = PDE_diffusion(
        data=data_ex,
        nfe_t=args.nfe_t,
        nfe_x=args.nfe_x,
        ncp_x=args.ncp_x,
        ncp_t=args.ncp_t,
    )

    doe_obj = DesignOfExperiments(
        experiment,
        fd_formula=fd_formula,
        step=step_size,
        objective_option=objective_option,
        use_grey_box_objective=True,
        scale_constant_value=1,
        scale_nominal_param_value=scale_nominal_param_value,
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_diagonal_lower_bound=1e-7,
        solver=make_ipopt(ipopt_out_path),  # SolverFactory('IPOPT'),
        tee=False,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    start_ts = time.time()
    print("Objective implementation: greybox (external)")

    doe_obj.use_grey_box = True
    model = doe_obj.model
    def _print_design(label):
        print(label)
        if hasattr(model, "scenario_blocks") and len(model.scenario_blocks) > 0:
            names = []
            vals = []
            for comp in model.scenario_blocks[0].experiment_inputs:
                if comp.value is None:
                    comp.set_value(0.0)
                names.append(comp.name)
                vals.append(pyo.value(comp))
            for n, v in zip(names, vals):
                print(f"{n} = {v}")
        else:
            print("(experiment_inputs unavailable before create_doe_model)")

    print("=== INITIAL DESIGN (greybox) [pre-create] ===")
    _print_design("design_values:")

    # Diagnostics (PDE_diffusion only)
    try:
        import pyomo as _pyomo
        print(f"sys.executable: {sys.executable}")
        print(f"pyomo.__file__: {_pyomo.__file__}")
        _solver = make_ipopt(ipopt_out_path)
        _exe = _solver.executable() if hasattr(_solver, "executable") else None
        print(f"ipopt executable: {_exe}")
        print(f"ipopt options: {_solver.options}")
        try:
            import idaes
            try:
                from idaes.core.util import paths as _idaes_paths
                print(f"idaes bin-directory: {_idaes_paths.bin_directory()}")
            except Exception as _e:
                print(f"idaes bin-directory: <unavailable> ({_e})")
        except Exception as _e:
            print(f"idaes import: <unavailable> ({_e})")
    except Exception as _e:
        print(f"diagnostics error: {_e}")

    try:
        if not doe_obj._built_scenarios:
            doe_obj.create_doe_model(model=model)
    except Exception:
        print("=== INITIAL DESIGN (greybox) [exception] ===")
        _print_design("design_values:")
        raise

    # Ensure identical initial design values and print them
    init_names = []
    init_vals = []
    for comp in model.scenario_blocks[0].experiment_inputs:
        if comp.value is None:
            comp.set_value(0.0)
        init_names.append(comp.name)
        init_vals.append(pyo.value(comp))
    print("=== INITIAL DESIGN (greybox) ===")
    for n, v in zip(init_names, init_vals):
        print(f"{n} = {v}")

    doe_obj.create_grey_box_objective_function(model=model)

    def _noop(*args, **kwargs):
        return None

    doe_obj.create_grey_box_objective_function = _noop
    doe_obj.run_doe(model=model)
    _print_model_sizes_from_ipopt()
    end_ts = time.time()

    print("=== FINAL DESIGN (greybox) ===")
    for n, v in zip(doe_obj.results["Experiment Design Names"], doe_obj.results["Experiment Design"]):
        print(f"{n} = {v}")

    """ Print out IPOPT log- written with the help of chatGPT 5.2"""

    import re  ## Module that helps search for the string of interest

    with open(ipopt_out_path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    ## Open the .out file in read-only mode, anything that is Nan or things like that should be ignored

    def grab_int(pat):
        m = re.search(pat, txt)
        return int(m.group(1)) if m else None

    def grab_float(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else None

    # Number of Iterations
    print("Number of Iterations....:", grab_int(r"Number of Iterations.*:\s+(\d+)"))

    # Scaled/unscaled final table (print block exactly like you showed)
    m = re.search(
        r"Objective\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+).*?"
        r"Dual infeasibility\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+).*?"
        r"Constraint violation\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+).*?"
        r"Complementarity\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+).*?"
        r"Overall NLP error\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)",
        txt,
        re.DOTALL,
    )
    if m:
        print("\n                                   (scaled)                 (unscaled)")
        print(f"Objective...............:  {m.group(1)}   {m.group(2)}")
        print(f"Dual infeasibility......:  {m.group(3)}   {m.group(4)}")
        print(f"Constraint violation....:  {m.group(5)}   {m.group(6)}")
        print(f"Complementarity.........:  {m.group(7)}   {m.group(8)}")
        print(f"Overall NLP error.......:  {m.group(9)}   {m.group(10)}\n")

    # Evaluation counts
    print(
        "Number of objective function evaluations             =",
        grab_int(r"Number of objective function evaluations\s*=\s*(\d+)"),
    )
    print(
        "Number of objective gradient evaluations             =",
        grab_int(r"Number of objective gradient evaluations\s*=\s*(\d+)"),
    )
    print(
        "Number of equality constraint evaluations            =",
        grab_int(r"Number of equality constraint evaluations\s*=\s*(\d+)"),
    )
    print(
        "Number of inequality constraint evaluations          =",
        grab_int(r"Number of inequality constraint evaluations\s*=\s*(\d+)"),
    )
    print(
        "Number of equality constraint Jacobian evaluations   =",
        grab_int(r"Number of equality constraint Jacobian evaluations\s*=\s*(\d+)"),
    )
    print(
        "Number of inequality constraint Jacobian evaluations =",
        grab_int(
            r"Number of inequality constraint Jacobian evaluations\s*=\s*(\d+)"
        ),
    )
    print(
        "Number of Lagrangian Hessian evaluations             =",
        grab_int(r"Number of Lagrangian Hessian evaluations\s*=\s*(\d+)"),
    )

    # CPU times
    print(
        "Total CPU secs in IPOPT (w/o function evaluations)   =",
        grab_float(
            r"Total CPU secs in IPOPT \(w/o function evaluations\)\s*=\s*([0-9.]+)"
        ),
    )
    print(
        "Total CPU secs in NLP function evaluations           =",
        grab_float(r"Total CPU secs in NLP function evaluations\s*=\s*([0-9.]+)"),
    )

    # EXIT line
    m = re.search(r"EXIT:\s*(.*)", txt)
    print("EXIT:", m.group(1).strip() if m else None)

    """ Print out a results summary"""

    # Print out a results summary
    print("Optimal experiment values: ")

    temps = doe_obj.results["Experiment Design"][0:]
    print("\tTemperature values: [" + ", ".join(f"{t:.2f}" for t in temps) + "]")

    print("FIM at optimal design:\n {}".format(np.array(doe_obj.results["FIM"])))
    print(
        "Objective value at optimal design: {:.2f}".format(
            pyo.value(doe_obj.model.objective)
        )
    )

    print(doe_obj.results["Experiment Design Names"])

    print(sorted(doe_obj.results.keys()))

    print("Solve time (s):", doe_obj.results["Solve Time"])
    print("Build time (s):", doe_obj.results["Build Time"])
    print("Initialization time (s):", doe_obj.results["Initialization Time"])
    print("Total wall time (s):", doe_obj.results["Wall-clock Time"])

    output_path = (
        Path(args.out_json)
        if args.out_json
        else run_dir
        / f"pde_diffusion_central_nfe_t{args.nfe_t}_nfe_x{args.nfe_x}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ipopt_stats = parse_ipopt_out(ipopt_out_path)
    result = {
        "method": "central",
        "nfe_x": args.nfe_x,
        "nfe_t": args.nfe_t,
        "doe_solve_time": doe_obj.results.get("Solve Time"),
        "doe_build_time": doe_obj.results.get("Build Time"),
        "doe_init_time": doe_obj.results.get("Initialization Time"),
        "doe_wall_time": doe_obj.results.get("Wall-clock Time"),
        "ipopt_iters": ipopt_stats.get("iterations"),
        "obj_eval": ipopt_stats.get("obj_eval_count"),
        "grad_eval": ipopt_stats.get("grad_eval_count"),
        "eq_con_eval": ipopt_stats.get("eq_con_eval_count"),
        "eq_jac_eval": ipopt_stats.get("eq_con_jac_eval_count"),
        "hess_eval": ipopt_stats.get("hess_eval_count"),
        "ipopt_cpu_wo_feval": ipopt_stats.get("ipopt_cpu_secs_no_eval"),
        "ipopt_cpu_nlp_feval": ipopt_stats.get("nlp_cpu_secs_eval"),
        "ipopt_exit": ipopt_stats.get("exit"),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

###################
# End optimal DoE
