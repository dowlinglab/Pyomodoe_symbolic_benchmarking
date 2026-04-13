#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
if os.path.exists("ipopt.out"):
    os.remove("ipopt.out")


"Modifications to avoid the IPOPT Error on CRC"

import shutil
import pyomo.environ as pyo

IPOPT_BIN = shutil.which("ipopt")

def make_ipopt():
    set = pyo.SolverFactory("ipopt", executable = IPOPT_BIN)
    set.options["linear_solver"] = "ma57"
    # # ---- IPOPT logging ----
    set.options["output_file"] = "ipopt.out"
    set.options["file_print_level"] = 12
    return set


# In[2]:


import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
import numpy as np
from pyomo.contrib.parmest.experiment import Experiment
## This code simulates xdot = -ax + bsin(wt) + u , a,b are the unknown parameter that are to be estimated. 
##The parameter w is known and set to 1.0


# ========================
class TwoParameterExperiment(Experiment):
    def __init__(self, data, nfe, ncp):
        """
        Arguments
        ---------
        data: object containing vital experimental information
        nfe: number of finite elements
        ncp: number of collocation points for the finite elements
        """
        self.data = data
        self.nfe = nfe
        self.ncp = ncp
        self.model = None

        #############################
        # End constructor definition

    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_experiment()
        return self.model

    # Create flexible model without data
    def create_model(self):
     

        m = self.model = pyo.ConcreteModel()

        # Model parameters
        m.w = pyo.Param(mutable=False, initialize=1.0)

        # Define model variables
        ########################
        # time
        m.t = ContinuousSet(bounds=[0, 1])

        # State and input
        m.x = pyo.Var(m.t, within=pyo.Reals)
        m.u = pyo.Var(m.t, within=pyo.Reals)

        # Unknown parameter
        m.a = pyo.Var(within=pyo.Reals)
        m.b = pyo.Var(within=pyo.Reals)

        # Differential variables wrt x
        m.dxdt = DerivativeVar(m.x, wrt=m.t)

        ########################
        # End variable def.

        # Equation definition
        ########################

        # Expression for the state evolution

        # State odes
        @m.Constraint(m.t)
        def x_ode(m, t):
            return m.dxdt[t] == m.a * m.x[t] + m.b * pyo.sin(m.w * m.x[t]) + m.u[t]


        ########################
        # End equation definition

    def finalize_model(self):
        """
        Example finalize model function. There are two main tasks
        here:

            1. Extracting useful information for the model to align
               with the experiment. (Here: CA0, t_final, t_control)
            2. Discretizing the model subject to this information.

        """
        m = self.model

        # Unpacking data before simulation
        control_points = self.data["control_points"]

        # Set initial state values for the experiment
        m.x[0].value = self.data["x0"]

        # Update model time `t` with time range and control time points
        m.t.update(self.data["t_range"])
        m.t.update(control_points)

        # Fix the unknown parameter values
        m.a.fix(self.data["a"])
        m.b.fix(self.data["b"])

        # Add upper and lower bounds to the design variable, CA[0]
        m.x[0].setlb(self.data["x_bounds"][0])
        m.x[0].setub(self.data["x_bounds"][1])

        m.t_control = control_points

        # Discretizing the model
        discr = pyo.TransformationFactory("dae.collocation")
        discr.apply_to(m, nfe=self.nfe, ncp=self.ncp, wrt=m.t)

        # Initializing control input in the model
        cv = None
        for t in m.t:
            if t in control_points:
                cv = control_points[t]
                m.u[t].fix()
            m.u[t].setlb(self.data["u_bounds"][0])
            m.u[t].setub(self.data["u_bounds"][1])
            m.u[t] = cv

        # Make a constraint that holds control input constant between control time points
        @m.Constraint(m.t - control_points)
        def u_control(m, t):
            """
            Piecewise constant control input between control points
            """
            neighbour_t = max(tc for tc in control_points if tc < t)
            return m.u[t] == m.u[neighbour_t]

        #########################
        # End model finalization

    def label_experiment(self):
        """
        Example for annotating (labeling) the model with a
        full experiment.
        """
        m = self.model

        # Set measurement labels
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add CA to experiment outputs
        m.experiment_outputs.update((m.x[t], None) for t in m.t_control)


        # Adding error for measurement values (assuming no covariance and constant error for all measurements)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        meas_error = 1e-2  # Error in state measurement
        # Add measurement error for CA
        m.measurement_error.update((m.x[t], meas_error) for t in m.t_control)
   

        # Identify design variables (experiment inputs) for the model
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add experimental input label for initial state
        m.experiment_inputs[m.x[m.t.first()]] = None
        # Add experimental input label for control input
        m.experiment_inputs.update((m.u[t], None) for t in m.t_control)

        # Add unknown parameter labels
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add labels to all unknown parameters with nominal value as the value
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.a])

        #########################
        # End model labeling


# In[3]:


from pyomo.common.dependencies import numpy as np

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



data_ex = {"x0": 0.5, "x_bounds": [0.5, 1.5], "t_range": [0, 1],
           "control_points": {"0": -1, "0.125": 1, "0.25": 1, "0.375": 1, "0.5": 1, "0.625": 1, "0.75": 1,
                              "0.875": 1, "1": 1}, "u_bounds": [-1.0, 1.0], "a": -0.1, "b":1.2}
# Put control input control time points into correct format for two parameter experiment
data_ex["control_points"] = {
    float(k): v for k, v in data_ex["control_points"].items()
}

# Create a two parameterExperiment object; data and discretization information are part
# of the constructor of this object
experiment = TwoParameterExperiment(data=data_ex, nfe=10, ncp=3)

# Use a central difference, with step size 1e-3
fd_formula = "central"
step_size = 1e-3

# Use the determinant objective with scaled sensitivity matrix
objective_option = "determinant"
scale_nominal_param_value = True

# Create the DesignOfExperiments object
# We will not be passing any prior information in this example.
# We also will rely on the initialization routine within
# the DesignOfExperiments class.


"Calling the Doe object 1000 times, saving to an excel file"
## Importing required packages

from pathlib import Path
from openpyxl import Workbook, load_workbook

# NOTEBOOK_ID = "Sine" # Defining the notebook ID for excel sheet tagging
# SCENARIO = "Symbolic" # Scenario implies the environment

# "Creating a path object and creating an excel file to save the output of the runs"
# RESULTS_DIR = Path("results") ## Foldername
# RESULTS_DIR.mkdir(exist_ok=True)
# XLSX_PATH = Path("results")/ f"{NOTEBOOK_ID}_{SCENARIO}.xlsx" ## Create an excel sheet with the 
# # notebook ID and scenario

# SHEET_NAME = "Data" ## Name of the sheet within the excel file


## Creating the excel file


# wb = Workbook()
# ws = wb.active
# ws.title = SHEET_NAME

# design_names = None # specifies the column names, will be filled at the first run


doe_obj = DesignOfExperiments(
experiment,
fd_formula=fd_formula,
step=step_size,
objective_option=objective_option,
gradient_method = "pynumero",
scale_constant_value=1,
scale_nominal_param_value=scale_nominal_param_value,
prior_FIM=None,
jac_initial=None,
fim_initial=None,
L_diagonal_lower_bound=1e-7,
solver= make_ipopt(),#SolverFactory('IPOPT'),
tee=False,
get_labeled_model_args=None,
_Cholesky_option=True,
_only_compute_fim_lower=True,
)
doe_obj.run_doe()

#     if run == 1: ## Do this only on run 1
#         ## Header names
#         HEADERS = [
#             "run",
#             "solve_time",
#             "build_time",
#             "init_time",
#             "wall_time",
#             "objective_value"
#         ] + list(doe_obj.results["Experiment Design Names"])
        
#         ws.append(HEADERS)

#     row = [
#         run,
#         doe_obj.results["Solve Time"],
#         doe_obj.results["Build Time"],
#         doe_obj.results["Initialization Time"],
#         doe_obj.results["Wall-clock Time"],
#         pyo.value(doe_obj.model.objective)
#     ] + list(doe_obj.results["Experiment Design"])

#     ws.append(row)
    
# wb.save(XLSX_PATH)


''' Print out IPOPT log- written with the help of chatGPT 5.2'''

import re ## Module that helps search for the string of interest

if os.path.exists("ipopt.out"):
    with open("ipopt.out", "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
else:
    print("WARNING: ipopt.out not found; skipping IPOPT log parse details.")
    txt = ""

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
print("Number of objective function evaluations             =", grab_int(r"Number of objective function evaluations\s*=\s*(\d+)"))
print("Number of objective gradient evaluations             =", grab_int(r"Number of objective gradient evaluations\s*=\s*(\d+)"))
print("Number of equality constraint evaluations            =", grab_int(r"Number of equality constraint evaluations\s*=\s*(\d+)"))
print("Number of inequality constraint evaluations          =", grab_int(r"Number of inequality constraint evaluations\s*=\s*(\d+)"))
print("Number of equality constraint Jacobian evaluations   =", grab_int(r"Number of equality constraint Jacobian evaluations\s*=\s*(\d+)"))
print("Number of inequality constraint Jacobian evaluations =", grab_int(r"Number of inequality constraint Jacobian evaluations\s*=\s*(\d+)"))
print("Number of Lagrangian Hessian evaluations             =", grab_int(r"Number of Lagrangian Hessian evaluations\s*=\s*(\d+)"))

# CPU times
print("Total CPU secs in IPOPT (w/o function evaluations)   =", grab_float(r"Total CPU secs in IPOPT \(w/o function evaluations\)\s*=\s*([0-9.]+)"))
print("Total CPU secs in NLP function evaluations           =", grab_float(r"Total CPU secs in NLP function evaluations\s*=\s*([0-9.]+)"))

# EXIT line
m = re.search(r"EXIT:\s*(.*)", txt)
print("EXIT:", m.group(1).strip() if m else None)


''' Print out a results summary'''



# Print out a results summary
print("Optimal experiment values: ")
print(
    "\tInitial state: {:.2f}".format(
        doe_obj.results["Experiment Design"][0]
    )
)
print(
    ("\t Control input values: [" + "{:.2f}, " * 8 + "{:.2f}]").format(
        *doe_obj.results["Experiment Design"][1:]
    )
)
print("FIM at optimal design:\n {}".format(np.array(doe_obj.results["FIM"])))
print(
    "Objective value at optimal design: {:.2f}".format(
        pyo.value(doe_obj.model.objective)
    )
)

print(doe_obj.results["Experiment Design Names"])



print("Solve time (s):", doe_obj.results["Solve Time"])
print("Build time (s):", doe_obj.results["Build Time"])
print("Initialization time (s):", doe_obj.results["Initialization Time"])
print("Total wall time (s):", doe_obj.results["Wall-clock Time"])

###################
# End optimal DoE


