#!/usr/bin/env python
# coding: utf-8

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
# 

# In[ ]:


"Modifications to avoid the IPOPT Error on CRC"

import shutil
import pyomo.environ as pyo

IPOPT_BIN = shutil.which("ipopt")

def make_ipopt():
    set = pyo.SolverFactory("ipopt", executable = IPOPT_BIN)
    set.options["linear_solver"] = "ma57"
    return set


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D ## Needed to create 3D plots

import shutil ## Shell utilities for python, allows to copy/move/rename files or delete directories
import sys ## interact with python interpreter itself
import os.path ## path manipulation utilities


# from pyomo.environ import *
# from pyomo.dae import *
from idaes.core.util import DiagnosticsToolbox

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
import numpy as np
from pyomo.contrib.parmest.experiment import Experiment


## Defining experiment class

class PDE_diffusion(Experiment):
    def __init__(self, data, nfe_t, nfe_x ,ncp_x, ncp_t):
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

        m.x = ContinuousSet(bounds = (0,1))
        m.t = ContinuousSet(bounds =(0,2))

        m.u = pyo.Var(m.t, within=pyo.Reals, initialize = 0.8, bounds = (0,1))

        m.T = Var(m.t,m.x)

        m.dTdt = DerivativeVar(m.T, wrt = m.t)

        m.dTdx = DerivativeVar(m.T, wrt = m.x)

        m.d2Tdx2 = DerivativeVar(m.dTdx, wrt = m.x)

        m.alpha = Var(within = pyo.Reals)

        @m.Constraint(m.t,m.x)

        def pde(m,t,x):
            if x == 0 or t == 0:
                return Constraint.Skip
            else: return m.dTdt[t,x] == m.alpha*m.d2Tdx2[t,x]

    def finalize_model(self):
        m = self.model

        m.alpha.fix(self.data["alpha"])

        m.u.fix()

        m.ic = Constraint(m.x,rule = lambda m, x: m.T[0,x] ==0 if x<m.x.last() else Constraint.Skip)

        ## Boundary conditions

        m.bc1 = Constraint(m.t, rule = lambda m, t: m.T[t,m.x.last()] == 1)

        m.bc2 = Constraint(m.t, rule = lambda m, t: m.dTdx[t,m.x.first()] == m.u[t])
        
        TransformationFactory("dae.finite_difference").apply_to(m,nfe = 20, wrt = m.x)
        TransformationFactory("dae.finite_difference").apply_to(m,nfe = 50, wrt = m.t)

        # TransformationFactory("dae.collocation").apply_to(m,nfe = 20,ncp = 3, wrt = m.x)
        # TransformationFactory("dae.collocation").apply_to(m,nfe = 40,ncp = 3, wrt = m.t)

    def label_experiment(self):
        m = self.model

        # control_points = self.data["control_points"]


        # Set measurement labels
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        
        m.experiment_outputs.update((m.T[t,x], None) for t in m.t for x in m.x)


        # Adding error for measurement values (assuming no covariance and constant error for all measurements)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        meas_error = 1e-2  # Error in state measurement
      
        m.measurement_error.update((m.T[t,x], meas_error) for t in m.t for x in m.x)
   

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
    0.0:   0.5,
    0.125: 0.5,
    0.25:  0.5,
    0.375: 0.5,
    0.5:   0,
    0.625: 0.5,
    0.75:  0.2,
    0.875: 0.5,
    1.0:   0.5,
}



data_ex = {"alpha": 0.8, "control_points": flat_u}

# Create an Experiment object; data and discretization information are part
# of the constructor of this object
experiment = PDE_diffusion(data=data_ex, nfe_t=2, nfe_x = 2, ncp_x = 2, ncp_t=2)

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







# "Calling the Doe object 1000 times, saving to an excel file"
## Importing required packages

from pathlib import Path
from openpyxl import Workbook, load_workbook

# NOTEBOOK_ID = "ScalarPDE" # Defining the notebook ID for excel sheet tagging
# SCENARIO = "Central" # Scenario implies the environment


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
scale_constant_value=1,
scale_nominal_param_value=scale_nominal_param_value,
prior_FIM=None,
jac_initial=None,
fim_initial=None,
L_diagonal_lower_bound=1e-7,
solver= make_ipopt(),#SolverFactory('IPOPT'), #, options={'linear_solver': 'mumps'}
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

# Print out a results summary
print("Optimal experiment values: ")
print(
    "\tInitial concentration: {:.2f}".format(
        doe_obj.results["Experiment Design"][0]
    )
)
print(
    ("\tTemperature values: [" + "{:.2f}, " * 8 + "{:.2f}]").format(
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

print(sorted(doe_obj.results.keys()))

print("Solve time (s):", doe_obj.results["Solve Time"])
print("Build time (s):", doe_obj.results["Build Time"])
print("Initialization time (s):", doe_obj.results["Initialization Time"])
print("Total wall time (s):", doe_obj.results["Wall-clock Time"])

###################
# End optimal DoE

