#!/usr/bin/env python
# coding: utf-8

# In[1]:


"Modifications to avoid the IPOPT Error on CRC"

import shutil
import pyomo.environ as pyo

IPOPT_BIN = shutil.which("ipopt")

def make_ipopt():
    set = pyo.SolverFactory("ipopt", executable = IPOPT_BIN)
    set.options["linear_solver"] = "ma57"
    return set


# In[2]:


#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
# === Required imports ===
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator

from pyomo.contrib.parmest.experiment import Experiment


# ========================
class ReactorExperiment(Experiment):
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
        """
        This is an example user model provided to DoE library.
        It is a dynamic problem solved by Pyomo.DAE.

        Return
        ------
        m: a Pyomo.DAE model
        """

        m = self.model = pyo.ConcreteModel()

        # Model parameters
        m.R = pyo.Param(mutable=False, initialize=8.314)

        # Define model variables
        ########################
        # time
        m.t = ContinuousSet(bounds=[0, 1])

        # Concentrations
        m.CA = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.CB = pyo.Var(m.t, within=pyo.NonNegativeReals)
        m.CC = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Temperature
        m.T = pyo.Var(m.t, within=pyo.NonNegativeReals)

        # Arrhenius rate law equations
        m.A1 = pyo.Var(within=pyo.NonNegativeReals)
        m.E1 = pyo.Var(within=pyo.NonNegativeReals)
        m.A2 = pyo.Var(within=pyo.NonNegativeReals)
        m.E2 = pyo.Var(within=pyo.NonNegativeReals)

        # Differential variables (Conc.)
        m.dCAdt = DerivativeVar(m.CA, wrt=m.t)
        m.dCBdt = DerivativeVar(m.CB, wrt=m.t)

        ########################
        # End variable def.

        # Equation definition
        ########################

        # Expression for rate constants
        @m.Expression(m.t)
        def k1(m, t):
            return m.A1 * pyo.exp(-m.E1 * 1000 / (m.R * m.T[t]))

        @m.Expression(m.t)
        def k2(m, t):
            return m.A2 * pyo.exp(-m.E2 * 1000 / (m.R * m.T[t]))

        # Concentration odes
        @m.Constraint(m.t)
        def CA_rxn_ode(m, t):
            return m.dCAdt[t] == -m.k1[t] * m.CA[t]

        @m.Constraint(m.t)
        def CB_rxn_ode(m, t):
            return m.dCBdt[t] == m.k1[t] * m.CA[t] - m.k2[t] * m.CB[t]

        # algebraic balance for concentration of C
        # Valid because the reaction system (A --> B --> C) is equimolar
        @m.Constraint(m.t)
        def CC_balance(m, t):
            return m.CA[0] == m.CA[t] + m.CB[t] + m.CC[t]

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

        # Set initial concentration values for the experiment
        m.CA[0].value = self.data["CA0"]
        m.CB[0].fix(self.data["CB0"])

        # Update model time `t` with time range and control time points
        m.t.update(self.data["t_range"])
        m.t.update(control_points)

        # Fix the unknown parameter values
        m.A1.fix(self.data["A1"])
        m.A2.fix(self.data["A2"])
        m.E1.fix(self.data["E1"])
        m.E2.fix(self.data["E2"])

        # Add upper and lower bounds to the design variable, CA[0]
        m.CA[0].setlb(self.data["CA_bounds"][0])
        m.CA[0].setub(self.data["CA_bounds"][1])

        m.t_control = control_points

        # Discretizing the model
        discr = pyo.TransformationFactory("dae.collocation")
        discr.apply_to(m, nfe=self.nfe, ncp=self.ncp, wrt=m.t)

        # Initializing Temperature in the model
        cv = None
        for t in m.t:
            if t in control_points:
                cv = control_points[t]
                m.T[t].fix()
            m.T[t].setlb(self.data["T_bounds"][0])
            m.T[t].setub(self.data["T_bounds"][1])
            m.T[t] = cv

        # Make a constraint that holds temperature constant between control time points
        @m.Constraint(m.t - control_points)
        def T_control(m, t):
            """
            Piecewise constant temperature between control points
            """
            neighbour_t = max(tc for tc in control_points if tc < t)
            return m.T[t] == m.T[neighbour_t]

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
        m.experiment_outputs.update((m.CA[t], None) for t in m.t_control)
        # Add CB to experiment outputs
        m.experiment_outputs.update((m.CB[t], None) for t in m.t_control)
        # Add CC to experiment outputs
        m.experiment_outputs.update((m.CC[t], None) for t in m.t_control)

        # Adding error for measurement values (assuming no covariance and constant error for all measurements)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        concentration_error = 1e-2  # Error in concentration measurement
        # Add measurement error for CA
        m.measurement_error.update((m.CA[t], concentration_error) for t in m.t_control)
        # Add measurement error for CB
        m.measurement_error.update((m.CB[t], concentration_error) for t in m.t_control)
        # Add measurement error for CC
        m.measurement_error.update((m.CC[t], concentration_error) for t in m.t_control)

        # Identify design variables (experiment inputs) for the model
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add experimental input label for initial concentration
        m.experiment_inputs[m.CA[m.t.first()]] = None
        # Add experimental input label for Temperature
        m.experiment_inputs.update((m.T[t], None) for t in m.t_control)

        # Add unknown parameter labels
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add labels to all unknown parameters with nominal value as the value
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.A1, m.A2, m.E1, m.E2])

        #########################
        # End model labeling


# In[3]:


#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
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



data_ex = {"CA0": 5.0, "CA_bounds": [1.0, 5.0], "CB0": 0.0, "CC0": 0.0, "t_range": [0, 1],
           "control_points": {"0": 500, "0.125": 300, "0.25": 300, "0.375": 300, "0.5": 300, "0.625": 300, "0.75": 300,
                              "0.875": 300, "1": 300}, "T_bounds": [300, 700], "A1": 84.79, "A2": 371.72, "E1": 7.78, "E2": 15.05}
# Put temperature control time points into correct format for reactor experiment
data_ex["control_points"] = {
    float(k): v for k, v in data_ex["control_points"].items()
}

# Create a ReactorExperiment object; data and discretization information are part
# of the constructor of this object
experiment = ReactorExperiment(data=data_ex, nfe=10, ncp=3)

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

NOTEBOOK_ID = "Reactor" # Defining the notebook ID for excel sheet tagging
SCENARIO = "Central" # Scenario implies the environment


"Creating a path object and creating an excel file to save the output of the runs"
RESULTS_DIR = Path("results") ## Foldername
RESULTS_DIR.mkdir(exist_ok=True)
XLSX_PATH = Path("results")/ f"{NOTEBOOK_ID}_{SCENARIO}.xlsx" ## Create an excel sheet with the 
# notebook ID and scenario

SHEET_NAME = "Data" ## Name of the sheet within the excel file


## Creating the excel file


wb = Workbook()
ws = wb.active
ws.title = SHEET_NAME

design_names = None # specifies the column names, will be filled at the first run



N_Runs = 1000 ## Number of runs in this code

for run in range(1, N_Runs+1):
    print(f"Run {run}/{N_Runs}")
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
    solver= make_ipopt(),#SolverFactory('IPOPT'),
    tee=True,
    get_labeled_model_args=None,
    _Cholesky_option=True,
    _only_compute_fim_lower=True,
    )
    doe_obj.run_doe()

    if run == 1: ## Do this only on run 1
        ## Header names
        HEADERS = [
            "run",
            "solve_time",
            "build_time",
            "init_time",
            "wall_time",
            "objective_value"
        ] + list(doe_obj.results["Experiment Design Names"])
        
        ws.append(HEADERS)

    row = [
        run,
        doe_obj.results["Solve Time"],
        doe_obj.results["Build Time"],
        doe_obj.results["Initialization Time"],
        doe_obj.results["Wall-clock Time"],
        pyo.value(doe_obj.model.objective)
    ] + list(doe_obj.results["Experiment Design"])

    ws.append(row)
    
wb.save(XLSX_PATH)

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


