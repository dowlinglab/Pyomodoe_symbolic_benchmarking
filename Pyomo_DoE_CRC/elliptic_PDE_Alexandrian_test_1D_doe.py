#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:30:29 2026

@author: snarasi2
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
x-grid: 10 pts
t-grid: 20 pts

Sensor locations: x = 0.25 and x = 0.75
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
        
        m.w1 = pyo.Var(bounds = (0,1), initialize = 1.0)
        
        m.w2 = pyo.Var(bounds = (0,1), initialize = 1.0)
        
        m.w3 = pyo.Var(bounds = (0,1), initialize = 1.0)
        
        m.w4 = pyo.Var(bounds = (0,1), initialize = 1.0)
        
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
        
        m.y1 = pyo.Expression(expr = m.w1*m.u[0.2,0.25])
        m.y2 = pyo.Expression(expr = m.w2*m.u[0.2,0.75])
        m.y3 = pyo.Expression(expr = m.w3*m.u[0.5,0.25])
        m.y4 = pyo.Expression(expr = m.w4*m.u[0.5,0.75])
        
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update((k, 0.0) for k in [m.y1, m.y2, m.y3, m.y4])
        
        m.measurement_error = pyo.Suffix(direction = pyo.Suffix.LOCAL)
        m.measurement_error.update((k, 1e-2) for k in [m.y1, m.y2, m.y3, m.y4])
        
        m.unknown_parameters = pyo.Suffix(direction = pyo.Suffix.LOCAL)
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.theta1, m.theta2])
        
        m.experiment_inputs = pyo.Suffix(direction= pyo.Suffix.LOCAL)
        m.experiment_inputs.update((k, None) for k in [m.w1, m.w2, m.w3, m.w4])
        
        
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


fd_formula = "central"
step_size = 1e-3
objective_option = "determinant"
scale_nominal_param_value = True

from pyomo.contrib.doe import DesignOfExperiments

doe_obj_det = DesignOfExperiments(
    experiment,
    fd_formula = fd_formula,
    step= step_size,
    objective_option = objective_option,
    scale_constant_value = 1,
    scale_nominal_param_value = scale_nominal_param_value,
    prior_FIM = None,
    jac_initial = None,
    fim_initial = None,
    L_diagonal_lower_bound = 1e-7,
    solver = pyo.SolverFactory("ipopt"),
    tee = False,
    get_labeled_model_args= None,
    _Cholesky_option = True,
    _only_compute_fim_lower = True,
    )


  
doe_obj_det.run_doe()


doe_obj_A = DesignOfExperiments(
    experiment,
    fd_formula = fd_formula,
    step= step_size,
    objective_option = "trace",
    scale_constant_value = 1,
    scale_nominal_param_value = scale_nominal_param_value,
    prior_FIM = None,
    jac_initial = None,
    fim_initial = None,
    L_diagonal_lower_bound = 1e-7,
    solver = pyo.SolverFactory("ipopt"),
    tee = False,
    get_labeled_model_args= None,
    _Cholesky_option = True,
    _only_compute_fim_lower = True,
    )


  
doe_obj_A.run_doe()


