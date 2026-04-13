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
## Defining model


def build_1D_2param_model(nfe_x = 10, nfe_t = 20, kappa = 0.025, T = 1.0):
    m = pyo.ConcreteModel()
    
    # Discretize space grid
    
    m.x = ContinuousSet(bounds = (0,1))
    m.t = ContinuousSet(bounds =(0,1))

    m.u = pyo.Var(m.t,m.x)

    m.dudt = DerivativeVar(m.u, wrt = m.t)

    m.dudx = DerivativeVar(m.u, wrt = m.x)

    m.d2udx2 = DerivativeVar(m.dudx, wrt = m.x)

    m.theta = pyo.Var([1,2], within = pyo.Reals)
    
    m.kappa = pyo.Param(initialize = 0.025)
    
    m.v = pyo.Param(initialize = 0.1)
    
    @m.Constraint(m.t,m.x)

    def pde(m,t,x):
        if x == m.x.first() or x == m.x.last() :
            return pyo.Constraint.Skip
        return m.dudt[t,x] == m.kappa*m.d2udx2[t,x] - m.v*m.dudx[t,x]
    
    ## Adding boundary conditions
    
    @m.Constraint(m.t)
    
    def bc_left(m,t):
        if t== m.t.first():
            return pyo.Constraint.Skip
        return m.dudx[t, m.x.first()] == 0.0
    
    @m.Constraint(m.t)
    
    def bc_right(m,t):
        if t == m.t.first():
            return pyo.Constraint.Skip
        return m.dudx[t, m.x.last()] == 0.0
    
    @m.Constraint(m.x)
    
    def ic(m,x):
        if x<0.5:
            return m.u[m.t.first(),x] == m.theta[1]
        return m.u[m.t.first(),x] == m.theta[2]
    
    ## Additional constraints to make the model square
    
    @m.Constraint(m.t)
    
    def bc_left_dummy(m,t):
        if t == m.t.first():
            return pyo.Constraint.Skip
        return m.d2udx2[t, m.x.first()] == 0.0
        
    @m.Constraint(m.t)
    
    def bc_right_dummy(m,t):
        if t == m.t.first():
            return pyo.Constraint.Skip
        return m.d2udx2[t, m.x.last()] == 0.0
    
    
    @m.Constraint()
    def bc_left_initial(m):
        return m.dudx[m.t.first(), m.x.first()] == 0.0

    
    # @m.Constraint(m.x)
    
    # def ic_dummy(m,x):
    #     return m.dudt[m.t.first(),x] == 0.0
    
    disc = pyo.TransformationFactory("dae.finite_difference")
    disc.apply_to(m, wrt = m.x, nfe = nfe_x, scheme = "BACKWARD")
    disc.apply_to(m, wrt = m.t, nfe = nfe_t, scheme = "BACKWARD")
    m.obj = pyo.Objective(expr = 0.0)
    
    return m

m = build_1D_2param_model()

m.theta[1].fix(1.0)
m.theta[2].fix(0.2)

solver = pyo.SolverFactory("ipopt")
results = solver.solve(m, tee=True)

# import idaes 
# from idaes.core.util.model_diagnostics import degrees_of_freedom
# print(degrees_of_freedom(m))

# from idaes.core.util.model_diagnostics import DiagnosticsToolbox

# dt = DiagnosticsToolbox(m)
# dt.report_structural_issues()
# dt.display_underconstrained_set()
# # dt.display_unused_variables()
