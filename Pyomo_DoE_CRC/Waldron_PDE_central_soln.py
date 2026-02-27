# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import shutil
import sys
import os.path

# if "google.colab" in sys.modules:
#     !wget 'https://raw.githubusercontent.com/IDAES/idaes-pse/main/scripts/colab_helper.py'
#     import colab_helper

#     colab_helper.install_idaes()
#     colab_helper.install_ipopt()

assert shutil.which("ipopt") or os.path.isfile("ipopt")
from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    Objective,
    TransformationFactory,
    SolverFactory,
)
from pyomo.dae import ContinuousSet, DerivativeVar


from itertools import product
import pandas as pd
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
from pyomo.opt import SolverFactory
from pyomo.contrib.parmest import parmest # import parmest
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe import DesignOfExperiments
import idaes.core.solvers.get_solver   # import the idaes solvers


'''
This code utilizes pyomo.dae to solve the PDEs in waldron et al, 2020
The paper considers the design of ramp PFR experiments using MBDOE. The
case study considered is Benzoic acid and Ethanol esterificatrion with
a sulfuric acid homogenous catalyst. The reaction is:
    C6H5COOH (BA) + C2H5OH(Ethoh) <-> C6H5COOC2H5(EB) + H2O(W)
 
    rBA = -k CBA
 
 The rate constant is reparameterized version of the Arrhenius equation
 
 k = exp(-KP1 - (KP2*10000)/R*[1/T - 1/TM])
 
 here KP1 and KP2 are the kinetic parameters to be estimated
 T is the reaction temperature
 TM = 378.15K is the mean temperature (avg of min and max temperatures)
 
 The pre-exponential factor k0 and activation energies are:
     
     k0 = exp(-KP1 + (KP2*10000)/(RTM))
     EA = KP2*10000
     
Reactor dimensions:
    
    L = 2m
    dia = 250 *10^-6 m
    
    
    An online HPLC is used to measure the concentration of Benzoic acid
    at the outlet for measuring the conversion rates of the transient
    experiment, the HPLC is connected to the reactor outlet by a section
    of tubing of volume 44.2 * 10^-6 L. The concentration of Benzoic acid 
    and ethyl benzoate was measured every 7 min, the measurement error
    is found to have a standard deviation of 0.03 M and 0.0165 M for 
    Benzoic acid and Ethyl Benzoate, respectively (based on repeated 
    experiments)
    
    The rate of reaction across the transient PFR is:
        
 
    \partial dCBA/dt = -nu \partial dCBA/dV + rBA
    
 '''
 
 
m = ConcreteModel()

m.r = ContinuousSet(bounds=(0, 1))
m.t = ContinuousSet(bounds=(0, 2))

m.T = Var(m.t, m.r)

m.dTdt = DerivativeVar(m.T, wrt=m.t)
m.dTdr = DerivativeVar(m.T, wrt=m.r)

m.d2Tdr2 = DerivativeVar(m.dTdr, wrt=m.r, initialize=0.0)


## 

# Uncomment the following line if you installed Ipopt via IDAES on your local machine
# but cannot find the Ipopt executable. Our Colab helper script takes care of the
# # environmental variables if you are  on Colab. Otherwise, you need to either import
# idaes or set the path environmental variable yourself.
#
# import idaes

# def model_plot(m):
#     r = sorted(m.r)
#     t = sorted(m.t)

#     rgrid = np.zeros((len(t), len(r)))
#     tgrid = np.zeros((len(t), len(r)))
#     Tgrid = np.zeros((len(t), len(r)))

#     for i in range(0, len(t)):
#         for j in range(0, len(r)):
#             rgrid[i, j] = r[j]
#             tgrid[i, j] = t[i]
#             Tgrid[i, j] = m.T[t[i], r[j]].value

#     fig = plt.figure(figsize=(10, 6))
#     ax = fig.add_subplot(1, 1, 1, projection="3d")
#     ax.set_xlabel("Distance r")
#     ax.set_ylabel("Time t")
#     ax.set_zlabel("Temperature T")
#     p = ax.plot_wireframe(rgrid, tgrid, Tgrid)
#     plt.show()
    
# m = ConcreteModel()

# m.r = ContinuousSet(bounds=(0, 1))
# m.t = ContinuousSet(bounds=(0, 2))

# m.T = Var(m.t, m.r)

# m.dTdt = DerivativeVar(m.T, wrt=m.t)
# m.dTdr = DerivativeVar(m.T, wrt=m.r)

# # This will not work because Pyomo.DAE does not know to use the boundary condition on the derivative
# # m.d2Tdr2 = DerivativeVar(m.T, wrt=(m.r, m.r), bounds=(-100,100))

# # This way Pyomo.DAE knows to use the boundary condition on the derivative at r=0
# m.d2Tdr2 = DerivativeVar(m.dTdr, wrt=m.r, initialize=0.0)


# @m.Constraint(m.t, m.r)
# def pde(m, t, r):
#     if t == 0:
#         return Constraint.Skip
#     if r == 0:
#         return Constraint.Skip
#     return m.dTdt[t, r] == m.d2Tdr2[t, r]


# m.obj = Objective(expr=1)

# # Initial condition
# m.ic = Constraint(m.r, rule=lambda m, r: m.T[0, r] == 0 if r < 1 else Constraint.Skip)

# # Boundary conditions on temperature
# m.bc1 = Constraint(m.t, rule=lambda m, t: m.T[t, 1] == 1)

# # Boundary conditions on temperature gradient
# # End is insulated
# m.bc2 = Constraint(m.t, rule=lambda m, t: m.dTdr[t, 0] == 0)

# # This needs to be backwards in time because:
# # 1. The initial condition is at t=0
# # 2. The insulated boundary condition is at r=0
# TransformationFactory("dae.finite_difference").apply_to(
#     m, nfe=50, scheme="BACKWARD", wrt=m.r
# )
# TransformationFactory("dae.finite_difference").apply_to(
#     m, nfe=50, scheme="BACKWARD", wrt=m.t
# )

# # If we do not use BACKWARDS AND BACKWARDS, we have extra degrees of freedom
# import idaes
# from idaes.core.util.model_diagnostics import degrees_of_freedom

# # print("Degrees of Freedom:", degrees_of_freedom(m))

# SolverFactory("ipopt").solve(m, tee=True).write()
# model_plot(m)