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
are chosen. First 1D version is solved



 """       
import os
import argparse
import json
import time
from pathlib import Path
"Modifications to avoid the IPOPT Error on CRC"

# import shutil
import pyomo.environ as pyo

# IPOPT_BIN = shutil.which("ipopt")
# IPOPT_LINEAR_SOLVER_DEFAULT = "ma57"

