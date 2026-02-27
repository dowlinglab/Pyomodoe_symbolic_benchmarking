import argparse
import os
import runpy
import numpy as np
import pyomo.environ as pyo
import math

parser = argparse.ArgumentParser()
parser.add_argument("--script", required=True)
parser.add_argument("--objective", required=True)
args = parser.parse_args()

os.environ["BENCH_OBJECTIVE_OPTION"] = args.objective

ctx = runpy.run_path(args.script)

doe_obj = ctx.get("doe_obj")
if doe_obj is None:
    raise SystemExit("doe_obj not found in script globals")

model = doe_obj.model

print("=== objective expr value ===")
try:
    print(pyo.value(model.objective.expr))
except Exception as e:
    print("error", e)

print("=== greybox outputs ===")
if hasattr(model, "obj_cons") and hasattr(model.obj_cons, "egb_fim_block"):
    egb = model.obj_cons.egb_fim_block
    for k in ["log-D-opt", "A-opt", "E-opt", "ME-opt"]:
        if k in egb.outputs:
            try:
                print(k, pyo.value(egb.outputs[k]))
            except Exception as e:
                print(k, "error", e)
else:
    print("no greybox outputs")

print("=== numpy reference from FIM ===")
FIM = np.array(doe_obj.results["FIM"], dtype=float)
print("FIM shape", FIM.shape)

# determinant
try:
    det = np.linalg.det(FIM)
    print("det(FIM)", det)
    if det > 0:
        print("log10(det)", math.log10(det))
        print("ln(det)", math.log(det))
    else:
        print("det <= 0")
except Exception as e:
    print("det error", e)

# trace
try:
    tr = np.trace(FIM)
    print("trace(FIM)", tr)
    if tr > 0:
        print("log10(trace)", math.log10(tr))
        print("ln(trace)", math.log(tr))
except Exception as e:
    print("trace error", e)

# trace of inverse
try:
    inv = np.linalg.inv(FIM)
    trinv = np.trace(inv)
    print("trace(inv(FIM))", trinv)
    if trinv > 0:
        print("log10(trace(inv))", math.log10(trinv))
        print("ln(trace(inv))", math.log(trinv))
except Exception as e:
    print("inv error", e)
