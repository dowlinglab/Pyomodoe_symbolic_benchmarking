import os

level = os.environ.get("BENCH_DERIV_CHECK", "none").strip().lower()
solver_name_env = os.environ.get("BENCH_SOLVER")
force_mumps = os.environ.get("BENCH_FORCE_MUMPS", "0") == "1"
tol = os.environ.get("BENCH_DERIV_TOL", "1e-4")

try:
    import pyomo.environ as pyo
    from pyomo.contrib.doe import DesignOfExperiments
except Exception:
    pyo = None
    DesignOfExperiments = None


def _wrap_solver(solver):
    try:
        options = solver.options
    except Exception:
        return solver

    if force_mumps and options.get("linear_solver") == "ma57":
        options["linear_solver"] = "mumps"

    if level and level != "none":
        options["derivative_test"] = "second-order" if level == "second-order" else "first-order"
        options["derivative_test_print_all"] = "yes"
        options["derivative_test_tol"] = float(tol)
        options["print_level"] = 5

    try:
        orig_solve = solver.solve

        def solve(*args, **kwargs):
            if force_mumps and options.get("linear_solver") == "ma57":
                options["linear_solver"] = "mumps"
            if level and level != "none":
                kwargs["tee"] = True
            return orig_solve(*args, **kwargs)

        solver.solve = solve
    except Exception:
        pass

    return solver


if pyo is not None:
    _orig_sf = pyo.SolverFactory

    def SolverFactory(name, *args, **kwargs):
        solver = _orig_sf(name, *args, **kwargs)
        try:
            solver_name = getattr(solver, "name", str(name)).lower()
        except Exception:
            solver_name = str(name).lower()
        if "ipopt" in solver_name:
            solver = _wrap_solver(solver)
        return solver

    pyo.SolverFactory = SolverFactory

if DesignOfExperiments is not None:
    try:
        import inspect

        sig = inspect.signature(DesignOfExperiments.__init__)
        if "gradient_method" not in sig.parameters or "use_grey_box_objective" not in sig.parameters:
            _orig_init = DesignOfExperiments.__init__

            def _init(self, *args, **kwargs):
                kwargs.pop("gradient_method", None)
                kwargs.pop("use_grey_box_objective", None)
                return _orig_init(self, *args, **kwargs)

            DesignOfExperiments.__init__ = _init
    except Exception:
        pass

try:
    import pyomo.opt

    _orig_sf_opt = pyomo.opt.SolverFactory

    def SolverFactory_opt(name, *args, **kwargs):
        solver = _orig_sf_opt(name, *args, **kwargs)
        try:
            solver_name = getattr(solver, "name", str(name)).lower()
        except Exception:
            solver_name = str(name).lower()
        if "ipopt" in solver_name:
            solver = _wrap_solver(solver)
        return solver

    pyomo.opt.SolverFactory = SolverFactory_opt
except Exception:
    pass
