import argparse
import json
import os
import platform
import re
import ast
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from bench.discovery import resolve_problem_file
from bench.ipopt_parse import parse_ipopt_out


def _git_commit(path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def _find_git_root(path: Path) -> Optional[Path]:
    cur = path.resolve()
    for _ in range(6):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _pyomo_git_commit() -> Optional[str]:
    try:
        import pyomo

        pyomo_path = Path(pyomo.__file__).resolve()
        root = _find_git_root(pyomo_path)
        if root:
            return _git_commit(root)
    except Exception:
        return None
    return None


def _package_versions() -> Dict[str, Optional[str]]:
    try:
        from importlib import metadata
    except Exception:
        return {}

    pkgs = [
        "pyomo",
        "numpy",
        "scipy",
        "pandas",
        "openpyxl",
        "cyipopt",
    ]
    out: Dict[str, Optional[str]] = {}
    for name in pkgs:
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = None
    return out


def _pip_freeze() -> Optional[str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def _parse_objective_from_log(log_path: Path) -> Optional[float]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    m = re.search(r"Objective value at optimal design:\s*([-+eE0-9.]+)", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _parse_fim_from_log(log_path: Path) -> Optional["np.ndarray"]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    m = re.search(r"FIM at optimal design:\s*(\[\[.*?\]\])", text, re.DOTALL)
    if not m:
        return None
    block = m.group(1)
    rows = []
    current = []
    in_row = False
    for line in block.splitlines():
        if "[" in line:
            in_row = True
        if in_row:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if nums:
                current.extend(float(x) for x in nums)
        if "]" in line and in_row:
            if current:
                rows.append(current)
            current = []
            in_row = False
    if not rows:
        return None
    try:
        import numpy as np
        return np.array(rows, dtype=float)
    except Exception:
        return None


def _compute_report_metrics(fim):
    if fim is None:
        return {
            "D_opt_report": None,
            "A_opt_report": None,
            "E_opt_report": None,
            "ME_opt_report": None,
        }
    try:
        import numpy as np
        import math

        det = float(np.linalg.det(fim))
        trace = float(np.trace(fim))
        eigvals = np.linalg.eigvals(fim)
        eigvals = np.real(eigvals)
        min_eig = float(np.min(eigvals))
        max_eig = float(np.max(eigvals))
        d_opt = math.log10(det) if det > 0 else None
        a_opt = math.log10(trace) if trace > 0 else None
        me_opt = (max_eig / min_eig) if min_eig != 0 else None
        return {
            "D_opt_report": d_opt,
            "A_opt_report": a_opt,
            "E_opt_report": min_eig,
            "ME_opt_report": me_opt,
        }
    except Exception:
        return {
            "D_opt_report": None,
            "A_opt_report": None,
            "E_opt_report": None,
            "ME_opt_report": None,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--objective-option", default=None)
    parser.add_argument(
        "--deriv-check",
        default="none",
        choices=["none", "first-order", "second-order"],
    )
    args = parser.parse_args()

    root = Path.cwd()
    script_path = resolve_problem_file(root, args.problem, args.mode)

    run_id = args.run_id or Path(args.out).stem
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ipopt_out = root / "ipopt.out"
    if ipopt_out.exists():
        try:
            ipopt_out.unlink()
        except Exception:
            pass

    env = os.environ.copy()
    env.update(
        {
            "BENCH_DERIV_CHECK": args.deriv_check,
            "BENCH_DERIV_TOL": "1e-4",
        }
    )
    # Do not force MUMPS; allow MA57 configuration to pass through.
    env["BENCH_FORCE_MUMPS"] = "0"
    if args.objective_option:
        env["BENCH_OBJECTIVE_OPTION"] = args.objective_option
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        conda_bin = f"{conda_prefix}/bin"
        path = env.get("PATH", "")
        if conda_bin not in path.split(os.pathsep):
            env["PATH"] = f"{conda_bin}{os.pathsep}{path}"
    env["BENCH_SOLVER"] = args.solver
    # Ensure sitecustomize is found so we can hook IPOPT options when requested.
    bench_path = root / "bench"
    env["PYTHONPATH"] = f"{bench_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.update(
        {
            "BENCH_PROBLEM": args.problem,
            "BENCH_INSTANCE": args.instance,
            "BENCH_MODE": args.mode,
            "BENCH_SOLVER": args.solver,
            "BENCH_RUN_ID": run_id,
        }
    )

    log_dir = root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"

    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    cmd = [sys.executable, str(script_path)]
    with log_path.open("w") as logf:
        try:
            ipopt_path = shutil.which("ipopt")
            logf.write(f"which ipopt: {ipopt_path}\n")
            if ipopt_path:
                ver = subprocess.run(["ipopt", "-v"], text=True, capture_output=True)
                if ver.stdout:
                    logf.write(ver.stdout)
                if ver.stderr:
                    logf.write(ver.stderr)
            logf.flush()
        except Exception as e:
            logf.write(f"ipopt version check failed: {e}\n")
            logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            text=True,
            stdout=logf,
            stderr=logf,
        )
    wall_time = time.monotonic() - start

    ipopt_dir = root / "results" / "ipopt"
    ipopt_dir.mkdir(parents=True, exist_ok=True)
    ipopt_copy = ipopt_dir / f"{run_id}.out"
    if ipopt_out.exists():
        try:
            ipopt_copy.write_text(ipopt_out.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
    ipopt_data = parse_ipopt_out(ipopt_copy if ipopt_copy.exists() else ipopt_out)
    try:
        from bench.deriv_parse import parse_deriv_log
    except Exception:
        parse_deriv_log = None

    bench_commit = _git_commit(root)
    pyomo_commit = _pyomo_git_commit()

    deriv_data = (
        parse_deriv_log(log_path, args.deriv_check)
        if parse_deriv_log
        else {
            "mode": args.mode,
            "level": args.deriv_check,
            "tol": 1e-4,
            "status": "unknown",
            "max_error": None,
            "worst_component": None,
            "log_file": str(log_path),
        }
    )
    if isinstance(deriv_data, dict):
        deriv_data.setdefault("mode", args.mode)

    obj_from_log = _parse_objective_from_log(log_path)
    objective_final = obj_from_log if obj_from_log is not None else ipopt_data.get("objective_final")
    objective_source = "log" if obj_from_log is not None else "ipopt"
    fim = _parse_fim_from_log(log_path)
    report_metrics = _compute_report_metrics(fim)

    status = "ok" if proc.returncode == 0 else "error"
    notes = None
    if (
        args.mode == "existing"
        and args.objective_option in {"minimum_eigenvalue", "condition_number"}
        and proc.returncode != 0
    ):
        status = "unsupported"
        notes = "E-opt/ME-opt unsupported for central objective"

    result: Dict[str, Any] = {
        "problem": args.problem,
        "instance": args.instance,
        "mode": args.mode,
        "solver": args.solver,
        "run_id": run_id,
        "status": status,
        "termination": ipopt_data.get("termination"),
        "objective_final": objective_final,
        "objective_source": objective_source,
        "native_obj": objective_final,
        **report_metrics,
        "iterations": ipopt_data.get("iterations"),
        "eval_counts": {
            "obj": ipopt_data.get("eval_obj"),
            "grad": ipopt_data.get("eval_grad"),
            "eq_con": ipopt_data.get("eval_eq_con"),
            "ineq_con": ipopt_data.get("eval_ineq_con"),
            "eq_jac": ipopt_data.get("eval_eq_jac"),
            "ineq_jac": ipopt_data.get("eval_ineq_jac"),
            "hess": ipopt_data.get("eval_hess"),
        },
        "wall_time_total_s": wall_time,
        "time_obj_eval_s": ipopt_data.get("time_nlp_eval"),
        "time_deriv_s": None,
        "model_sizes": {
            "n_vars": ipopt_data.get("n_vars"),
            "n_cons": ipopt_data.get("n_cons"),
            "nnz_jac": ipopt_data.get("nnz_jac"),
            "nnz_hess": ipopt_data.get("nnz_hess"),
        },
        "deriv_check": deriv_data,
        "git_commit": {
            "bench": bench_commit,
            "pyomo": pyomo_commit,
        },
        "python_version": platform.python_version(),
        "python_version_full": sys.version.replace("\n", " "),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "package_versions": _package_versions(),
        "pip_freeze": _pip_freeze(),
        "started_at": started_at,
        "stdout_tail": None,
        "stderr_tail": None,
        "exit_code": proc.returncode,
    }
    if notes:
        result["notes"] = notes

    out_path.write_text(json.dumps(result, indent=2))
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
