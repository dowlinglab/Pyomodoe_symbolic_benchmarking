import re
from pathlib import Path
from typing import Any, Dict, Optional


_INT_PATTERNS = {
    "iterations": r"Number of Iterations.*:\s+(\d+)",
    "eval_obj": r"Number of objective function evaluations\s*=\s*(\d+)",
    "eval_grad": r"Number of objective gradient evaluations\s*=\s*(\d+)",
    "eval_eq_con": r"Number of equality constraint evaluations\s*=\s*(\d+)",
    "eval_ineq_con": r"Number of inequality constraint evaluations\s*=\s*(\d+)",
    "eval_eq_jac": r"Number of equality constraint Jacobian evaluations\s*=\s*(\d+)",
    "eval_ineq_jac": r"Number of inequality constraint Jacobian evaluations\s*=\s*(\d+)",
    "eval_hess": r"Number of Lagrangian Hessian evaluations\s*=\s*(\d+)",
    "n_vars": r"(?:Total number of variables|Number of variables)\.*:\s*(\d+)",
    "n_cons": r"(?:Total number of equality constraints|Number of constraints|Number of equality constraints)\.*:\s*(\d+)",
    "nnz_jac": r"Number of nonzeros in (?:equality constraint )?Jacobian\.*:\s*(\d+)",
    "nnz_hess": r"Number of nonzeros in (?:Lagrangian )?Hessian\.*:\s*(\d+)",
    "nnz_jac_eq": r"Number of nonzeros in equality constraint Jacobian\.*:\s*(\d+)",
    "nnz_jac_ineq": r"Number of nonzeros in inequality constraint Jacobian\.*:\s*(\d+)",
}

_FLOAT_PATTERNS = {
    "time_nlp_eval": r"Total CPU secs in NLP function evaluations\s*=\s*([0-9.]+)",
    "time_ipopt_wo_eval": r"Total CPU secs in IPOPT \(w/o function evaluations\)\s*=\s*([0-9.]+)",
}


_OBJECTIVE_BLOCK = re.compile(
    r"Objective\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)", re.DOTALL
)


_EXIT = re.compile(r"EXIT:\s*(.*)")


def _grab_int(text: str, pat: str) -> Optional[int]:
    m = re.search(pat, text)
    return int(m.group(1)) if m else None


def _grab_float(text: str, pat: str) -> Optional[float]:
    m = re.search(pat, text)
    return float(m.group(1)) if m else None


def parse_ipopt_out(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    data: Dict[str, Any] = {}

    for key, pat in _INT_PATTERNS.items():
        data[key] = _grab_int(text, pat)

    if data.get("nnz_jac") is None:
        eq = data.get("nnz_jac_eq")
        ineq = data.get("nnz_jac_ineq")
        if eq is not None or ineq is not None:
            data["nnz_jac"] = (eq or 0) + (ineq or 0)

    for key, pat in _FLOAT_PATTERNS.items():
        data[key] = _grab_float(text, pat)

    m = _OBJECTIVE_BLOCK.search(text)
    if m:
        # Use unscaled objective if present (second capture)
        try:
            data["objective_final"] = float(m.group(2))
        except Exception:
            pass

    m = _EXIT.search(text)
    if m:
        data["termination"] = m.group(1).strip()

    return data
