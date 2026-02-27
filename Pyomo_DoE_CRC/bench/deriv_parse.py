import re
from pathlib import Path
from typing import Any, Dict


_PASS_PAT = re.compile(
    r"derivative test.*(passed|success|ok)|no errors detected by derivative checker",
    re.IGNORECASE,
)
_FAIL_PAT = re.compile(
    r"derivative test.*(failed|error)|error in derivative|derivative checker.*(failed|error)",
    re.IGNORECASE,
)
_WARN_PAT = re.compile(
    r"derivative test.*warning|warning:.*derivative|inconsistent with finite differences",
    re.IGNORECASE,
)
_MAX_ERR_PAT = re.compile(r"max(?:imum)?\s+(?:relative\s+)?error\s*=\s*([0-9.eE+-]+)")
_WORST_PAT = re.compile(r"worst\s+component\s*:\s*(.*)")


def parse_deriv_log(path: Path, level: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "mode": None,
        "level": level,
        "tol": 1e-4,
        "status": "unknown",
        "max_error": None,
        "worst_component": None,
        "log_file": str(path),
    }

    if not path.exists():
        return data

    text = path.read_text(encoding="utf-8", errors="replace")

    if level == "none":
        data["status"] = "not_requested"
        return data

    if _FAIL_PAT.search(text) or _WARN_PAT.search(text):
        data["status"] = "fail"
    elif _PASS_PAT.search(text):
        data["status"] = "pass"
    else:
        data["status"] = "fail"

    m = _MAX_ERR_PAT.search(text)
    if m:
        try:
            data["max_error"] = float(m.group(1))
        except Exception:
            pass

    m = _WORST_PAT.search(text)
    if m:
        data["worst_component"] = m.group(1).strip()

    return data
