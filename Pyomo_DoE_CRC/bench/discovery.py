import os
from pathlib import Path
from typing import Dict


MODE_TO_SUFFIX = {
    "existing": "_central",
    "greybox": "_central_greybox",
    "central": "_central",
}


def discover_problems(root: Path) -> Dict[str, Dict[str, Path]]:
    root = root.resolve()
    problems: Dict[str, Dict[str, Path]] = {}
    for path in root.glob("*_central.py"):
        name = path.stem[: -len("_central")]
        problems.setdefault(name, {})["central"] = path
    for path in root.glob("*_central_greybox.py"):
        name = path.stem[: -len("_central_greybox")]
        problems.setdefault(name, {})["central_greybox"] = path
    return problems


def resolve_problem_file(root: Path, problem: str, mode: str) -> Path:
    problems = discover_problems(root)
    if problem not in problems:
        available = ", ".join(sorted(problems.keys()))
        raise ValueError(f"Unknown problem '{problem}'. Available: {available}")

    suffix = MODE_TO_SUFFIX.get(mode)
    if not suffix:
        raise ValueError(f"Unknown mode '{mode}'. Expected existing|greybox|central")

    if suffix in ("_central", "_central_greybox"):
        key = "central" if suffix == "_central" else "central_greybox"
    if key not in problems[problem]:
        raise ValueError(f"Problem '{problem}' does not have a '{key}' script")

    return problems[problem][key]
