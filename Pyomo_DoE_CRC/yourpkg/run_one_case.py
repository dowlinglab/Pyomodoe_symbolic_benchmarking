#!/usr/bin/env python3
"""Run one PDE DoE case in FD or symbolic mode and emit JSON metrics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fd", "symbolic"], required=True)
    parser.add_argument("--nfe-x", type=int, required=True)
    parser.add_argument("--nfe-t", type=int, required=True)
    parser.add_argument("--ipopt-exe", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = Path.cwd()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = out_path.parent

    method = "central" if args.mode == "fd" else "symbolic"
    ipopt_out = run_dir / f"{method}_{args.nfe_x}_{args.nfe_t}.ipopt.out"

    env = os.environ.copy()
    env["IPOPT_BIN_OVERRIDE"] = args.ipopt_exe

    cmd = [
        sys.executable,
        str(root / "pde_doe_execution_helper.py"),
        "--method",
        method,
        "--mode",
        "baseline",
        "--nfe_x",
        str(args.nfe_x),
        "--nfe_t",
        str(args.nfe_t),
        "--out_json",
        str(out_path),
        "--run_dir",
        str(run_dir),
        "--ipopt_out",
        str(ipopt_out),
        "--ipopt_exe",
        args.ipopt_exe,
    ]
    subprocess.run(cmd, check=True, env=env)

    # Ensure required solver metadata is present in artifact JSON.
    data = json.loads(out_path.read_text(encoding="utf-8"))
    data["ipopt_exe"] = args.ipopt_exe
    data["ipopt_linear_solver"] = os.environ.get("IPOPT_LINEAR_SOLVER")
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
