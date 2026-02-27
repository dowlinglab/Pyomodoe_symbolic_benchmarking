#!/usr/bin/env python
"""
PDE diffusion benchmark sweep for 5 fixed discretization cases.

How to run:
  python bench_sweep_5cases.py --run_all
  python bench_sweep_5cases.py --case_index 1

Design notes:
- Each case runs once per method (central/symbolic).
- Uses `bash -lc` + `conda activate` for strict env separation.
- Writes per-case JSONs + CSV + per-case overlay bar plots.
- Intended for extension: add cases, add metrics, or switch plotting.
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt

# Fixed 5 cases requested by the user (nfe_x, nfe_t). nfe_x <= nfe_t.
CASES = [
    (2, 2),
    (5, 10),
    (10, 20),
    (25, 40),
    (50, 75),
]

# Metric ordering for plotting (1..12) and CSV output.
METRIC_ORDER = [
    "doe_build_time",
    "doe_init_time",
    "doe_solve_time",
    "ipopt_cpu_nlp_feval",
    "ipopt_cpu_wo_feval",
    "doe_wall_time",
    "ipopt_iters",
    "obj_eval",
    "grad_eval",
    "eq_con_eval",
    "eq_jac_eval",
    "hess_eval",
]

METRIC_LABELS = [
    "Build",
    "Init.",
    "Solve",
    "NLP eval.",
    "CPU",
    "Wall clck.",
    "Iterations",
    "Obj. fun.",
    "Obj. grad.",
    "Eq. constr.",
    "Eq. jac.",
    "Lag. Hess.",
]


def parse_args():
    """Parse command-line arguments.

    By default, no cases run unless --run_all or --case_index is provided.
    This prevents accidental long runs when editing/testing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all 5 cases (10 total solves).",
    )
    parser.add_argument(
        "--case_index",
        type=int,
        default=0,
        help="Run a single case by index (1..5).",
    )
    return parser.parse_args()


def run_case(env_name, script_path, nfe_x, nfe_t, run_dir):
    """Run a single PDE script in a specific conda env and return JSON.

    Uses `bash -lc` + `conda activate` to honor environment separation
    without merging dependencies between the two environments.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / f"{Path(script_path).stem}.json"
    ipopt_out = run_dir / "ipopt.out"

    cmd = (
        f"source $(conda info --base)/etc/profile.d/conda.sh "
        f"&& conda activate {env_name} "
        f"&& python {script_path} "
        f"--nfe_x {nfe_x} --nfe_t {nfe_t} "
        f"--out_json {json_path} --run_dir {run_dir}"
    )

    start = time.time()
    subprocess.run(["bash", "-lc", cmd], check=True)
    elapsed = time.time() - start

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data, elapsed


def plot_overlay_bars(out_path, central_vals, sym_vals):
    """Create overlaid bar chart for one case.

    Requirements enforced:
    - x-axis 1..12
    - overlay bars at same x (not grouped)
    - styles for central vs symbolic
    - decryption legend mapping 1..12 -> label
    - short metric labels above bars (rotated 45 deg)
    - y-axis label exact text
    """
    x = list(range(1, len(METRIC_ORDER) + 1))

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x,
        central_vals,
        color="white",
        alpha=0.1,
        edgecolor="black",
        label="Central FD",
        linewidth=1.0,
    )
    ax.bar(
        x,
        sym_vals,
        color="gray",
        alpha=0.5,
        edgecolor="black",
        label="Symbolic",
        linewidth=1.0,
    )

    ax.set_xlabel("Metric index")
    ax.set_ylabel("Counts (optimizer iterations)/computational time")
    ax.set_xticks(x)

    # Annotate short labels above each bar (use the max of the two values).
    for i, (c_val, s_val, label) in enumerate(zip(central_vals, sym_vals, METRIC_LABELS), start=1):
        top = max(c_val, s_val)
        ax.text(
            i,
            top if top != 0 else 0.0,
            label,
            ha="center",
            va="bottom",
            rotation=45,
            fontsize=9,
        )

    ax.legend(loc="upper left")

    mapping_lines = [f"{i}. {name}" for i, name in enumerate(METRIC_LABELS, start=1)]
    mapping_text = "Metric index mapping:\n" + "\n".join(mapping_lines)
    ax.text(
        1.02,
        0.5,
        mapping_text,
        transform=ax.transAxes,
        fontsize=9,
        va="center",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    if not args.run_all and args.case_index <= 0:
        print("No cases selected. Use --run_all or --case_index 1..5.")
        return

    if args.case_index and (args.case_index < 1 or args.case_index > len(CASES)):
        raise SystemExit("case_index must be in 1..5")

    out_root = Path("out")
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "results.csv"
    csv_rows = []

    case_indices = [args.case_index] if args.case_index else list(range(1, len(CASES) + 1))

    for case_i in case_indices:
        nfe_x, nfe_t = CASES[case_i - 1]
        case_dir = out_root / f"case_{case_i}_nfe_x{nfe_x}_nfe_t{nfe_t}"
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"CASE {case_i}/5: nfe_x={nfe_x}, nfe_t={nfe_t}")
        print(f"  outputs: {case_dir}")

        print("  running central...")
        central_json, central_sec = run_case(
            "pyomo",
            "PDE_diffusion_central.py",
            nfe_x,
            nfe_t,
            case_dir / "central",
        )
        print(f"  central done (seconds={central_sec:.2f})")

        print("  running symbolic...")
        sym_json, sym_sec = run_case(
            "pyomosym",
            "PDE_diffusion_sym.py",
            nfe_x,
            nfe_t,
            case_dir / "symbolic",
        )
        print(f"  symbolic done (seconds={sym_sec:.2f})")
        print(f"CASE {case_i}/5 complete.")

        # Write per-case CSV row with central/symbolic/delta for each metric.
        row = {
            "case": case_i,
            "nfe_x": nfe_x,
            "nfe_t": nfe_t,
        }
        for metric in METRIC_ORDER:
            c_val = central_json.get(metric)
            s_val = sym_json.get(metric)
            row[f"central_{metric}"] = c_val
            row[f"symbolic_{metric}"] = s_val
            row[f"delta_{metric}"] = (s_val - c_val) if (s_val is not None and c_val is not None) else None
        csv_rows.append(row)

        # Plot overlay bars.
        central_vals = [central_json.get(m, 0.0) or 0.0 for m in METRIC_ORDER]
        sym_vals = [sym_json.get(m, 0.0) or 0.0 for m in METRIC_ORDER]
        plot_overlay_bars(
            out_root / "plots" / f"case_{case_i}_nfe_x{nfe_x}_nfe_t{nfe_t}.png",
            central_vals,
            sym_vals,
        )

    # Write CSV (one row per case with central/symbolic/delta columns).
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"Wrote results: {csv_path}")


if __name__ == "__main__":
    main()
