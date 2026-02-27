#!/usr/bin/env python
import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfe_t_min", type=int, default=2)
    parser.add_argument("--nfe_t_max", type=int, default=100)
    parser.add_argument("--nfe_x_min", type=int, default=2)
    parser.add_argument("--nfe_x_max", type=int, default=75)
    parser.add_argument("--out_dir", type=str, default="bench_logs/pde_sweep")
    parser.add_argument("--csv_path", type=str, default="bench_logs/pde_sweep_diff.csv")
    parser.add_argument("--max_pairs", type=int, default=0, help="If > 0, limit number of (nfe_t, nfe_x) pairs.")
    return parser.parse_args()


def flatten_metrics(data, prefix=""):
    out = {}
    if isinstance(data, dict):
        for k, v in data.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            out.update(flatten_metrics(v, key))
    else:
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            out[prefix] = float(data)
    return out


def run_one(env_name, script, nfe_t, nfe_x, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"nfe_t{nfe_t}_nfe_x{nfe_x}"
    json_path = out_dir / f"{Path(script).stem}_{tag}.json"
    ipopt_out = out_dir / f"{Path(script).stem}_{tag}.ipopt.out"

    cmd = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        script,
        "--nfe_t",
        str(nfe_t),
        "--nfe_x",
        str(nfe_x),
        "--output_json",
        str(json_path),
        "--ipopt_out",
        str(ipopt_out),
    ]
    subprocess.run(cmd, check=True)

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    pair_count = 0

    for nfe_t in range(args.nfe_t_min, args.nfe_t_max + 1):
        for nfe_x in range(args.nfe_x_min, args.nfe_x_max + 1):
            pair_count += 1
            print(f"Running nfe_t={nfe_t}, nfe_x={nfe_x} ({pair_count})")

            central = run_one("pyomo", "PDE_diffusion_central.py", nfe_t, nfe_x, out_dir)
            sym = run_one("pyomosym", "PDE_diffusion_sym.py", nfe_t, nfe_x, out_dir)

            c_flat = flatten_metrics(central)
            s_flat = flatten_metrics(sym)

            for key in sorted(set(c_flat) & set(s_flat)):
                c_val = c_flat[key]
                s_val = s_flat[key]
                rows.append(
                    {
                        "nfe_t": nfe_t,
                        "nfe_x": nfe_x,
                        "metric": key,
                        "central": c_val,
                        "sym": s_val,
                        "sym_minus_central": s_val - c_val,
                    }
                )

            if args.max_pairs > 0 and pair_count >= args.max_pairs:
                print(f"Reached max_pairs={args.max_pairs}; stopping early.")
                break
        if args.max_pairs > 0 and pair_count >= args.max_pairs:
            break

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["nfe_t", "nfe_x", "metric", "central", "sym", "sym_minus_central"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} metric rows to {csv_path}")


if __name__ == "__main__":
    main()
