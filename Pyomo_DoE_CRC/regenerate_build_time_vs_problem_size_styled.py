#!/usr/bin/env python3
"""
Regenerate build_time vs problem size with locked styling.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("out/symbolic_measurements_20260226_145344/summary_partial.csv"),
        help="Path to summary_partial.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("out"),
        help="Output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"ERROR: CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if "n_cons_x_n_vars" not in df.columns or "build_time" not in df.columns:
        raise SystemExit("ERROR: CSV must contain n_cons_x_n_vars and build_time columns")
    if len(df) < 2:
        raise SystemExit("ERROR: CSV needs at least 2 rows to omit the last row")

    plot_df = df.iloc[:-1].copy()

    x = pd.to_numeric(plot_df["n_cons_x_n_vars"], errors="coerce")
    y = pd.to_numeric(plot_df["build_time"], errors="coerce")
    valid = x.notna() & y.notna() & (x > 0) & (y > 0)
    x = x[valid]
    y = y[valid]
    if x.empty:
        raise SystemExit("ERROR: no valid positive data after filtering")

    plt.figure(figsize=(14, 8))
    plt.plot(x, y, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$n_{\mathrm{cons}} \times n_{\mathrm{vars}}$", fontsize=18)
    plt.ylabel(r"$t_{\mathrm{build}}\;(\mathrm{s})$", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6)
    plt.tight_layout()

    today = datetime.now().strftime("%Y-%m-%d")
    out_path = args.outdir / f"build_time_vs_problem_size_styled_{today}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Input CSV: {args.csv}")
    print(f"Output figure: {out_path}")


if __name__ == "__main__":
    main()
