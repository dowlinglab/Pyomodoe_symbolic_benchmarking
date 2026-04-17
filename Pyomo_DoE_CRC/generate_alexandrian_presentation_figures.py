#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the two presentation figures for the 1D Alexandrian DoE story.

Figure 1:
    Runtime comparison for the corrected 1D solved case
    (central finite difference vs symbolic / pynumero)

Figure 2:
    Model-size scaling at fixed n_p = 2, using the measured counts from the
    current 1D model and the fitted exact formulas for the DoE builds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTDIR = Path("results_plots/presentation")
OUTDIR.mkdir(parents=True, exist_ok=True)


def save_all(fig, stem: str) -> None:
    fig.savefig(OUTDIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.eps", bbox_inches="tight")
    plt.close(fig)


def runtime_figure() -> None:
    # Corrected runs after replacing Expression outputs with auxiliary Vars.
    central = {
        "Build": 0.6246989169994777,
        "Init": 0.30229154100015876,
        "Solve": 0.71174954199887,
        "Wall": 1.6387399999985064,
    }
    symbolic = {
        "Build": 33.73411816599764,
        "Init": 1.7199765420009498,
        "Solve": 1.9615593330017873,
        "Wall": 37.41565404100038,
    }

    labels = list(central.keys())
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(
        x - width / 2,
        [central[k] for k in labels],
        width=width,
        facecolor="white",
        edgecolor="black",
        linewidth=2.0,
        label="Central finite difference",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        [symbolic[k] for k in labels],
        width=width,
        color="#B7D7F0",
        edgecolor="#7AA7D1",
        linewidth=1.0,
        label="Symbolic (pynumero)",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time (s)")
    ax.set_title("1D Solved Case: Runtime Comparison (n_p = 2)")
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.legend(frameon=True)

    save_all(fig, "alexandrian_runtime_comparison")


def scaling_figure() -> None:
    # Measured base labeled counts from the corrected 1D model.
    # Each tuple: (nfe_x, nfe_t, base_eq, central_eq)
    measured = [
        (10, 40, 1805, 7243),
        (20, 40, 3445, 13803),
        (20, 80, 6805, 27243),
        (40, 80, 13285, 53163),
    ]

    nxy = np.array([nx * nt for nx, nt, _, _ in measured], dtype=float)
    base_eq = np.array([beq for _, _, beq, _ in measured], dtype=float)
    central_eq = np.array([ceq for _, _, _, ceq in measured], dtype=float)

    # From exact measured symbolic counts at (10,40) and (20,40):
    #   symbolic_eq = 3 * base_eq + 11
    symbolic_eq = 3.0 * base_eq + 11.0

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        nxy,
        central_eq,
        marker="o",
        markersize=7,
        linewidth=2.2,
        color="black",
        label="Central FD DoE equations",
    )
    ax.plot(
        nxy,
        symbolic_eq,
        marker="s",
        markersize=7,
        linewidth=2.2,
        color="#4F8FCB",
        label="Symbolic DoE equations",
    )

    for nx, nt, beq, _ in measured:
        ax.annotate(
            f"({nx},{nt})",
            xy=(nx * nt, beq * 4.0 + 23.0),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(r"$n_x \times n_t$  (fixed $n_p = 2$)")
    ax.set_ylabel("Number of equality constraints")
    ax.set_title("Alex Scaling Plot: DoE Model Size vs Discretization")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=True)

    save_all(fig, "alexandrian_model_size_scaling")


def main() -> None:
    runtime_figure()
    scaling_figure()
    print(f"Wrote figures to {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
