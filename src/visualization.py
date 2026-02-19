"""
Visualization utilities (publication-ready).

Outputs:
- PDF + SVG for each figure (plus optional PNG preview)
- consistent style: titles, labels, legends, grids, and footer with key parameters

This file does NOT run simulations; it only plots from inputs (cfg, tech, sim results).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .scaling_laws import TierATechnology
from .simulation import SimulationResult, SimulationGrids


def _save_figure(fig: plt.Figure, outpath_base: Path, *, save_png: bool = True, dpi: int = 300) -> None:
    """
    Save figure as PDF and SVG (and optionally PNG) for LaTeX + Word workflows.
    """
    outpath_base.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(outpath_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outpath_base.with_suffix(".svg"), bbox_inches="tight")
    if save_png:
        fig.savefig(outpath_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def _footer_from_config(cfg: Dict[str, Any]) -> str:
    """
    Compact parameter footer for reproducibility (appears under the plot).
    Keep it short, but include the anchors and key exponents.
    """
    exp = cfg["exponents"]
    sup = cfg["anchors_supervised"]
    gap = cfg["anchors_gap"]
    econ = cfg["economics"]

    footer = (
        f"Exponents: alpha={exp['alpha']}, beta={exp['beta']}, gamma={exp['gamma']} | "
        f"alpha'={exp['alpha_p']}, beta'={exp['beta_p']}, gamma'={exp['gamma_p']}\n"
        f"Anchors: (D0,L0)=({sup['D0']},{sup['L0']}), (D1,L1)=({sup['D1']},{sup['L1']}), "
        f"rho={gap['rho']}, q={gap['q']} | "
        f"Costs: k={econ['k']}, c_T={econ['c_T']}, b={econ['b']}"
    )
    return footer


def plot_scaling_curves(
    *,
    cfg: Dict[str, Any],
    tech: TierATechnology,
    N: float,
    grids: SimulationGrids,
    outdir: Path,
    save_png: bool = True,
) -> None:
    """
    Figure 1: supervised frontier vs distilled loss and gap as functions of D.
    """
    D = grids.D_plot_grid

    Ltilde = np.array([tech.L_tilde(N, float(d)) for d in D])
    Lstud = np.array([tech.L_student(N, float(d)) for d in D])
    gap = Lstud - Ltilde

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(D, Ltilde, label=r"Supervised frontier $\tilde{L}_S(N,D)$")
    ax.plot(D, Lstud, label=r"Distilled loss $L_S(N,D)$ (Tier A)")
    ax.plot(D, gap, label=r"Gap $\Delta(D)=L_S-\tilde{L}_S$")

    ax.set_xscale("log")
    ax.set_xlabel(r"Training tokens $D$ (log scale)")
    ax.set_ylabel("Cross-entropy loss (normalized units)")
    ax.set_title("Technology block: supervised frontier vs distilled loss")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    fig.text(0.01, -0.08, _footer_from_config(cfg), ha="left", va="top", fontsize=8)

    _save_figure(fig, outdir / "fig_01_scaling_curves", save_png=save_png)
    plt.close(fig)


def plot_student_profit_slices(
    *,
    cfg: Dict[str, Any],
    tech: TierATechnology,
    N: float,
    grids: SimulationGrids,
    p_values: List[float],
    outdir: Path,
    save_png: bool = True,
) -> None:
    """
    Figure 2: student payoff Pi_S(D) for selected prices p (interior optimum sanity check).
    """
    D = grids.D_plot_grid

    econ = cfg["economics"]
    a = float(econ.get("a", 0.0))
    b = float(econ["b"])
    k = float(econ["k"])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for p in p_values:
        Ls = np.array([tech.L_student(N, float(d)) for d in D])
        Pi = (a - b * Ls) - (p + k) * D
        ax.plot(D, Pi, label=fr"$p={p:.3g}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"Training tokens $D$ (log scale)")
    ax.set_ylabel(r"Student payoff $\Pi_S(D)$ (normalized units)")
    ax.set_title("Student payoff profiles (sanity check for interior optimum)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(title="Teacher token price")

    fig.text(0.01, -0.08, _footer_from_config(cfg), ha="left", va="top", fontsize=8)

    _save_figure(fig, outdir / "fig_02_student_profit_slices", save_png=save_png)
    plt.close(fig)


def plot_demand_curve(
    *,
    cfg: Dict[str, Any],
    sim: SimulationResult,
    outdir: Path,
    save_png: bool = True,
) -> None:
    """
    Figure 3: demand curve D*(p).
    """
    p = np.array([r.p for r in sim.demand_rows])
    D = np.array([r.D_star for r in sim.demand_rows])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(p, D, label=r"Student demand $D^*(p)$")
    ax.set_yscale("log")

    ax.set_xlabel(r"Teacher token price $p$")
    ax.set_ylabel(r"Optimal training tokens $D^*(p)$ (log scale)")
    ax.set_title("Student demand for teacher tokens")

    # boundary share annotation (useful diagnostic)
    ax.text(
        0.02, 0.02,
        f"Boundary solutions share: {sim.boundary_share:.2%}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, linewidth=0.5),
    )

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    fig.text(0.01, -0.08, _footer_from_config(cfg), ha="left", va="top", fontsize=8)

    _save_figure(fig, outdir / "fig_03_demand_curve", save_png=save_png)
    plt.close(fig)


def plot_teacher_profit(
    *,
    cfg: Dict[str, Any],
    sim: SimulationResult,
    outdir: Path,
    save_png: bool = True,
) -> None:
    """
    Figure 4: teacher profit Pi_T(p) with p* highlighted.
    """
    p = np.array([r.p for r in sim.demand_rows])
    PiT = np.array([r.pi_teacher for r in sim.demand_rows])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(p, PiT, label=r"Teacher profit $\Pi_T(p)$")
    ax.axvline(sim.p_star, linestyle="--", linewidth=1.0, label=fr"$p^*={sim.p_star:.3g}$")
    ax.scatter([sim.p_star], [sim.pi_teacher_star], zorder=5)

    ax.set_xlabel(r"Teacher token price $p$")
    ax.set_ylabel(r"Teacher payoff $\Pi_T(p)$ (normalized units)")
    ax.set_title("Teacher pricing objective and optimal price")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    fig.text(0.01, -0.08, _footer_from_config(cfg), ha="left", va="top", fontsize=8)

    _save_figure(fig, outdir / "fig_04_teacher_profit", save_png=save_png)
    plt.close(fig)
