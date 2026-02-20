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

import numpy as np
import pandas as pd


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
    p = np.array([r.p for r in sim.demand_rows])
    D = np.array([r.D_star for r in sim.demand_rows])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(p, D, label=r"Student demand $D(p)$ (hard outside option)")

    # symlog handles zeros cleanly, while still showing wide ranges
    ax.set_yscale("symlog", linthresh=1e-6)

    ax.set_xlabel(r"Teacher token price $p$")
    ax.set_ylabel(r"Optimal training tokens $D(p)$ (symlog scale)")
    ax.set_title("Student demand for teacher tokens (with opt-out)")

    ax.text(
        0.02, 0.02,
        f"Opt-out share: {sim.optout_share:.2%}\nBoundary share: {sim.boundary_share:.2%}",
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

def plot_student_indirect_payoff(
    *,
    cfg: Dict[str, Any],
    sim: SimulationResult,
    outdir: Path,
    save_png: bool = True,
) -> None:
    """
    Figure: Student indirect payoff Pi_S^*(p) over the price grid.

    Uses sim.demand_rows[*].pi_student, which should already be the optimized value
    (and equals 0 under hard outside option if you coded it that way).
    """
    p = np.array([r.p for r in sim.demand_rows], dtype=float)
    piS = np.array([r.pi_student for r in sim.demand_rows], dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(p, piS, label=r"Student indirect payoff $\Pi_S^*(p)$")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, label="Outside option (0)")

    # Mark teacher optimum p* and Pi_S*(p*)
    p_star = float(sim.p_star)
    # Find nearest grid index to p_star
    idx = int(np.argmin(np.abs(p - p_star)))
    ax.scatter([p[idx]], [piS[idx]], zorder=5)
    ax.axvline(p_star, linestyle="--", linewidth=1.0, label=fr"$p^*={p_star:.3g}$")

    ax.set_xlabel(r"Teacher token price $p$")
    ax.set_ylabel(r"Student payoff at optimum $\Pi_S^*(p)$ (normalized units)")
    ax.set_title("Student participation and payoff across teacher prices")

    # Optional: annotate value at p*
    ax.text(
        0.02, 0.98,
        fr"$\Pi_S^*(p^*)={piS[idx]:.3g}$",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, linewidth=0.5),
    )

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    fig.text(0.01, -0.08, _footer_from_config(cfg), ha="left", va="top", fontsize=8)

    _save_figure(fig, outdir / "fig_05_student_indirect_payoff", save_png=save_png)
    plt.close(fig)
    
def plot_soft_demand_curve(*, cfg: Dict[str, Any], df: "pd.DataFrame", outdir: Path, save_png: bool = True) -> None:
    p = df["p"].to_numpy()
    Dsoft = df["D_soft"].to_numpy()
    s = df["s_enter"].to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(p, Dsoft, label=r"Effective demand $D^{soft}(p)=s(p)\cdot D^*(p)$")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel(r"Teacher token price $p$")
    ax.set_ylabel(r"Effective tokens $D^{soft}(p)$ (symlog)")
    ax.set_title("Soft participation: effective token demand")

    ax2 = ax.twinx()
    ax2.plot(p, s, linestyle="--", label=r"Enter prob $s(p)$")
    ax2.set_ylabel(r"Enter probability $s(p)$")

    # legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.text(0.01, -0.08, _footer_from_config(cfg), ha="left", va="top", fontsize=8)
    _save_figure(fig, outdir / "fig_soft_01_demand", save_png=save_png)
    plt.close(fig)


def plot_soft_teacher_profit(*, cfg: Dict[str, Any], df: "pd.DataFrame", outdir: Path, save_png: bool = True) -> None:
    p = df["p"].to_numpy()
    piT = df["pi_teacher_soft"].to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(p, piT, label=r"Teacher profit $\Pi_T^{soft}(p)$")
    ax.set_xlabel(r"Teacher token price $p$")
    ax.set_ylabel(r"Teacher profit (normalized units)")
    ax.set_title("Teacher pricing under soft participation")

    # mark p*
    idx = int(np.argmax(piT))
    ax.axvline(float(p[idx]), linestyle="--", linewidth=1.0, label=fr"$p^*={p[idx]:.3g}$")
    ax.scatter([p[idx]], [piT[idx]], zorder=5)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.text(0.01, -0.08, _footer_from_config(cfg), ha="left", va="top", fontsize=8)
    _save_figure(fig, outdir / "fig_soft_02_teacher_profit", save_png=save_png)
    plt.close(fig)