"""
Simulation routines for baseline (Tier A).

Core outputs:
- demand table over a price grid: p, D*(p), losses, profits, boundary diagnostics
- teacher profit over p and grid-based maximizer p*

Design:
- Solver-agnostic: calls best-response solver from src/model.py
- Technology-agnostic: relies on the technology interface methods (L_tilde, gap_term, L_student)

This file does NOT do any plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .scaling_laws import TierATechnology
from .model import (
    EconomicsParams,
    GridsParams,
    SolverParams,
    StudentBestResponseResult,
    build_params_from_config,
    solve_student_best_response_direct,
    teacher_profit,
)


@dataclass(frozen=True)
class SimulationGrids:
    """Numerical grids used in the baseline simulation."""
    p_grid: np.ndarray
    D_plot_grid: np.ndarray


def build_simulation_grids(cfg: Dict[str, Any]) -> SimulationGrids:
    grids = cfg["grids"]
    p_min = float(grids["p_min"])
    p_max = float(grids["p_max"])
    p_points = int(grids["p_points"])
    if p_max <= p_min:
        raise ValueError("Require p_max > p_min.")
    if p_points < 10:
        raise ValueError("Require p_points >= 10 for smooth curves.")

    p_grid = np.linspace(p_min, p_max, p_points)

    # Separate grid for plotting curves against D (log-spaced is best for scaling laws)
    D_plot_min = float(grids.get("D_plot_min", grids["D_min"]))
    D_plot_max = float(grids.get("D_plot_max", grids["D_max"]))
    if not (D_plot_min > 0 and D_plot_max > D_plot_min):
        raise ValueError("Require D_plot_max > D_plot_min > 0.")
    D_plot_grid = np.logspace(np.log10(D_plot_min), np.log10(D_plot_max), 200)

    return SimulationGrids(p_grid=p_grid, D_plot_grid=D_plot_grid)


@dataclass
class DemandRow:
    """One row per price p for tidy output."""
    p: float
    D_star: float

    # Technology outcomes at D_star
    L_student: float
    L_tilde: float
    gap: float

    # Payoffs
    pi_student: float
    pi_teacher: float

    # Diagnostics
    success: bool
    nfev: int
    is_boundary: bool
    boundary_side: str  # "", "min", "max"
    message: str


@dataclass
class SimulationResult:
    """Container of baseline simulation outputs."""
    demand_rows: List[DemandRow]

    # Teacher optimum on the p-grid
    p_star: float
    D_star_at_p_star: float
    pi_teacher_star: float

    # Summaries useful in logs
    boundary_share: float  # fraction of p-grid points with boundary solutions


def run_baseline_grid_simulation(
    *,
    cfg: Dict[str, Any],
    tech: TierATechnology,
    N: float,
) -> Tuple[SimulationResult, SimulationGrids, Tuple[EconomicsParams, GridsParams, SolverParams]]:
    """
    Run baseline simulation over the configured price grid.
    Returns:
      (SimulationResult, SimulationGrids, (econ, grids, solver))
    """
    econ, grids, solver = build_params_from_config(cfg)
    sim_grids = build_simulation_grids(cfg)

    rows: List[DemandRow] = []

    best_piT = -np.inf
    best_p: Optional[float] = None
    best_D: Optional[float] = None

    boundary_count = 0

    for p in sim_grids.p_grid:
        br: StudentBestResponseResult = solve_student_best_response_direct(
            N=N, p=float(p), tech=tech, econ=econ, grids=grids, solver=solver
        )

        D_star = float(br.D_star)

        Ls = float(tech.L_student(N, D_star))
        Lt = float(tech.L_tilde(N, D_star))
        gap = float(tech.gap_term(N, D_star))

        piS = float(br.pi_star)
        piT = float(teacher_profit(float(p), D_of_p=D_star, econ=econ))

        boundary_side = br.boundary_side or ""
        is_boundary = bool(br.is_boundary)
        if is_boundary:
            boundary_count += 1

        row = DemandRow(
            p=float(p),
            D_star=D_star,
            L_student=Ls,
            L_tilde=Lt,
            gap=gap,
            pi_student=piS,
            pi_teacher=piT,
            success=bool(br.success),
            nfev=int(br.nfev),
            is_boundary=is_boundary,
            boundary_side=str(boundary_side),
            message=str(br.message),
        )
        rows.append(row)

        if piT > best_piT:
            best_piT = piT
            best_p = float(p)
            best_D = D_star

    assert best_p is not None and best_D is not None

    boundary_share = boundary_count / max(1, len(sim_grids.p_grid))

    sim = SimulationResult(
        demand_rows=rows,
        p_star=float(best_p),
        D_star_at_p_star=float(best_D),
        pi_teacher_star=float(best_piT),
        boundary_share=float(boundary_share),
    )
    return sim, sim_grids, (econ, grids, solver)


def to_dataframe(sim: SimulationResult):
    """
    Convenience: convert demand rows to a pandas DataFrame.
    Imported lazily to avoid hard dependency inside core simulation logic.
    """
    import pandas as pd
    return pd.DataFrame([asdict(r) for r in sim.demand_rows])
