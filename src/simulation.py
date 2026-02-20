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
        raise ValueError("Require p_points >= 10.")

    p_grid = np.linspace(p_min, p_max, p_points)

    D_plot_min = float(grids.get("D_plot_min", grids["D_min"]))
    D_plot_max = float(grids.get("D_plot_max", grids["D_max"]))
    if not (D_plot_min > 0 and D_plot_max > D_plot_min):
        raise ValueError("Require D_plot_max > D_plot_min > 0.")
    D_plot_grid = np.logspace(np.log10(D_plot_min), np.log10(D_plot_max), 200)

    return SimulationGrids(p_grid=p_grid, D_plot_grid=D_plot_grid)


@dataclass
class DemandRow:
    p: float
    D_star: float

    # Tech outcomes (NaN if D_star==0 due to opt-out)
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
    boundary_side: str
    message: str

    # New: opt-out flag
    opted_out: bool


@dataclass
class SimulationResult:
    demand_rows: List[DemandRow]
    p_star: float
    D_star_at_p_star: float
    pi_teacher_star: float
    boundary_share: float
    optout_share: float


def run_baseline_grid_simulation(
    *,
    cfg: Dict[str, Any],
    tech: TierATechnology,
    N: float,
) -> Tuple[SimulationResult, SimulationGrids, Tuple[EconomicsParams, GridsParams, SolverParams]]:
    econ, grids, solver = build_params_from_config(cfg)
    sim_grids = build_simulation_grids(cfg)

    rows: List[DemandRow] = []
    best_piT = -np.inf
    best_p: Optional[float] = None
    best_D: Optional[float] = None

    boundary_count = 0
    optout_count = 0

    for p in sim_grids.p_grid:
        br: StudentBestResponseResult = solve_student_best_response_direct(
            N=N, p=float(p), tech=tech, econ=econ, grids=grids, solver=solver
        )

        # HARD OUTSIDE OPTION:
        # If the student's optimized payoff is negative, assume the student opts out.
        if br.pi_star < 0:
            optout_count += 1
            D_star = 0.0
            piS = 0.0
            piT = 0.0

            # Losses are undefined at D=0 in our scaling-law form (D^{-beta}),
            # so store NaN (plots that rely on losses should filter NaNs).
            Ls = float("nan")
            Lt = float("nan")
            gap = float("nan")

            rows.append(DemandRow(
                p=float(p),
                D_star=D_star,
                L_student=Ls,
                L_tilde=Lt,
                gap=gap,
                pi_student=piS,
                pi_teacher=piT,
                success=bool(br.success),
                nfev=int(br.nfev),
                is_boundary=False,
                boundary_side="",
                message="Opted out (hard outside option).",
                opted_out=True,
            ))
            continue

        # Otherwise, student enters and uses the computed best response
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

        rows.append(DemandRow(
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
            opted_out=False,
        ))

        if piT > best_piT:
            best_piT = piT
            best_p = float(p)
            best_D = D_star

    # If everyone opts out, teacher profit is identically 0
    if best_p is None:
        best_p = float(sim_grids.p_grid[0])
        best_D = 0.0
        best_piT = 0.0

    boundary_share = boundary_count / max(1, len(sim_grids.p_grid))
    optout_share = optout_count / max(1, len(sim_grids.p_grid))

    sim = SimulationResult(
        demand_rows=rows,
        p_star=float(best_p),
        D_star_at_p_star=float(best_D),
        pi_teacher_star=float(best_piT),
        boundary_share=float(boundary_share),
        optout_share=float(optout_share),
    )
    return sim, sim_grids, (econ, grids, solver)


def to_dataframe(sim: SimulationResult):
    import pandas as pd
    return pd.DataFrame([asdict(r) for r in sim.demand_rows])