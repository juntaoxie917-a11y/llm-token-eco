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
from .simulation import build_simulation_grids, SimulationGrids  # reuse your grid builder


def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


@dataclass
class SoftDemandRow:
    p: float

    # Raw best response (conditional on entering)
    D_star: float
    pi_student_star: float

    # Soft participation
    s_enter: float
    D_soft: float
    pi_student_soft: float
    pi_teacher_soft: float

    # Tech outcomes evaluated at D_star (optional; can be NaN if D_star is tiny)
    L_student: float
    L_tilde: float
    gap: float

    # Diagnostics
    success: bool
    nfev: int
    is_boundary: bool
    boundary_side: str
    message: str


@dataclass
class SoftSimulationResult:
    demand_rows: List[SoftDemandRow]
    p_star: float
    D_soft_at_p_star: float
    pi_teacher_star: float
    avg_enter_prob: float


def run_soft_grid_simulation(
    *,
    cfg: Dict[str, Any],
    tech: TierATechnology,
    N: float,
) -> Tuple[SoftSimulationResult, SimulationGrids, Tuple[EconomicsParams, GridsParams, SolverParams]]:
    econ, grids, solver = build_params_from_config(cfg)
    sim_grids = build_simulation_grids(cfg)

    soft_cfg = cfg.get("soft_outside", {})
    tau = float(soft_cfg.get("tau", 0.2))
    if tau <= 0:
        raise ValueError("soft_outside.tau must be > 0.")

    rows: List[SoftDemandRow] = []

    best_piT = -np.inf
    best_p: Optional[float] = None
    best_Dsoft: Optional[float] = None

    enter_probs = []

    for p in sim_grids.p_grid:
        br: StudentBestResponseResult = solve_student_best_response_direct(
            N=N, p=float(p), tech=tech, econ=econ, grids=grids, solver=solver
        )

        D_star = float(br.D_star)
        piS_star = float(br.pi_star)

        # soft participation probability
        s = float(sigmoid(np.array([piS_star / tau]))[0])
        enter_probs.append(s)

        D_soft = s * D_star
        piT_soft = float((float(p) - econ.c_T) * D_soft)

        # simplest "expected" student payoff (outside option=0)
        piS_soft = s * piS_star

        # evaluate tech at D_star (not D_soft), since D_soft is an expectation proxy
        # If D_star is extremely small, these may blow up; guard lightly.
        if D_star <= 0:
            Ls = float("nan")
            Lt = float("nan")
            gap = float("nan")
        else:
            Ls = float(tech.L_student(N, D_star))
            Lt = float(tech.L_tilde(N, D_star))
            gap = float(tech.gap_term(N, D_star))

        rows.append(SoftDemandRow(
            p=float(p),
            D_star=D_star,
            pi_student_star=piS_star,
            s_enter=s,
            D_soft=float(D_soft),
            pi_student_soft=float(piS_soft),
            pi_teacher_soft=float(piT_soft),
            L_student=Ls,
            L_tilde=Lt,
            gap=gap,
            success=bool(br.success),
            nfev=int(br.nfev),
            is_boundary=bool(br.is_boundary),
            boundary_side=str(br.boundary_side or ""),
            message=str(br.message),
        ))

        if piT_soft > best_piT:
            best_piT = piT_soft
            best_p = float(p)
            best_Dsoft = float(D_soft)

    if best_p is None:
        best_p = float(sim_grids.p_grid[0])
        best_Dsoft = 0.0
        best_piT = 0.0

    sim = SoftSimulationResult(
        demand_rows=rows,
        p_star=float(best_p),
        D_soft_at_p_star=float(best_Dsoft),
        pi_teacher_star=float(best_piT),
        avg_enter_prob=float(np.mean(enter_probs)) if enter_probs else 0.0,
    )
    return sim, sim_grids, (econ, grids, solver)


def to_dataframe(sim: SoftSimulationResult):
    import pandas as pd
    return pd.DataFrame([asdict(r) for r in sim.demand_rows])