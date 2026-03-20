"""Stage 4: competition simulation runner.

Grid-based upstream pricing loop (teacher Stage-1), nesting:
1) competition-aware student best response over D,
2) downstream equilibrium at D*.

No plotting and no output-saving logic in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .competition_downstream_solver import DownstreamSolverParams
from .competition_static import CompetitionParams
from .competition_student import (
    CompetitionStudentBestResponseResult,
    solve_student_best_response_competition,
)
from .model import EconomicsParams, GridsParams, SolverParams, build_params_from_config
from .scaling_laws import TierATechnology
from .simulation import SimulationGrids, build_simulation_grids


@dataclass
class CompetitionSimulationRow:
    p: float

    D_star: float
    pi_student_total: float
    pi_student_downstream: float

    pi_teacher_upstream: float
    pi_teacher_downstream: float
    pi_teacher_total: float

    P_T_down_star: float
    P_S_down_star: float
    s_T_down_star: float
    s_S_down_star: float
    s_0_down_star: float

    # Stage-3 diagnostics
    br_success: bool
    br_message: str
    br_nfev: int
    br_is_boundary: bool
    br_boundary_side: str
    downstream_calls: int
    downstream_failures: int
    downstream_success_rate: float
    cache_hits: int
    cache_misses: int

    # Downstream equilibrium diagnostics at D*
    down_success: bool
    down_method: str
    down_message: str
    down_nfev: int
    down_iterations: int
    down_residual_norm: float
    down_hit_bounds: bool


@dataclass
class CompetitionSimulationResult:
    rows: List[CompetitionSimulationRow]

    p_star: float
    D_star_at_p_star: float

    P_T_down_at_p_star: float
    P_S_down_at_p_star: float

    pi_teacher_total_star: float
    pi_teacher_upstream_at_p_star: float
    pi_teacher_downstream_at_p_star: float

    br_success_rate: float
    down_success_rate: float
    boundary_share: float


def run_competition_grid_simulation(
    *,
    cfg: Dict[str, Any],
    tech: TierATechnology,
    N: float,
    comp: CompetitionParams,
    downstream_solver_params: DownstreamSolverParams,
    p_grid_override: Optional[Sequence[float]] = None,
    use_student_cache: bool = True,
    student_cache_precision: int = 8,
) -> Tuple[CompetitionSimulationResult, SimulationGrids, Tuple[EconomicsParams, GridsParams, SolverParams]]:
    """Run Stage-4 competition simulation over upstream price grid."""
    econ, grids, solver = build_params_from_config(cfg)
    sim_grids = build_simulation_grids(cfg)

    p_grid: Iterable[float]
    if p_grid_override is None:
        p_grid = sim_grids.p_grid
        effective_sim_grids = sim_grids
    else:
        p_grid_arr = np.asarray([float(x) for x in p_grid_override], dtype=float)
        if p_grid_arr.size == 0:
            raise ValueError("p_grid_override must contain at least one price point.")
        p_grid = p_grid_arr
        effective_sim_grids = SimulationGrids(
            p_grid=p_grid_arr,
            D_plot_grid=sim_grids.D_plot_grid,
        )

    rows: List[CompetitionSimulationRow] = []

    best_pi = -np.inf
    best_idx = -1

    warm_down_prices: Optional[Tuple[float, float]] = None

    br_success_count = 0
    down_success_count = 0
    boundary_count = 0

    for p in p_grid:
        br: CompetitionStudentBestResponseResult = solve_student_best_response_competition(
            N=N,
            p=float(p),
            tech=tech,
            econ=econ,
            grids=grids,
            solver=solver,
            comp=comp,
            downstream_solver_params=downstream_solver_params,
            initial_downstream_prices=warm_down_prices,
            use_cache=use_student_cache,
            cache_precision=student_cache_precision,
        )

        eq = br.downstream_eq_at_star
        if eq.success:
            warm_down_prices = (eq.P_T_star, eq.P_S_star)

        pi_teacher_up = float((float(p) - econ.c_T) * br.D_star)
        pi_teacher_down = float(eq.pi_T_down_star)
        pi_teacher_total = float(pi_teacher_up + pi_teacher_down)

        if br.success:
            br_success_count += 1
        if eq.success:
            down_success_count += 1
        if br.is_boundary:
            boundary_count += 1

        row = CompetitionSimulationRow(
            p=float(p),
            D_star=float(br.D_star),
            pi_student_total=float(br.pi_student_total_star),
            pi_student_downstream=float(br.pi_student_down_star),
            pi_teacher_upstream=pi_teacher_up,
            pi_teacher_downstream=pi_teacher_down,
            pi_teacher_total=pi_teacher_total,
            P_T_down_star=float(eq.P_T_star),
            P_S_down_star=float(eq.P_S_star),
            s_T_down_star=float(eq.s_T_star),
            s_S_down_star=float(eq.s_S_star),
            s_0_down_star=float(eq.s_0_star),
            br_success=bool(br.success),
            br_message=str(br.message),
            br_nfev=int(br.nfev),
            br_is_boundary=bool(br.is_boundary),
            br_boundary_side=str(br.boundary_side or ""),
            downstream_calls=int(br.downstream_calls),
            downstream_failures=int(br.downstream_failures),
            downstream_success_rate=float(br.downstream_success_rate),
            cache_hits=int(br.cache_hits),
            cache_misses=int(br.cache_misses),
            down_success=bool(eq.success),
            down_method=str(eq.method_used),
            down_message=str(eq.message),
            down_nfev=int(eq.nfev),
            down_iterations=int(eq.iterations),
            down_residual_norm=float(eq.residual_norm),
            down_hit_bounds=bool(eq.hit_bounds),
        )
        rows.append(row)

        if pi_teacher_total > best_pi:
            best_pi = pi_teacher_total
            best_idx = len(rows) - 1

    if best_idx < 0:
        raise RuntimeError("No rows produced in competition simulation.")

    best_row = rows[best_idx]
    total = max(1, len(rows))
    result = CompetitionSimulationResult(
        rows=rows,
        p_star=float(best_row.p),
        D_star_at_p_star=float(best_row.D_star),
        P_T_down_at_p_star=float(best_row.P_T_down_star),
        P_S_down_at_p_star=float(best_row.P_S_down_star),
        pi_teacher_total_star=float(best_row.pi_teacher_total),
        pi_teacher_upstream_at_p_star=float(best_row.pi_teacher_upstream),
        pi_teacher_downstream_at_p_star=float(best_row.pi_teacher_downstream),
        br_success_rate=float(br_success_count / total),
        down_success_rate=float(down_success_count / total),
        boundary_share=float(boundary_count / total),
    )
    return result, effective_sim_grids, (econ, grids, solver)


def to_dataframe(sim: CompetitionSimulationResult):
    import pandas as pd

    return pd.DataFrame([asdict(r) for r in sim.rows])
