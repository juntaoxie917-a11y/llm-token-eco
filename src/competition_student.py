"""Stage 3: competition-aware student best response.

This module solves the student's Stage-2 problem for fixed upstream token price p:

    max_D  pi_S_down_eq(D) - (p + k) D

where pi_S_down_eq(D) is obtained from the Stage-2 downstream pricing-subgame solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from scipy.optimize import minimize_scalar

from .competition_downstream_solver import (
    DownstreamEquilibriumResult,
    DownstreamSolverParams,
    solve_downstream_equilibrium_at_D,
)
from .competition_static import CompetitionParams
from .model import EconomicsParams, GridsParams, SolverParams
from .scaling_laws import TierATechnology


@dataclass(frozen=True)
class CompetitionStudentBestResponseResult:
    p: float
    D_star: float
    pi_student_total_star: float
    pi_student_down_star: float

    downstream_eq_at_star: DownstreamEquilibriumResult

    is_boundary: bool
    boundary_side: Optional[str]

    nfev: int
    success: bool
    message: str

    downstream_calls: int
    downstream_failures: int
    downstream_success_rate: float
    cache_hits: int
    cache_misses: int


def solve_student_best_response_competition(
    *,
    N: float,
    p: float,
    tech: TierATechnology,
    econ: EconomicsParams,
    grids: GridsParams,
    solver: SolverParams,
    comp: CompetitionParams,
    downstream_solver_params: DownstreamSolverParams,
    initial_downstream_prices: Optional[Tuple[float, float]] = None,
    use_cache: bool = True,
    cache_precision: int = 8,
) -> CompetitionStudentBestResponseResult:
    """Solve competition-aware student best response over D in [D_min, D_max]."""
    if p < 0:
        raise ValueError("Require p >= 0.")

    state = {
        "calls": 0,
        "failures": 0,
        "last_prices": initial_downstream_prices,
        "cache_hits": 0,
        "cache_misses": 0,
    }
    cache: Dict[float, DownstreamEquilibriumResult] = {}

    def _cache_key(d: float) -> float:
        return float(round(float(d), int(cache_precision)))

    def objective(D: float) -> float:
        # Minimize negative of student total payoff.
        state["calls"] += 1

        if use_cache:
            key = _cache_key(float(D))
            if key in cache:
                eq = cache[key]
                state["cache_hits"] += 1
            else:
                state["cache_misses"] += 1
                try:
                    eq = solve_downstream_equilibrium_at_D(
                        D=float(D),
                        N=N,
                        tech=tech,
                        comp=comp,
                        sp=downstream_solver_params,
                        initial_prices=state["last_prices"],
                    )
                except Exception:
                    state["failures"] += 1
                    return 1e12 + (p + econ.k) * float(D)
                cache[key] = eq
        else:
            try:
                eq = solve_downstream_equilibrium_at_D(
                    D=float(D),
                    N=N,
                    tech=tech,
                    comp=comp,
                    sp=downstream_solver_params,
                    initial_prices=state["last_prices"],
                )
            except Exception:
                state["failures"] += 1
                return 1e12 + (p + econ.k) * float(D)

        if eq.success:
            state["last_prices"] = (eq.P_T_star, eq.P_S_star)
        else:
            state["failures"] += 1

        pi_student_total = float(eq.pi_S_down_star - (p + econ.k) * float(D))

        if not eq.success:
            # Penalize non-converged inner solves so optimizer favors reliable points.
            return -pi_student_total + 1e8 + 1e4 * float(eq.residual_norm)
        return -pi_student_total

    res = minimize_scalar(
        objective,
        bounds=(grids.D_min, grids.D_max),
        method="bounded",
        options={"xatol": solver.xtol, "maxiter": solver.max_iter},
    )

    D_star = float(res.x)
    if use_cache:
        key_star = _cache_key(D_star)
        if key_star in cache:
            eq_star = cache[key_star]
            state["cache_hits"] += 1
        else:
            state["cache_misses"] += 1
            eq_star = solve_downstream_equilibrium_at_D(
                D=D_star,
                N=N,
                tech=tech,
                comp=comp,
                sp=downstream_solver_params,
                initial_prices=state["last_prices"],
            )
            cache[key_star] = eq_star
    else:
        eq_star = solve_downstream_equilibrium_at_D(
            D=D_star,
            N=N,
            tech=tech,
            comp=comp,
            sp=downstream_solver_params,
            initial_prices=state["last_prices"],
        )
    if not eq_star.success:
        state["failures"] += 1

    pi_down = float(eq_star.pi_S_down_star)
    pi_total = float(pi_down - (p + econ.k) * D_star)

    tol = 1e-10 * max(1.0, grids.D_max)
    is_min = abs(D_star - grids.D_min) <= tol
    is_max = abs(D_star - grids.D_max) <= tol
    is_boundary = bool(is_min or is_max)
    boundary_side = "min" if is_min else ("max" if is_max else None)

    calls = int(state["calls"])
    failures = int(state["failures"])
    success_rate = (calls - failures) / max(1, calls)

    outer_success = bool(getattr(res, "success", True)) and bool(eq_star.success)
    outer_msg = str(getattr(res, "message", ""))
    if not eq_star.success:
        outer_msg = f"outer optimizer done, but downstream at D* failed: {eq_star.message}"

    return CompetitionStudentBestResponseResult(
        p=float(p),
        D_star=D_star,
        pi_student_total_star=pi_total,
        pi_student_down_star=pi_down,
        downstream_eq_at_star=eq_star,
        is_boundary=is_boundary,
        boundary_side=boundary_side,
        nfev=int(getattr(res, "nfev", -1)),
        success=outer_success,
        message=outer_msg,
        downstream_calls=calls,
        downstream_failures=failures,
        downstream_success_rate=float(success_rate),
        cache_hits=int(state["cache_hits"]),
        cache_misses=int(state["cache_misses"]),
    )
