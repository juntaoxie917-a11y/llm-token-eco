"""Stage 2: standalone downstream pricing-subgame solver.

This module solves the downstream Nash pricing game at fixed training level D.
It does NOT implement student best response over D or outer upstream price loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from .competition_static import (
    CompetitionParams,
    downstream_outcomes_from_prices,
    student_quality_from_loss,
)
from .scaling_laws import TierATechnology


@dataclass(frozen=True)
class DownstreamSolverParams:
    P_T_min: float
    P_T_max: float
    P_S_min: float
    P_S_max: float
    fd_eps: float
    root_tol: float
    max_nfev: int
    br_max_iter: int
    br_tol: float


@dataclass(frozen=True)
class DownstreamEquilibriumResult:
    D: float
    q_T: float
    q_S: float

    P_T_star: float
    P_S_star: float

    s_T_star: float
    s_S_star: float
    s_0_star: float

    pi_T_down_star: float
    pi_S_down_star: float

    success: bool
    message: str
    method_used: str
    nfev: int
    iterations: int
    residual_norm: float
    foc_T: float
    foc_S: float
    hit_bounds: bool


def build_downstream_solver_params_from_config(cfg: Dict[str, Any]) -> DownstreamSolverParams:
    """Load downstream solver controls from config with safe defaults."""
    scfg = cfg.get("competition", {}).get("downstream_solver", {})
    params = DownstreamSolverParams(
        P_T_min=float(scfg.get("P_T_min", 0.0)),
        P_T_max=float(scfg.get("P_T_max", 20.0)),
        P_S_min=float(scfg.get("P_S_min", 0.0)),
        P_S_max=float(scfg.get("P_S_max", 20.0)),
        fd_eps=float(scfg.get("fd_eps", 1e-5)),
        root_tol=float(scfg.get("root_tol", 1e-6)),
        max_nfev=int(scfg.get("max_nfev", 400)),
        br_max_iter=int(scfg.get("br_max_iter", 80)),
        br_tol=float(scfg.get("br_tol", 1e-7)),
    )
    validate_downstream_solver_params(params)
    return params


def validate_downstream_solver_params(params: DownstreamSolverParams) -> None:
    if not (params.P_T_max > params.P_T_min):
        raise ValueError("Require P_T_max > P_T_min.")
    if not (params.P_S_max > params.P_S_min):
        raise ValueError("Require P_S_max > P_S_min.")
    if params.fd_eps <= 0:
        raise ValueError("fd_eps must be > 0.")
    if params.root_tol <= 0:
        raise ValueError("root_tol must be > 0.")
    if params.max_nfev <= 0:
        raise ValueError("max_nfev must be > 0.")
    if params.br_max_iter <= 0:
        raise ValueError("br_max_iter must be > 0.")
    if params.br_tol <= 0:
        raise ValueError("br_tol must be > 0.")


def _q_s_from_D(*, D: float, N: float, tech: TierATechnology, comp: CompetitionParams) -> float:
    if D <= 0:
        raise ValueError("Require D > 0 in downstream solver.")
    loss = float(tech.L_student(N, D))
    return float(
        student_quality_from_loss(
            loss,
            mode=comp.quality_map,
            scale=comp.quality_scale,
            shift=comp.quality_shift,
        )
    )


def _payoffs(
    *,
    P_T: float,
    P_S: float,
    q_T: float,
    q_S: float,
    comp: CompetitionParams,
) -> Tuple[float, float, float, float, float]:
    shares, profits, _ = downstream_outcomes_from_prices(
        P_T=P_T,
        P_S=P_S,
        q_T=q_T,
        q_S=q_S,
        params=comp,
    )
    return (
        float(profits.pi_T_down),
        float(profits.pi_S_down),
        float(shares.s_T),
        float(shares.s_S),
        float(shares.s_0),
    )


def _focs(
    x: np.ndarray,
    *,
    q_T: float,
    q_S: float,
    comp: CompetitionParams,
    sp: DownstreamSolverParams,
) -> np.ndarray:
    P_T = float(x[0])
    P_S = float(x[1])
    h = sp.fd_eps

    # Central finite differences around candidate prices.
    piT_plus, _, _, _, _ = _payoffs(P_T=P_T + h, P_S=P_S, q_T=q_T, q_S=q_S, comp=comp)
    piT_minus, _, _, _, _ = _payoffs(P_T=P_T - h, P_S=P_S, q_T=q_T, q_S=q_S, comp=comp)
    d_piT_d_PT = (piT_plus - piT_minus) / (2.0 * h)

    _, piS_plus, _, _, _ = _payoffs(P_T=P_T, P_S=P_S + h, q_T=q_T, q_S=q_S, comp=comp)
    _, piS_minus, _, _, _ = _payoffs(P_T=P_T, P_S=P_S - h, q_T=q_T, q_S=q_S, comp=comp)
    d_piS_d_PS = (piS_plus - piS_minus) / (2.0 * h)

    return np.array([d_piT_d_PT, d_piS_d_PS], dtype=float)


def _best_response_fallback(
    *,
    q_T: float,
    q_S: float,
    comp: CompetitionParams,
    sp: DownstreamSolverParams,
    P_T_init: float,
    P_S_init: float,
) -> Tuple[float, float, int]:
    P_T = float(P_T_init)
    P_S = float(P_S_init)

    for it in range(1, sp.br_max_iter + 1):
        prev_T, prev_S = P_T, P_S

        def neg_piT(p_t: float) -> float:
            piT, _, _, _, _ = _payoffs(P_T=float(p_t), P_S=P_S, q_T=q_T, q_S=q_S, comp=comp)
            return -piT

        resT = minimize_scalar(
            neg_piT,
            bounds=(sp.P_T_min, sp.P_T_max),
            method="bounded",
            options={"xatol": sp.br_tol, "maxiter": 200},
        )
        P_T = float(resT.x)

        def neg_piS(p_s: float) -> float:
            _, piS, _, _, _ = _payoffs(P_T=P_T, P_S=float(p_s), q_T=q_T, q_S=q_S, comp=comp)
            return -piS

        resS = minimize_scalar(
            neg_piS,
            bounds=(sp.P_S_min, sp.P_S_max),
            method="bounded",
            options={"xatol": sp.br_tol, "maxiter": 200},
        )
        P_S = float(resS.x)

        if max(abs(P_T - prev_T), abs(P_S - prev_S)) <= sp.br_tol:
            return P_T, P_S, it

    return P_T, P_S, sp.br_max_iter


def solve_downstream_equilibrium_at_D(
    *,
    D: float,
    N: float,
    tech: TierATechnology,
    comp: CompetitionParams,
    sp: DownstreamSolverParams,
    initial_prices: Optional[Tuple[float, float]] = None,
) -> DownstreamEquilibriumResult:
    """Solve downstream Nash pricing equilibrium at fixed D.

    Primary method: bounded least-squares root search on two FOCs.
    Fallback method: iterative best responses with bounded scalar maximization.
    """
    q_S = _q_s_from_D(D=D, N=N, tech=tech, comp=comp)
    q_T = float(comp.q_T)

    if initial_prices is None:
        x0 = np.array([
            0.5 * (sp.P_T_min + sp.P_T_max),
            0.5 * (sp.P_S_min + sp.P_S_max),
        ])
    else:
        x0 = np.array([float(initial_prices[0]), float(initial_prices[1])], dtype=float)

    x0[0] = np.clip(x0[0], sp.P_T_min, sp.P_T_max)
    x0[1] = np.clip(x0[1], sp.P_S_min, sp.P_S_max)

    lsq = least_squares(
        lambda x: _focs(x, q_T=q_T, q_S=q_S, comp=comp, sp=sp),
        x0=x0,
        bounds=([sp.P_T_min, sp.P_S_min], [sp.P_T_max, sp.P_S_max]),
        xtol=sp.root_tol,
        ftol=sp.root_tol,
        gtol=sp.root_tol,
        max_nfev=sp.max_nfev,
    )

    x_lsq = np.array(lsq.x, dtype=float)
    foc_lsq = _focs(x_lsq, q_T=q_T, q_S=q_S, comp=comp, sp=sp)
    residual_lsq = float(np.linalg.norm(foc_lsq, ord=2))

    use_lsq = bool(lsq.success and residual_lsq <= 10.0 * sp.root_tol)

    if use_lsq:
        P_T_star, P_S_star = float(x_lsq[0]), float(x_lsq[1])
        method_used = "least_squares"
        iterations = int(getattr(lsq, "njev", 0) or 0)
        nfev = int(getattr(lsq, "nfev", 0) or 0)
        message = f"least_squares converged: {lsq.message}"
        foc_T, foc_S = float(foc_lsq[0]), float(foc_lsq[1])
        residual_norm = residual_lsq
        success = True
    else:
        P_T_star, P_S_star, br_iters = _best_response_fallback(
            q_T=q_T,
            q_S=q_S,
            comp=comp,
            sp=sp,
            P_T_init=float(x_lsq[0]),
            P_S_init=float(x_lsq[1]),
        )
        foc_br = _focs(np.array([P_T_star, P_S_star]), q_T=q_T, q_S=q_S, comp=comp, sp=sp)
        residual_norm = float(np.linalg.norm(foc_br, ord=2))
        foc_T, foc_S = float(foc_br[0]), float(foc_br[1])
        method_used = "best_response_fallback"
        iterations = int(br_iters)
        nfev = int(getattr(lsq, "nfev", 0) or 0)
        success = bool(residual_norm <= 50.0 * sp.root_tol)
        message = (
            "fallback used after least_squares non-convergence; "
            f"lsq_success={lsq.success}, lsq_residual={residual_lsq:.3e}, "
            f"fallback_residual={residual_norm:.3e}"
        )

    piT, piS, sT, sS, s0 = _payoffs(P_T=P_T_star, P_S=P_S_star, q_T=q_T, q_S=q_S, comp=comp)

    tol_b = 1e-8
    hit_bounds = bool(
        abs(P_T_star - sp.P_T_min) <= tol_b
        or abs(P_T_star - sp.P_T_max) <= tol_b
        or abs(P_S_star - sp.P_S_min) <= tol_b
        or abs(P_S_star - sp.P_S_max) <= tol_b
    )

    return DownstreamEquilibriumResult(
        D=float(D),
        q_T=q_T,
        q_S=q_S,
        P_T_star=float(P_T_star),
        P_S_star=float(P_S_star),
        s_T_star=float(sT),
        s_S_star=float(sS),
        s_0_star=float(s0),
        pi_T_down_star=float(piT),
        pi_S_down_star=float(piS),
        success=bool(success),
        message=str(message),
        method_used=str(method_used),
        nfev=int(nfev),
        iterations=int(iterations),
        residual_norm=float(residual_norm),
        foc_T=float(foc_T),
        foc_S=float(foc_S),
        hit_bounds=hit_bounds,
    )
