"""Economic model block for baseline (and future extensions).

Step 3 responsibility:
- Define payoffs given a Technology block.
- Provide a robust 1D solver for the student's best response D*(p) via direct maximization.

Design goals:
- Keep payoffs separate from the scaling-law technology implementation.
- Provide diagnostics that help validate interior solutions and catch boundary optima.
- Make the solver reusable in extensions (competition may require multiple best responses).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

# SciPy is recommended for robust optimization; if unavailable, we can fall back later.
try:
    from scipy.optimize import minimize_scalar
except ImportError as e:
    raise ImportError(
        "scipy is required for Step 3 (direct maximization). Install with: pip install scipy"
    ) from e

from .scaling_laws import TierATechnology


@dataclass(frozen=True)
class EconomicsParams:
    """Baseline economic parameters."""
    a: float  # level term in V(L)=a-bL (does not affect choices)
    b: float  # slope (must be >0)
    k: float  # student compute cost per token
    c_T: float  # teacher marginal cost per token


@dataclass(frozen=True)
class GridsParams:
    """Bounds and grids used in simulation."""
    D_min: float
    D_max: float


@dataclass(frozen=True)
class SolverParams:
    """Solver controls for the direct maximization routine."""
    xtol: float
    max_iter: int


def build_params_from_config(cfg: Dict[str, Any]) -> Tuple[EconomicsParams, GridsParams, SolverParams]:
    """Extract economic, grid, and solver parameters from config."""
    econ = cfg["economics"]
    grids = cfg["grids"]
    solver = cfg["solver"]

    ep = EconomicsParams(
        a=float(econ.get("a", 0.0)),
        b=float(econ["b"]),
        k=float(econ["k"]),
        c_T=float(econ["c_T"]),
    )
    if ep.b <= 0:
        raise ValueError("economics.b must be > 0.")
    if ep.k < 0:
        raise ValueError("economics.k must be >= 0.")
    if ep.c_T < 0:
        raise ValueError("economics.c_T must be >= 0.")

    gp = GridsParams(
        D_min=float(grids["D_min"]),
        D_max=float(grids["D_max"]),
    )
    if not (0 < gp.D_min < gp.D_max):
        raise ValueError("Require 0 < D_min < D_max.")

    sp = SolverParams(
        xtol=float(solver.get("xtol", 1e-10)),
        max_iter=int(solver.get("max_iter", 200)),
    )
    return ep, gp, sp


# ---- Value and profit functions ----
def downstream_value_linear(L: float, *, a: float, b: float) -> float:
    """Baseline downstream value function: V(L) = a - b L."""
    return a - b * L


def student_profit(D: float, *, N: float, p: float, tech: TierATechnology, econ: EconomicsParams) -> float:
    """Student profit: Pi_S(D)=V(L_student(N,D)) - (p+k)D."""
    if D <= 0:
        # infeasible; return very low utility to keep optimizer away
        return -1e300
    Ls = tech.L_student(N, D)
    V = downstream_value_linear(Ls, a=econ.a, b=econ.b)
    return V - (p + econ.k) * D


@dataclass(frozen=True)
class StudentBestResponseResult:
    D_star: float
    pi_star: float
    is_boundary: bool
    boundary_side: Optional[str]  # "min", "max", or None
    nfev: int
    success: bool
    message: str


def solve_student_best_response_direct(
    *,
    N: float,
    p: float,
    tech: TierATechnology,
    econ: EconomicsParams,
    grids: GridsParams,
    solver: SolverParams,
) -> StudentBestResponseResult:
    """Compute D*(p) by directly maximizing student profit over [D_min, D_max].

    We maximize Pi_S(D) equivalently by minimizing -Pi_S(D).

    Diagnostics:
    - whether the optimum lies at a boundary (D_min or D_max)
    - function evaluation count
    """
    if p < 0:
        raise ValueError("Require p>=0.")

    def objective(D: float) -> float:
        return -student_profit(D, N=N, p=p, tech=tech, econ=econ)

    res = minimize_scalar(
        objective,
        bounds=(grids.D_min, grids.D_max),
        method="bounded",
        options={"xatol": solver.xtol, "maxiter": solver.max_iter},
    )
    D_star = float(res.x)
    pi_star = -float(res.fun)

    # boundary detection (tolerance)
    tol = 1e-10 * max(1.0, grids.D_max)
    is_min = abs(D_star - grids.D_min) <= tol
    is_max = abs(D_star - grids.D_max) <= tol
    is_boundary = bool(is_min or is_max)
    boundary_side = "min" if is_min else ("max" if is_max else None)

    msg = getattr(res, "message", "")
    return StudentBestResponseResult(
        D_star=D_star,
        pi_star=pi_star,
        is_boundary=is_boundary,
        boundary_side=boundary_side,
        nfev=int(getattr(res, "nfev", -1)),
        success=bool(getattr(res, "success", True)),
        message=str(msg),
    )


# ---- Teacher profit (given student demand) ----
def teacher_profit(p: float, *, D_of_p: float, econ: EconomicsParams) -> float:
    """Teacher profit: Pi_T(p) = (p - c_T) * D(p)."""
    return (p - econ.c_T) * D_of_p
