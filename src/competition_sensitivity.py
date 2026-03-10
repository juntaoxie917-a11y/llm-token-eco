"""Stage 3: non-invasive 1D sensitivity runners for competition model.

This module wraps existing competition simulation logic and does not modify
solver mechanics. It runs comparative-statics sweeps by overriding one
parameter at a time in-memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable, List, Optional, Sequence

from .competition_downstream_solver import DownstreamSolverParams
from .competition_static import CompetitionParams
from .competition_threshold import ThresholdInteriorSettings, evaluate_market_size_once
from .scaling_laws import TierATechnology


@dataclass(frozen=True)
class CompetitionSensitivityRow:
    parameter_name: str
    parameter_value: float
    M: float

    equilibrium_exists: bool
    interior_equilibrium: bool
    interior_equilibrium_weak: Optional[bool]

    p_star: Optional[float]
    D_star_at_p_star: Optional[float]

    pi_teacher_star: Optional[float]
    pi_teacher_upstream_at_p_star: Optional[float]
    pi_teacher_downstream_at_p_star: Optional[float]
    pi_student_star: Optional[float]

    P_T_star: Optional[float]
    P_S_star: Optional[float]
    s_T_star: Optional[float]
    s_S_star: Optional[float]
    s_0_star: Optional[float]

    success: bool
    message: str

    teacher_solver_ok: bool
    student_solver_ok: bool
    downstream_solver_ok: bool

    teacher_interior: bool
    student_interior: bool
    downstream_interior: bool

    teacher_reason: Optional[str]
    student_reason: Optional[str]
    downstream_reason: Optional[str]

    used_fallback: bool
    downstream_residual: Optional[float]
    min_share: Optional[float]
    price_distance_to_boundary: Optional[float]
    demand_distance_to_boundary: Optional[float]

    br_success_rate: float
    down_success_rate: float
    boundary_share: float


@dataclass(frozen=True)
class CompetitionSensitivitySweepResult:
    parameter_name: str
    rows: List[CompetitionSensitivityRow]


def _message_from_reasons(reasons: Sequence[str]) -> str:
    if not reasons:
        return "ok"
    return "; ".join(str(r) for r in reasons)


def _evaluate_parameter_once(
    *,
    cfg: dict,
    tech: TierATechnology,
    N: float,
    comp: CompetitionParams,
    downstream_solver_params: DownstreamSolverParams,
    threshold_settings: ThresholdInteriorSettings,
    parameter_name: str,
    parameter_value: float,
    p_grid_override: Optional[Sequence[float]] = None,
    include_weak: bool = True,
    use_student_cache: bool = True,
    student_cache_precision: int = 8,
) -> CompetitionSensitivityRow:
    # Reuse the same single-point evaluator as the M-threshold workflow to
    # guarantee identical interior-equilibrium classification semantics.
    m_eval = evaluate_market_size_once(
        cfg=cfg,
        tech=tech,
        N=float(N),
        base_comp=comp,
        downstream_solver_params=downstream_solver_params,
        market_size=float(comp.M),
        p_grid_override=p_grid_override,
        threshold_settings=threshold_settings,
        use_student_cache=use_student_cache,
        student_cache_precision=student_cache_precision,
        include_weak=include_weak,
    )

    equilibrium_exists = bool(m_eval.teacher_solver_ok)
    success = bool(m_eval.teacher_solver_ok and m_eval.student_solver_ok and m_eval.downstream_solver_ok)
    message = _message_from_reasons(m_eval.reasons)

    return CompetitionSensitivityRow(
        parameter_name=str(parameter_name),
        parameter_value=float(parameter_value),
        M=float(m_eval.market_size),
        equilibrium_exists=equilibrium_exists,
        interior_equilibrium=bool(m_eval.overall_interior_strict),
        interior_equilibrium_weak=m_eval.overall_interior_weak,
        p_star=m_eval.p_star,
        D_star_at_p_star=m_eval.D_star,
        pi_teacher_star=m_eval.pi_teacher_total_star,
        pi_teacher_upstream_at_p_star=m_eval.pi_teacher_upstream_at_p_star,
        pi_teacher_downstream_at_p_star=m_eval.pi_teacher_downstream_at_p_star,
        pi_student_star=m_eval.pi_student_total_at_p_star,
        P_T_star=m_eval.P_T_star,
        P_S_star=m_eval.P_S_star,
        s_T_star=m_eval.s_T,
        s_S_star=m_eval.s_S,
        s_0_star=m_eval.s_0,
        success=success,
        message=message,
        teacher_solver_ok=bool(m_eval.teacher_solver_ok),
        student_solver_ok=bool(m_eval.student_solver_ok),
        downstream_solver_ok=bool(m_eval.downstream_solver_ok),
        teacher_interior=bool(m_eval.teacher_interior),
        student_interior=bool(m_eval.student_interior),
        downstream_interior=bool(m_eval.downstream_interior),
        teacher_reason=m_eval.teacher_reason,
        student_reason=m_eval.student_reason,
        downstream_reason=m_eval.downstream_reason,
        used_fallback=bool(m_eval.used_fallback),
        downstream_residual=m_eval.downstream_residual,
        min_share=m_eval.min_share,
        price_distance_to_boundary=m_eval.price_distance_to_boundary,
        demand_distance_to_boundary=m_eval.demand_distance_to_boundary,
        br_success_rate=float(m_eval.br_success_rate),
        down_success_rate=float(m_eval.down_success_rate),
        boundary_share=float(m_eval.boundary_share),
    )


def _run_single_parameter_sensitivity(
    *,
    cfg: dict,
    tech: TierATechnology,
    N: float,
    base_comp: CompetitionParams,
    downstream_solver_params: DownstreamSolverParams,
    threshold_settings: ThresholdInteriorSettings,
    parameter_name: str,
    parameter_grid: Iterable[float],
    p_grid_override: Optional[Sequence[float]] = None,
    include_weak: bool = True,
    use_student_cache: bool = True,
    student_cache_precision: int = 8,
) -> CompetitionSensitivitySweepResult:
    rows: List[CompetitionSensitivityRow] = []

    for value in parameter_grid:
        v = float(value)
        if parameter_name == "u0":
            comp_local = replace(base_comp, u0=v)
        elif parameter_name == "tau":
            if v <= 0:
                raise ValueError("tau sweep values must be > 0.")
            comp_local = replace(base_comp, tau=v)
        else:
            raise ValueError("parameter_name must be 'u0' or 'tau'.")

        row = _evaluate_parameter_once(
            cfg=cfg,
            tech=tech,
            N=N,
            comp=comp_local,
            downstream_solver_params=downstream_solver_params,
            threshold_settings=threshold_settings,
            parameter_name=parameter_name,
            parameter_value=v,
            p_grid_override=p_grid_override,
            include_weak=include_weak,
            use_student_cache=use_student_cache,
            student_cache_precision=student_cache_precision,
        )
        rows.append(row)

    rows = sorted(rows, key=lambda r: float(r.parameter_value))
    return CompetitionSensitivitySweepResult(parameter_name=parameter_name, rows=rows)


def run_u0_sensitivity(
    *,
    cfg: dict,
    tech: TierATechnology,
    N: float,
    base_comp: CompetitionParams,
    downstream_solver_params: DownstreamSolverParams,
    threshold_settings: ThresholdInteriorSettings,
    u0_grid: Iterable[float],
    p_grid_override: Optional[Sequence[float]] = None,
    include_weak: bool = True,
    use_student_cache: bool = True,
    student_cache_precision: int = 8,
) -> CompetitionSensitivitySweepResult:
    """Run 1D sensitivity sweep over outside-option utility `u0`."""
    return _run_single_parameter_sensitivity(
        cfg=cfg,
        tech=tech,
        N=N,
        base_comp=base_comp,
        downstream_solver_params=downstream_solver_params,
        threshold_settings=threshold_settings,
        parameter_name="u0",
        parameter_grid=u0_grid,
        p_grid_override=p_grid_override,
        include_weak=include_weak,
        use_student_cache=use_student_cache,
        student_cache_precision=student_cache_precision,
    )


def run_tau_sensitivity(
    *,
    cfg: dict,
    tech: TierATechnology,
    N: float,
    base_comp: CompetitionParams,
    downstream_solver_params: DownstreamSolverParams,
    threshold_settings: ThresholdInteriorSettings,
    tau_grid: Iterable[float],
    p_grid_override: Optional[Sequence[float]] = None,
    include_weak: bool = True,
    use_student_cache: bool = True,
    student_cache_precision: int = 8,
) -> CompetitionSensitivitySweepResult:
    """Run 1D sensitivity sweep over logit price sensitivity `tau`."""
    return _run_single_parameter_sensitivity(
        cfg=cfg,
        tech=tech,
        N=N,
        base_comp=base_comp,
        downstream_solver_params=downstream_solver_params,
        threshold_settings=threshold_settings,
        parameter_name="tau",
        parameter_grid=tau_grid,
        p_grid_override=p_grid_override,
        include_weak=include_weak,
        use_student_cache=use_student_cache,
        student_cache_precision=student_cache_precision,
    )


def sensitivity_results_to_records(rows: Sequence[CompetitionSensitivityRow]) -> List[dict]:
    """Convert sensitivity rows to plain records without extra dependencies."""
    return [asdict(r) for r in rows]
