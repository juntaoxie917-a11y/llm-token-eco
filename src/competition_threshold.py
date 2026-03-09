"""Stage 1: reusable interior-equilibrium classification helpers.

This module classifies whether one full competition-run equilibrium is interior
under a three-layer criterion:
1) teacher outer optimum interiority,
2) student best-response interiority,
3) downstream subgame interiority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .competition_downstream_solver import DownstreamSolverParams
from .competition_simulation import run_competition_grid_simulation
from .competition_simulation import CompetitionSimulationResult, CompetitionSimulationRow
from .competition_static import CompetitionParams
from .model import GridsParams
from .scaling_laws import TierATechnology
from .simulation import SimulationGrids


@dataclass(frozen=True)
class ThresholdInteriorSettings:
    """Tolerances for strict/weak interior-equilibrium classification."""

    price_boundary_tol: float = 1e-8
    d_boundary_tol: float = 1e-8
    downstream_price_boundary_tol: float = 1e-8
    share_tol: float = 1e-8
    solver_residual_tol: float = 1e-4
    weak_share_tol: float = 1e-10


@dataclass(frozen=True)
class InteriorClassification:
    overall_interior_strict: bool
    overall_interior_weak: Optional[bool]

    teacher_interior: bool
    student_interior: bool
    downstream_interior: bool

    teacher_reason: Optional[str]
    student_reason: Optional[str]
    downstream_reason: Optional[str]

    p_star: Optional[float]
    D_star: Optional[float]
    P_T_star: Optional[float]
    P_S_star: Optional[float]
    s_T: Optional[float]
    s_S: Optional[float]
    s_0: Optional[float]

    teacher_solver_ok: bool
    student_solver_ok: bool
    downstream_solver_ok: bool

    used_fallback: bool
    min_share: Optional[float]
    price_distance_to_boundary: Optional[float]
    demand_distance_to_boundary: Optional[float]
    downstream_residual: Optional[float]

    teacher_price_at_lower_boundary: bool
    teacher_price_at_upper_boundary: bool
    student_D_at_lower_boundary: bool
    student_D_at_upper_boundary: bool
    downstream_price_at_boundary: bool

    share_teacher_positive: bool
    share_student_positive: bool
    share_outside_positive: bool

    reasons: List[str]


@dataclass(frozen=True)
class MarketSizeEvaluationResult:
    market_size: float

    overall_interior_strict: bool
    overall_interior_weak: Optional[bool]
    reasons: List[str]

    teacher_interior: bool
    student_interior: bool
    downstream_interior: bool

    p_star: Optional[float]
    D_star: Optional[float]
    P_T_star: Optional[float]
    P_S_star: Optional[float]
    s_T: Optional[float]
    s_S: Optional[float]
    s_0: Optional[float]

    pi_teacher_total_star: Optional[float]
    pi_teacher_upstream_at_p_star: Optional[float]
    pi_teacher_downstream_at_p_star: Optional[float]
    pi_student_total_at_p_star: Optional[float]

    teacher_price_at_lower_boundary: bool
    teacher_price_at_upper_boundary: bool
    student_D_at_lower_boundary: bool
    student_D_at_upper_boundary: bool
    downstream_price_at_boundary: bool

    teacher_solver_ok: bool
    student_solver_ok: bool
    downstream_solver_ok: bool

    used_fallback: bool
    downstream_residual: Optional[float]
    min_share: Optional[float]
    price_distance_to_boundary: Optional[float]
    demand_distance_to_boundary: Optional[float]

    br_success_rate: float
    down_success_rate: float
    boundary_share: float

    teacher_reason: Optional[str]
    student_reason: Optional[str]
    downstream_reason: Optional[str]


@dataclass(frozen=True)
class ThresholdPatternSummary:
    is_monotone_nonincreasing: bool
    transition_count: int
    transition_intervals: List[Tuple[float, float]]
    supports_single_threshold: bool
    message: str

    # Failure-cause summary near transitions.
    near_transition_failure_cause_counts: Dict[str, int]


@dataclass(frozen=True)
class MarketSizeSweepResult:
    rows: List[MarketSizeEvaluationResult]
    pattern: ThresholdPatternSummary


@dataclass(frozen=True)
class ThresholdRefinementSummary:
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    midpoint_estimate: Optional[float]
    interval_width: Optional[float]
    steps: int
    trustworthy: bool
    skipped: bool
    message: str


@dataclass(frozen=True)
class ThresholdRefinementResult:
    summary: ThresholdRefinementSummary
    history: List[MarketSizeEvaluationResult]


def _first_reason(reasons: Sequence[str], allowed_prefix: str) -> Optional[str]:
    for reason in reasons:
        if reason.startswith(allowed_prefix):
            return reason
    return None


def _find_star_row(sim: CompetitionSimulationResult) -> Optional[CompetitionSimulationRow]:
    if not sim.rows:
        return None

    # Prefer exact p* row; fallback to teacher-profit argmax if needed.
    for row in sim.rows:
        if abs(float(row.p) - float(sim.p_star)) <= 1e-12:
            return row

    return max(sim.rows, key=lambda r: float(r.pi_teacher_total))


def classify_interior_equilibrium(
    *,
    sim: CompetitionSimulationResult,
    sim_grids: SimulationGrids,
    model_grids: GridsParams,
    downstream_solver_params: DownstreamSolverParams,
    threshold_settings: ThresholdInteriorSettings,
    include_weak: bool = True,
) -> InteriorClassification:
    """Classify interiority for one full competition simulation result.

    Args:
        sim: Full competition run output (rows + optimum summary).
        sim_grids: Upstream price grid used in the run.
        model_grids: Student demand bounds used by the solver.
        downstream_solver_params: Downstream price bounds/tolerances.
        threshold_settings: Interiority tolerances.
        include_weak: If True, return a weak diagnostic classification too.
    """
    reasons: List[str] = []

    if not sim.rows:
        reasons.append("teacher_solver_failed")
        teacher_reason = "teacher_solver_failed"
        return InteriorClassification(
            overall_interior_strict=False,
            overall_interior_weak=False if include_weak else None,
            teacher_interior=False,
            student_interior=False,
            downstream_interior=False,
            teacher_reason=teacher_reason,
            student_reason="student_solver_failed",
            downstream_reason="downstream_solver_failed",
            p_star=None,
            D_star=None,
            P_T_star=None,
            P_S_star=None,
            s_T=None,
            s_S=None,
            s_0=None,
            teacher_solver_ok=False,
            student_solver_ok=False,
            downstream_solver_ok=False,
            used_fallback=False,
            min_share=None,
            price_distance_to_boundary=None,
            demand_distance_to_boundary=None,
            downstream_residual=None,
            teacher_price_at_lower_boundary=False,
            teacher_price_at_upper_boundary=False,
            student_D_at_lower_boundary=False,
            student_D_at_upper_boundary=False,
            downstream_price_at_boundary=False,
            share_teacher_positive=False,
            share_student_positive=False,
            share_outside_positive=False,
            reasons=reasons,
        )

    row = _find_star_row(sim)
    if row is None:
        reasons.append("teacher_solver_failed")
        return InteriorClassification(
            overall_interior_strict=False,
            overall_interior_weak=False if include_weak else None,
            teacher_interior=False,
            student_interior=False,
            downstream_interior=False,
            teacher_reason="teacher_solver_failed",
            student_reason="student_solver_failed",
            downstream_reason="downstream_solver_failed",
            p_star=None,
            D_star=None,
            P_T_star=None,
            P_S_star=None,
            s_T=None,
            s_S=None,
            s_0=None,
            teacher_solver_ok=False,
            student_solver_ok=False,
            downstream_solver_ok=False,
            used_fallback=False,
            min_share=None,
            price_distance_to_boundary=None,
            demand_distance_to_boundary=None,
            downstream_residual=None,
            teacher_price_at_lower_boundary=False,
            teacher_price_at_upper_boundary=False,
            student_D_at_lower_boundary=False,
            student_D_at_upper_boundary=False,
            downstream_price_at_boundary=False,
            share_teacher_positive=False,
            share_student_positive=False,
            share_outside_positive=False,
            reasons=reasons,
        )

    p_star = float(row.p)
    d_star = float(row.D_star)

    p_min = float(sim_grids.p_grid.min())
    p_max = float(sim_grids.p_grid.max())
    d_min = float(model_grids.D_min)
    d_max = float(model_grids.D_max)

    dist_p_low = p_star - p_min
    dist_p_high = p_max - p_star
    dist_d_low = d_star - d_min
    dist_d_high = d_max - d_star

    teacher_price_at_lower_boundary = dist_p_low <= threshold_settings.price_boundary_tol
    teacher_price_at_upper_boundary = dist_p_high <= threshold_settings.price_boundary_tol

    student_D_at_lower_boundary = dist_d_low <= threshold_settings.d_boundary_tol
    student_D_at_upper_boundary = dist_d_high <= threshold_settings.d_boundary_tol

    p_t_star = float(row.P_T_down_star)
    p_s_star = float(row.P_S_down_star)

    dist_pt_low = p_t_star - float(downstream_solver_params.P_T_min)
    dist_pt_high = float(downstream_solver_params.P_T_max) - p_t_star
    dist_ps_low = p_s_star - float(downstream_solver_params.P_S_min)
    dist_ps_high = float(downstream_solver_params.P_S_max) - p_s_star

    downstream_price_at_boundary = bool(
        row.down_hit_bounds
        or dist_pt_low <= threshold_settings.downstream_price_boundary_tol
        or dist_pt_high <= threshold_settings.downstream_price_boundary_tol
        or dist_ps_low <= threshold_settings.downstream_price_boundary_tol
        or dist_ps_high <= threshold_settings.downstream_price_boundary_tol
    )

    s_t = float(row.s_T_down_star)
    s_s = float(row.s_S_down_star)
    s_0 = float(row.s_0_down_star)
    min_share = min(s_t, s_s, s_0)

    share_teacher_positive = s_t > threshold_settings.share_tol
    share_student_positive = s_s > threshold_settings.share_tol
    share_outside_positive = s_0 > threshold_settings.share_tol

    student_solver_ok = bool(row.br_success)
    downstream_solver_ok = bool(row.down_success)
    teacher_solver_ok = bool(len(sim.rows) > 0)

    used_fallback = str(row.down_method) == "best_response_fallback"
    residual_too_large = float(row.down_residual_norm) > float(threshold_settings.solver_residual_tol)

    if teacher_price_at_lower_boundary:
        reasons.append("teacher_price_at_lower_boundary")
    if teacher_price_at_upper_boundary:
        reasons.append("teacher_price_at_upper_boundary")

    if not student_solver_ok:
        reasons.append("student_solver_failed")
    if student_D_at_lower_boundary:
        reasons.append("student_D_at_lower_boundary")
    if student_D_at_upper_boundary:
        reasons.append("student_D_at_upper_boundary")

    if not downstream_solver_ok:
        reasons.append("downstream_solver_failed")
    if used_fallback:
        reasons.append("downstream_used_fallback")
    if downstream_price_at_boundary:
        reasons.append("downstream_price_at_boundary")
    if not (share_teacher_positive and share_student_positive and share_outside_positive):
        reasons.append("downstream_share_too_small")
    if residual_too_large:
        reasons.append("residual_too_large")

    teacher_interior = not (teacher_price_at_lower_boundary or teacher_price_at_upper_boundary)
    student_interior = bool(student_solver_ok and not (student_D_at_lower_boundary or student_D_at_upper_boundary))
    downstream_interior = bool(
        downstream_solver_ok
        and (not used_fallback)
        and (not downstream_price_at_boundary)
        and share_teacher_positive
        and share_student_positive
        and share_outside_positive
        and (not residual_too_large)
    )

    overall_interior_strict = bool(teacher_interior and student_interior and downstream_interior)

    if include_weak:
        weak_share_teacher_positive = s_t > threshold_settings.weak_share_tol
        weak_share_student_positive = s_s > threshold_settings.weak_share_tol
        weak_share_outside_positive = s_0 > threshold_settings.weak_share_tol

        overall_interior_weak = bool(
            teacher_interior
            and student_solver_ok
            and (not student_D_at_lower_boundary)
            and (not student_D_at_upper_boundary)
            and downstream_solver_ok
            and (not downstream_price_at_boundary)
            and weak_share_teacher_positive
            and weak_share_student_positive
            and weak_share_outside_positive
        )
    else:
        overall_interior_weak = None

    teacher_reason = _first_reason(reasons, "teacher_")
    student_reason = _first_reason(reasons, "student_")
    downstream_reason = _first_reason(reasons, "downstream_")
    if downstream_reason is None and "residual_too_large" in reasons:
        downstream_reason = "residual_too_large"

    return InteriorClassification(
        overall_interior_strict=overall_interior_strict,
        overall_interior_weak=overall_interior_weak,
        teacher_interior=teacher_interior,
        student_interior=student_interior,
        downstream_interior=downstream_interior,
        teacher_reason=teacher_reason,
        student_reason=student_reason,
        downstream_reason=downstream_reason,
        p_star=p_star,
        D_star=d_star,
        P_T_star=p_t_star,
        P_S_star=p_s_star,
        s_T=s_t,
        s_S=s_s,
        s_0=s_0,
        teacher_solver_ok=teacher_solver_ok,
        student_solver_ok=student_solver_ok,
        downstream_solver_ok=downstream_solver_ok,
        used_fallback=used_fallback,
        min_share=min_share,
        price_distance_to_boundary=min(dist_p_low, dist_p_high),
        demand_distance_to_boundary=min(dist_d_low, dist_d_high),
        downstream_residual=float(row.down_residual_norm),
        teacher_price_at_lower_boundary=teacher_price_at_lower_boundary,
        teacher_price_at_upper_boundary=teacher_price_at_upper_boundary,
        student_D_at_lower_boundary=student_D_at_lower_boundary,
        student_D_at_upper_boundary=student_D_at_upper_boundary,
        downstream_price_at_boundary=downstream_price_at_boundary,
        share_teacher_positive=share_teacher_positive,
        share_student_positive=share_student_positive,
        share_outside_positive=share_outside_positive,
        reasons=reasons,
    )


def evaluate_market_size_once(
    *,
    cfg: dict,
    tech: TierATechnology,
    N: float,
    base_comp: CompetitionParams,
    downstream_solver_params: DownstreamSolverParams,
    market_size: float,
    threshold_settings: ThresholdInteriorSettings,
    p_grid_override: Optional[Sequence[float]] = None,
    include_weak: bool = True,
    use_student_cache: bool = True,
    student_cache_precision: int = 8,
) -> MarketSizeEvaluationResult:
    """Evaluate the competition model at one market size and classify interiority.

    This wrapper changes only downstream market size `M` and reuses the existing
    competition simulation runner as a black box.
    """
    m_val = float(market_size)
    if m_val <= 0:
        raise ValueError("market_size must be > 0.")

    comp_local = replace(base_comp, M=m_val)

    sim, sim_grids, (_econ, grids, _solver) = run_competition_grid_simulation(
        cfg=cfg,
        tech=tech,
        N=float(N),
        comp=comp_local,
        downstream_solver_params=downstream_solver_params,
        p_grid_override=p_grid_override,
        use_student_cache=use_student_cache,
        student_cache_precision=student_cache_precision,
    )

    cls = classify_interior_equilibrium(
        sim=sim,
        sim_grids=sim_grids,
        model_grids=grids,
        downstream_solver_params=downstream_solver_params,
        threshold_settings=threshold_settings,
        include_weak=include_weak,
    )

    row_star = _find_star_row(sim)
    pi_student_total_at_p_star = float(row_star.pi_student_total) if row_star is not None else None

    return MarketSizeEvaluationResult(
        market_size=m_val,
        overall_interior_strict=cls.overall_interior_strict,
        overall_interior_weak=cls.overall_interior_weak,
        reasons=list(cls.reasons),
        teacher_interior=cls.teacher_interior,
        student_interior=cls.student_interior,
        downstream_interior=cls.downstream_interior,
        p_star=cls.p_star,
        D_star=cls.D_star,
        P_T_star=cls.P_T_star,
        P_S_star=cls.P_S_star,
        s_T=cls.s_T,
        s_S=cls.s_S,
        s_0=cls.s_0,
        pi_teacher_total_star=float(sim.pi_teacher_total_star),
        pi_teacher_upstream_at_p_star=float(sim.pi_teacher_upstream_at_p_star),
        pi_teacher_downstream_at_p_star=float(sim.pi_teacher_downstream_at_p_star),
        pi_student_total_at_p_star=pi_student_total_at_p_star,
        teacher_price_at_lower_boundary=cls.teacher_price_at_lower_boundary,
        teacher_price_at_upper_boundary=cls.teacher_price_at_upper_boundary,
        student_D_at_lower_boundary=cls.student_D_at_lower_boundary,
        student_D_at_upper_boundary=cls.student_D_at_upper_boundary,
        downstream_price_at_boundary=cls.downstream_price_at_boundary,
        teacher_solver_ok=cls.teacher_solver_ok,
        student_solver_ok=cls.student_solver_ok,
        downstream_solver_ok=cls.downstream_solver_ok,
        used_fallback=cls.used_fallback,
        downstream_residual=cls.downstream_residual,
        min_share=cls.min_share,
        price_distance_to_boundary=cls.price_distance_to_boundary,
        demand_distance_to_boundary=cls.demand_distance_to_boundary,
        br_success_rate=float(sim.br_success_rate),
        down_success_rate=float(sim.down_success_rate),
        boundary_share=float(sim.boundary_share),
        teacher_reason=cls.teacher_reason,
        student_reason=cls.student_reason,
        downstream_reason=cls.downstream_reason,
    )


def _classify_failure_cause(row: MarketSizeEvaluationResult) -> str:
    teacher_issue = bool(row.teacher_price_at_lower_boundary or row.teacher_price_at_upper_boundary)
    student_issue = bool(row.student_D_at_lower_boundary or row.student_D_at_upper_boundary)
    downstream_issue = bool(
        (not row.downstream_solver_ok)
        or row.downstream_price_at_boundary
        or (row.downstream_reason is not None)
        or ("downstream_share_too_small" in row.reasons)
        or ("residual_too_large" in row.reasons)
    )

    active = int(teacher_issue) + int(student_issue) + int(downstream_issue)
    if active >= 2:
        return "mixed"
    if teacher_issue:
        return "teacher_boundary"
    if student_issue:
        return "student_boundary"
    if downstream_issue:
        return "downstream_failure"
    return "unknown"


def check_threshold_pattern(rows: Sequence[MarketSizeEvaluationResult]) -> ThresholdPatternSummary:
    """Check whether strict interior classification matches a single-threshold pattern.

    Expected pattern for a clean threshold interpretation:
    - small market sizes: strict interior=True
    - large market sizes: strict interior=False
    - at most one True->False transition.
    """
    if not rows:
        return ThresholdPatternSummary(
            is_monotone_nonincreasing=False,
            transition_count=0,
            transition_intervals=[],
            supports_single_threshold=False,
            message="No rows provided.",
            near_transition_failure_cause_counts={},
        )

    rows_sorted = sorted(rows, key=lambda r: float(r.market_size))
    flags = [bool(r.overall_interior_strict) for r in rows_sorted]

    transitions_idx: List[int] = []
    for i in range(len(flags) - 1):
        if flags[i] != flags[i + 1]:
            transitions_idx.append(i)

    monotone_nonincreasing = all((not flags[i + 1]) or flags[i] for i in range(len(flags) - 1))

    transition_intervals: List[Tuple[float, float]] = [
        (float(rows_sorted[i].market_size), float(rows_sorted[i + 1].market_size))
        for i in transitions_idx
    ]

    has_true = any(flags)
    has_false = any(not x for x in flags)
    supports_single_threshold = bool(
        monotone_nonincreasing and len(transition_intervals) <= 1 and has_true and has_false
    )

    cause_counts: Dict[str, int] = {}
    for i in transitions_idx:
        lo = max(0, i - 1)
        hi = min(len(rows_sorted) - 1, i + 2)
        for j in range(lo, hi + 1):
            row = rows_sorted[j]
            if row.overall_interior_strict:
                continue
            cause = _classify_failure_cause(row)
            cause_counts[cause] = int(cause_counts.get(cause, 0) + 1)

    if supports_single_threshold:
        msg = "Pattern is consistent with a single threshold (strict classification)."
    elif monotone_nonincreasing and (not has_true or not has_false):
        msg = "Pattern is monotone but no interior/non-interior switch is observed in this grid."
    elif not monotone_nonincreasing:
        msg = "Pattern is non-monotone; single-threshold interpretation is not supported."
    else:
        msg = "Pattern is not suitable for a unique-threshold claim."

    return ThresholdPatternSummary(
        is_monotone_nonincreasing=bool(monotone_nonincreasing),
        transition_count=int(len(transition_intervals)),
        transition_intervals=transition_intervals,
        supports_single_threshold=supports_single_threshold,
        message=msg,
        near_transition_failure_cause_counts=cause_counts,
    )


def run_market_size_sweep(
    *,
    cfg: dict,
    tech: TierATechnology,
    N: float,
    base_comp: CompetitionParams,
    downstream_solver_params: DownstreamSolverParams,
    market_size_grid: Iterable[float],
    threshold_settings: ThresholdInteriorSettings,
    p_grid_override: Optional[Sequence[float]] = None,
    include_weak: bool = True,
    use_student_cache: bool = True,
    student_cache_precision: int = 8,
    output_csv_path: Optional[str] = None,
) -> MarketSizeSweepResult:
    """Run a coarse market-size sweep using the Stage-2 single-point wrapper."""
    rows: List[MarketSizeEvaluationResult] = []
    for m in market_size_grid:
        row = evaluate_market_size_once(
            cfg=cfg,
            tech=tech,
            N=N,
            base_comp=base_comp,
            downstream_solver_params=downstream_solver_params,
            market_size=float(m),
            threshold_settings=threshold_settings,
            p_grid_override=p_grid_override,
            include_weak=include_weak,
            use_student_cache=use_student_cache,
            student_cache_precision=student_cache_precision,
        )
        rows.append(row)

    rows = sorted(rows, key=lambda r: float(r.market_size))
    pattern = check_threshold_pattern(rows)

    if output_csv_path is not None:
        df = sweep_results_to_dataframe(rows)
        df.to_csv(output_csv_path, index=False)

    return MarketSizeSweepResult(rows=rows, pattern=pattern)


def sweep_results_to_dataframe(rows: Sequence[MarketSizeEvaluationResult]):
    import pandas as pd

    return pd.DataFrame([asdict(r) for r in rows])


def refine_market_size_threshold(
    *,
    cfg: dict,
    tech: TierATechnology,
    N: float,
    base_comp: CompetitionParams,
    downstream_solver_params: DownstreamSolverParams,
    threshold_settings: ThresholdInteriorSettings,
    coarse_sweep: MarketSizeSweepResult,
    refinement_tol: float,
    max_refinement_steps: int,
    p_grid_override: Optional[Sequence[float]] = None,
    include_weak: bool = True,
    use_student_cache: bool = True,
    student_cache_precision: int = 8,
) -> ThresholdRefinementResult:
    """Refine the threshold interval using strict-classification bisection.

    Refinement runs only when coarse sweep supports a single-threshold
    interpretation and provides exactly one transition interval.
    """
    if refinement_tol <= 0:
        raise ValueError("refinement_tol must be > 0.")
    if max_refinement_steps <= 0:
        raise ValueError("max_refinement_steps must be > 0.")

    pattern = coarse_sweep.pattern
    if not pattern.supports_single_threshold:
        summary = ThresholdRefinementSummary(
            lower_bound=None,
            upper_bound=None,
            midpoint_estimate=None,
            interval_width=None,
            steps=0,
            trustworthy=False,
            skipped=True,
            message=(
                "Refinement skipped: coarse sweep does not support a single-threshold "
                "interpretation."
            ),
        )
        return ThresholdRefinementResult(summary=summary, history=[])

    if len(pattern.transition_intervals) != 1:
        summary = ThresholdRefinementSummary(
            lower_bound=None,
            upper_bound=None,
            midpoint_estimate=None,
            interval_width=None,
            steps=0,
            trustworthy=False,
            skipped=True,
            message="Refinement skipped: expected exactly one transition interval.",
        )
        return ThresholdRefinementResult(summary=summary, history=[])

    left_m, right_m = pattern.transition_intervals[0]
    if not (right_m > left_m):
        summary = ThresholdRefinementSummary(
            lower_bound=None,
            upper_bound=None,
            midpoint_estimate=None,
            interval_width=None,
            steps=0,
            trustworthy=False,
            skipped=True,
            message="Refinement skipped: invalid transition interval.",
        )
        return ThresholdRefinementResult(summary=summary, history=[])

    # Reuse already-computed coarse points where possible.
    coarse_map = {float(r.market_size): r for r in coarse_sweep.rows}

    left_row = coarse_map.get(float(left_m))
    if left_row is None:
        left_row = evaluate_market_size_once(
            cfg=cfg,
            tech=tech,
            N=N,
            base_comp=base_comp,
            downstream_solver_params=downstream_solver_params,
            market_size=float(left_m),
            threshold_settings=threshold_settings,
            p_grid_override=p_grid_override,
            include_weak=include_weak,
            use_student_cache=use_student_cache,
            student_cache_precision=student_cache_precision,
        )

    right_row = coarse_map.get(float(right_m))
    if right_row is None:
        right_row = evaluate_market_size_once(
            cfg=cfg,
            tech=tech,
            N=N,
            base_comp=base_comp,
            downstream_solver_params=downstream_solver_params,
            market_size=float(right_m),
            threshold_settings=threshold_settings,
            p_grid_override=p_grid_override,
            include_weak=include_weak,
            use_student_cache=use_student_cache,
            student_cache_precision=student_cache_precision,
        )

    # Guard against inconsistent endpoint signs.
    if not left_row.overall_interior_strict or right_row.overall_interior_strict:
        summary = ThresholdRefinementSummary(
            lower_bound=float(left_m),
            upper_bound=float(right_m),
            midpoint_estimate=float(0.5 * (left_m + right_m)),
            interval_width=float(right_m - left_m),
            steps=0,
            trustworthy=False,
            skipped=True,
            message=(
                "Refinement skipped: transition endpoints do not bracket a True->False "
                "strict-classification change."
            ),
        )
        return ThresholdRefinementResult(summary=summary, history=[left_row, right_row])

    history: List[MarketSizeEvaluationResult] = [left_row, right_row]
    cache: Dict[float, MarketSizeEvaluationResult] = {
        float(left_row.market_size): left_row,
        float(right_row.market_size): right_row,
    }

    steps = 0
    left = float(left_m)
    right = float(right_m)

    while (right - left) > float(refinement_tol) and steps < int(max_refinement_steps):
        mid = 0.5 * (left + right)
        if mid in cache:
            mid_row = cache[mid]
        else:
            mid_row = evaluate_market_size_once(
                cfg=cfg,
                tech=tech,
                N=N,
                base_comp=base_comp,
                downstream_solver_params=downstream_solver_params,
                market_size=float(mid),
                threshold_settings=threshold_settings,
                p_grid_override=p_grid_override,
                include_weak=include_weak,
                use_student_cache=use_student_cache,
                student_cache_precision=student_cache_precision,
            )
            cache[mid] = mid_row
            history.append(mid_row)

        # Keep invariant: left=True(interior), right=False(non-interior).
        if mid_row.overall_interior_strict:
            left = float(mid)
        else:
            right = float(mid)

        steps += 1

    width = float(right - left)
    midpoint = float(0.5 * (left + right))
    trustworthy = bool(width <= float(refinement_tol))

    if trustworthy:
        msg = "Refinement converged to requested interval tolerance."
    else:
        msg = "Refinement stopped at max_refinement_steps before reaching tolerance."

    summary = ThresholdRefinementSummary(
        lower_bound=float(left),
        upper_bound=float(right),
        midpoint_estimate=midpoint,
        interval_width=width,
        steps=int(steps),
        trustworthy=trustworthy,
        skipped=False,
        message=msg,
    )

    history_sorted = sorted(history, key=lambda r: float(r.market_size))
    return ThresholdRefinementResult(summary=summary, history=history_sorted)


def refinement_history_to_dataframe(rows: Sequence[MarketSizeEvaluationResult]):
    import pandas as pd

    return pd.DataFrame([asdict(r) for r in rows])
