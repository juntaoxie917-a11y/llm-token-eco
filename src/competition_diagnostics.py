"""Stage 6 diagnostics and trustworthiness checks for competition outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .competition_downstream_solver import DownstreamSolverParams, solve_downstream_equilibrium_at_D
from .competition_static import CompetitionParams
from .scaling_laws import TierATechnology


@dataclass(frozen=True)
class DiagnosticsThresholds:
    share_sum_tol: float = 1e-8
    profit_identity_tol: float = 1e-8
    monotonicity_tol: float = 1e-10


def compute_core_diagnostics(
    *,
    df: pd.DataFrame,
    econ_k: float,
    solver_bounds: DownstreamSolverParams,
    thresholds: DiagnosticsThresholds,
) -> Dict[str, object]:
    """Compute core trustworthiness checks from per-price simulation rows."""
    required = [
        "p",
        "D_star",
        "pi_student_total",
        "pi_student_downstream",
        "pi_teacher_upstream",
        "pi_teacher_downstream",
        "pi_teacher_total",
        "P_T_down_star",
        "P_S_down_star",
        "s_T_down_star",
        "s_S_down_star",
        "s_0_down_star",
        "br_success",
        "br_is_boundary",
        "down_success",
        "down_method",
        "down_hit_bounds",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for diagnostics: {missing}")

    share_sum = df["s_T_down_star"] + df["s_S_down_star"] + df["s_0_down_star"]
    share_sum_err = (share_sum - 1.0).abs()

    teacher_identity_err = (df["pi_teacher_total"] - (df["pi_teacher_upstream"] + df["pi_teacher_downstream"])).abs()
    student_identity_err = (
        df["pi_student_total"] - (df["pi_student_downstream"] - (df["p"] + float(econ_k)) * df["D_star"])
    ).abs()

    p = df["p"].to_numpy(dtype=float)
    D = df["D_star"].to_numpy(dtype=float)
    dD = np.diff(D)
    monotone_viol = int(np.sum(dD > thresholds.monotonicity_tol))

    pt = df["P_T_down_star"]
    ps = df["P_S_down_star"]
    pt_outside = int(((pt < solver_bounds.P_T_min - 1e-10) | (pt > solver_bounds.P_T_max + 1e-10)).sum())
    ps_outside = int(((ps < solver_bounds.P_S_min - 1e-10) | (ps > solver_bounds.P_S_max + 1e-10)).sum())

    fallback_count = int((df["down_method"].astype(str) == "best_response_fallback").sum())
    down_fail_count = int((~df["down_success"].astype(bool)).sum())
    br_fail_count = int((~df["br_success"].astype(bool)).sum())
    boundary_count = int(df["br_is_boundary"].astype(bool).sum())
    hit_bounds_count = int(df["down_hit_bounds"].astype(bool).sum())

    total = max(1, len(df))
    diagnostics = {
        "checks": {
            "shares_sum_to_one": {
                "ok": bool(float(share_sum_err.max()) <= thresholds.share_sum_tol),
                "max_error": float(share_sum_err.max()),
                "tol": thresholds.share_sum_tol,
            },
            "admissible_price_region": {
                "ok": bool(pt_outside == 0 and ps_outside == 0),
                "teacher_price_outside_count": pt_outside,
                "student_price_outside_count": ps_outside,
                "bounds": {
                    "P_T_min": solver_bounds.P_T_min,
                    "P_T_max": solver_bounds.P_T_max,
                    "P_S_min": solver_bounds.P_S_min,
                    "P_S_max": solver_bounds.P_S_max,
                },
            },
            "profit_identities": {
                "teacher_ok": bool(float(teacher_identity_err.max()) <= thresholds.profit_identity_tol),
                "student_ok": bool(float(student_identity_err.max()) <= thresholds.profit_identity_tol),
                "teacher_max_error": float(teacher_identity_err.max()),
                "student_max_error": float(student_identity_err.max()),
                "tol": thresholds.profit_identity_tol,
            },
        },
        "failure_and_boundary_summary": {
            "br_fail_count": br_fail_count,
            "downstream_fail_count": down_fail_count,
            "fallback_count": fallback_count,
            "boundary_count": boundary_count,
            "downstream_hit_bounds_count": hit_bounds_count,
            "br_fail_share": float(br_fail_count / total),
            "downstream_fail_share": float(down_fail_count / total),
            "fallback_share": float(fallback_count / total),
            "boundary_share": float(boundary_count / total),
            "downstream_hit_bounds_share": float(hit_bounds_count / total),
        },
        "smoke_test_from_grid": {
            "higher_p_usually_lowers_D": bool(monotone_viol == 0),
            "monotonicity_violations": monotone_viol,
            "max_positive_jump_in_D": float(np.max(dD)) if len(dD) else 0.0,
        },
    }
    return diagnostics


def smoke_test_quality_vs_share(
    *,
    D_values: List[float],
    N: float,
    tech: TierATechnology,
    comp: CompetitionParams,
    sp: DownstreamSolverParams,
) -> Dict[str, object]:
    """Check whether higher student quality tends to raise student share."""
    rows = []
    warm = None
    for D in D_values:
        eq = solve_downstream_equilibrium_at_D(
            D=float(D),
            N=N,
            tech=tech,
            comp=comp,
            sp=sp,
            initial_prices=warm,
        )
        if eq.success:
            warm = (eq.P_T_star, eq.P_S_star)
        rows.append({
            "D": float(D),
            "q_S": float(eq.q_S),
            "s_S": float(eq.s_S_star),
            "success": bool(eq.success),
            "residual_norm": float(eq.residual_norm),
        })

    success = all(r["success"] for r in rows)
    q = [r["q_S"] for r in rows]
    s = [r["s_S"] for r in rows]
    monotone = all(s[i + 1] >= s[i] - 1e-8 for i in range(len(s) - 1))

    return {
        "ok": bool(success and monotone),
        "all_solver_success": success,
        "higher_quality_raises_student_share": monotone,
        "rows": rows,
    }


def smoke_test_outside_option(
    *,
    D: float,
    N: float,
    tech: TierATechnology,
    comp: CompetitionParams,
    sp: DownstreamSolverParams,
    u0_low: float,
    u0_high: float,
) -> Dict[str, object]:
    """Check that a more attractive outside option reduces inside shares."""
    if u0_high <= u0_low:
        raise ValueError("Require u0_high > u0_low.")

    comp_low = CompetitionParams(
        M=comp.M,
        m_T=comp.m_T,
        m_S=comp.m_S,
        u0=float(u0_low),
        tau=comp.tau,
        q_T=comp.q_T,
        quality_map=comp.quality_map,
        quality_scale=comp.quality_scale,
        quality_shift=comp.quality_shift,
    )
    comp_high = CompetitionParams(
        M=comp.M,
        m_T=comp.m_T,
        m_S=comp.m_S,
        u0=float(u0_high),
        tau=comp.tau,
        q_T=comp.q_T,
        quality_map=comp.quality_map,
        quality_scale=comp.quality_scale,
        quality_shift=comp.quality_shift,
    )

    eq_low = solve_downstream_equilibrium_at_D(D=D, N=N, tech=tech, comp=comp_low, sp=sp)
    eq_high = solve_downstream_equilibrium_at_D(D=D, N=N, tech=tech, comp=comp_high, sp=sp)

    inside_low = float(eq_low.s_T_star + eq_low.s_S_star)
    inside_high = float(eq_high.s_T_star + eq_high.s_S_star)
    decreases = inside_high <= inside_low + 1e-8

    return {
        "ok": bool(eq_low.success and eq_high.success and decreases),
        "all_solver_success": bool(eq_low.success and eq_high.success),
        "inside_share_low_u0": inside_low,
        "inside_share_high_u0": inside_high,
        "more_attractive_outside_reduces_inside_share": bool(decreases),
        "u0_low": float(u0_low),
        "u0_high": float(u0_high),
        "residual_low": float(eq_low.residual_norm),
        "residual_high": float(eq_high.residual_norm),
    }


def summarize_overall_status(report: Dict[str, object]) -> Dict[str, object]:
    """Create top-level pass/fail flags and warnings for a Stage-6 report."""
    checks = report["core"]["checks"]
    smoke = report["smoke_tests"]

    pass_core = bool(
        checks["shares_sum_to_one"]["ok"]
        and checks["admissible_price_region"]["ok"]
        and checks["profit_identities"]["teacher_ok"]
        and checks["profit_identities"]["student_ok"]
    )
    pass_smoke = bool(
        report["core"]["smoke_test_from_grid"]["higher_p_usually_lowers_D"]
        and smoke["quality_vs_share"]["ok"]
        and smoke["outside_option"]["ok"]
    )

    warnings: List[str] = []
    fb = report["core"]["failure_and_boundary_summary"]["fallback_share"]
    if fb > 0.2:
        warnings.append("Fallback share above 20%; solver robustness may depend on fallback path.")
    if report["core"]["failure_and_boundary_summary"]["downstream_fail_count"] > 0:
        warnings.append("Downstream failures detected; inspect affected rows before trusting optimum.")
    if report["core"]["failure_and_boundary_summary"]["br_fail_count"] > 0:
        warnings.append("Student BR failures detected; outer curve may be contaminated.")

    return {
        "pass_core_checks": pass_core,
        "pass_smoke_tests": pass_smoke,
        "pass_all": bool(pass_core and pass_smoke),
        "warnings": warnings,
    }
