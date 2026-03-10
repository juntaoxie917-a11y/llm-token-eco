"""Static building blocks for the competition extension (Stage 1 only).

This module intentionally excludes equilibrium solvers and simulation loops.
It only provides:
- parameter objects and config parsing,
- quality mapping helpers,
- stable downstream logit share helpers,
- downstream profit helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Literal, Tuple

import numpy as np

QualityMapMode = Literal["neg_loss", "affine_neg_loss"]


@dataclass(frozen=True)
class CompetitionParams:
    """Competition-specific static parameters for downstream demand/profits."""

    M: float
    m_T: float
    m_S: float
    u0: float
    tau: float
    q_T: float
    quality_map: QualityMapMode
    quality_scale: float
    quality_shift: float


@dataclass(frozen=True)
class DownstreamShares:
    s_T: float
    s_S: float
    s_0: float


@dataclass(frozen=True)
class DownstreamProfits:
    pi_T_down: float
    pi_S_down: float


@dataclass(frozen=True)
class ShareCheck:
    ok: bool
    total: float
    min_share: float
    max_share: float
    error_to_one: float
    message: str


def build_competition_params_from_config(cfg: Dict[str, Any]) -> CompetitionParams:
    """Build competition static parameters from config.

    Expected config shape:
      competition:
        M, m_T, m_S, u0, tau, q_T
        quality_mapping:
          mode: neg_loss | affine_neg_loss
          scale: float
          shift: float
    """
    comp = cfg.get("competition", {})
    qm = comp.get("quality_mapping", {})

    mode_raw = str(qm.get("mode", "neg_loss"))
    if mode_raw not in {"neg_loss", "affine_neg_loss"}:
        raise ValueError("competition.quality_mapping.mode must be 'neg_loss' or 'affine_neg_loss'.")

    params = CompetitionParams(
        M=float(comp.get("M", 1.0)),
        m_T=float(comp.get("m_T", 0.0)),
        m_S=float(comp.get("m_S", 0.0)),
        u0=float(comp.get("u0", 0.0)),
        tau=float(comp.get("tau", 1.0)),
        q_T=float(comp.get("q_T", 0.0)),
        quality_map=mode_raw,
        quality_scale=float(qm.get("scale", 1.0)),
        quality_shift=float(qm.get("shift", 0.0)),
    )
    validate_competition_params(params)
    return params


def validate_competition_params(params: CompetitionParams) -> None:
    if params.M <= 0:
        raise ValueError("competition.M must be > 0.")
    if not math.isfinite(params.u0):
        raise ValueError("competition.u0 must be finite.")
    if params.tau <= 0:
        raise ValueError("competition.tau must be > 0.")
    if params.quality_scale <= 0:
        raise ValueError("competition.quality_mapping.scale must be > 0.")


def student_quality_from_loss(
    loss: float,
    *,
    mode: QualityMapMode = "neg_loss",
    scale: float = 1.0,
    shift: float = 0.0,
) -> float:
    """Map student loss to downstream quality.

    Default is `q_S = -loss`.
    Optional affine form: `q_S = shift - scale * loss`.
    """
    if loss < 0:
        raise ValueError("loss should be non-negative in this model.")
    if scale <= 0:
        raise ValueError("scale must be > 0.")

    if mode == "neg_loss":
        return -loss
    if mode == "affine_neg_loss":
        return shift - scale * loss
    raise ValueError(f"Unknown quality map mode: {mode}")


def downstream_utilities(
    *,
    P_T: float,
    P_S: float,
    q_T: float,
    q_S: float,
    u0: float,
    tau: float,
) -> Tuple[float, float, float]:
    """Compute logit indices (scaled utilities) for teacher/student/outside.

    Index convention:
      v_T = (q_T - P_T) / tau
      v_S = (q_S - P_S) / tau
      v_0 = u0 / tau
    """
    if tau <= 0:
        raise ValueError("tau must be > 0.")
    v_T = (q_T - P_T) / tau
    v_S = (q_S - P_S) / tau
    v_0 = u0 / tau
    return float(v_T), float(v_S), float(v_0)


def stable_softmax3(v_T: float, v_S: float, v_0: float) -> DownstreamShares:
    """Numerically stable softmax for three alternatives."""
    vals = np.array([v_T, v_S, v_0], dtype=float)
    vmax = float(np.max(vals))
    exps = np.exp(vals - vmax)
    denom = float(np.sum(exps))
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Invalid softmax denominator.")
    probs = exps / denom
    return DownstreamShares(s_T=float(probs[0]), s_S=float(probs[1]), s_0=float(probs[2]))


def compute_downstream_shares(
    *,
    P_T: float,
    P_S: float,
    q_T: float,
    q_S: float,
    u0: float,
    tau: float,
) -> DownstreamShares:
    """Convenience wrapper from prices/qualities to three shares."""
    v_T, v_S, v_0 = downstream_utilities(P_T=P_T, P_S=P_S, q_T=q_T, q_S=q_S, u0=u0, tau=tau)
    return stable_softmax3(v_T, v_S, v_0)


def check_shares(shares: DownstreamShares, tol: float = 1e-10) -> ShareCheck:
    total = shares.s_T + shares.s_S + shares.s_0
    min_share = min(shares.s_T, shares.s_S, shares.s_0)
    max_share = max(shares.s_T, shares.s_S, shares.s_0)
    err = abs(total - 1.0)
    within_bounds = (min_share >= -tol) and (max_share <= 1.0 + tol)
    ok = bool(within_bounds and (err <= tol))
    msg = "ok" if ok else "share constraints violated"
    return ShareCheck(
        ok=ok,
        total=float(total),
        min_share=float(min_share),
        max_share=float(max_share),
        error_to_one=float(err),
        message=msg,
    )


def downstream_profits(
    *,
    P_T: float,
    P_S: float,
    shares: DownstreamShares,
    M: float,
    m_T: float,
    m_S: float,
) -> DownstreamProfits:
    """Compute downstream operating profits for teacher and student."""
    if M <= 0:
        raise ValueError("M must be > 0.")
    pi_T = (P_T - m_T) * M * shares.s_T
    pi_S = (P_S - m_S) * M * shares.s_S
    return DownstreamProfits(pi_T_down=float(pi_T), pi_S_down=float(pi_S))


def downstream_outcomes_from_prices(
    *,
    P_T: float,
    P_S: float,
    q_T: float,
    q_S: float,
    params: CompetitionParams,
) -> Tuple[DownstreamShares, DownstreamProfits, ShareCheck]:
    """Single-call helper used by Stage 1 demos and tests."""
    shares = compute_downstream_shares(
        P_T=P_T,
        P_S=P_S,
        q_T=q_T,
        q_S=q_S,
        u0=params.u0,
        tau=params.tau,
    )
    profits = downstream_profits(
        P_T=P_T,
        P_S=P_S,
        shares=shares,
        M=params.M,
        m_T=params.m_T,
        m_S=params.m_S,
    )
    share_check = check_shares(shares)
    return shares, profits, share_check
