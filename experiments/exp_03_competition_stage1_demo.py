from __future__ import annotations

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.competition_static import (
    CompetitionParams,
    student_quality_from_loss,
    downstream_outcomes_from_prices,
)


def main() -> None:
    # Manual inputs as required by Stage 1 (no equilibrium solving).
    params = CompetitionParams(
        M=1_000_000.0,
        m_T=3.0,
        m_S=2.5,
        u0=0.0,
        tau=1.0,
        q_T=2.0,
        quality_map="affine_neg_loss",
        quality_scale=1.0,
        quality_shift=0.0,
    )

    # Fixed downstream prices for demonstration.
    P_T = 5.0
    P_S = 4.5

    # Two quality states from different student losses.
    loss_high = 1.4  # worse model
    loss_low = 0.8   # better model
    q_s_high = student_quality_from_loss(
        loss_high,
        mode=params.quality_map,
        scale=params.quality_scale,
        shift=params.quality_shift,
    )
    q_s_low = student_quality_from_loss(
        loss_low,
        mode=params.quality_map,
        scale=params.quality_scale,
        shift=params.quality_shift,
    )

    shares_high, profits_high, chk_high = downstream_outcomes_from_prices(
        P_T=P_T,
        P_S=P_S,
        q_T=params.q_T,
        q_S=q_s_high,
        params=params,
    )
    shares_low, profits_low, chk_low = downstream_outcomes_from_prices(
        P_T=P_T,
        P_S=P_S,
        q_T=params.q_T,
        q_S=q_s_low,
        params=params,
    )

    # Stage 1 completion checks
    if not chk_high.ok or not chk_low.ok:
        raise RuntimeError("Share validation failed in Stage 1 demo.")
    if shares_low.s_S <= shares_high.s_S:
        raise RuntimeError("Expected student share to increase when student quality improves.")

    result = {
        "params": {
            "M": params.M,
            "m_T": params.m_T,
            "m_S": params.m_S,
            "u0": params.u0,
            "tau": params.tau,
            "q_T": params.q_T,
            "quality_map": params.quality_map,
            "quality_scale": params.quality_scale,
            "quality_shift": params.quality_shift,
        },
        "manual_inputs": {
            "P_T": P_T,
            "P_S": P_S,
            "loss_high": loss_high,
            "loss_low": loss_low,
            "q_s_high": q_s_high,
            "q_s_low": q_s_low,
        },
        "case_high_loss": {
            "shares": shares_high.__dict__,
            "profits": profits_high.__dict__,
            "share_check": chk_high.__dict__,
        },
        "case_low_loss": {
            "shares": shares_low.__dict__,
            "profits": profits_low.__dict__,
            "share_check": chk_low.__dict__,
        },
        "comparative_static": {
            "student_share_increases_when_loss_falls": True,
            "delta_student_share": shares_low.s_S - shares_high.s_S,
        },
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
