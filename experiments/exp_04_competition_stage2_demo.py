from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_loader import load_and_validate
from src.scaling_laws import build_tierA_from_config
from src.competition_static import CompetitionParams
from src.competition_downstream_solver import (
    DownstreamSolverParams,
    solve_downstream_equilibrium_at_D,
)


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = load_and_validate(os.path.join(base_dir, "config", "base.yaml"))
    tech = build_tierA_from_config(cfg)
    N = float(cfg["student"]["N0"])

    # Stage 2 uses manual competition params (keeps baseline config untouched).
    comp = CompetitionParams(
        M=1_000_000.0,
        m_T=2.0,
        m_S=2.0,
        u0=0.0,
        tau=1.0,
        q_T=-1.0,
        quality_map="neg_loss",
        quality_scale=1.0,
        quality_shift=0.0,
    )
    sp = DownstreamSolverParams(
        P_T_min=0.0,
        P_T_max=20.0,
        P_S_min=0.0,
        P_S_max=20.0,
        fd_eps=1e-5,
        root_tol=1e-6,
        max_nfev=500,
        br_max_iter=80,
        br_tol=1e-7,
    )

    D_values = [5.0, 20.0, 128.0, 500.0]

    results = []
    warm_start = None
    for D in D_values:
        eq = solve_downstream_equilibrium_at_D(
            D=D,
            N=N,
            tech=tech,
            comp=comp,
            sp=sp,
            initial_prices=warm_start,
        )
        results.append(eq)
        warm_start = (eq.P_T_star, eq.P_S_star)

    success_rate = sum(int(r.success) for r in results) / len(results)
    all_success = all(r.success for r in results)
    max_resid = max(r.residual_norm for r in results)

    q_series = [r.q_S for r in results]
    s_series = [r.s_S_star for r in results]
    share_tends_up = all(s_series[i + 1] >= s_series[i] - 1e-8 for i in range(len(s_series) - 1))

    payload = {
        "stage": "stage_2_downstream_subgame_demo",
        "D_values": D_values,
        "summary": {
            "all_success": all_success,
            "success_rate": success_rate,
            "max_residual_norm": max_resid,
            "student_share_non_decreasing_with_quality": share_tends_up,
            "q_S_series": q_series,
            "s_S_series": s_series,
        },
        "rows": [asdict(r) for r in results],
    }

    # Hard fail only on solver failures; share monotonicity is reported as a diagnostic tendency.
    if not all_success:
        raise RuntimeError("At least one downstream equilibrium solve failed in Stage 2 demo.")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
