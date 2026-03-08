from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.competition_downstream_solver import DownstreamSolverParams
from src.competition_static import CompetitionParams
from src.competition_student import solve_student_best_response_competition
from src.config_loader import load_and_validate
from src.model import build_params_from_config
from src.scaling_laws import build_tierA_from_config


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = load_and_validate(os.path.join(base_dir, "config", "base.yaml"))
    tech = build_tierA_from_config(cfg)
    econ, grids, solver = build_params_from_config(cfg)
    N = float(cfg["student"]["N0"])

    # Keep Stage 3 self-contained: manual competition params, no baseline rewrites.
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

    p_values = [0.0, 1.0, 2.0, 4.0, 8.0]

    rows = []
    warm_prices = None
    for p in p_values:
        br = solve_student_best_response_competition(
            N=N,
            p=p,
            tech=tech,
            econ=econ,
            grids=grids,
            solver=solver,
            comp=comp,
            downstream_solver_params=sp,
            initial_downstream_prices=warm_prices,
        )
        rows.append(br)
        if br.downstream_eq_at_star.success:
            warm_prices = (br.downstream_eq_at_star.P_T_star, br.downstream_eq_at_star.P_S_star)

    all_success = all(r.success for r in rows)
    boundary_count = sum(int(r.is_boundary) for r in rows)
    boundary_share = boundary_count / max(1, len(rows))
    avg_inner_success_rate = sum(r.downstream_success_rate for r in rows) / max(1, len(rows))

    D_series = [r.D_star for r in rows]
    dD = [D_series[i + 1] - D_series[i] for i in range(len(D_series) - 1)]
    non_increasing = all(x <= 1e-10 for x in dD)
    violations = sum(int(x > 1e-10) for x in dD)

    payload = {
        "stage": "stage_3_student_best_response_demo",
        "p_values": p_values,
        "summary": {
            "all_success": all_success,
            "boundary_count": boundary_count,
            "boundary_share": boundary_share,
            "avg_inner_success_rate": avg_inner_success_rate,
            "D_star_non_increasing_in_p": non_increasing,
            "D_star_monotonicity_violations": violations,
        },
        "rows": [asdict(r) for r in rows],
    }

    if not all_success:
        raise RuntimeError("At least one Stage 3 best-response solve failed.")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
