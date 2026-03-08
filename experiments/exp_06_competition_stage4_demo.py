from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.competition_downstream_solver import DownstreamSolverParams
from src.competition_simulation import run_competition_grid_simulation, to_dataframe
from src.competition_static import CompetitionParams
from src.config_loader import load_and_validate
from src.scaling_laws import build_tierA_from_config


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = load_and_validate(os.path.join(base_dir, "config", "base.yaml"))
    tech = build_tierA_from_config(cfg)
    N = float(cfg["student"]["N0"])

    # Stage 4 keeps competition parameters local and additive.
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

    # Keep demo fast while preserving Stage-4 grid structure.
    p_grid = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]

    sim, _sim_grids, _params = run_competition_grid_simulation(
        cfg=cfg,
        tech=tech,
        N=N,
        comp=comp,
        downstream_solver_params=sp,
        p_grid_override=p_grid,
    )

    df = to_dataframe(sim)

    payload = {
        "stage": "stage_4_competition_simulation_demo",
        "summary": {
            "p_star": sim.p_star,
            "D_star_at_p_star": sim.D_star_at_p_star,
            "P_T_down_at_p_star": sim.P_T_down_at_p_star,
            "P_S_down_at_p_star": sim.P_S_down_at_p_star,
            "pi_teacher_total_star": sim.pi_teacher_total_star,
            "pi_teacher_upstream_at_p_star": sim.pi_teacher_upstream_at_p_star,
            "pi_teacher_downstream_at_p_star": sim.pi_teacher_downstream_at_p_star,
            "br_success_rate": sim.br_success_rate,
            "down_success_rate": sim.down_success_rate,
            "boundary_share": sim.boundary_share,
            "num_rows": int(len(df)),
        },
        "rows": [asdict(r) for r in sim.rows],
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
