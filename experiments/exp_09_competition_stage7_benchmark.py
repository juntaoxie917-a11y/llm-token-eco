from __future__ import annotations

import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.competition_downstream_solver import DownstreamSolverParams
from src.competition_simulation import run_competition_grid_simulation, to_dataframe
from src.competition_static import CompetitionParams
from src.config_loader import load_and_validate
from src.scaling_laws import build_tierA_from_config


def _max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = load_and_validate(os.path.join(base_dir, "config", "base.yaml"))
    tech = build_tierA_from_config(cfg)
    N = float(cfg["student"]["N0"])

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

    # Use a medium grid for benchmark repeatability and reasonable runtime.
    p_grid = list(np.linspace(0.0, 50.0, 160))

    t0 = time.perf_counter()
    sim_no_cache, _, _ = run_competition_grid_simulation(
        cfg=cfg,
        tech=tech,
        N=N,
        comp=comp,
        downstream_solver_params=sp,
        p_grid_override=p_grid,
        use_student_cache=False,
        student_cache_precision=8,
    )
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    sim_cache, _, _ = run_competition_grid_simulation(
        cfg=cfg,
        tech=tech,
        N=N,
        comp=comp,
        downstream_solver_params=sp,
        p_grid_override=p_grid,
        use_student_cache=True,
        student_cache_precision=6,
    )
    t3 = time.perf_counter()

    df_no = to_dataframe(sim_no_cache).sort_values("p").reset_index(drop=True)
    df_ca = to_dataframe(sim_cache).sort_values("p").reset_index(drop=True)

    diffs = {
        "D_star": _max_abs_diff(df_no["D_star"], df_ca["D_star"]),
        "pi_student_total": _max_abs_diff(df_no["pi_student_total"], df_ca["pi_student_total"]),
        "pi_teacher_total": _max_abs_diff(df_no["pi_teacher_total"], df_ca["pi_teacher_total"]),
        "P_T_down_star": _max_abs_diff(df_no["P_T_down_star"], df_ca["P_T_down_star"]),
        "P_S_down_star": _max_abs_diff(df_no["P_S_down_star"], df_ca["P_S_down_star"]),
        "s_T_down_star": _max_abs_diff(df_no["s_T_down_star"], df_ca["s_T_down_star"]),
        "s_S_down_star": _max_abs_diff(df_no["s_S_down_star"], df_ca["s_S_down_star"]),
        "s_0_down_star": _max_abs_diff(df_no["s_0_down_star"], df_ca["s_0_down_star"]),
    }

    runtime_no = float(t1 - t0)
    runtime_cache = float(t3 - t2)
    speedup = float(runtime_no / max(1e-12, runtime_cache))

    # Economically meaningful "materially unchanged" thresholds.
    thresholds = {
        "same_p_star": True,
        "pi_teacher_total_max_abs": 0.1,
        "pi_student_total_max_abs": 1e-3,
        "P_T_down_star_max_abs": 1e-5,
        "P_S_down_star_max_abs": 1e-5,
        "shares_max_abs": 1e-5,
    }
    same_p_star = bool(sim_no_cache.p_star == sim_cache.p_star)
    materially_unchanged = bool(
        same_p_star
        and diffs["pi_teacher_total"] <= thresholds["pi_teacher_total_max_abs"]
        and diffs["pi_student_total"] <= thresholds["pi_student_total_max_abs"]
        and diffs["P_T_down_star"] <= thresholds["P_T_down_star_max_abs"]
        and diffs["P_S_down_star"] <= thresholds["P_S_down_star_max_abs"]
        and diffs["s_T_down_star"] <= thresholds["shares_max_abs"]
        and diffs["s_S_down_star"] <= thresholds["shares_max_abs"]
        and diffs["s_0_down_star"] <= thresholds["shares_max_abs"]
    )

    cache_hits_total = int(df_ca["cache_hits"].sum()) if "cache_hits" in df_ca.columns else 0
    cache_misses_total = int(df_ca["cache_misses"].sum()) if "cache_misses" in df_ca.columns else 0

    report = {
        "stage": "stage_7_benchmark",
        "grid_points": len(p_grid),
        "runtime_seconds": {
            "no_cache": runtime_no,
            "cache_enabled": runtime_cache,
            "speedup_ratio_no_over_cache": speedup,
        },
        "output_difference_max_abs": diffs,
        "material_change_thresholds": thresholds,
        "materially_unchanged": materially_unchanged,
        "optima_compare": {
            "p_star_no_cache": sim_no_cache.p_star,
            "p_star_cache": sim_cache.p_star,
            "pi_teacher_total_star_no_cache": sim_no_cache.pi_teacher_total_star,
            "pi_teacher_total_star_cache": sim_cache.pi_teacher_total_star,
        },
        "cache_usage": {
            "cache_hits_total": cache_hits_total,
            "cache_misses_total": cache_misses_total,
        },
    }

    out_path = os.path.join(base_dir, "results", "tables", "competition_stage7_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Stage 7 benchmark completed.")
    print("Report:", out_path)
    print("speedup=", speedup)
    print("materially_unchanged=", materially_unchanged)


if __name__ == "__main__":
    main()
