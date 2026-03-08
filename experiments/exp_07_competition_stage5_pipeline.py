from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.competition_downstream_solver import DownstreamSolverParams
from src.competition_downstream_solver import build_downstream_solver_params_from_config
from src.competition_simulation import run_competition_grid_simulation, to_dataframe
from src.competition_static import CompetitionParams
from src.competition_static import build_competition_params_from_config
from src.competition_visualization import (
    load_competition_results_csv,
    plot_competition_d_star_vs_p,
    plot_competition_downstream_prices_vs_p,
    plot_competition_downstream_shares_vs_p,
    plot_competition_teacher_profit_vs_p,
)
from src.config_loader import load_and_validate, load_yaml
from src.scaling_laws import build_tierA_from_config


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    competition_cfg_path = os.path.join(base_dir, "config", "competition.yaml")
    competition_cfg = load_yaml(competition_cfg_path)

    base_cfg_rel = str(competition_cfg.get("run", {}).get("base_config", "config/base.yaml"))
    cfg_path = os.path.join(base_dir, base_cfg_rel)
    cfg = load_and_validate(cfg_path)
    tech = build_tierA_from_config(cfg)
    N = float(cfg["student"]["N0"])

    comp: CompetitionParams = build_competition_params_from_config(competition_cfg)
    sp: DownstreamSolverParams = build_downstream_solver_params_from_config(competition_cfg)

    # For Stage 5 pipeline we run the full configured p-grid by default.
    sim, _sim_grids, _params = run_competition_grid_simulation(
        cfg=cfg,
        tech=tech,
        N=N,
        comp=comp,
        downstream_solver_params=sp,
        p_grid_override=None,
    )

    out_tables = os.path.join(base_dir, "results", "tables")
    out_figs = os.path.join(base_dir, "results", "figures")
    out_logs = os.path.join(base_dir, "results", "logs")
    os.makedirs(out_tables, exist_ok=True)
    os.makedirs(out_figs, exist_ok=True)
    os.makedirs(out_logs, exist_ok=True)

    csv_path = os.path.join(out_tables, "competition_stage5_grid_results.csv")
    summary_path = os.path.join(out_tables, "competition_stage5_optimum.json")
    diagnostics_path = os.path.join(out_tables, "competition_stage5_diagnostics.json")
    log_path = os.path.join(out_logs, "exp_07_competition_stage5_run_log.json")

    df = to_dataframe(sim)
    df.to_csv(csv_path, index=False)

    summary = {
        "p_star": sim.p_star,
        "D_star_at_p_star": sim.D_star_at_p_star,
        "P_T_down_at_p_star": sim.P_T_down_at_p_star,
        "P_S_down_at_p_star": sim.P_S_down_at_p_star,
        "pi_teacher_total_star": sim.pi_teacher_total_star,
        "pi_teacher_upstream_at_p_star": sim.pi_teacher_upstream_at_p_star,
        "pi_teacher_downstream_at_p_star": sim.pi_teacher_downstream_at_p_star,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    share_sum = df["s_T_down_star"] + df["s_S_down_star"] + df["s_0_down_star"]
    diagnostics = {
        "mode": "competition_stage5",
        "grid_info": {
            "p_min": float(df["p"].min()),
            "p_max": float(df["p"].max()),
            "p_points": int(len(df)),
        },
        "solver": {
            "br_success_rate": float(sim.br_success_rate),
            "down_success_rate": float(sim.down_success_rate),
            "boundary_share": float(sim.boundary_share),
            "avg_downstream_residual_norm": float(df["down_residual_norm"].mean()),
            "max_downstream_residual_norm": float(df["down_residual_norm"].max()),
        },
        "consistency_checks": {
            "share_sum_error_max": float((share_sum - 1.0).abs().max()),
            "share_min": float(df[["s_T_down_star", "s_S_down_star", "s_0_down_star"]].min().min()),
            "share_max": float(df[["s_T_down_star", "s_S_down_star", "s_0_down_star"]].max().max()),
        },
        "teacher_optimum": summary,
        "artifacts": {
            "results_csv": csv_path,
            "summary_json": summary_path,
            "diagnostics_json": diagnostics_path,
        },
    }
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    # Stage 5 requirement: plot from saved outputs where practical.
    df_saved = load_competition_results_csv(csv_path)
    out_figs_path = Path(os.path.abspath(out_figs))
    plot_competition_d_star_vs_p(df=df_saved, outdir=out_figs_path)
    plot_competition_teacher_profit_vs_p(df=df_saved, outdir=out_figs_path)
    plot_competition_downstream_prices_vs_p(df=df_saved, outdir=out_figs_path)
    plot_competition_downstream_shares_vs_p(df=df_saved, outdir=out_figs_path)

    run_log = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": cfg_path,
        "competition_config_path": competition_cfg_path,
        "competition_params": comp.__dict__,
        "downstream_solver_params": sp.__dict__,
        "summary": summary,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)

    print("Stage 5 pipeline completed.")
    print("Saved:")
    print(" -", csv_path)
    print(" -", summary_path)
    print(" -", diagnostics_path)
    print("Figures:")
    print(" -", os.path.join(out_figs, "fig_comp_01_dstar_vs_p.pdf"))
    print(" -", os.path.join(out_figs, "fig_comp_02_teacher_profit_vs_p.pdf"))
    print(" -", os.path.join(out_figs, "fig_comp_03_downstream_prices_vs_p.pdf"))
    print(" -", os.path.join(out_figs, "fig_comp_04_downstream_shares_vs_p.pdf"))


if __name__ == "__main__":
    main()
