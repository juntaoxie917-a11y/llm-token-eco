from __future__ import annotations

import json
import time
from pathlib import Path

try:
    from experiments._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(__file__)

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
    plot_competition_student_profit_vs_p,
    plot_competition_teacher_profit_vs_p,
)
from src.config_loader import load_with_base_config, load_yaml
from src.scaling_laws import build_tierA_from_config


def main() -> None:
    base_dir = PROJECT_ROOT
    competition_cfg_path = base_dir / "config" / "competition.yaml"
    competition_cfg = load_yaml(competition_cfg_path)

    cfg = load_with_base_config(competition_cfg_path, project_root=base_dir)
    base_cfg_rel = Path(str(competition_cfg.get("run", {}).get("base_config", "config/base.yaml")))
    cfg_path = base_dir / base_cfg_rel
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

    out_tables = base_dir / "results" / "tables"
    out_figs = base_dir / "results" / "figures" / "competition" / "base"
    out_logs = base_dir / "results" / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    csv_path = out_tables / "competition_stage5_grid_results.csv"
    summary_path = out_tables / "competition_stage5_optimum.json"
    diagnostics_path = out_tables / "competition_stage5_diagnostics.json"
    log_path = out_logs / "exp_30_competition_base_pipeline_run_log.json"

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
        "tau_semantics": "competition uses price sensitivity in utility q - tau * P",
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
            "results_csv": str(csv_path),
            "summary_json": str(summary_path),
            "diagnostics_json": str(diagnostics_path),
        },
    }
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    # Stage 5 requirement: plot from saved outputs where practical.
    df_saved = load_competition_results_csv(csv_path)
    out_figs_path = out_figs
    plot_competition_d_star_vs_p(df=df_saved, outdir=out_figs_path)
    plot_competition_teacher_profit_vs_p(df=df_saved, outdir=out_figs_path)
    plot_competition_downstream_prices_vs_p(df=df_saved, outdir=out_figs_path)
    plot_competition_downstream_shares_vs_p(df=df_saved, outdir=out_figs_path)
    plot_competition_student_profit_vs_p(df=df_saved, outdir=out_figs_path)

    run_log = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(cfg_path),
        "competition_config_path": str(competition_cfg_path),
        "tau_semantics": "competition uses price sensitivity in utility q - tau * P",
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
    print(" -", out_figs / "fig_comp_01_dstar_vs_p.pdf")
    print(" -", out_figs / "fig_comp_02_teacher_profit_vs_p.pdf")
    print(" -", out_figs / "fig_comp_03_downstream_prices_vs_p.pdf")
    print(" -", out_figs / "fig_comp_04_downstream_shares_vs_p.pdf")
    print(" -", out_figs / "fig_comp_05_student_profit_vs_p.pdf")


if __name__ == "__main__":
    main()
