from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.competition_downstream_solver import build_downstream_solver_params_from_config
from src.competition_static import build_competition_params_from_config
from src.competition_threshold import (
    build_threshold_settings_from_config,
    build_threshold_summary,
    refine_market_size_threshold,
    run_market_size_sweep,
    save_threshold_outputs,
    sweep_results_to_dataframe,
)
from src.competition_visualization import plot_competition_threshold_suite
from src.config_loader import load_with_base_config, load_yaml
from src.scaling_laws import build_tierA_from_config


def _build_market_size_grid(th_cfg: dict) -> list[float]:
    if "market_size_grid" in th_cfg:
        grid = [float(x) for x in th_cfg.get("market_size_grid", [])]
        if len(grid) < 2:
            raise ValueError("competition.threshold_analysis.market_size_grid must have >= 2 points.")
        return sorted(grid)

    m_min = float(th_cfg.get("market_size_min", 1000.0))
    m_max = float(th_cfg.get("market_size_max", 100000.0))
    m_points = int(th_cfg.get("market_size_points", 20))
    if not (m_max > m_min):
        raise ValueError("Require market_size_max > market_size_min.")
    if m_points < 2:
        raise ValueError("Require market_size_points >= 2.")

    return [float(x) for x in np.linspace(m_min, m_max, m_points)]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    competition_cfg_path = project_root / "config" / "competition.yaml"

    competition_cfg = load_yaml(competition_cfg_path)
    cfg = load_with_base_config(competition_cfg_path, project_root=project_root)

    tech = build_tierA_from_config(cfg)
    N = float(cfg["student"]["N0"])

    comp = build_competition_params_from_config(competition_cfg)
    sp = build_downstream_solver_params_from_config(competition_cfg)

    th_cfg = competition_cfg.get("competition", {}).get("threshold_analysis", {})
    threshold_settings = build_threshold_settings_from_config(competition_cfg)

    market_size_grid = _build_market_size_grid(th_cfg)
    include_weak = bool(th_cfg.get("include_weak", True))
    run_refinement_flag = bool(th_cfg.get("run_refinement", True))
    refinement_tol = float(th_cfg.get("refinement_tol", 1e3))
    max_refinement_steps = int(th_cfg.get("max_refinement_steps", 20))

    sweep = run_market_size_sweep(
        cfg=cfg,
        tech=tech,
        N=N,
        base_comp=comp,
        downstream_solver_params=sp,
        market_size_grid=market_size_grid,
        threshold_settings=threshold_settings,
        include_weak=include_weak,
        output_csv_path=None,
    )

    refinement = None
    if run_refinement_flag:
        refinement = refine_market_size_threshold(
            cfg=cfg,
            tech=tech,
            N=N,
            base_comp=comp,
            downstream_solver_params=sp,
            threshold_settings=threshold_settings,
            coarse_sweep=sweep,
            refinement_tol=refinement_tol,
            max_refinement_steps=max_refinement_steps,
            include_weak=include_weak,
        )

    out_tables = project_root / "results" / "tables"
    out_figs = project_root / "results" / "figures" / "threshold"
    out_logs = project_root / "results" / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    artifacts = save_threshold_outputs(
        sweep=sweep,
        refinement=refinement,
        tables_dir=str(out_tables),
        stem="competition_threshold",
    )

    df = sweep_results_to_dataframe(sweep.rows)
    plot_competition_threshold_suite(df=df, outdir=out_figs, include_weak=include_weak)

    summary = build_threshold_summary(sweep=sweep, refinement=refinement)
    run_log = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "competition_config_path": str(competition_cfg_path),
        "threshold_analysis_config": th_cfg,
        "summary": summary,
        "artifacts": {
            "sweep_csv_path": artifacts.sweep_csv_path,
            "summary_json_path": artifacts.summary_json_path,
            "refinement_history_csv_path": artifacts.refinement_history_csv_path,
            "fig_dir": str(out_figs),
        },
    }
    with open(out_logs / "exp_08_competition_threshold_run_log.json", "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)

    print("Stage 6 threshold pipeline completed.")
    print("Coarse pattern:", sweep.pattern.message)
    if refinement is not None:
        rs = refinement.summary
        print(
            "Refinement:",
            {
                "skipped": rs.skipped,
                "trustworthy": rs.trustworthy,
                "lower": rs.lower_bound,
                "upper": rs.upper_bound,
                "midpoint": rs.midpoint_estimate,
                "width": rs.interval_width,
                "steps": rs.steps,
            },
        )
    print("Saved:")
    print(" -", artifacts.sweep_csv_path)
    print(" -", artifacts.summary_json_path)
    if artifacts.refinement_history_csv_path is not None:
        print(" -", artifacts.refinement_history_csv_path)
    print("Figures:")
    print(" -", out_figs)


if __name__ == "__main__":
    main()
