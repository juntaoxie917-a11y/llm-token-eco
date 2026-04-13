from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from experiments._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(__file__)

from src.competition_downstream_solver import build_downstream_solver_params_from_config
from src.competition_simulation import run_competition_grid_simulation, to_dataframe
from src.competition_static import build_competition_params_from_config
from src.competition_threshold import (
    MarketSizeEvaluationResult,
    build_threshold_settings_from_config,
    build_threshold_summary,
    refine_market_size_threshold,
    refine_market_size_threshold_in_interval,
    run_market_size_sweep,
    save_threshold_outputs,
    sweep_results_to_dataframe,
)
from src.competition_visualization import (
    plot_competition_student_profit_vs_p_multi_market_size,
    plot_competition_teacher_profit_vs_p_multi_market_size,
    plot_competition_threshold_suite,
)
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


def _merge_rows_by_market_size(rows: list[MarketSizeEvaluationResult], *, precision: int = 10) -> list[MarketSizeEvaluationResult]:
    merged = {}
    for r in rows:
        key = round(float(r.market_size), precision)
        merged[key] = r
    return sorted(merged.values(), key=lambda x: float(x.market_size))


def _find_transition_x_for_step_plot(rows: list[MarketSizeEvaluationResult]) -> float | None:
    """For step(where='post'), visual jump happens at the right point of True->False pair."""
    if len(rows) < 2:
        return None
    rows_sorted = sorted(rows, key=lambda r: float(r.market_size))
    flags = [bool(r.overall_interior_strict) for r in rows_sorted]
    for i in range(len(flags) - 1):
        if flags[i] and (not flags[i + 1]):
            return float(rows_sorted[i + 1].market_size)
    return None


def _representative_market_sizes_for_price_scan(
    *,
    market_size_grid: list[float],
    requested: list[float] | None,
) -> list[float]:
    if requested:
        values = sorted({float(x) for x in requested})
        return [v for v in values if v > 0]

    return sorted({float(x) for x in market_size_grid if float(x) > 0})


def main() -> None:
    project_root = PROJECT_ROOT
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
    refinement_interval = th_cfg.get("refinement_interval")

    # Optional local densification for plotting near the critical interval.
    dense_plot_near_critical = bool(th_cfg.get("dense_plot_near_critical", True))
    near_critical_points = int(th_cfg.get("near_critical_points", 31))
    near_critical_margin = float(th_cfg.get("near_critical_margin", 0.0))

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
        if isinstance(refinement_interval, list) and len(refinement_interval) == 2:
            refinement = refine_market_size_threshold_in_interval(
                cfg=cfg,
                tech=tech,
                N=N,
                base_comp=comp,
                downstream_solver_params=sp,
                threshold_settings=threshold_settings,
                interval_low=float(refinement_interval[0]),
                interval_high=float(refinement_interval[1]),
                refinement_tol=refinement_tol,
                max_refinement_steps=max_refinement_steps,
                include_weak=include_weak,
            )
        else:
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
    out_figs = project_root / "results" / "figures" / "competition" / "sensitivity" / "threshold"
    out_logs = project_root / "results" / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    artifacts = save_threshold_outputs(
        sweep=sweep,
        refinement=refinement,
        tables_dir=str(out_tables),
        stem="competition_sensitivity_threshold",
    )

    plot_rows = list(sweep.rows)

    if (
        dense_plot_near_critical
        and refinement is not None
        and (not refinement.summary.skipped)
        and refinement.summary.lower_bound is not None
        and refinement.summary.upper_bound is not None
        and near_critical_points >= 3
    ):
        local_lo = max(1e-12, float(refinement.summary.lower_bound) - near_critical_margin)
        local_hi = float(refinement.summary.upper_bound) + near_critical_margin
        if local_hi > local_lo:
            local_grid = [float(x) for x in np.linspace(local_lo, local_hi, near_critical_points)]
            local_sweep = run_market_size_sweep(
                cfg=cfg,
                tech=tech,
                N=N,
                base_comp=comp,
                downstream_solver_params=sp,
                market_size_grid=local_grid,
                threshold_settings=threshold_settings,
                include_weak=include_weak,
                output_csv_path=None,
            )
            plot_rows.extend(local_sweep.rows)

        # Also include bisection history points so plotted curve reflects refined checks.
        plot_rows.extend(refinement.history)

    plot_rows = _merge_rows_by_market_size(plot_rows)
    df = sweep_results_to_dataframe(plot_rows)

    critical_m = None
    critical_interval = None
    if refinement is not None and (not refinement.summary.skipped):
        # Align marker with plotted step transition to avoid visual offset.
        critical_m = _find_transition_x_for_step_plot(plot_rows)
        if critical_m is None:
            critical_m = refinement.summary.midpoint_estimate
        if refinement.summary.lower_bound is not None and refinement.summary.upper_bound is not None:
            critical_interval = (refinement.summary.lower_bound, refinement.summary.upper_bound)

    plot_competition_threshold_suite(
        df=df,
        outdir=out_figs,
        include_weak=include_weak,
        critical_m=critical_m,
        critical_interval=critical_interval,
    )

    rep_market_sizes = _representative_market_sizes_for_price_scan(
        market_size_grid=market_size_grid,
        requested=th_cfg.get("price_domain_representative_market_sizes"),
    )
    price_domain_curves: list[tuple[float, object]] = []
    for market_size in rep_market_sizes:
        comp_rep = replace(comp, M=float(market_size))
        sim_rep, _sim_grids, _params = run_competition_grid_simulation(
            cfg=cfg,
            tech=tech,
            N=N,
            comp=comp_rep,
            downstream_solver_params=sp,
            p_grid_override=None,
        )
        df_rep = to_dataframe(sim_rep)
        price_domain_curves.append((float(market_size), df_rep))

    plot_competition_teacher_profit_vs_p_multi_market_size(
        curves=price_domain_curves,
        outdir=out_figs,
        stem="fig_comp_threshold_09_teacher_profit_vs_p_multi_market_size",
    )
    plot_competition_student_profit_vs_p_multi_market_size(
        curves=price_domain_curves,
        outdir=out_figs,
        stem="fig_comp_threshold_10_student_profit_vs_p_multi_market_size",
    )

    summary = build_threshold_summary(sweep=sweep, refinement=refinement)
    run_log = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "competition_config_path": str(competition_cfg_path),
        "tau_semantics": "competition uses price sensitivity in utility q - tau * P",
        "threshold_analysis_config": th_cfg,
        "summary": summary,
        "artifacts": {
            "sweep_csv_path": artifacts.sweep_csv_path,
            "summary_json_path": artifacts.summary_json_path,
            "refinement_history_csv_path": artifacts.refinement_history_csv_path,
            "fig_dir": str(out_figs),
            "representative_market_sizes_for_price_domain": rep_market_sizes,
        },
    }
    with open(out_logs / "exp_45_competition_sensitivity_threshold_run_log.json", "w", encoding="utf-8") as f:
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
