from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.competition_downstream_solver import build_downstream_solver_params_from_config
from src.competition_sensitivity import run_tau_sensitivity, sensitivity_results_to_records
from src.competition_sensitivity_config import build_competition_sensitivity_config
from src.competition_simulation import run_competition_grid_simulation, to_dataframe
from src.competition_static import build_competition_params_from_config
from src.competition_threshold import build_threshold_settings_from_config
from src.competition_visualization import (
    load_competition_results_csv,
    plot_competition_sensitivity_d_star_vs_parameter,
    plot_competition_sensitivity_downstream_prices_vs_parameter,
    plot_competition_sensitivity_downstream_shares_vs_parameter,
    plot_competition_sensitivity_interior_indicator_vs_parameter,
    plot_competition_sensitivity_p_star_vs_parameter,
    plot_competition_sensitivity_student_payoff_vs_parameter,
    plot_competition_sensitivity_teacher_payoff_vs_parameter,
    plot_competition_student_profit_vs_p_multi_tau,
    plot_competition_teacher_profit_vs_p_multi_tau,
)
from src.config_loader import load_with_base_config, load_yaml
from src.scaling_laws import build_tierA_from_config


def _as_float_or_none(value):
    if value is None:
        return None
    v = float(value)
    if np.isnan(v):
        return None
    return v


def _is_monotone(values: Iterable[float], *, tol: float = 1e-10) -> str:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return "insufficient_points"

    nondec = all((vals[i + 1] - vals[i]) >= -tol for i in range(len(vals) - 1))
    noninc = all((vals[i + 1] - vals[i]) <= tol for i in range(len(vals) - 1))

    if nondec and noninc:
        return "constant"
    if nondec:
        return "nondecreasing"
    if noninc:
        return "nonincreasing"
    return "nonmonotone"


def _build_tau_summary(df) -> dict:
    data = df.sort_values(by="parameter_value").copy()
    interior = data[data["interior_equilibrium"] == True]  # noqa: E712

    first_tau = _as_float_or_none(interior["parameter_value"].iloc[0]) if len(interior) > 0 else None
    last_tau = _as_float_or_none(interior["parameter_value"].iloc[-1]) if len(interior) > 0 else None

    summary = {
        "runs": int(len(data)),
        "interior_count": int(data["interior_equilibrium"].sum()),
        "interior_share": float(data["interior_equilibrium"].mean()) if len(data) > 0 else 0.0,
        "failed_runs": int((~data["success"]).sum()),
        "first_tau_with_interior": first_tau,
        "last_tau_with_interior": last_tau,
        "monotonicity": {
            "p_star": _is_monotone(data["p_star"].values),
            "D_star_at_p_star": _is_monotone(data["D_star_at_p_star"].values),
            "pi_teacher_star": _is_monotone(data["pi_teacher_star"].values),
            "pi_student_star": _is_monotone(data["pi_student_star"].values),
            "s_T_star": _is_monotone(data["s_T_star"].values),
            "s_S_star": _is_monotone(data["s_S_star"].values),
            "s_0_star": _is_monotone(data["s_0_star"].values),
        },
    }
    return summary


def _resolve_market_size_for_tau(comp_cfg: dict, base_market_size: float) -> float:
    tau_cfg = (
        comp_cfg.get("competition", {})
        .get("sensitivity_analysis", {})
        .get("tau_sweep", {})
    )

    if "fixed_market_size" in tau_cfg:
        m = float(tau_cfg["fixed_market_size"])
        if m <= 0:
            raise ValueError("competition.sensitivity_analysis.tau_sweep.fixed_market_size must be > 0.")
        return m

    if bool(tau_cfg.get("use_threshold_midpoint", False)):
        summary_path = Path(
            tau_cfg.get(
                "threshold_summary_path",
                os.path.join("results", "tables", "competition_threshold_summary.json"),
            )
        )
        if not summary_path.is_absolute():
            summary_path = Path.cwd() / summary_path
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            midpoint = (
                payload.get("refinement", {}).get("midpoint_estimate")
                if isinstance(payload, dict)
                else None
            )
            if midpoint is not None:
                midpoint = float(midpoint)
                if midpoint > 0:
                    return midpoint

    return float(base_market_size)


def _representative_tau_values(grid: List[float], requested: List[float] | None) -> List[float]:
    if requested:
        return sorted({float(x) for x in requested})

    return sorted({float(x) for x in grid})


def _small_tau_instability_report(df, *, tau_instability_threshold: float) -> dict:
    data = df.sort_values(by="parameter_value").copy()
    small = data[data["parameter_value"] <= float(tau_instability_threshold)]

    unstable = small[
        (~small["success"])
        | (~small["downstream_solver_ok"])
        | (small["used_fallback"] == True)  # noqa: E712
    ]

    return {
        "threshold": float(tau_instability_threshold),
        "small_tau_points": [float(x) for x in small["parameter_value"].tolist()],
        "unstable_tau_points": [float(x) for x in unstable["parameter_value"].tolist()],
        "unstable_count": int(len(unstable)),
        "unstable_messages": [str(x) for x in unstable["message"].tolist()],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    competition_cfg_path = project_root / "config" / "competition.yaml"

    competition_cfg = load_yaml(competition_cfg_path)
    cfg = load_with_base_config(competition_cfg_path, project_root=project_root)

    tech = build_tierA_from_config(cfg)
    N = float(cfg["student"]["N0"])

    comp = build_competition_params_from_config(competition_cfg)
    sp = build_downstream_solver_params_from_config(competition_cfg)
    threshold_settings = build_threshold_settings_from_config(competition_cfg)
    sens_cfg = build_competition_sensitivity_config(competition_cfg)

    tau_grid = [float(x) for x in sens_cfg.tau_sweep.grid]
    if len(tau_grid) < 2:
        raise ValueError("tau sensitivity requires at least two tau points.")
    if any(t <= 0 for t in tau_grid):
        raise ValueError("tau sensitivity grid must satisfy tau > 0.")

    market_size = _resolve_market_size_for_tau(competition_cfg, base_market_size=float(comp.M))
    comp_local = replace(comp, M=market_size)

    sweep = run_tau_sensitivity(
        cfg=cfg,
        tech=tech,
        N=N,
        base_comp=comp_local,
        downstream_solver_params=sp,
        threshold_settings=threshold_settings,
        tau_grid=tau_grid,
    )

    out_tables_root = project_root / "results" / "tables"
    out_tables = out_tables_root / "tau_sensitivity"
    out_figs = project_root / "results" / "figures" / "tau_sensitivity"
    out_logs = project_root / "results" / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    results_csv_path = out_tables / "tau_sensitivity_results.csv"
    summary_json_path = out_tables / "tau_sensitivity_summary.json"
    diagnostics_json_path = out_tables / "tau_sensitivity_diagnostics.json"

    legacy_results_csv_path = out_tables_root / "tau_sensitivity_results.csv"
    legacy_summary_json_path = out_tables_root / "tau_sensitivity_summary.json"
    legacy_diagnostics_json_path = out_tables_root / "tau_sensitivity_diagnostics.json"

    records = sensitivity_results_to_records(sweep.rows)
    if len(records) == 0:
        raise RuntimeError("tau sensitivity produced no rows.")

    with results_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    df = load_competition_results_csv(results_csv_path)

    summary = _build_tau_summary(df)
    summary["M_used"] = float(market_size)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tau_cfg = (
        competition_cfg.get("competition", {})
        .get("sensitivity_analysis", {})
        .get("tau_sweep", {})
    )
    instability_threshold = float(tau_cfg.get("instability_tau_threshold", 0.3))
    instability_report = _small_tau_instability_report(df, tau_instability_threshold=instability_threshold)

    diagnostics = {
        "experiment_name": "tau_sensitivity",
        "varied_parameter": "tau",
        "tau_grid": tau_grid,
        "fixed_parameters": {
            "M": float(market_size),
            "u0": float(comp_local.u0),
            "q_T": float(comp_local.q_T),
            "m_T": float(comp_local.m_T),
            "m_S": float(comp_local.m_S),
        },
        "runs": int(len(df)),
        "successful_runs": int(df["success"].sum()),
        "failed_runs": int((~df["success"]).sum()),
        "interior_runs": int(df["interior_equilibrium"].sum()),
        "boundary_or_degenerate_runs": int((~df["interior_equilibrium"]).sum()),
        "small_tau_instability": instability_report,
        "notes": [
            "Existing market-size threshold analysis was not modified.",
            "u0 and tau sensitivities are implemented as separate workflows.",
            "Competition tau denotes price sensitivity in utility q - tau * P.",
        ],
    }
    diagnostics_json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    # Keep top-level filenames for backward compatibility with existing checks.
    shutil.copyfile(results_csv_path, legacy_results_csv_path)
    shutil.copyfile(summary_json_path, legacy_summary_json_path)
    shutil.copyfile(diagnostics_json_path, legacy_diagnostics_json_path)

    plot_competition_sensitivity_p_star_vs_parameter(
        df=df,
        outdir=out_figs,
        parameter_col="parameter_value",
        x_label=r"price sensitivity $\tau$",
        stem="fig_tau_01_p_star_vs_tau",
    )
    plot_competition_sensitivity_teacher_payoff_vs_parameter(
        df=df,
        outdir=out_figs,
        parameter_col="parameter_value",
        x_label=r"price sensitivity $\tau$",
        stem="fig_tau_02_teacher_payoff_vs_tau",
    )
    plot_competition_sensitivity_student_payoff_vs_parameter(
        df=df,
        outdir=out_figs,
        parameter_col="parameter_value",
        x_label=r"price sensitivity $\tau$",
        stem="fig_tau_03_student_payoff_vs_tau",
    )
    plot_competition_sensitivity_d_star_vs_parameter(
        df=df,
        outdir=out_figs,
        parameter_col="parameter_value",
        x_label=r"price sensitivity $\tau$",
        stem="fig_tau_04_d_star_vs_tau",
    )
    plot_competition_sensitivity_interior_indicator_vs_parameter(
        df=df,
        outdir=out_figs,
        parameter_col="parameter_value",
        x_label=r"price sensitivity $\tau$",
        stem="fig_tau_05_interior_indicator_vs_tau",
    )
    plot_competition_sensitivity_downstream_prices_vs_parameter(
        df=df,
        outdir=out_figs,
        parameter_col="parameter_value",
        x_label=r"price sensitivity $\tau$",
        stem="fig_tau_06_downstream_prices_vs_tau",
    )
    plot_competition_sensitivity_downstream_shares_vs_parameter(
        df=df,
        outdir=out_figs,
        parameter_col="parameter_value",
        x_label=r"price sensitivity $\tau$",
        stem="fig_tau_07_downstream_shares_vs_tau",
    )

    rep_taus = _representative_tau_values(tau_grid, tau_cfg.get("price_domain_representative_values"))
    price_domain_curves: list[tuple[float, object]] = []
    for tau_val in rep_taus:
        comp_rep = replace(comp_local, tau=float(tau_val))
        sim_rep, _sim_grids, _params = run_competition_grid_simulation(
            cfg=cfg,
            tech=tech,
            N=N,
            comp=comp_rep,
            downstream_solver_params=sp,
            p_grid_override=None,
        )
        df_rep = to_dataframe(sim_rep)
        price_domain_curves.append((float(tau_val), df_rep))

    plot_competition_teacher_profit_vs_p_multi_tau(
        curves=price_domain_curves,
        outdir=out_figs,
        stem="fig_tau_price_teacher_vs_p_multi_tau",
    )
    plot_competition_student_profit_vs_p_multi_tau(
        curves=price_domain_curves,
        outdir=out_figs,
        stem="fig_tau_price_student_vs_p_multi_tau",
    )

    run_log = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "competition_config_path": str(competition_cfg_path),
        "tau_grid": tau_grid,
        "M_used": float(market_size),
        "representative_tau_for_price_domain": rep_taus,
        "artifacts": {
            "results_csv": str(results_csv_path),
            "summary_json": str(summary_json_path),
            "diagnostics_json": str(diagnostics_json_path),
            "legacy_results_csv": str(legacy_results_csv_path),
            "legacy_summary_json": str(legacy_summary_json_path),
            "legacy_diagnostics_json": str(legacy_diagnostics_json_path),
            "fig_dir": str(out_figs),
        },
    }

    log_path = out_logs / "exp_10_competition_tau_sensitivity_run_log.json"
    log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")

    print("Stage 6 tau sensitivity completed.")
    print("Saved:")
    print(" -", results_csv_path)
    print(" -", summary_json_path)
    print(" -", diagnostics_json_path)
    print("Figures:")
    print(" -", out_figs)


if __name__ == "__main__":
    main()
