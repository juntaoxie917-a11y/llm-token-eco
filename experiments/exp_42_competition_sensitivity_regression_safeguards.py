from __future__ import annotations

import hashlib
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

try:
    from experiments._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(__file__)

from experiments.exp_40_competition_sensitivity_u0 import main as run_u0_sensitivity_exp
from experiments.exp_41_competition_sensitivity_tau import main as run_tau_sensitivity_exp
from src.competition_downstream_solver import build_downstream_solver_params_from_config
from src.competition_static import build_competition_params_from_config
from src.competition_threshold import (
    build_threshold_settings_from_config,
    evaluate_market_size_once,
    refine_market_size_threshold,
    run_market_size_sweep,
    save_threshold_outputs,
)
from src.config_loader import load_with_base_config, load_yaml
from src.scaling_laws import build_tierA_from_config


def _build_market_size_grid(th_cfg: dict) -> List[float]:
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

    if m_points == 2:
        return [m_min, m_max]

    step = (m_max - m_min) / float(m_points - 1)
    return [m_min + i * step for i in range(m_points)]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_files(paths: List[Path]) -> Dict[str, dict]:
    snap: Dict[str, dict] = {}
    for p in paths:
        if p.exists():
            st = p.stat()
            snap[str(p)] = {
                "exists": True,
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
                "sha256": _file_sha256(p),
            }
        else:
            snap[str(p)] = {
                "exists": False,
                "size": None,
                "mtime_ns": None,
                "sha256": None,
            }
    return snap


def _legacy_strict_from_row(row, threshold_settings) -> bool:
    # Reconstruct the pre-helper strict criterion from row-level diagnostics.
    min_share = float(row.min_share) if row.min_share is not None else float("-inf")
    residual = float(row.downstream_residual) if row.downstream_residual is not None else float("inf")

    return bool(
        row.teacher_solver_ok
        and row.student_solver_ok
        and row.downstream_solver_ok
        and (not row.teacher_price_at_lower_boundary)
        and (not row.teacher_price_at_upper_boundary)
        and (not row.student_D_at_lower_boundary)
        and (not row.student_D_at_upper_boundary)
        and (not row.downstream_price_at_boundary)
        and (not row.used_fallback)
        and (min_share > float(threshold_settings.share_tol))
        and (residual <= float(threshold_settings.solver_residual_tol))
    )


def main() -> None:
    project_root = PROJECT_ROOT
    competition_cfg_path = project_root / "config" / "competition.yaml"

    competition_cfg = load_yaml(competition_cfg_path)
    cfg = load_with_base_config(competition_cfg_path, project_root=project_root)

    tech = build_tierA_from_config(cfg)
    N = float(cfg["student"]["N0"])

    comp = build_competition_params_from_config(competition_cfg)
    sp = build_downstream_solver_params_from_config(competition_cfg)
    threshold_settings = build_threshold_settings_from_config(competition_cfg)

    th_cfg = competition_cfg.get("competition", {}).get("threshold_analysis", {})
    include_weak = bool(th_cfg.get("include_weak", True))
    refinement_tol = float(th_cfg.get("refinement_tol", 1e3))
    max_refinement_steps = int(th_cfg.get("max_refinement_steps", 20))

    market_size_grid = _build_market_size_grid(th_cfg)

    out_tables = project_root / "results" / "tables"
    out_logs = project_root / "results" / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    # Stage 9 check #1: smoke/validation run for old M experiment path.
    sweep = run_market_size_sweep(
        cfg=cfg,
        tech=tech,
        N=N,
        base_comp=comp,
        downstream_solver_params=sp,
        market_size_grid=market_size_grid,
        threshold_settings=threshold_settings,
        include_weak=include_weak,
    )
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

    smoke_artifacts = save_threshold_outputs(
        sweep=sweep,
        refinement=refinement,
        tables_dir=str(out_tables),
        stem="competition_threshold_stage9_smoke",
    )

    # Stage 9 check #2: old expected columns still exist.
    expected_threshold_columns = [
        "market_size",
        "overall_interior_strict",
        "overall_interior_weak",
        "p_star",
        "D_star",
        "pi_teacher_total_star",
        "pi_student_total_at_p_star",
        "P_T_star",
        "P_S_star",
        "s_T",
        "s_S",
        "s_0",
        "teacher_solver_ok",
        "student_solver_ok",
        "downstream_solver_ok",
        "used_fallback",
        "downstream_residual",
        "min_share",
        "price_distance_to_boundary",
        "demand_distance_to_boundary",
    ]

    threshold_csv = project_root / "results" / "tables" / "competition_sensitivity_threshold_sweep_results.csv"
    if not threshold_csv.exists():
        # Fall back to smoke CSV if the canonical file is missing.
        threshold_csv = Path(smoke_artifacts.sweep_csv_path)

    with threshold_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        actual_columns = list(reader.fieldnames or [])
    missing_threshold_columns = [c for c in expected_threshold_columns if c not in actual_columns]

    # Stage 9 check #3: helper classification equals legacy criterion on known calibration.
    calib_market_size = float(comp.M)
    eval_row = evaluate_market_size_once(
        cfg=cfg,
        tech=tech,
        N=N,
        base_comp=comp,
        downstream_solver_params=sp,
        market_size=calib_market_size,
        threshold_settings=threshold_settings,
        include_weak=include_weak,
    )
    helper_strict = bool(eval_row.overall_interior_strict)
    legacy_strict = _legacy_strict_from_row(eval_row, threshold_settings)

    # Stage 9 check #4: running u0/tau scripts must not overwrite old M outputs.
    # Competition tau semantics: price sensitivity in q - tau * P (not temperature).
    protected_m_outputs = [
        project_root / "results" / "tables" / "competition_sensitivity_threshold_sweep_results.csv",
        project_root / "results" / "tables" / "competition_sensitivity_threshold_summary.json",
        project_root / "results" / "tables" / "competition_sensitivity_threshold_refinement_history.csv",
    ]
    before = _snapshot_files(protected_m_outputs)

    run_u0_sensitivity_exp()
    run_tau_sensitivity_exp()

    after = _snapshot_files(protected_m_outputs)

    overwritten_paths: List[str] = []
    for p in protected_m_outputs:
        key = str(p)
        b = before[key]
        a = after[key]
        if b["exists"] and a["exists"] and b["sha256"] != a["sha256"]:
            overwritten_paths.append(key)

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "stage_9_regression_safeguards",
        "tau_semantics": "competition uses price sensitivity in utility q - tau * P",
        "checks": {
            "m_smoke_run": {
                "passed": True,
                "market_points": int(len(sweep.rows)),
                "pattern_message": str(sweep.pattern.message),
                "smoke_artifacts": asdict(smoke_artifacts),
            },
            "old_columns_exist": {
                "passed": len(missing_threshold_columns) == 0,
                "checked_file": str(threshold_csv),
                "missing_columns": missing_threshold_columns,
            },
            "interior_helper_consistency": {
                "passed": helper_strict == legacy_strict,
                "market_size": calib_market_size,
                "helper_strict": helper_strict,
                "legacy_strict": legacy_strict,
            },
            "u0_tau_do_not_overwrite_m_outputs": {
                "passed": len(overwritten_paths) == 0,
                "protected_outputs": [str(p) for p in protected_m_outputs],
                "overwritten_paths": overwritten_paths,
            },
        },
    }

    report["all_passed"] = bool(
        report["checks"]["m_smoke_run"]["passed"]
        and report["checks"]["old_columns_exist"]["passed"]
        and report["checks"]["interior_helper_consistency"]["passed"]
        and report["checks"]["u0_tau_do_not_overwrite_m_outputs"]["passed"]
    )

    report_path = out_tables / "competition_stage9_regression_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    run_log_path = out_logs / "exp_42_competition_sensitivity_regression_safeguards_run_log.json"
    run_log = {
        "timestamp_utc": report["timestamp_utc"],
        "report_path": str(report_path),
        "all_passed": report["all_passed"],
        "tau_semantics": report["tau_semantics"],
    }
    run_log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")

    print("Stage 9 regression safeguards completed.")
    print("Report:", report_path)
    print("All passed:", report["all_passed"])


if __name__ == "__main__":
    main()
