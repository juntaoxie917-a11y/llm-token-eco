from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.competition_downstream_solver import build_downstream_solver_params_from_config
from src.competition_sensitivity import run_tau_sensitivity, run_u0_sensitivity
from src.competition_sensitivity_config import build_competition_sensitivity_config
from src.competition_static import build_competition_params_from_config
from src.competition_threshold import (
    build_threshold_settings_from_config,
    run_market_size_sweep,
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


def _resolve_market_size_for_u0(comp_cfg: dict, base_market_size: float) -> float:
    u0_cfg = comp_cfg.get("competition", {}).get("sensitivity_analysis", {}).get("u0_sweep", {})

    if "fixed_market_size" in u0_cfg:
        m = float(u0_cfg["fixed_market_size"])
        if m <= 0:
            raise ValueError("competition.sensitivity_analysis.u0_sweep.fixed_market_size must be > 0.")
        return m

    if bool(u0_cfg.get("use_threshold_midpoint", False)):
        summary_path = Path(
            u0_cfg.get(
                "threshold_summary_path",
                os.path.join("results", "tables", "competition_threshold_summary.json"),
            )
        )
        if not summary_path.is_absolute():
            summary_path = Path.cwd() / summary_path
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            midpoint = payload.get("refinement", {}).get("midpoint_estimate") if isinstance(payload, dict) else None
            if midpoint is not None:
                midpoint = float(midpoint)
                if midpoint > 0:
                    return midpoint

    return float(base_market_size)


def _resolve_market_size_for_tau(comp_cfg: dict, base_market_size: float) -> float:
    tau_cfg = comp_cfg.get("competition", {}).get("sensitivity_analysis", {}).get("tau_sweep", {})

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
            midpoint = payload.get("refinement", {}).get("midpoint_estimate") if isinstance(payload, dict) else None
            if midpoint is not None:
                midpoint = float(midpoint)
                if midpoint > 0:
                    return midpoint

    return float(base_market_size)


def _first_last_interior(values: Sequence[float], flags: Sequence[bool]) -> tuple[float | None, float | None]:
    interior_values = [float(v) for v, ok in zip(values, flags) if bool(ok)]
    if not interior_values:
        return None, None
    return float(interior_values[0]), float(interior_values[-1])


def _p_grid(*, p_min: float, p_max: float, p_points: int) -> list[float]:
    if p_points < 10:
        raise ValueError("p_points must be >= 10.")
    if p_max <= p_min:
        raise ValueError("Require p_max > p_min.")
    return [float(x) for x in np.linspace(float(p_min), float(p_max), int(p_points))]


def _label_from_flags(
    *,
    interior_equilibrium: bool,
    teacher_price_at_upper_boundary: bool,
    success: bool,
    teacher_solver_ok: bool,
    student_solver_ok: bool,
    downstream_solver_ok: bool,
) -> str:
    if not (success and teacher_solver_ok and student_solver_ok and downstream_solver_ok):
        return "unresolved"
    if interior_equilibrium:
        return "interior"
    if teacher_price_at_upper_boundary:
        return "non_interior_bound_limited"
    return "non_interior_pmax_insensitive"


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

    th_cfg = competition_cfg.get("competition", {}).get("threshold_analysis", {})
    include_weak = bool(th_cfg.get("include_weak", True))

    p_min = float(cfg["grids"]["p_min"])
    p_points = int(cfg["grids"]["p_points"])

    stability_cfg = competition_cfg.get("competition", {}).get("unconstrained_like_stability", {})
    p_max_grid = [float(x) for x in stability_cfg.get("p_max_grid", [50.0, 80.0, 120.0, 200.0])]
    p_max_grid = sorted({float(x) for x in p_max_grid})

    market_size_grid = _build_market_size_grid(th_cfg)
    u0_grid = [float(x) for x in sens_cfg.u0_sweep.grid]
    tau_grid = [float(x) for x in sens_cfg.tau_sweep.grid]

    if len(u0_grid) < 2:
        raise ValueError("u0 sensitivity requires at least two u0 points.")
    if len(tau_grid) < 2:
        raise ValueError("tau sensitivity requires at least two tau points.")
    if any(t <= 0 for t in tau_grid):
        raise ValueError("tau sensitivity grid must satisfy tau > 0.")

    market_size_for_u0 = _resolve_market_size_for_u0(competition_cfg, base_market_size=float(comp.M))
    market_size_for_tau = _resolve_market_size_for_tau(competition_cfg, base_market_size=float(comp.M))

    comp_u0 = replace(comp, M=float(market_size_for_u0))
    comp_tau = replace(comp, M=float(market_size_for_tau))

    summary_rows: list[dict] = []
    panel_rows: list[dict] = []

    for p_max in p_max_grid:
        p_grid_override = _p_grid(p_min=p_min, p_max=float(p_max), p_points=p_points)

        m_sweep = run_market_size_sweep(
            cfg=cfg,
            tech=tech,
            N=N,
            base_comp=comp,
            downstream_solver_params=sp,
            market_size_grid=market_size_grid,
            threshold_settings=threshold_settings,
            p_grid_override=p_grid_override,
            include_weak=include_weak,
        )
        m_values = [float(r.market_size) for r in m_sweep.rows]
        m_flags = [bool(r.overall_interior_strict) for r in m_sweep.rows]
        m_first, m_last = _first_last_interior(m_values, m_flags)
        m_upper_hits = int(sum(1 for r in m_sweep.rows if str(r.teacher_reason or "") == "teacher_price_at_upper_boundary"))

        for r in m_sweep.rows:
            teacher_boundary = bool(r.teacher_price_at_upper_boundary)
            label = _label_from_flags(
                interior_equilibrium=bool(r.overall_interior_strict),
                teacher_price_at_upper_boundary=teacher_boundary,
                success=bool(r.teacher_solver_ok and r.student_solver_ok and r.downstream_solver_ok),
                teacher_solver_ok=bool(r.teacher_solver_ok),
                student_solver_ok=bool(r.student_solver_ok),
                downstream_solver_ok=bool(r.downstream_solver_ok),
            )
            panel_rows.append(
                {
                    "parameter": "market_size",
                    "parameter_value": float(r.market_size),
                    "p_max": float(p_max),
                    "label": label,
                    "interior_equilibrium": bool(r.overall_interior_strict),
                    "teacher_reason": str(r.teacher_reason or ""),
                    "teacher_price_at_upper_boundary": teacher_boundary,
                    "student_reason": str(r.student_reason or ""),
                    "downstream_reason": str(r.downstream_reason or ""),
                    "success": bool(r.teacher_solver_ok and r.student_solver_ok and r.downstream_solver_ok),
                    "teacher_solver_ok": bool(r.teacher_solver_ok),
                    "student_solver_ok": bool(r.student_solver_ok),
                    "downstream_solver_ok": bool(r.downstream_solver_ok),
                }
            )

        summary_rows.append(
            {
                "parameter": "market_size",
                "p_max": float(p_max),
                "runs": int(len(m_sweep.rows)),
                "interior_count": int(sum(m_flags)),
                "interior_share": float(sum(m_flags) / max(1, len(m_flags))),
                "first_interior": m_first,
                "last_interior": m_last,
                "teacher_upper_boundary_hits": m_upper_hits,
                "supports_single_threshold": bool(m_sweep.pattern.supports_single_threshold),
                "transition_count": int(m_sweep.pattern.transition_count),
                "pattern_message": str(m_sweep.pattern.message),
            }
        )

        u0_sweep = run_u0_sensitivity(
            cfg=cfg,
            tech=tech,
            N=N,
            base_comp=comp_u0,
            downstream_solver_params=sp,
            threshold_settings=threshold_settings,
            u0_grid=u0_grid,
            p_grid_override=p_grid_override,
            include_weak=include_weak,
        )
        u0_values = [float(r.parameter_value) for r in u0_sweep.rows]
        u0_flags = [bool(r.interior_equilibrium) for r in u0_sweep.rows]
        u0_first, u0_last = _first_last_interior(u0_values, u0_flags)
        u0_upper_hits = int(sum(1 for r in u0_sweep.rows if str(r.teacher_reason or "") == "teacher_price_at_upper_boundary"))

        for r in u0_sweep.rows:
            teacher_boundary = bool(str(r.teacher_reason or "") == "teacher_price_at_upper_boundary")
            label = _label_from_flags(
                interior_equilibrium=bool(r.interior_equilibrium),
                teacher_price_at_upper_boundary=teacher_boundary,
                success=bool(r.success),
                teacher_solver_ok=bool(r.teacher_solver_ok),
                student_solver_ok=bool(r.student_solver_ok),
                downstream_solver_ok=bool(r.downstream_solver_ok),
            )
            panel_rows.append(
                {
                    "parameter": "u0",
                    "parameter_value": float(r.parameter_value),
                    "p_max": float(p_max),
                    "label": label,
                    "interior_equilibrium": bool(r.interior_equilibrium),
                    "teacher_reason": str(r.teacher_reason or ""),
                    "teacher_price_at_upper_boundary": teacher_boundary,
                    "student_reason": str(r.student_reason or ""),
                    "downstream_reason": str(r.downstream_reason or ""),
                    "success": bool(r.success),
                    "teacher_solver_ok": bool(r.teacher_solver_ok),
                    "student_solver_ok": bool(r.student_solver_ok),
                    "downstream_solver_ok": bool(r.downstream_solver_ok),
                }
            )

        summary_rows.append(
            {
                "parameter": "u0",
                "p_max": float(p_max),
                "runs": int(len(u0_sweep.rows)),
                "interior_count": int(sum(u0_flags)),
                "interior_share": float(sum(u0_flags) / max(1, len(u0_flags))),
                "first_interior": u0_first,
                "last_interior": u0_last,
                "teacher_upper_boundary_hits": u0_upper_hits,
                "supports_single_threshold": None,
                "transition_count": None,
                "pattern_message": None,
            }
        )

        tau_sweep = run_tau_sensitivity(
            cfg=cfg,
            tech=tech,
            N=N,
            base_comp=comp_tau,
            downstream_solver_params=sp,
            threshold_settings=threshold_settings,
            tau_grid=tau_grid,
            p_grid_override=p_grid_override,
            include_weak=include_weak,
        )
        tau_values = [float(r.parameter_value) for r in tau_sweep.rows]
        tau_flags = [bool(r.interior_equilibrium) for r in tau_sweep.rows]
        tau_first, tau_last = _first_last_interior(tau_values, tau_flags)
        tau_upper_hits = int(sum(1 for r in tau_sweep.rows if str(r.teacher_reason or "") == "teacher_price_at_upper_boundary"))

        for r in tau_sweep.rows:
            teacher_boundary = bool(str(r.teacher_reason or "") == "teacher_price_at_upper_boundary")
            label = _label_from_flags(
                interior_equilibrium=bool(r.interior_equilibrium),
                teacher_price_at_upper_boundary=teacher_boundary,
                success=bool(r.success),
                teacher_solver_ok=bool(r.teacher_solver_ok),
                student_solver_ok=bool(r.student_solver_ok),
                downstream_solver_ok=bool(r.downstream_solver_ok),
            )
            panel_rows.append(
                {
                    "parameter": "tau",
                    "parameter_value": float(r.parameter_value),
                    "p_max": float(p_max),
                    "label": label,
                    "interior_equilibrium": bool(r.interior_equilibrium),
                    "teacher_reason": str(r.teacher_reason or ""),
                    "teacher_price_at_upper_boundary": teacher_boundary,
                    "student_reason": str(r.student_reason or ""),
                    "downstream_reason": str(r.downstream_reason or ""),
                    "success": bool(r.success),
                    "teacher_solver_ok": bool(r.teacher_solver_ok),
                    "student_solver_ok": bool(r.student_solver_ok),
                    "downstream_solver_ok": bool(r.downstream_solver_ok),
                }
            )

        summary_rows.append(
            {
                "parameter": "tau",
                "p_max": float(p_max),
                "runs": int(len(tau_sweep.rows)),
                "interior_count": int(sum(tau_flags)),
                "interior_share": float(sum(tau_flags) / max(1, len(tau_flags))),
                "first_interior": tau_first,
                "last_interior": tau_last,
                "teacher_upper_boundary_hits": tau_upper_hits,
                "supports_single_threshold": None,
                "transition_count": None,
                "pattern_message": None,
            }
        )

    out_tables = project_root / "results" / "tables" / "unconstrained_like"
    out_logs = project_root / "results" / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    summary_csv_path = out_tables / "competition_unconstrained_like_stability_summary.csv"
    summary_json_path = out_tables / "competition_unconstrained_like_stability_summary.json"
    panel_csv_path = out_tables / "competition_unconstrained_like_panel.csv"
    unresolved_csv_path = out_tables / "competition_unconstrained_like_unresolved_points.csv"

    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with panel_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(panel_rows[0].keys()))
        writer.writeheader()
        writer.writerows(panel_rows)

    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in panel_rows:
        key = (str(row["parameter"]), float(row["parameter_value"]))
        grouped[key].append(row)

    unresolved_rows: list[dict] = []
    for (parameter, parameter_value), rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda x: float(x["p_max"]))
        labels = [str(r["label"]) for r in rows_sorted]
        label_set = set(labels)
        last_label = labels[-1]
        is_unresolved = bool(
            ("unresolved" in label_set)
            or (last_label == "non_interior_bound_limited")
            or (len(label_set) >= 2 and "interior" in label_set and "non_interior_pmax_insensitive" in label_set)
        )
        if is_unresolved:
            unresolved_rows.append(
                {
                    "parameter": parameter,
                    "parameter_value": float(parameter_value),
                    "labels_over_pmax": ",".join(labels),
                    "last_label": last_label,
                    "max_tested_pmax": float(rows_sorted[-1]["p_max"]),
                }
            )

    with unresolved_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "parameter",
            "parameter_value",
            "labels_over_pmax",
            "last_label",
            "max_tested_pmax",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unresolved_rows)

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "competition_config_path": str(competition_cfg_path),
        "p_min": p_min,
        "p_points": p_points,
        "p_max_grid": p_max_grid,
        "market_size_grid": market_size_grid,
        "u0_grid": u0_grid,
        "tau_grid": tau_grid,
        "market_size_for_u0": float(market_size_for_u0),
        "market_size_for_tau": float(market_size_for_tau),
        "summary_rows": summary_rows,
        "panel_rows_count": int(len(panel_rows)),
        "unresolved_points_count": int(len(unresolved_rows)),
    }
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    run_log = {
        "timestamp_utc": payload["timestamp_utc"],
        "experiment": "exp_12_competition_unconstrained_like_stability",
        "artifacts": {
            "summary_csv": str(summary_csv_path),
            "summary_json": str(summary_json_path),
            "panel_csv": str(panel_csv_path),
            "unresolved_csv": str(unresolved_csv_path),
        },
    }
    run_log_path = out_logs / "exp_12_competition_unconstrained_like_stability_run_log.json"
    run_log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")

    print("Stage 12 unconstrained-like stability check completed.")
    print("Saved:")
    print(" -", summary_csv_path)
    print(" -", summary_json_path)
    print(" -", panel_csv_path)
    print(" -", unresolved_csv_path)


if __name__ == "__main__":
    main()
