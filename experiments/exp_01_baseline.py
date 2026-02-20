"""Experiment runner: exp_01_baseline (Tier A).

Run from project root:
    python experiments/exp_01_baseline.py
"""
from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from src.config_loader import load_and_validate
from src.scaling_laws import build_tierA_from_config
from src.simulation import run_baseline_grid_simulation
from src.visualization import (
    plot_scaling_curves,
    plot_student_profit_slices,
    plot_demand_curve,
    plot_teacher_profit,
    plot_student_indirect_payoff,
)

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "config" / "base.yaml"
    cfg = load_and_validate(cfg_path)

    tech = build_tierA_from_config(cfg)

    sup = cfg["anchors_supervised"]
    gap = cfg["anchors_gap"]
    anchor_errors = tech.check_anchors(
        D0=float(sup["D0"]),
        D1=float(sup["D1"]),
        L0=float(sup["L0"]),
        L1=float(sup["L1"]),
        rho=float(gap["rho"]),
        q=float(gap["q"]),
    )

    N = float(cfg["student"]["N0"])  # baseline uses N=N0 (n=1)
    sim, sim_grids, _ = run_baseline_grid_simulation(cfg=cfg, tech=tech, N=N)

    out_tables = project_root / "results" / "tables"
    out_figs = project_root / "results" / "figures"
    out_logs = project_root / "results" / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([r.__dict__ for r in sim.demand_rows])
    df.to_csv(out_tables / "baseline_demand_curve.csv", index=False)

    # --- Diagnostics (research-friendly) ---
    p = df["p"].to_numpy()
    D = df["D_star"].to_numpy()

    # opt-out diagnostics (only present after enabling hard outside option in simulation.py)
    if "opted_out" in df.columns:
        opted_out = df["opted_out"].astype(bool).to_numpy()
        optout_share = float(opted_out.mean())
        optout_count = int(opted_out.sum())
    else:
        optout_share = 0.0
        optout_count = 0

    # monotonicity: D(p) should be non-increasing in p on the grid
    dD = np.diff(D)
    is_nonincreasing = bool(np.all(dD <= 1e-12))  # small tolerance
    num_violations = int(np.sum(dD > 1e-12))

    diagnostics = {
        "anchor_errors": anchor_errors,
        "grid_info": {
            "p_min": float(p.min()),
            "p_max": float(p.max()),
            "p_points": int(len(p)),
        },
        "student_best_response": {
            "success_rate": float(df["success"].mean()),
            "boundary_share": float(df["is_boundary"].mean()),
            "optout_share": optout_share,
            "optout_count": optout_count,
            "boundary_counts": {
                "min": int((df["boundary_side"] == "min").sum()),
                "max": int((df["boundary_side"] == "max").sum()),
            },
            "D_star_summary": {
                "min": float(np.min(D)),
                "median": float(np.median(D)),
                "max": float(np.max(D)),
            },
            "nfev_summary": {
                "min": int(df["nfev"].min()),
                "median": float(df["nfev"].median()),
                "max": int(df["nfev"].max()),
            },
        },
        "demand_monotonicity_check": {
            "is_nonincreasing": is_nonincreasing,
            "num_violations": num_violations,
            "max_positive_jump": float(np.max(dD)) if len(dD) else 0.0,
        },
        "teacher_optimum": {
            "p_star": sim.p_star,
            "D_star_at_p_star": sim.D_star_at_p_star,
            "pi_teacher_star": sim.pi_teacher_star,
        },
        "notes": "All quantities computed on the configured p-grid; p* is grid-based in baseline.",
    }

    warnings = []
    if diagnostics["student_best_response"]["boundary_share"] > 0.05:
        warnings.append("Boundary solutions exceed 5% of grid points; consider expanding [D_min, D_max] or revisiting anchors.")
    if diagnostics["student_best_response"]["success_rate"] < 0.99:
        warnings.append("Optimizer success rate below 99%; check objective shape and bounds.")
    if not diagnostics["demand_monotonicity_check"]["is_nonincreasing"]:
        warnings.append("Demand monotonicity violated on the grid; increase p-grid resolution or inspect numerical issues.")
    if diagnostics["student_best_response"].get("optout_share", 0.0) > 0.95:
        warnings.append("Opt-out share exceeds 95%: the student almost always opts out (check 'a', 'b', and cost scale).")
    if warnings:
        diagnostics["warnings"] = warnings

    (out_tables / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    summary = {
        "p_star": sim.p_star,
        "D_star_at_p_star": sim.D_star_at_p_star,
        "pi_teacher_star": sim.pi_teacher_star,
        "anchor_errors": anchor_errors,
        "config_path": str(cfg_path),
    }
    (out_tables / "baseline_optimum.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log_obj = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": cfg,
        "anchor_errors": anchor_errors,
        "summary": summary,
    }
    (out_logs / "exp_01_run_log.json").write_text(json.dumps(log_obj, indent=2), encoding="utf-8")

    plot_scaling_curves(cfg=cfg, tech=tech, N=N, outdir=out_figs, grids=sim_grids)
    plot_student_profit_slices(
        cfg=cfg, tech=tech, N=N,
        p_values=[0.01, 0.05, 0.2, 1.0],
        outdir=out_figs,
        grids=sim_grids,
    )
    plot_demand_curve(sim=sim, outdir=out_figs, cfg=cfg)
    plot_teacher_profit(sim=sim, outdir=out_figs, cfg=cfg)
    plot_student_indirect_payoff(cfg=cfg, sim=sim, outdir=out_figs)

    print("Done. Outputs written to:")
    print(" -", out_tables)
    print(" -", out_figs)
    print(" -", out_logs)
    print("Key results:", summary)

if __name__ == "__main__":
    main()
