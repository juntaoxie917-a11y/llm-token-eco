"""Experiment runner: exp_01_baseline (Tier A).

Run from project root:
    python experiments/exp_01_baseline.py
"""
from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path
import json
import time
import pandas as pd

from src.config_loader import load_and_validate
from src.scaling_laws import build_tierA_from_config
from src.simulation import run_baseline_grid_simulation
from src.visualization import (
    plot_scaling_curves,
    plot_student_profit_slices,
    plot_demand_curve,
    plot_teacher_profit,
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
    sim = run_baseline_grid_simulation(cfg=cfg, tech=tech, N=N)

    out_tables = project_root / "results" / "tables"
    out_figs = project_root / "results" / "figures"
    out_logs = project_root / "results" / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([r.__dict__ for r in sim.demand_rows])
    df.to_csv(out_tables / "baseline_demand_curve.csv", index=False)

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

    plot_scaling_curves(cfg=cfg, tech=tech, N=N, outdir=out_figs)
    plot_student_profit_slices(
        cfg=cfg, tech=tech, N=N,
        p_values=[0.01, 0.05, 0.2, 1.0],
        econ=cfg["economics"],
        outdir=out_figs,
    )
    plot_demand_curve(sim=sim, outdir=out_figs)
    plot_teacher_profit(sim=sim, outdir=out_figs)

    print("Done. Outputs written to:")
    print(" -", out_tables)
    print(" -", out_figs)
    print(" -", out_logs)
    print("Key results:", summary)

if __name__ == "__main__":
    main()
