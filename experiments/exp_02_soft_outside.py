from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from pathlib import Path

from src.config_loader import load_and_validate
from src.scaling_laws import build_tierA_from_config
from src.simulation_soft import run_soft_grid_simulation, to_dataframe as soft_to_df
from src.visualization import (
    plot_scaling_curves,
    plot_soft_demand_curve,
    plot_soft_teacher_profit,
    plot_soft_student_payoff,
)

# Optional: if you want soft-specific plots, we will add them later


def main():
    base_dir = Path(__file__).resolve().parents[1]
    cfg_path = base_dir / "config" / "soft.yaml"
    cfg = load_and_validate(cfg_path)

    out_tables = base_dir / "results" / "tables"
    out_figs = base_dir / "results" / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    tech = build_tierA_from_config(cfg)
    anchor_errors = tech.check_anchors(
        D0=float(cfg["anchors_supervised"]["D0"]),
        D1=float(cfg["anchors_supervised"]["D1"]),
        L0=float(cfg["anchors_supervised"]["L0"]),
        L1=float(cfg["anchors_supervised"]["L1"]),
        rho=float(cfg["anchors_gap"]["rho"]),
        q=float(cfg["anchors_gap"]["q"]),
    )

    N = float(cfg["student"]["N0"])

    sim, sim_grids, (econ, grids, solver) = run_soft_grid_simulation(cfg=cfg, tech=tech, N=N)

    df = soft_to_df(sim)
    df.to_csv(out_tables / "soft_demand_curve.csv", index=False)

    diagnostics = {
        "mode": "soft_outside",
        "soft_outside": cfg.get("soft_outside", {}),
        "anchor_errors": anchor_errors,
        "teacher_optimum": {
            "p_star": sim.p_star,
            "D_soft_at_p_star": sim.D_soft_at_p_star,
            "pi_teacher_star": sim.pi_teacher_star,
        },
        "avg_enter_prob": sim.avg_enter_prob,
        "grid_info": {
            "p_min": float(sim_grids.p_grid.min()),
            "p_max": float(sim_grids.p_grid.max()),
            "p_points": int(len(sim_grids.p_grid)),
        },
    }
    (out_tables / "soft_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    # Reuse existing scaling curve plots (technology)
    plot_scaling_curves(cfg=cfg, tech=tech, N=N, grids=sim_grids, outdir=out_figs)
    plot_soft_demand_curve(cfg=cfg, df=df, outdir=out_figs)
    plot_soft_teacher_profit(cfg=cfg, df=df, outdir=out_figs)
    plot_soft_student_payoff(cfg=cfg, df=df, outdir=out_figs)

    # IMPORTANT: plot_demand_curve/plot_teacher_profit currently read sim.demand_rows fields:
    # they expect DemandRow with D_star and pi_teacher. Soft rows have different field names.
    # So we will NOT call those hard plots here.
    # Instead, we will add soft-specific plot functions next (recommended).

    # We can still reuse the student payoff plot if you adapt it to read pi_student_star/soft.
    # We'll add dedicated soft plot functions in visualization in the next step.

    print("Soft experiment finished.")
    print(f"p*: {sim.p_star:.4g}, teacher profit*: {sim.pi_teacher_star:.4g}, avg enter prob: {sim.avg_enter_prob:.2%}")


if __name__ == "__main__":
    main()