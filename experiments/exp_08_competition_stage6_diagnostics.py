from __future__ import annotations

import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.competition_diagnostics import (
    DiagnosticsThresholds,
    compute_core_diagnostics,
    smoke_test_outside_option,
    smoke_test_quality_vs_share,
    summarize_overall_status,
)
from src.competition_downstream_solver import DownstreamSolverParams
from src.competition_static import CompetitionParams
from src.config_loader import load_and_validate
from src.model import build_params_from_config
from src.scaling_laws import build_tierA_from_config


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = load_and_validate(os.path.join(base_dir, "config", "base.yaml"))
    tech = build_tierA_from_config(cfg)
    econ, _grids, _solver = build_params_from_config(cfg)
    N = float(cfg["student"]["N0"])

    csv_path = os.path.join(base_dir, "results", "tables", "competition_stage5_grid_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Missing Stage 5 results CSV. Run experiments/exp_07_competition_stage5_pipeline.py first."
        )
    df = pd.read_csv(csv_path)

    comp = CompetitionParams(
        M=1_000_000.0,
        m_T=2.0,
        m_S=2.0,
        u0=0.0,
        tau=1.0,
        q_T=-1.0,
        quality_map="neg_loss",
        quality_scale=1.0,
        quality_shift=0.0,
    )
    sp = DownstreamSolverParams(
        P_T_min=0.0,
        P_T_max=20.0,
        P_S_min=0.0,
        P_S_max=20.0,
        fd_eps=1e-5,
        root_tol=1e-6,
        max_nfev=500,
        br_max_iter=80,
        br_tol=1e-7,
    )
    thresholds = DiagnosticsThresholds(
        share_sum_tol=1e-8,
        profit_identity_tol=1e-8,
        monotonicity_tol=1e-10,
    )

    core = compute_core_diagnostics(
        df=df,
        econ_k=econ.k,
        solver_bounds=sp,
        thresholds=thresholds,
    )
    smoke_quality = smoke_test_quality_vs_share(
        D_values=[5.0, 20.0, 128.0, 500.0],
        N=N,
        tech=tech,
        comp=comp,
        sp=sp,
    )
    smoke_outside = smoke_test_outside_option(
        D=128.0,
        N=N,
        tech=tech,
        comp=comp,
        sp=sp,
        u0_low=-0.5,
        u0_high=0.5,
    )

    report = {
        "stage": "stage_6_competition_diagnostics",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_artifact": csv_path,
        "core": core,
        "smoke_tests": {
            "quality_vs_share": smoke_quality,
            "outside_option": smoke_outside,
        },
    }
    report["status"] = summarize_overall_status(report)

    out_path = os.path.join(base_dir, "results", "tables", "competition_stage6_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Stage 6 diagnostics completed.")
    print("Report:", out_path)
    print("pass_all=", report["status"]["pass_all"])
    print("warnings=", len(report["status"]["warnings"]))


if __name__ == "__main__":
    main()
