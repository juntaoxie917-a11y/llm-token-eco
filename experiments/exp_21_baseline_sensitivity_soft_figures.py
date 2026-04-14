from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from experiments._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(__file__)

from src.config_loader import load_with_base_config
from src.scaling_laws import build_tierA_from_config
from src.simulation_soft import run_soft_grid_simulation


def set_soft_param(cfg: Dict[str, Any], var: str, value: float) -> None:
    if var in {"alpha", "beta", "gamma"}:
        cfg["exponents"][var] = float(value)
    elif var in {"k", "c_T"}:
        cfg["economics"][var] = float(value)
    elif var == "tau":
        cfg.setdefault("soft_outside", {})["tau"] = float(value)
    else:
        raise KeyError(f"Unknown parameter: {var}")


def param_grid(cfg: Dict[str, Any], var: str, n: int) -> np.ndarray:
    if var in {"alpha", "beta", "gamma"}:
        base = float(cfg["exponents"][var])
        return np.linspace(0.8 * base, 1.2 * base, n)
    if var in {"k", "c_T"}:
        base = float(cfg["economics"][var])
        return np.linspace(0.8 * base, 1.2 * base, n)
    if var == "tau":
        # Match user's preferred visual style: a wide tau ladder.
        return np.linspace(0.2, 2.0, n)
    raise KeyError(f"Unknown parameter: {var}")


def run_soft_curve_table(
    *,
    cfg_base: Dict[str, Any],
    var: str,
    values: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for i, v in enumerate(values, 1):
        cfg = copy.deepcopy(cfg_base)
        set_soft_param(cfg, var, float(v))

        tech = build_tierA_from_config(cfg)
        N = float(cfg["student"]["N0"])
        sim, _, _ = run_soft_grid_simulation(cfg=cfg, tech=tech, N=N)

        for r in sim.demand_rows:
            rows.append(
                {
                    "var": var,
                    "value": float(v),
                    "p": float(r.p),
                    "D_star": float(r.D_star),
                    "D_soft": float(r.D_soft),
                    "pi_student_star": float(r.pi_student_star),
                    "pi_student_soft": float(r.pi_student_soft),
                    "pi_teacher_soft": float(r.pi_teacher_soft),
                    "s_enter": float(r.s_enter),
                    "L_student": float(r.L_student),
                    "L_tilde": float(r.L_tilde),
                    "gap": float(r.gap),
                }
            )

        print(f"[soft-curves] {var}: {i}/{len(values)}", flush=True)

    return pd.DataFrame(rows)


def y_label(metric: str) -> str:
    labels = {
        "pi_student_soft": r"Student total payoff $\Pi_S^{\mathrm{soft}}(p)$",
        "pi_teacher_soft": r"Teacher total payoff $\Pi_T^{\mathrm{soft}}(p)$",
        "D_soft": r"Soft demand $D^{\mathrm{eff}}(p)$",
        "s_enter": r"Entry probability $s(p)$",
        "D_star": r"Conditional demand $D^*(p)$",
        "pi_student_star": r"Conditional student payoff $\Pi_S^*(p)$",
        "L_student": r"Student loss $L_S(p)$",
    }
    return labels[metric]


def pretty_param_label(var: str, value: float) -> str:
    if var == "alpha":
        return rf"$\alpha={value:.4g}$"
    if var == "beta":
        return rf"$\beta={value:.4g}$"
    if var == "gamma":
        return rf"$\gamma={value:.4g}$"
    if var == "tau":
        return rf"$\tau={value:.4g}$"
    if var == "c_T":
        return rf"$c_T={value:.4g}$"
    if var == "k":
        return rf"$k={value:.4g}$"
    return f"{var}={value:.4g}"


def draw_metric_family(
    *,
    df: pd.DataFrame,
    var: str,
    metric: str,
    out_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    vals = sorted(df["value"].dropna().unique().tolist())
    for v in vals:
        sub = df[df["value"] == v].sort_values("p")
        label = pretty_param_label(var, float(v))

        ax.plot(
            sub["p"].to_numpy(dtype=float),
            sub[metric].to_numpy(dtype=float),
            linewidth=2.0,
            label=label,
        )

    ax.set_xlabel(r"Upstream token price $p$", fontsize=14)
    ax.set_ylabel(y_label(metric), fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=2, fontsize=10, frameon=True)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Soft-only sensitivity curve-family plotting")
    parser.add_argument("--curve-levels", type=int, default=10, help="Number of parameter values per parameter")
    parser.add_argument("--p-points", type=int, default=220, help="Price grid points for smooth curves")
    args = parser.parse_args()

    root = PROJECT_ROOT
    cfg = load_with_base_config(root / "config" / "soft.yaml", project_root=root)

    # Keep fine p-grid for smoother curves.
    cfg["grids"]["p_points"] = int(max(120, args.p_points))

    out_tables = root / "results" / "tables"
    out_fig_dir = root / "results" / "figures" / "baseline" / "sensitivity"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    params = ["alpha", "beta", "gamma", "k", "c_T", "tau"]
    metrics = [
        "pi_student_soft",
        "pi_teacher_soft",
        "D_soft",
        "s_enter",
        "D_star",
        "pi_student_star",
        "L_student",
    ]

    frames: List[pd.DataFrame] = []
    figure_paths: List[str] = []

    for p in params:
        vals = param_grid(cfg, p, n=int(args.curve_levels))
        print(f"[param] start {p} with {len(vals)} curves", flush=True)
        df = run_soft_curve_table(cfg_base=cfg, var=p, values=vals)
        frames.append(df)

        for m in metrics:
            base = out_fig_dir / p / f"soft_sens_{p}_{m}"
            draw_metric_family(df=df, var=p, metric=m, out_base=base)
            figure_paths.extend([str(base.with_suffix(".pdf")), str(base.with_suffix(".png")), str(base.with_suffix(".svg"))])

    full = pd.concat(frames, ignore_index=True)
    curve_path = out_tables / "sens_soft_curve_families_full.csv"
    full.to_csv(curve_path, index=False)

    summary = {
        "mode": "soft_only",
        "params": params,
        "metrics": metrics,
        "figure_layout": "results/figures/baseline/sensitivity/<param>/soft_sens_<param>_<metric>.<ext>",
        "curve_count_per_param": int(args.curve_levels),
        "p_points": int(cfg["grids"]["p_points"]),
        "rows": int(len(full)),
        "figure_count": int(len(figure_paths)),
        "curve_table": str(curve_path),
        "figure_dir": str(out_fig_dir),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path = out_tables / "sens_soft_curve_families_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(str(curve_path))
    print(str(summary_path))
    print(str(out_fig_dir))


if __name__ == "__main__":
    main()
