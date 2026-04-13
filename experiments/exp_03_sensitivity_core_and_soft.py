from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_loader import load_and_validate
from src.scaling_laws import build_tierA_from_config
from src.simulation import run_baseline_grid_simulation
from src.simulation_soft import run_soft_grid_simulation


try:
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze
except ImportError as e:
    raise ImportError("SALib is required. Install with: pip install SALib") from e


@dataclass(frozen=True)
class RunMetrics:
    p_star: float
    D_star_metric: float
    pi_teacher_star: float
    optout_or_enter_metric: float
    boundary_share: float


def _set_param(cfg: Dict[str, Any], var: str, value: float, *, mode: str) -> None:
    if var in {"alpha", "beta", "gamma"}:
        cfg["exponents"][var] = float(value)
    elif var in {"k", "c_T"}:
        cfg["economics"][var] = float(value)
    elif var == "tau":
        if mode != "soft":
            raise ValueError("tau is only valid in soft mode")
        cfg.setdefault("soft_outside", {})["tau"] = float(value)
    else:
        raise KeyError(f"Unknown parameter: {var}")


def _baseline_values(cfg_hard: Dict[str, Any], cfg_soft: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    hard = {
        "alpha": float(cfg_hard["exponents"]["alpha"]),
        "beta": float(cfg_hard["exponents"]["beta"]),
        "gamma": float(cfg_hard["exponents"]["gamma"]),
        "k": float(cfg_hard["economics"]["k"]),
        "c_T": float(cfg_hard["economics"]["c_T"]),
    }
    soft = {
        "alpha": float(cfg_soft["exponents"]["alpha"]),
        "beta": float(cfg_soft["exponents"]["beta"]),
        "gamma": float(cfg_soft["exponents"]["gamma"]),
        "k": float(cfg_soft["economics"]["k"]),
        "c_T": float(cfg_soft["economics"]["c_T"]),
        "tau": float(cfg_soft.get("soft_outside", {}).get("tau", 0.2)),
    }
    return hard, soft


def _build_ranges(base_vals: Dict[str, float], *, mode: str) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for k, v0 in base_vals.items():
        if k == "tau":
            lo = max(1e-4, 0.5 * v0)
            hi = 2.0 * v0
        else:
            lo = max(1e-8, 0.8 * v0)
            hi = 1.2 * v0
        out[k] = (float(lo), float(hi))

    if mode == "hard" and "tau" in out:
        out.pop("tau", None)

    return out


def _run_hard(cfg: Dict[str, Any]) -> RunMetrics:
    tech = build_tierA_from_config(cfg)
    N = float(cfg["student"]["N0"])
    sim, _, _ = run_baseline_grid_simulation(cfg=cfg, tech=tech, N=N)
    return RunMetrics(
        p_star=float(sim.p_star),
        D_star_metric=float(sim.D_star_at_p_star),
        pi_teacher_star=float(sim.pi_teacher_star),
        optout_or_enter_metric=float(sim.optout_share),
        boundary_share=float(sim.boundary_share),
    )


def _run_soft(cfg: Dict[str, Any]) -> RunMetrics:
    tech = build_tierA_from_config(cfg)
    N = float(cfg["student"]["N0"])
    sim, _, _ = run_soft_grid_simulation(cfg=cfg, tech=tech, N=N)
    return RunMetrics(
        p_star=float(sim.p_star),
        D_star_metric=float(sim.D_soft_at_p_star),
        pi_teacher_star=float(sim.pi_teacher_star),
        optout_or_enter_metric=float(sim.avg_enter_prob),
        boundary_share=float("nan"),
    )


def _safe_run(
    cfg_base: Dict[str, Any],
    mode: str,
    assign: Dict[str, float],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg_base)
    for k, v in assign.items():
        _set_param(cfg, k, float(v), mode=mode)

    try:
        m = _run_hard(cfg) if mode == "hard" else _run_soft(cfg)
        return {
            "success": True,
            "error": "",
            "p_star": m.p_star,
            "D_star_metric": m.D_star_metric,
            "pi_teacher_star": m.pi_teacher_star,
            "optout_or_enter_metric": m.optout_or_enter_metric,
            "boundary_share": m.boundary_share,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "error": repr(e),
            "p_star": np.nan,
            "D_star_metric": np.nan,
            "pi_teacher_star": np.nan,
            "optout_or_enter_metric": np.nan,
            "boundary_share": np.nan,
        }


def run_oat(
    *,
    mode: str,
    cfg_base: Dict[str, Any],
    base_vals: Dict[str, float],
    ranges: Dict[str, Tuple[float, float]],
    points: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    target_vars = list(ranges.keys())
    total = len(target_vars) * points
    done = 0

    for var in target_vars:
        lo, hi = ranges[var]
        grid = np.linspace(lo, hi, points)
        for val in grid:
            assign = {var: float(val)}
            rec = _safe_run(cfg_base=cfg_base, mode=mode, assign=assign)
            rec.update({
                "mode": mode,
                "var": var,
                "value": float(val),
                "value_ratio": float(val / base_vals[var]),
            })
            rows.append(rec)
            done += 1
            if done % max(1, total // 8) == 0 or done == total:
                print(f"[OAT:{mode}] {done}/{total}", flush=True)

    return pd.DataFrame(rows)


def run_sobol(
    *,
    mode: str,
    cfg_base: Dict[str, Any],
    ranges: Dict[str, Tuple[float, float]],
    n_base: int,
    seed: int,
) -> Tuple[pd.DataFrame, int, int]:
    names = list(ranges.keys())
    bounds = [list(ranges[n]) for n in names]
    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
    }

    X = sobol_sample.sample(problem, N=n_base, calc_second_order=False, seed=seed)

    y = np.empty(X.shape[0], dtype=float)
    fail_count = 0
    progress_step = max(1, X.shape[0] // 10)

    for i, x in enumerate(X):
        assign = {k: float(v) for k, v in zip(names, x)}
        rec = _safe_run(cfg_base=cfg_base, mode=mode, assign=assign)
        if rec["success"] and np.isfinite(rec["pi_teacher_star"]):
            y[i] = float(rec["pi_teacher_star"])
        else:
            fail_count += 1
            y[i] = np.nan

        if (i + 1) % progress_step == 0 or (i + 1) == X.shape[0]:
            print(f"[SOBOL:{mode}] {i + 1}/{X.shape[0]}", flush=True)

    if np.isnan(y).any():
        finite_vals = y[np.isfinite(y)]
        fill = float(np.nanmedian(finite_vals)) if finite_vals.size else 0.0
        y = np.where(np.isfinite(y), y, fill)

    si = sobol_analyze.analyze(problem, y, calc_second_order=False, print_to_console=False)

    df = pd.DataFrame({
        "var": names,
        "S1": si["S1"],
        "ST": si["ST"],
    })
    return df, int(X.shape[0]), int(fail_count)


def _oat_top_effect(df: pd.DataFrame, baseline_pi: float) -> pd.DataFrame:
    tmp = df.copy()
    if baseline_pi != 0 and np.isfinite(baseline_pi):
        tmp["rel_effect"] = (tmp["pi_teacher_star"] - baseline_pi) / baseline_pi
    else:
        tmp["rel_effect"] = np.nan

    grp = tmp.groupby("var")["rel_effect"].apply(
        lambda s: float(np.nanmax(np.abs(s.to_numpy(dtype=float))))
    )
    out = grp.reset_index()
    out.columns = ["var", "max_abs_rel_effect"]
    out = out.sort_values("max_abs_rel_effect", ascending=False)
    return out


def plot_sensitivity_curves(
    *,
    oat_hard: pd.DataFrame,
    oat_soft: pd.DataFrame,
    hard_baseline_pi: float,
    soft_baseline_pi: float,
    out_base: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, df, mode, base_pi in [
        (axes[0], oat_hard, "hard", hard_baseline_pi),
        (axes[1], oat_soft, "soft", soft_baseline_pi),
    ]:
        vars_sorted = sorted(df["var"].dropna().unique().tolist())
        for var in vars_sorted:
            sub = df[df["var"] == var].sort_values("value_ratio")
            y = sub["pi_teacher_star"].to_numpy(dtype=float)
            if base_pi != 0 and np.isfinite(base_pi):
                y = y / base_pi
            ax.plot(
                sub["value_ratio"].to_numpy(dtype=float),
                y,
                marker="o",
                linewidth=1.5,
                markersize=3.0,
                label=var,
            )

        ax.set_xlabel("parameter ratio (value / baseline)")
        ax.set_ylabel("normalized teacher optimum payoff")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=8)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def build_student_payoff_curve_rows(
    *,
    mode: str,
    cfg_base: Dict[str, Any],
    base_vals: Dict[str, float],
    ranges: Dict[str, Tuple[float, float]],
    curve_levels: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    vars_list = list(ranges.keys())
    total_runs = len(vars_list) * curve_levels
    done = 0

    for var in vars_list:
        lo, hi = ranges[var]
        grid = np.linspace(lo, hi, curve_levels)
        for val in grid:
            cfg = copy.deepcopy(cfg_base)
            _set_param(cfg, var, float(val), mode=mode)

            tech = build_tierA_from_config(cfg)
            N = float(cfg["student"]["N0"])

            if mode == "hard":
                sim, _, _ = run_baseline_grid_simulation(cfg=cfg, tech=tech, N=N)
                for r in sim.demand_rows:
                    rows.append({
                        "mode": mode,
                        "var": var,
                        "value": float(val),
                        "value_ratio": float(val / base_vals[var]),
                        "p": float(r.p),
                        "student_total_payoff": float(r.pi_student),
                        "teacher_total_payoff": float(r.pi_teacher),
                    })
            else:
                sim, _, _ = run_soft_grid_simulation(cfg=cfg, tech=tech, N=N)
                for r in sim.demand_rows:
                    rows.append({
                        "mode": mode,
                        "var": var,
                        "value": float(val),
                        "value_ratio": float(val / base_vals[var]),
                        "p": float(r.p),
                        "student_total_payoff": float(r.pi_student_soft),
                        "teacher_total_payoff": float(r.pi_teacher_soft),
                    })

            done += 1
            if done % max(1, total_runs // 8) == 0 or done == total_runs:
                print(f"[CURVE:{mode}] {done}/{total_runs}", flush=True)

    return pd.DataFrame(rows)


def _pretty_param_label(var: str) -> str:
    if var == "alpha":
        return r"$\alpha$"
    if var == "beta":
        return r"$\beta$"
    if var == "gamma":
        return r"$\gamma$"
    if var == "tau":
        return r"$\tau$"
    if var == "c_T":
        return r"$c_T$"
    return var


def plot_student_payoff_curves_by_param(
    *,
    curve_df: pd.DataFrame,
    mode: str,
    out_dir: Path,
) -> List[Path]:
    out_paths: List[Path] = []
    vars_list = sorted(curve_df["var"].dropna().unique().tolist())

    for var in vars_list:
        sub = curve_df[curve_df["var"] == var].copy()
        values = sorted(sub["value"].dropna().unique().tolist())

        fig = plt.figure(figsize=(9, 5), constrained_layout=True)
        ax = fig.add_subplot(111)

        for val in values:
            s = sub[sub["value"] == val].sort_values("p")
            label = f"{_pretty_param_label(var)}={val:.4g}"
            ax.plot(
                s["p"].to_numpy(dtype=float),
                s["student_total_payoff"].to_numpy(dtype=float),
                linewidth=1.6,
                label=label,
            )

        ax.set_xlabel(r"Upstream token price $p$")
        ax.set_ylabel("Student total payoff")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(ncol=2, fontsize=8, loc="best")

        out_base = out_dir / f"fig_07_{mode}_student_payoff_curves_{var}"
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)

        out_paths.extend([
            out_base.with_suffix(".pdf"),
            out_base.with_suffix(".png"),
            out_base.with_suffix(".svg"),
        ])

    return out_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sensitivity for alpha/beta/gamma/k/c_T and tau(soft)")
    parser.add_argument("--oat-points", type=int, default=17, help="OAT points per parameter")
    parser.add_argument("--sobol-n", type=int, default=64, help="Sobol base N (increase to 256 for full run)")
    parser.add_argument("--p-points", type=int, default=200, help="Teacher price grid points override for faster sensitivity runs")
    parser.add_argument("--curve-levels", type=int, default=11, help="Number of parameter values (curves) per parameter in student-payoff-vs-price figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_hard = load_and_validate(project_root / "config" / "base.yaml")
    cfg_soft = load_and_validate(project_root / "config" / "soft.yaml")

    # Runtime control for sensitivity batches: keep price range unchanged, only adjust grid density.
    cfg_hard["grids"]["p_points"] = int(args.p_points)
    cfg_soft["grids"]["p_points"] = int(args.p_points)

    seed = int(cfg_hard.get("experiment", {}).get("seed", 42))
    np.random.seed(seed)

    hard_base_vals, soft_base_vals = _baseline_values(cfg_hard, cfg_soft)
    hard_ranges = _build_ranges(hard_base_vals, mode="hard")
    soft_ranges = _build_ranges(soft_base_vals, mode="soft")

    out_tables = project_root / "results" / "tables"
    out_figs = project_root / "results" / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    hard_baseline = _safe_run(cfg_base=cfg_hard, mode="hard", assign={})
    soft_baseline = _safe_run(cfg_base=cfg_soft, mode="soft", assign={})

    print("[stage] OAT hard", flush=True)

    oat_hard = run_oat(
        mode="hard",
        cfg_base=cfg_hard,
        base_vals=hard_base_vals,
        ranges=hard_ranges,
        points=int(args.oat_points),
    )
    print("[stage] OAT soft", flush=True)
    oat_soft = run_oat(
        mode="soft",
        cfg_base=cfg_soft,
        base_vals=soft_base_vals,
        ranges=soft_ranges,
        points=int(args.oat_points),
    )

    oat_hard_path = out_tables / "sens_hard_oat.csv"
    oat_soft_path = out_tables / "sens_soft_oat.csv"
    oat_hard.to_csv(oat_hard_path, index=False)
    oat_soft.to_csv(oat_soft_path, index=False)

    print("[stage] Sobol hard", flush=True)
    sobol_hard, sobol_hard_samples, sobol_hard_fail = run_sobol(
        mode="hard",
        cfg_base=cfg_hard,
        ranges=hard_ranges,
        n_base=int(args.sobol_n),
        seed=seed,
    )
    print("[stage] Sobol soft", flush=True)
    sobol_soft, sobol_soft_samples, sobol_soft_fail = run_sobol(
        mode="soft",
        cfg_base=cfg_soft,
        ranges=soft_ranges,
        n_base=int(args.sobol_n),
        seed=seed,
    )

    sobol_hard_path = out_tables / "sens_hard_sobol.csv"
    sobol_soft_path = out_tables / "sens_soft_sobol.csv"
    sobol_hard.to_csv(sobol_hard_path, index=False)
    sobol_soft.to_csv(sobol_soft_path, index=False)

    print("[stage] Curve families hard", flush=True)
    curve_hard = build_student_payoff_curve_rows(
        mode="hard",
        cfg_base=cfg_hard,
        base_vals=hard_base_vals,
        ranges=hard_ranges,
        curve_levels=int(args.curve_levels),
    )
    print("[stage] Curve families soft", flush=True)
    curve_soft = build_student_payoff_curve_rows(
        mode="soft",
        cfg_base=cfg_soft,
        base_vals=soft_base_vals,
        ranges=soft_ranges,
        curve_levels=int(args.curve_levels),
    )

    curve_hard_path = out_tables / "sens_hard_student_payoff_curves.csv"
    curve_soft_path = out_tables / "sens_soft_student_payoff_curves.csv"
    curve_hard.to_csv(curve_hard_path, index=False)
    curve_soft.to_csv(curve_soft_path, index=False)

    hard_top = _oat_top_effect(oat_hard, baseline_pi=float(hard_baseline["pi_teacher_star"]))
    soft_top = _oat_top_effect(oat_soft, baseline_pi=float(soft_baseline["pi_teacher_star"]))

    top_hard_oat = hard_top.iloc[0]["var"] if len(hard_top) else ""
    top_soft_oat = soft_top.iloc[0]["var"] if len(soft_top) else ""
    top_hard_sobol = sobol_hard.sort_values("ST", ascending=False).iloc[0]["var"] if len(sobol_hard) else ""
    top_soft_sobol = sobol_soft.sort_values("ST", ascending=False).iloc[0]["var"] if len(sobol_soft) else ""

    fig_base = out_figs / "fig_06_sensitivity_curves_core_soft"

    summary = {
        "baselines": {
            "hard": hard_base_vals,
            "soft": soft_base_vals,
            "hard_baseline_metrics": hard_baseline,
            "soft_baseline_metrics": soft_baseline,
        },
        "ranges": {
            "hard": {k: [v[0], v[1]] for k, v in hard_ranges.items()},
            "soft": {k: [v[0], v[1]] for k, v in soft_ranges.items()},
        },
        "oat": {
            "hard_top_by_abs_effect": hard_top.to_dict(orient="records"),
            "soft_top_by_abs_effect": soft_top.to_dict(orient="records"),
        },
        "sobol": {
            "hard": sobol_hard.to_dict(orient="records"),
            "soft": sobol_soft.to_dict(orient="records"),
            "hard_top_var_by_ST": top_hard_sobol,
            "soft_top_var_by_ST": top_soft_sobol,
        },
        "run_meta": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "oat_rows_hard": int(len(oat_hard)),
            "oat_rows_soft": int(len(oat_soft)),
            "sobol_samples_hard": sobol_hard_samples,
            "sobol_samples_soft": sobol_soft_samples,
            "sobol_failures_hard": sobol_hard_fail,
            "sobol_failures_soft": sobol_soft_fail,
            "oat_points": int(args.oat_points),
            "sobol_n": int(args.sobol_n),
            "curve_levels": int(args.curve_levels),
            "curve_rows_hard": int(len(curve_hard)),
            "curve_rows_soft": int(len(curve_soft)),
        },
    }

    summary_path = out_tables / "sens_core_soft_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_error = ""
    curve_plot_error = ""
    curve_plot_paths: List[Path] = []
    try:
        plot_sensitivity_curves(
            oat_hard=oat_hard,
            oat_soft=oat_soft,
            hard_baseline_pi=float(hard_baseline["pi_teacher_star"]),
            soft_baseline_pi=float(soft_baseline["pi_teacher_star"]),
            out_base=fig_base,
        )
    except Exception as e:  # noqa: BLE001
        plot_error = repr(e)

    try:
        curve_plot_paths.extend(
            plot_student_payoff_curves_by_param(curve_df=curve_hard, mode="hard", out_dir=out_figs)
        )
        curve_plot_paths.extend(
            plot_student_payoff_curves_by_param(curve_df=curve_soft, mode="soft", out_dir=out_figs)
        )
    except Exception as e:  # noqa: BLE001
        curve_plot_error = repr(e)

    print(str(oat_hard_path))
    print(str(sobol_hard_path))
    print(str(oat_soft_path))
    print(str(sobol_soft_path))
    print(str(curve_hard_path))
    print(str(curve_soft_path))
    print(str(summary_path))
    print(str(fig_base.with_suffix(".pdf")))
    print(str(fig_base.with_suffix(".png")))
    print(str(fig_base.with_suffix(".svg")))
    print(f"hard top-1 oat: {top_hard_oat}")
    print(f"hard top-1 sobol ST: {top_hard_sobol}")
    print(f"soft top-1 oat: {top_soft_oat}")
    print(f"soft top-1 sobol ST: {top_soft_sobol}")
    if plot_error:
        print(f"plot error: {plot_error}")
    if curve_plot_error:
        print(f"curve plot error: {curve_plot_error}")
    else:
        for p in curve_plot_paths:
            print(str(p))


if __name__ == "__main__":
    main()
