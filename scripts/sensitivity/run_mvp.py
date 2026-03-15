import os
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze.sobol import analyze as sobol_analyze


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_problem(cfg):
    names = [v["name"] for v in cfg["variables"]]
    bounds = [v["bounds"] for v in cfg["variables"]]
    return {"num_vars": len(names), "names": names, "bounds": bounds}


def evaluate_one(alpha, beta, quality_coef, token_mix, cfg, rng):
    m = cfg["model"]
    s = cfg["search"]
    mix_gain = float(cfg["token_mix"]["mix_gain"])

    q_eff = float(quality_coef) * (1.0 + mix_gain * (float(token_mix) - 0.5))
    q_eff = max(q_eff, 1e-6)

    C = float(m["total_compute"])
    k = float(m["compute_coeff"])

    N_min = float(s["N_min"])
    N_max = float(s["N_max"])
    grid_size = int(s["grid_size"])

    N_grid = np.logspace(np.log10(N_min), np.log10(N_max), grid_size)
    D_grid = C / (k * N_grid)

    loss = float(m["A"]) * np.power(N_grid, -float(alpha)) + float(m["B"]) * np.power(D_grid, -float(beta)) / q_eff + float(m["E"])

    noise = rng.normal(0.0, 0.01 * np.mean(loss))
    loss_noisy = loss + noise

    idx = int(np.argmin(loss_noisy))
    best_loss = float(loss_noisy[idx])
    best_N = float(N_grid[idx])
    best_D = float(D_grid[idx])
    best_ratio = best_N / best_D
    qpc = 1.0 / best_loss

    return best_loss, best_ratio, qpc, best_N, best_D


def bootstrap_ci(x, rounds=1000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = []
    for _ in range(rounds):
        sample = rng.choice(x, size=n, replace=True)
        stats.append(np.median(sample))
    lo = np.percentile(stats, 100 * (alpha / 2))
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(np.median(stats)), float(lo), float(hi), np.array(stats)


def plot_sobol(df_sobol, out_png):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(df_sobol))
    w = 0.35
    ax.bar(x - w / 2, df_sobol["S1"], width=w, label="S1")
    ax.bar(x + w / 2, df_sobol["ST"], width=w, label="ST")
    ax.set_xticks(x)
    ax.set_xticklabels(df_sobol["var"], rotation=0)
    ax.set_ylabel("Sobol index")
    ax.set_title("Global Sensitivity (S1 / ST)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_ratio_drift(df, out_png):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sc = ax.scatter(df["alpha"], df["opt_ratio"], c=df["quality_coef"], s=16, alpha=0.65)
    ax.set_xlabel("alpha")
    ax.set_ylabel("Optimal N:D ratio (N/D)")
    ax.set_title("Optimal Ratio Drift Across Assumptions")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("quality_coef")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_ci(df, out_png, bootstrap_rounds, seed):
    plt.style.use("ggplot")
    q = df["best_loss"].quantile(0.2)
    sub = df[df["best_loss"] <= q].copy()

    alpha_med, alpha_lo, alpha_hi, _ = bootstrap_ci(sub["alpha"].values, rounds=bootstrap_rounds, seed=seed)
    beta_med, beta_lo, beta_hi, _ = bootstrap_ci(sub["beta"].values, rounds=bootstrap_rounds, seed=seed + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(sub["alpha"], bins=24, alpha=0.8)
    axes[0].axvline(alpha_med, linestyle="--", linewidth=2)
    axes[0].axvspan(alpha_lo, alpha_hi, alpha=0.2)
    axes[0].set_title(f"alpha CI: {alpha_med:.3f} [{alpha_lo:.3f}, {alpha_hi:.3f}]")

    axes[1].hist(sub["beta"], bins=24, alpha=0.8)
    axes[1].axvline(beta_med, linestyle="--", linewidth=2)
    axes[1].axvspan(beta_lo, beta_hi, alpha=0.2)
    axes[1].set_title(f"beta CI: {beta_med:.3f} [{beta_lo:.3f}, {beta_hi:.3f}]")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _get_midpoint_cfg(cfg):
    mids = {}
    for v in cfg["variables"]:
        lo, hi = v["bounds"]
        mids[v["name"]] = 0.5 * (lo + hi)
    return mids


def generate_oat_curves(cfg, curve_csv_dir: Path, figures_dir: Path, n_points=60):
    base = _get_midpoint_cfg(cfg)
    rng = np.random.default_rng(cfg["seed"] + 2026)

    for v in cfg["variables"]:
        name = v["name"]
        lo, hi = v["bounds"]
        xs = np.linspace(lo, hi, n_points)

        rows = []
        for x in xs:
            params = base.copy()
            params[name] = float(x)

            losses, ratios, qpcs = [], [], []
            for _ in range(cfg["n_repeats"]):
                best_loss, opt_ratio, qpc, _, _ = evaluate_one(
                    params["alpha"], params["beta"], params["quality_coef"], params["token_mix"], cfg, rng
                )
                losses.append(best_loss)
                ratios.append(opt_ratio)
                qpcs.append(qpc)

            rows.append(
                {
                    name: x,
                    "best_loss_mean": float(np.mean(losses)),
                    "best_loss_std": float(np.std(losses)),
                    "opt_ratio_mean": float(np.mean(ratios)),
                    "opt_ratio_std": float(np.std(ratios)),
                    "qpc_mean": float(np.mean(qpcs)),
                }
            )

        df_curve = pd.DataFrame(rows)
        df_curve.to_csv(curve_csv_dir / f"curve_{name}.csv", index=False)

        plt.style.use("ggplot")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(df_curve[name], df_curve["best_loss_mean"], linewidth=2)
        axes[0].fill_between(
            df_curve[name],
            df_curve["best_loss_mean"] - df_curve["best_loss_std"],
            df_curve["best_loss_mean"] + df_curve["best_loss_std"],
            alpha=0.2,
        )
        axes[0].set_xlabel(name)
        axes[0].set_ylabel("best_loss")
        axes[0].set_title(f"Response Curve: {name} -> best_loss")

        axes[1].plot(df_curve[name], df_curve["opt_ratio_mean"], linewidth=2)
        axes[1].fill_between(
            df_curve[name],
            df_curve["opt_ratio_mean"] - df_curve["opt_ratio_std"],
            df_curve[name] * 0 + (df_curve["opt_ratio_mean"] + df_curve["opt_ratio_std"]) - df_curve[name] * 0,  # 兼容写法
            alpha=0.2,
        )
        axes[1].set_xlabel(name)
        axes[1].set_ylabel("optimal N:D ratio")
        axes[1].set_title(f"Response Curve: {name} -> opt_ratio")

        fig.tight_layout()
        fig.savefig(figures_dir / f"fig_curve_{name}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sensitivity_mvp.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    out_dir = Path(cfg["output_dir"])   # results
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    logs_dir = out_dir / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])
    problem = build_problem(cfg)

    X = sobol_sample(problem, cfg["n_samples"], calc_second_order=False)

    rows = []
    for i, x in enumerate(X):
        alpha, beta, quality_coef, token_mix = x
        for r in range(cfg["n_repeats"]):
            best_loss, opt_ratio, qpc, best_N, best_D = evaluate_one(
                alpha, beta, quality_coef, token_mix, cfg, rng
            )
            rows.append(
                {
                    "sample_id": i,
                    "repeat_id": r,
                    "alpha": alpha,
                    "beta": beta,
                    "quality_coef": quality_coef,
                    "token_mix": token_mix,
                    "best_loss": best_loss,
                    "opt_ratio": opt_ratio,
                    "qpc": qpc,
                    "best_N": best_N,
                    "best_D": best_D,
                }
            )

    df = pd.DataFrame(rows)
    df_mean = (
        df.groupby("sample_id", as_index=False)
        .agg(
            alpha=("alpha", "mean"),
            beta=("beta", "mean"),
            quality_coef=("quality_coef", "mean"),
            token_mix=("token_mix", "mean"),
            best_loss=("best_loss", "mean"),
            opt_ratio=("opt_ratio", "mean"),
            qpc=("qpc", "mean"),
        )
    )

    si = sobol_analyze(problem, df_mean["best_loss"].values, calc_second_order=False, print_to_console=False)
    df_sobol = pd.DataFrame({"var": problem["names"], "S1": si["S1"], "ST": si["ST"]}).fillna(0.0)

    df.to_csv(tables_dir / "raw_runs.csv", index=False)
    df_mean.to_csv(tables_dir / "sample_mean.csv", index=False)
    df_sobol.to_csv(tables_dir / "sobol_best_loss.csv", index=False)

    plot_sobol(df_sobol, figures_dir / "fig_sobol_bar.png")
    plot_ratio_drift(df_mean, figures_dir / "fig_opt_ratio_drift.png")
    plot_ci(df_mean, figures_dir / "fig_alpha_beta_ci.png", cfg["bootstrap_rounds"], cfg["seed"])
    generate_oat_curves(cfg, tables_dir, figures_dir, n_points=80)

    print(f"[OK] figures: {figures_dir}")


if __name__ == "__main__":
    main()