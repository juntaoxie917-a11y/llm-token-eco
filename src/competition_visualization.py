"""Stage 5 plotting helpers for competition outputs.

These functions consume saved tabular results (DataFrame/CSV) and only plot.
They never run equilibrium solvers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_figure(fig: plt.Figure, outpath_base: Path, *, save_png: bool = True, dpi: int = 300) -> None:
    outpath_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outpath_base.with_suffix(".svg"), bbox_inches="tight")
    if save_png:
        fig.savefig(outpath_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def load_competition_results_csv(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def plot_competition_d_star_vs_p(*, df: pd.DataFrame, outdir: Path, stem: str = "fig_comp_01_dstar_vs_p") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(df["p"], df["D_star"], label=r"$D^*(p)$")
    ax.set_xlabel(r"Upstream token price $p$")
    ax.set_ylabel(r"Student best-response tokens $D^*$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    _save_figure(fig, outdir / stem)
    plt.close(fig)


def plot_competition_teacher_profit_vs_p(
    *,
    df: pd.DataFrame,
    outdir: Path,
    stem: str = "fig_comp_02_teacher_profit_vs_p",
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    p = df["p"]
    y = df["pi_teacher_total"]
    idx = int(y.idxmax())

    ax.plot(p, y, label=r"Teacher total profit")
    ax.axvline(float(df.loc[idx, "p"]), linestyle="--", linewidth=1.0, label=fr"$p^*={df.loc[idx, 'p']:.3g}$")
    ax.scatter([df.loc[idx, "p"]], [df.loc[idx, "pi_teacher_total"]], zorder=5)

    ax.set_xlabel(r"Upstream token price $p$")
    ax.set_ylabel(r"Teacher total profit")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    _save_figure(fig, outdir / stem)
    plt.close(fig)


def plot_competition_student_profit_vs_p(
    *,
    df: pd.DataFrame,
    outdir: Path,
    stem: str = "fig_comp_05_student_profit_vs_p",
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    p = df["p"]
    y = df["pi_student_total"]
    idx_nash = int(df["pi_teacher_total"].idxmax())
    p_nash = float(df.loc[idx_nash, "p"])
    y_nash = float(df.loc[idx_nash, "pi_student_total"])

    ax.plot(p, y, label=r"Student total profit")
    ax.axvline(p_nash, linestyle="--", linewidth=1.0, label=fr"$p^*={p_nash:.3g}$")
    ax.scatter([p_nash], [y_nash], zorder=5)

    ax.set_xlabel(r"Upstream token price $p$")
    ax.set_ylabel(r"Student total profit")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    _save_figure(fig, outdir / stem)
    plt.close(fig)


def plot_competition_downstream_prices_vs_p(
    *,
    df: pd.DataFrame,
    outdir: Path,
    stem: str = "fig_comp_03_downstream_prices_vs_p",
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(df["p"], df["P_T_down_star"], label=r"$P_T^*(p)$")
    ax.plot(df["p"], df["P_S_down_star"], label=r"$P_S^*(p)$")

    ax.set_xlabel(r"Upstream token price $p$")
    ax.set_ylabel("Downstream equilibrium prices")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    _save_figure(fig, outdir / stem)
    plt.close(fig)


def plot_competition_downstream_shares_vs_p(
    *,
    df: pd.DataFrame,
    outdir: Path,
    stem: str = "fig_comp_04_downstream_shares_vs_p",
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(df["p"], df["s_T_down_star"], label=r"$s_T^*(p)$")
    ax.plot(df["p"], df["s_S_down_star"], label=r"$s_S^*(p)$")
    ax.plot(df["p"], df["s_0_down_star"], label=r"$s_0^*(p)$")

    ax.set_xlabel(r"Upstream token price $p$")
    ax.set_ylabel("Downstream market shares")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    _save_figure(fig, outdir / stem)
    plt.close(fig)


def plot_competition_threshold_strict_vs_market_size(
    *,
    df: pd.DataFrame,
    outdir: Path,
    market_col: str = "market_size",
    strict_col: str = "overall_interior_strict",
    stem: str = "fig_comp_threshold_01_strict_vs_market_size",
) -> None:
    """Plot strict interior classification (0/1) against market size."""
    data = df.sort_values(by=market_col).copy()
    y = data[strict_col].astype(int)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(data[market_col], y, where="post", label="Strict interior (1=yes)")
    ax.scatter(data[market_col], y, s=20)

    ax.set_xlabel("Downstream market size")
    ax.set_ylabel("Strict interior classification")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    _save_figure(fig, outdir / stem)
    plt.close(fig)
