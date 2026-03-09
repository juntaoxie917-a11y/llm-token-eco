"""Stage 5 plotting helpers for competition outputs.

These functions consume saved tabular results (DataFrame/CSV) and only plot.
They never run equilibrium solvers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

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


def _annotate_critical_m(
    ax: plt.Axes,
    *,
    critical_m: Optional[float],
    critical_interval: Optional[Tuple[float, float]] = None,
) -> None:
    """Add critical-market-size marker in the same spirit as p* annotations."""
    if critical_m is None:
        return

    m_star = float(critical_m)
    ax.axvline(m_star, linestyle="--", linewidth=1.0, label=fr"$m_c\approx{m_star:.4g}$")

    y0, y1 = ax.get_ylim()
    y_text = y0 + 0.92 * (y1 - y0)
    if critical_interval is None:
        text = fr"$m_c\approx {m_star:.4g}$"
    else:
        lo, hi = float(critical_interval[0]), float(critical_interval[1])
        text = fr"$m_c\in[{lo:.4g}, {hi:.4g}]$"

    ax.text(
        m_star,
        y_text,
        text,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, linewidth=0.5),
    )


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
    critical_m: Optional[float] = None,
    critical_interval: Optional[Tuple[float, float]] = None,
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
    _annotate_critical_m(ax, critical_m=critical_m, critical_interval=critical_interval)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    _save_figure(fig, outdir / stem)
    plt.close(fig)


def plot_competition_threshold_weak_vs_market_size(
    *,
    df: pd.DataFrame,
    outdir: Path,
    market_col: str = "market_size",
    weak_col: str = "overall_interior_weak",
    critical_m: Optional[float] = None,
    critical_interval: Optional[Tuple[float, float]] = None,
    stem: str = "fig_comp_threshold_02_weak_vs_market_size",
) -> None:
    """Plot weak interior classification (0/1) against market size."""
    if weak_col not in df.columns:
        return

    data = df.sort_values(by=market_col).copy()
    # Handle optional None values robustly.
    y = data[weak_col].fillna(False).astype(int)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(data[market_col], y, where="post", label="Weak interior (1=yes)")
    ax.scatter(data[market_col], y, s=20)

    ax.set_xlabel("Downstream market size")
    ax.set_ylabel("Weak interior classification")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    _annotate_critical_m(ax, critical_m=critical_m, critical_interval=critical_interval)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    _save_figure(fig, outdir / stem)
    plt.close(fig)


def plot_competition_threshold_p_star_vs_market_size(
    *,
    df: pd.DataFrame,
    outdir: Path,
    market_col: str = "market_size",
    p_col: str = "p_star",
    critical_m: Optional[float] = None,
    critical_interval: Optional[Tuple[float, float]] = None,
    stem: str = "fig_comp_threshold_03_p_star_vs_market_size",
) -> None:
    data = df.sort_values(by=market_col).copy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[market_col], data[p_col], marker="o", label=r"$p^*(M)$")
    _annotate_critical_m(ax, critical_m=critical_m, critical_interval=critical_interval)

    ax.set_xlabel("Downstream market size")
    ax.set_ylabel("Teacher optimal upstream price")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    _save_figure(fig, outdir / stem)
    plt.close(fig)


def plot_competition_threshold_d_star_vs_market_size(
    *,
    df: pd.DataFrame,
    outdir: Path,
    market_col: str = "market_size",
    d_col: str = "D_star",
    critical_m: Optional[float] = None,
    critical_interval: Optional[Tuple[float, float]] = None,
    stem: str = "fig_comp_threshold_04_d_star_vs_market_size",
) -> None:
    data = df.sort_values(by=market_col).copy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[market_col], data[d_col], marker="o", label=r"$D^*(M)$")
    _annotate_critical_m(ax, critical_m=critical_m, critical_interval=critical_interval)

    ax.set_xlabel("Downstream market size")
    ax.set_ylabel("Student equilibrium training demand")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    _save_figure(fig, outdir / stem)
    plt.close(fig)


def plot_competition_threshold_teacher_payoff_vs_market_size(
    *,
    df: pd.DataFrame,
    outdir: Path,
    market_col: str = "market_size",
    payoff_col: str = "pi_teacher_total_star",
    critical_m: Optional[float] = None,
    critical_interval: Optional[Tuple[float, float]] = None,
    stem: str = "fig_comp_threshold_05_teacher_payoff_vs_market_size",
) -> None:
    data = df.sort_values(by=market_col).copy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[market_col], data[payoff_col], marker="o", label="Teacher total payoff at optimum")
    _annotate_critical_m(ax, critical_m=critical_m, critical_interval=critical_interval)

    ax.set_xlabel("Downstream market size")
    ax.set_ylabel("Teacher total payoff")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    _save_figure(fig, outdir / stem)
    plt.close(fig)


def plot_competition_threshold_distance_diagnostics_vs_market_size(
    *,
    df: pd.DataFrame,
    outdir: Path,
    market_col: str = "market_size",
    price_dist_col: str = "price_distance_to_boundary",
    demand_dist_col: str = "demand_distance_to_boundary",
    critical_m: Optional[float] = None,
    critical_interval: Optional[Tuple[float, float]] = None,
    stem: str = "fig_comp_threshold_06_distances_vs_market_size",
) -> None:
    data = df.sort_values(by=market_col).copy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[market_col], data[price_dist_col], marker="o", label="Price distance to boundary")
    ax.plot(data[market_col], data[demand_dist_col], marker="o", label="Demand distance to boundary")
    _annotate_critical_m(ax, critical_m=critical_m, critical_interval=critical_interval)

    ax.set_xlabel("Downstream market size")
    ax.set_ylabel("Distance to boundary")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    _save_figure(fig, outdir / stem)
    plt.close(fig)


def plot_competition_threshold_min_share_vs_market_size(
    *,
    df: pd.DataFrame,
    outdir: Path,
    market_col: str = "market_size",
    min_share_col: str = "min_share",
    critical_m: Optional[float] = None,
    critical_interval: Optional[Tuple[float, float]] = None,
    stem: str = "fig_comp_threshold_07_min_share_vs_market_size",
) -> None:
    data = df.sort_values(by=market_col).copy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[market_col], data[min_share_col], marker="o", label="Min downstream share")
    _annotate_critical_m(ax, critical_m=critical_m, critical_interval=critical_interval)

    ax.set_xlabel("Downstream market size")
    ax.set_ylabel("Minimum of (s_T, s_S, s_0)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    _save_figure(fig, outdir / stem)
    plt.close(fig)


def plot_competition_threshold_suite(
    *,
    df: pd.DataFrame,
    outdir: Path,
    include_weak: bool = True,
    critical_m: Optional[float] = None,
    critical_interval: Optional[Tuple[float, float]] = None,
) -> None:
    """Generate the Stage-5 threshold figure suite from a saved table/DataFrame."""
    plot_competition_threshold_strict_vs_market_size(
        df=df,
        outdir=outdir,
        critical_m=critical_m,
        critical_interval=critical_interval,
    )
    if include_weak:
        plot_competition_threshold_weak_vs_market_size(
            df=df,
            outdir=outdir,
            critical_m=critical_m,
            critical_interval=critical_interval,
        )
    plot_competition_threshold_p_star_vs_market_size(
        df=df,
        outdir=outdir,
        critical_m=critical_m,
        critical_interval=critical_interval,
    )
    plot_competition_threshold_d_star_vs_market_size(
        df=df,
        outdir=outdir,
        critical_m=critical_m,
        critical_interval=critical_interval,
    )
    plot_competition_threshold_teacher_payoff_vs_market_size(
        df=df,
        outdir=outdir,
        critical_m=critical_m,
        critical_interval=critical_interval,
    )
    plot_competition_threshold_distance_diagnostics_vs_market_size(
        df=df,
        outdir=outdir,
        critical_m=critical_m,
        critical_interval=critical_interval,
    )
    plot_competition_threshold_min_share_vs_market_size(
        df=df,
        outdir=outdir,
        critical_m=critical_m,
        critical_interval=critical_interval,
    )
