from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from experiments._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(__file__)


def _save_figure(fig: plt.Figure, outpath_base: Path, *, save_png: bool = True, dpi: int = 300) -> None:
    outpath_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outpath_base.with_suffix(".svg"), bbox_inches="tight")
    if save_png:
        fig.savefig(outpath_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def _label_to_code(label: str) -> int:
    mapping = {
        "Interior": 0,
        "Boundary in other variables": 1,
        "Boundary at $p_{\\text{max}}$": 2,
        "Unresolved": 3,
    }
    return int(mapping.get(str(label), 3))


def _plot_label_heatmap(*, panel: pd.DataFrame, parameter: str, outdir: Path) -> None:
    data = panel[panel["parameter"] == parameter].copy()
    if data.empty:
        return

    data["label_code"] = data["label"].map(_label_to_code)
    pivot = (
        data.pivot_table(
            index="p_max",
            columns="parameter_value",
            values="label_code",
            aggfunc="first",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Fixed category palette: interior/bound-limited/pmax-insensitive/unresolved.
    cmap = plt.matplotlib.colors.ListedColormap(["#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    x_vals = [float(x) for x in pivot.columns.tolist()]
    y_vals = [float(y) for y in pivot.index.tolist()]

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=(min(x_vals), max(x_vals), min(y_vals), max(y_vals)),
    )

    ax.set_xlabel({"market_size": "downstream market size $M$", "u0": "outside option utility $u_0$", "tau": r"price sensitivity $\tau$"}[parameter])
    ax.set_ylabel(r"upstream price cap $p_{\max}$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([
        "Interior",
        "Boundary in other variables",
        "Boundary at $p_{\\text{max}}$",
        "Unresolved",
    ])

    stem = {
        "market_size": "fig_unconstrained_M_label_heatmap",
        "u0": "fig_unconstrained_u0_label_heatmap",
        "tau": "fig_unconstrained_tau_label_heatmap",
    }[parameter]
    _save_figure(fig, outdir / stem)
    plt.close(fig)


def _plot_threshold_endpoint_vs_pmax(*, summary: pd.DataFrame, outdir: Path) -> None:
    data = summary.copy()
    data = data[data["parameter"].isin(["market_size", "u0", "tau"])].copy()
    data = data.sort_values(by=["parameter", "p_max"])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for parameter, group in data.groupby("parameter"):
        x = group["p_max"].astype(float).values
        y_last = pd.to_numeric(group["last_interior"], errors="coerce").values
        y_first = pd.to_numeric(group["first_interior"], errors="coerce").values

        label_last = {
            "market_size": r"$M$: last interior",
            "u0": r"$u_0$: last interior",
            "tau": r"$\tau$: last interior",
        }[parameter]
        ax.plot(x, y_last, marker="o", label=label_last)

        # Plot first endpoint with dashed style to show interval span evolution.
        label_first = {
            "market_size": r"$M$: first interior",
            "u0": r"$u_0$: first interior",
            "tau": r"$\tau$: first interior",
        }[parameter]
        ax.plot(x, y_first, marker="o", linestyle="--", alpha=0.7, label=label_first)

    ax.set_xlabel(r"upstream price cap $p_{\max}$")
    ax.set_ylabel("interior interval endpoints")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(ncols=2, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    _save_figure(fig, outdir / "fig_unconstrained_threshold_endpoint_vs_pmax")
    plt.close(fig)


def main() -> None:
    project_root = PROJECT_ROOT
    out_figs = project_root / "results" / "figures" / "competition" / "sensitivity" / "unconstrained_like"
    out_tables = project_root / "results" / "tables" / "unconstrained_like"

    panel_csv = out_tables / "competition_unconstrained_like_panel.csv"
    summary_csv = out_tables / "competition_unconstrained_like_stability_summary.csv"

    if not panel_csv.exists():
        raise FileNotFoundError(f"Missing panel table: {panel_csv}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary table: {summary_csv}")

    panel = pd.read_csv(panel_csv)
    summary = pd.read_csv(summary_csv)

    _plot_label_heatmap(panel=panel, parameter="market_size", outdir=out_figs)
    _plot_label_heatmap(panel=panel, parameter="u0", outdir=out_figs)
    _plot_label_heatmap(panel=panel, parameter="tau", outdir=out_figs)
    _plot_threshold_endpoint_vs_pmax(summary=summary, outdir=out_figs)

    print("Stage 13 unconstrained-like plots completed.")
    print("Saved:")
    print(" -", out_figs / "fig_unconstrained_M_label_heatmap")
    print(" -", out_figs / "fig_unconstrained_u0_label_heatmap")
    print(" -", out_figs / "fig_unconstrained_tau_label_heatmap")
    print(" -", out_figs / "fig_unconstrained_threshold_endpoint_vs_pmax")


if __name__ == "__main__":
    main()
