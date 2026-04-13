from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def pretty_label(var: str) -> str:
    mapping = {
        "alpha": r"$\alpha$",
        "beta": r"$\beta$",
        "gamma": r"$\gamma$",
        "tau": r"$\tau$",
        "c_T": r"$c_T$",
        "k": r"$k$",
    }
    return mapping.get(var, var)


def render_metric(mode: str, df: pd.DataFrame, out_dir: Path, fig_prefix: str, metric_col: str, ylabel: str) -> None:
    vars_list = sorted(df["var"].dropna().unique().tolist())
    out_dir.mkdir(parents=True, exist_ok=True)

    for var in vars_list:
        sub = df[df["var"] == var].copy()
        values = sorted(sub["value"].dropna().unique().tolist())

        fig = plt.figure(figsize=(9, 5), constrained_layout=True)
        ax = fig.add_subplot(111)

        for v in values:
            s = sub[sub["value"] == v].sort_values("p")
            ax.plot(
                s["p"].to_numpy(float),
                s[metric_col].to_numpy(float),
                linewidth=1.8,
                label=f"{pretty_label(var)}={v:.4g}",
            )

        ax.set_xlabel(r"Upstream token price $p$")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=8, ncol=2)

        out_base = out_dir / f"{fig_prefix}_{mode}_{'student' if metric_col == 'student_total_payoff' else 'teacher'}_payoff_curves_{var}"
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)

        print("saved", out_base.with_suffix(".pdf"))


def load_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"skip missing file: {path}")
        return None
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render payoff-curve families from saved sensitivity tables")
    parser.add_argument("--modes", nargs="+", choices=["soft", "hard"], default=["soft"], help="Which modes to render")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    tables = root / "results" / "tables"
    figs = root / "results" / "figures"

    for mode in args.modes:
        table_path = tables / f"sens_{mode}_student_payoff_curves.csv"
        df = load_if_exists(table_path)
        if df is None:
            continue

        required = {"p", "var", "value", "student_total_payoff", "teacher_total_payoff"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing columns in {table_path}: {missing}")

        render_metric(mode, df, figs, "fig_08" if mode == "soft" else "fig_07", "student_total_payoff", "Student total payoff")
        render_metric(mode, df, figs, "fig_09" if mode == "soft" else "fig_08", "teacher_total_payoff", "Teacher total payoff")


if __name__ == "__main__":
    main()
