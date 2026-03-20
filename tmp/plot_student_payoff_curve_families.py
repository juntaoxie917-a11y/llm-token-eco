from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def pretty_label(var: str) -> str:
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


def render(mode: str, df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
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
                s["student_total_payoff"].to_numpy(float),
                linewidth=1.8,
                label=f"{pretty_label(var)}={v:.4g}",
            )

        ax.set_xlabel(r"Upstream token price $p$")
        ax.set_ylabel("Student total payoff")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=8, ncol=2)

        out_base = out_dir / f"{prefix}_{mode}_student_payoff_curves_{var}"
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)

        print("saved", out_base.with_suffix(".pdf"))
        print("saved", out_base.with_suffix(".png"))
        print("saved", out_base.with_suffix(".svg"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tables = root / "results" / "tables"
    figs = root / "results" / "figures"

    hard = pd.read_csv(tables / "sens_hard_student_payoff_curves.csv")
    soft = pd.read_csv(tables / "sens_soft_student_payoff_curves.csv")

    render("hard", hard, figs, "fig_07")
    render("soft", soft, figs, "fig_08")


if __name__ == "__main__":
    main()
