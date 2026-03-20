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
        "k": r"$k$",
        "c_T": r"$c_T$",
        "tau": r"$\tau$",
    }
    return mapping.get(var, var)


def make_family(df: pd.DataFrame, var: str, metric: str, ylabel: str, out_path: Path) -> None:
    sub = df[df["var"] == var].copy()
    vals = sorted(sub["value"].dropna().unique().tolist())

    fig, ax = plt.subplots(figsize=(10, 8))
    for v in vals:
        s = sub[sub["value"] == v].sort_values("p")
        ax.plot(
            s["p"].to_numpy(float),
            s[metric].to_numpy(float),
            linewidth=2.0,
            label=f"{pretty_label(var)}={v:.3g}",
        )

    ax.set_xlabel(r"Upstream token price $p$", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=2, fontsize=10, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


root = Path(__file__).resolve().parents[1]
curve_csv = root / "results" / "tables" / "sens_soft_curve_families_full.csv"
out_dir = root / "results" / "figures" / "sensitivity_soft"

df = pd.read_csv(curve_csv)

vars_list = ["alpha", "beta", "gamma", "k", "c_T", "tau"]
metric_specs = [
    ("pi_teacher_soft", "Teacher total payoff", "fig_09_soft_teacher_payoff_curves"),
    ("D_soft", "Soft demand", "fig_10_soft_dsoft_curves"),
    ("s_enter", "Entry probability", "fig_11_soft_entry_prob_curves"),
    ("D_star", "Conditional demand D*", "fig_12_soft_dstar_curves"),
    ("pi_student_star", "Conditional student payoff", "fig_13_soft_student_star_payoff_curves"),
    ("L_student", "Student loss", "fig_14_soft_student_loss_curves"),
    ("L_tilde", "Supervised frontier loss", "fig_15_soft_ltilde_curves"),
    ("gap", "Distillation gap", "fig_16_soft_gap_curves"),
]

count = 0
for metric, ylabel, prefix in metric_specs:
    for var in vars_list:
        out = out_dir / f"{prefix}_{var}.png"
        make_family(df=df, var=var, metric=metric, ylabel=ylabel, out_path=out)
        count += 1
        print("saved", out)

print("total_png", count)
