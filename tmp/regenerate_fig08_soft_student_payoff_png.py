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


root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "results" / "tables" / "sens_soft_curve_families_full.csv")
out_dir = root / "results" / "figures" / "sensitivity_soft"
out_dir.mkdir(parents=True, exist_ok=True)

vars_list = ["alpha", "beta", "gamma", "k", "c_T", "tau"]

for var in vars_list:
    sub = df[df["var"] == var].copy()
    vals = sorted(sub["value"].dropna().unique().tolist())

    fig, ax = plt.subplots(figsize=(10, 8))
    for v in vals:
        s = sub[sub["value"] == v].sort_values("p")
        label = f"{pretty_label(var)}={v:.3g}"
        ax.plot(
            s["p"].to_numpy(float),
            s["pi_student_soft"].to_numpy(float),
            linewidth=2.0,
            label=label,
        )

    ax.set_xlabel(r"Upstream token price $p$", fontsize=14)
    ax.set_ylabel("Student total payoff", fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=2, fontsize=10, frameon=True)

    out = out_dir / f"fig_08_soft_student_payoff_curves_{var}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
