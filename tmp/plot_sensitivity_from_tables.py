import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parents[1]

hard_df = pd.read_csv(root / "results" / "tables" / "sens_hard_oat.csv")
soft_df = pd.read_csv(root / "results" / "tables" / "sens_soft_oat.csv")
summary_text = (root / "results" / "tables" / "sens_core_soft_summary.json").read_text(encoding="utf-8")
summary = json.loads(summary_text.replace("NaN", "null"))

hard_base_pi = float(summary["baselines"]["hard_baseline_metrics"]["pi_teacher_star"])
soft_base_pi = float(summary["baselines"]["soft_baseline_metrics"]["pi_teacher_star"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for ax, df, mode, base_pi in [
    (axes[0], hard_df, "hard", hard_base_pi),
    (axes[1], soft_df, "soft", soft_base_pi),
]:
    for var in sorted(df["var"].dropna().unique().tolist()):
        sub = df[df["var"] == var].sort_values("value_ratio")
        ax.plot(
            sub["value_ratio"],
            sub["pi_teacher_star"] / base_pi,
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=var,
        )

    ax.set_title(f"{mode.upper()} OAT Curves")
    ax.set_xlabel("parameter ratio (value / baseline)")
    ax.set_ylabel("normalized teacher optimum payoff")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8)

out_base = root / "results" / "figures" / "fig_06_sensitivity_curves_core_soft"
out_base.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)

print("saved", out_base.with_suffix(".pdf"))
print("saved", out_base.with_suffix(".png"))
print("saved", out_base.with_suffix(".svg"))
