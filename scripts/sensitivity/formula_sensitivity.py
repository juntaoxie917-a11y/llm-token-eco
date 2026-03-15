import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def L_tilde(N, D, A, B, E, alpha, beta, gamma):
    U = A * (N ** (-alpha)) + B * (D ** (-beta))
    return E + (U ** gamma), U


def partials(N, D, A, B, E, alpha, beta, gamma):
    L, U = L_tilde(N, D, A, B, E, alpha, beta, gamma)
    eps = 1e-12
    U = max(U, eps)
    L = max(L, eps)

    dN = gamma * (U ** (gamma - 1.0)) * (-A * alpha * (N ** (-alpha - 1.0)))
    dD = gamma * (U ** (gamma - 1.0)) * (-B * beta * (D ** (-beta - 1.0)))
    dA = gamma * (U ** (gamma - 1.0)) * (N ** (-alpha))
    dB = gamma * (U ** (gamma - 1.0)) * (D ** (-beta))
    dE = 1.0
    da = gamma * (U ** (gamma - 1.0)) * (-A * np.log(N) * (N ** (-alpha)))
    db = gamma * (U ** (gamma - 1.0)) * (-B * np.log(D) * (D ** (-beta)))
    dg = (U ** gamma) * np.log(U)

    grad = {
        "N_s": dN, "D": dD, "A": dA, "B": dB, "E": dE,
        "alpha": da, "beta": db, "gamma": dg
    }

    # 归一化敏感度: S_x = (x/L) * dL/dx
    S = {
        "N_s": (N / L) * dN,
        "D": (D / L) * dD,
        "A": (A / L) * dA,
        "B": (B / L) * dB,
        "alpha": (alpha / L) * da if alpha > 0 else np.nan,
        "beta": (beta / L) * db if beta > 0 else np.nan,
        "gamma": (gamma / L) * dg if gamma > 0 else np.nan,
    }
    return L, grad, S


def tornado_plot(s_dict, out_png):
    items = [(k, abs(v)) for k, v in s_dict.items() if np.isfinite(v)]
    items = sorted(items, key=lambda x: x[1], reverse=True)

    names = [k for k, _ in items]
    vals = [s_dict[k] for k in names]

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(names))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized sensitivity S_x = (x/L) dL/dx")
    ax.set_title("Local Sensitivity Tornado (at baseline)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def curve_plot(x, y, xlabel, ylabel, title, out_png):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sensitivity_mvp.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    out_dir = Path(cfg.get("output_dir", "results"))
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    m = cfg["model"]
    fs = cfg["formula_sensitivity"]
    vars_cfg = {v["name"]: v["bounds"] for v in cfg["variables"]}

    A = float(m["A"]); B = float(m["B"]); E = float(m["E"]); gamma = float(m["gamma"])
    alpha = float(np.mean(vars_cfg["alpha"]))
    beta = float(np.mean(vars_cfg["beta"]))

    N0 = float(fs["baseline"]["N_s"])
    D0 = float(fs["baseline"]["D"])

    # 基线点局部敏感度
    L0, grad0, S0 = partials(N0, D0, A, B, E, alpha, beta, gamma)
    df_local = pd.DataFrame({
        "var": list(S0.keys()),
        "normalized_sensitivity": [S0[k] for k in S0.keys()],
        "abs_normalized_sensitivity": [abs(S0[k]) for k in S0.keys()],
    }).sort_values("abs_normalized_sensitivity", ascending=False)
    df_local.to_csv(tables_dir / "formula_local_sensitivity.csv", index=False)
    tornado_plot(S0, figures_dir / "fig_formula_tornado.png")

    # N 扫描：看 S_N 与 L 的变化
    N_scan = np.logspace(
        np.log10(float(fs["N_scan"]["min"])),
        np.log10(float(fs["N_scan"]["max"])),
        int(fs["N_scan"]["points"]),
    )
    rows_n = []
    for N in N_scan:
        L, _, S = partials(float(N), D0, A, B, E, alpha, beta, gamma)
        rows_n.append({"N_s": N, "L": L, "S_N_s": S["N_s"]})
    df_n = pd.DataFrame(rows_n)
    df_n.to_csv(tables_dir / "formula_curve_N.csv", index=False)
    curve_plot(df_n["N_s"], df_n["S_N_s"], "N_s", "S_N_s", "Sensitivity curve: S_N_s vs N_s", figures_dir / "fig_formula_curve_N.png")

    # D 扫描：看 S_D 与 L 的变化
    D_scan = np.logspace(
        np.log10(float(fs["D_scan"]["min"])),
        np.log10(float(fs["D_scan"]["max"])),
        int(fs["D_scan"]["points"]),
    )
    rows_d = []
    for D in D_scan:
        L, _, S = partials(N0, float(D), A, B, E, alpha, beta, gamma)
        rows_d.append({"D": D, "L": L, "S_D": S["D"]})
    df_d = pd.DataFrame(rows_d)
    df_d.to_csv(tables_dir / "formula_curve_D.csv", index=False)
    curve_plot(df_d["D"], df_d["S_D"], "D", "S_D", "Sensitivity curve: S_D vs D", figures_dir / "fig_formula_curve_D.png")

    print(f"[OK] figures -> {figures_dir}")
    print(f"[OK] tables  -> {tables_dir}")


if __name__ == "__main__":
    main()