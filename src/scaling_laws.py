"""Scaling-law technology block (Tier A).

Provides callable functions:
  - supervised frontier L_tilde(N,D)
  - gap term gap_term(N,D)
  - distilled loss L_student(N,D) = L_tilde + gap_term

Built from anchor-normalized coefficients in `normalization.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from .normalization import SupervisedCoeffs, GapCoeffs, solve_supervised_coeffs, solve_gap_coeffs


@dataclass(frozen=True)
class Exponents:
    alpha: float
    beta: float
    gamma: float
    alpha_p: float
    beta_p: float
    gamma_p: float


@dataclass(frozen=True)
class TierATechnology:
    N0: float
    exponents: Exponents
    supervised: SupervisedCoeffs
    gap: GapCoeffs

    def n(self, N: float) -> float:
        if N <= 0:
            raise ValueError("Require N>0.")
        return N / self.N0

    def L_tilde(self, N: float, D: float) -> float:
        if D <= 0:
            raise ValueError("Require D>0.")
        n = self.n(N)
        a, b, g = self.exponents.alpha, self.exponents.beta, self.exponents.gamma
        E, A, B = self.supervised.E, self.supervised.A, self.supervised.B
        inside = (A / (n ** a)) + B * (D ** (-b))
        if inside <= 0:
            raise ValueError("Invalid inside term for L_tilde (<=0). Check parameters.")
        return E + (inside ** g)

    def gap_term(self, N: float, D: float) -> float:
        if D <= 0:
            raise ValueError("Require D>0.")
        n = self.n(N)
        ap, bp, gp = self.exponents.alpha_p, self.exponents.beta_p, self.exponents.gamma_p
        A_p, B_p, lam = self.gap.A_p, self.gap.B_p, self.gap.lam
        inside = (A_p / (n ** ap)) + B_p * (D ** (-bp))
        if inside <= 0:
            raise ValueError("Invalid inside term for gap_term (<=0). Check parameters.")
        return lam * (inside ** gp)

    def L_student(self, N: float, D: float) -> float:
        return self.L_tilde(N, D) + self.gap_term(N, D)

    def check_anchors(self, *, D0: float, D1: float, L0: float, L1: float, rho: float, q: float) -> Dict[str, float]:
        N = self.N0
        Lt0 = self.L_tilde(N, D0)
        Lt1 = self.L_tilde(N, D1)
        g0 = self.gap_term(N, D0)
        g1 = self.gap_term(N, D1)
        return {
            "Ltilde_D0_error": float(Lt0 - L0),
            "Ltilde_D1_error": float(Lt1 - L1),
            "gap_D0_error": float(g0 - rho * L0),
            "gap_D1_error": float(g1 - (q * rho * L0)),
        }


def build_tierA_from_config(cfg: Dict[str, Any]) -> TierATechnology:
    N0 = float(cfg["student"]["N0"])
    exp = cfg["exponents"]
    exponents = Exponents(
        alpha=float(exp["alpha"]),
        beta=float(exp["beta"]),
        gamma=float(exp["gamma"]),
        alpha_p=float(exp["alpha_p"]),
        beta_p=float(exp["beta_p"]),
        gamma_p=float(exp["gamma_p"]),
    )

    sup = cfg["anchors_supervised"]
    D0, D1 = float(sup["D0"]), float(sup["D1"])
    L0, L1 = float(sup["L0"]), float(sup["L1"])
    E = float(sup.get("E", 0.0))
    supervised = solve_supervised_coeffs(D0=D0, D1=D1, L0=L0, L1=L1, beta=exponents.beta, gamma=exponents.gamma, E=E)

    gap_cfg = cfg["anchors_gap"]
    rho, q = float(gap_cfg["rho"]), float(gap_cfg["q"])
    A_p = float(gap_cfg.get("A_p", 1.0))
    gap = solve_gap_coeffs(D0=D0, D1=D1, L0=L0, rho=rho, q=q, A_p=A_p, beta_p=exponents.beta_p, gamma_p=exponents.gamma_p)

    return TierATechnology(N0=N0, exponents=exponents, supervised=supervised, gap=gap)
