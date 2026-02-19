"""Anchor-based normalization for Tier A.

Supervised frontier (baseline uses E=0):
    Ltilde(D) = E + (A + B * D^{-beta})^{gamma}
Anchors:
    Ltilde(D0)=L0, Ltilde(D1)=L1
=> Solve A,B in closed form (when E=0).

Gap term (Tier A reduced form):
    gap(D) = lam * (A_p + B_p * D^{-beta_p})^{gamma_p}
Anchors:
    gap(D0)=rho*L0
    gap(D1)=q*gap(D0)
=> Solve B_p and lam in closed form (A_p fixed).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SupervisedCoeffs:
    E: float
    A: float
    B: float


@dataclass(frozen=True)
class GapCoeffs:
    A_p: float
    B_p: float
    lam: float


def solve_supervised_coeffs(*, D0: float, D1: float, L0: float, L1: float, beta: float, gamma: float, E: float = 0.0) -> SupervisedCoeffs:
    if abs(E) > 1e-15:
        raise ValueError("Closed-form solver assumes E=0. For E>0, add a third anchor or set E=0.")
    if not (D0 > 0 and D1 > D0):
        raise ValueError("Require D1>D0>0.")
    if not (L0 > 0 and L1 > 0 and L1 < L0):
        raise ValueError("Require L0>0, L1>0 and L1<L0.")
    if beta <= 0 or gamma <= 0:
        raise ValueError("Require beta>0 and gamma>0.")

    t0 = L0 ** (1.0 / gamma)
    t1 = L1 ** (1.0 / gamma)
    denom = (D0 ** (-beta) - D1 ** (-beta))
    if abs(denom) < 1e-18:
        raise ValueError("Degenerate anchors: D0^{-beta} == D1^{-beta}.")

    B = (t0 - t1) / denom
    A = t0 - B * (D0 ** (-beta))
    if A <= 0:
        raise ValueError(f"Solved A<=0 (A={A}). Adjust anchors/exponents.")
    if B <= 0:
        raise ValueError(f"Solved B<=0 (B={B}). Adjust anchors/exponents.")
    return SupervisedCoeffs(E=float(E), A=float(A), B=float(B))


def solve_gap_coeffs(*, D0: float, D1: float, L0: float, rho: float, q: float, A_p: float, beta_p: float, gamma_p: float) -> GapCoeffs:
    if not (D0 > 0 and D1 > D0):
        raise ValueError("Require D1>D0>0.")
    if not (L0 > 0):
        raise ValueError("Require L0>0.")
    if not (0 < rho < 1):
        raise ValueError("Require 0<rho<1.")
    if not (0 < q < 1):
        raise ValueError("Require 0<q<1.")
    if A_p <= 0:
        raise ValueError("Require A_p>0.")
    if beta_p <= 0 or gamma_p <= 0:
        raise ValueError("Require beta_p>0 and gamma_p>0.")

    r = q ** (1.0 / gamma_p)
    denom = (D1 ** (-beta_p) - r * (D0 ** (-beta_p)))
    num = r - 1.0
    if abs(denom) < 1e-18:
        raise ValueError("Degenerate anchors for gap term; adjust D1/D0 or beta_p/gamma_p.")

    B_p = (num * A_p) / denom
    if A_p + B_p * (D0 ** (-beta_p)) <= 0:
        raise ValueError("Invalid gap parameters: A_p + B_p D0^{-beta_p} <= 0.")

    lam = (rho * L0) / ((A_p + B_p * (D0 ** (-beta_p))) ** gamma_p)
    if lam <= 0:
        raise ValueError("Solved lam<=0; check anchors.")
    return GapCoeffs(A_p=float(A_p), B_p=float(B_p), lam=float(lam))
