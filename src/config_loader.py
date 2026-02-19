"""Configuration loader and lightweight validation.

Loads YAML configuration and performs minimal validation to catch common mistakes early.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # PyYAML
except ImportError as e:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping/dict.")
    return cfg


def _require(cfg: Dict[str, Any], keys: list[str], ctx: str = "") -> None:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            prefix = f"[{ctx}] " if ctx else ""
            raise KeyError(prefix + "Missing required key: " + "/".join(keys))
        cur = cur[k]


def validate_base_config(cfg: Dict[str, Any]) -> None:
    _require(cfg, ["student", "N0"], ctx="student")
    _require(cfg, ["exponents"], ctx="exponents")
    _require(cfg, ["anchors_supervised"], ctx="anchors_supervised")
    _require(cfg, ["anchors_gap"], ctx="anchors_gap")
    _require(cfg, ["economics"], ctx="economics")
    _require(cfg, ["grids"], ctx="grids")
    _require(cfg, ["solver"], ctx="solver")

    for k in ["alpha", "beta", "gamma", "alpha_p", "beta_p", "gamma_p"]:
        _require(cfg, ["exponents", k], ctx="exponents")

    for k in ["D0", "D1", "L0", "L1", "E"]:
        _require(cfg, ["anchors_supervised", k], ctx="anchors_supervised")

    for k in ["rho", "q", "A_p"]:
        _require(cfg, ["anchors_gap", k], ctx="anchors_gap")

    for k in ["D_min", "D_max", "p_min", "p_max", "p_points"]:
        _require(cfg, ["grids", k], ctx="grids")

    D0 = float(cfg["anchors_supervised"]["D0"])
    D1 = float(cfg["anchors_supervised"]["D1"])
    if not (D0 > 0 and D1 > D0):
        raise ValueError("Require D1 > D0 > 0 in anchors_supervised.")

    L0 = float(cfg["anchors_supervised"]["L0"])
    L1 = float(cfg["anchors_supervised"]["L1"])
    if not (L0 > 0 and L1 > 0 and L1 < L0):
        raise ValueError("Require L0>0, L1>0 and L1 < L0 in anchors_supervised.")

    rho = float(cfg["anchors_gap"]["rho"])
    q = float(cfg["anchors_gap"]["q"])
    if not (0 < rho < 1):
        raise ValueError("Require 0<rho<1 in anchors_gap.")
    if not (0 < q < 1):
        raise ValueError("Require 0<q<1 in anchors_gap.")

    N0 = float(cfg["student"]["N0"])
    if N0 <= 0:
        raise ValueError("student.N0 must be > 0.")

    D_min = float(cfg["grids"]["D_min"])
    D_max = float(cfg["grids"]["D_max"])
    if not (0 < D_min < D_max):
        raise ValueError("Require 0 < D_min < D_max in grids.")

    p_min = float(cfg["grids"]["p_min"])
    p_max = float(cfg["grids"]["p_max"])
    if not (p_max > p_min):
        raise ValueError("Require p_max > p_min.")

    p_points = int(cfg["grids"]["p_points"])
    if p_points < 10:
        raise ValueError("p_points should be >= 10 for a smooth demand curve.")


def load_and_validate(path: str | Path) -> Dict[str, Any]:
    cfg = load_yaml(path)
    validate_base_config(cfg)
    return cfg
