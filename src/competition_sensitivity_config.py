"""Stage 2: config parsing helpers for competition sensitivity analysis.

This module only parses and validates sensitivity sweep config.
It does not run any solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class SensitivitySweep1D:
    parameter_name: str
    grid: List[float]
    enabled: bool


@dataclass(frozen=True)
class CompetitionSensitivityConfig:
    u0_sweep: SensitivitySweep1D
    tau_sweep: SensitivitySweep1D


def _read_grid(section: Dict[str, Any], *, key_prefix: str) -> List[float]:
    """Read grid from either explicit list or min/max/points controls."""
    if "grid" in section:
        grid = [float(x) for x in section.get("grid", [])]
        if len(grid) < 2:
            raise ValueError(f"competition.sensitivity_analysis.{key_prefix}.grid must have >= 2 points.")
        return grid

    if {"min", "max", "points"}.issubset(section.keys()):
        lo = float(section["min"])
        hi = float(section["max"])
        pts = int(section["points"])
        if not (hi > lo):
            raise ValueError(f"competition.sensitivity_analysis.{key_prefix}.max must be > min.")
        if pts < 2:
            raise ValueError(f"competition.sensitivity_analysis.{key_prefix}.points must be >= 2.")
        return [float(x) for x in np.linspace(lo, hi, pts)]

    raise ValueError(
        "Missing sweep grid controls for "
        f"competition.sensitivity_analysis.{key_prefix}. "
        "Provide either `grid`, or (`min`, `max`, `points`)."
    )


def _validate_u0_grid(grid: List[float]) -> None:
    for v in grid:
        if not math.isfinite(v):
            raise ValueError("u0 sweep values must be finite.")


def _validate_tau_grid(grid: List[float]) -> None:
    for v in grid:
        if not math.isfinite(v):
            raise ValueError("tau sweep values must be finite.")
        if v <= 0:
            raise ValueError("tau sweep values must be > 0.")


def build_competition_sensitivity_config(competition_cfg: Dict[str, Any]) -> CompetitionSensitivityConfig:
    """Build separate sweep controls for `u0` and `tau` sensitivity.

    Expected optional path:
      competition.sensitivity_analysis.u0_sweep
      competition.sensitivity_analysis.tau_sweep

    If section is missing, default to disabled sweeps with baseline singleton grids.
    """
    comp = competition_cfg.get("competition", {})
    sens = comp.get("sensitivity_analysis", {})

    base_u0 = float(comp.get("u0", 0.0))
    base_tau = float(comp.get("tau", 1.0))

    u0_section = sens.get("u0_sweep", {})
    tau_section = sens.get("tau_sweep", {})

    if u0_section:
        u0_grid = _read_grid(u0_section, key_prefix="u0_sweep")
    else:
        u0_grid = [base_u0]

    if tau_section:
        tau_grid = _read_grid(tau_section, key_prefix="tau_sweep")
    else:
        tau_grid = [base_tau]

    _validate_u0_grid(u0_grid)
    _validate_tau_grid(tau_grid)

    u0_enabled = bool(u0_section.get("enabled", False)) if u0_section else False
    tau_enabled = bool(tau_section.get("enabled", False)) if tau_section else False

    return CompetitionSensitivityConfig(
        u0_sweep=SensitivitySweep1D(parameter_name="u0", grid=u0_grid, enabled=u0_enabled),
        tau_sweep=SensitivitySweep1D(parameter_name="tau", grid=tau_grid, enabled=tau_enabled),
    )
