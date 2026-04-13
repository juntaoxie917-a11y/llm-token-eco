"""Shared I/O helpers for cross-platform, reproducible outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as Path."""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_figure_bundle(
    fig: Any,
    outpath_base: str | Path,
    *,
    save_png: bool = True,
    dpi: int = 300,
) -> Path:
    """Save one figure as PDF/SVG and optionally PNG using a common base path."""
    base = Path(outpath_base)
    ensure_dir(base.parent)

    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    if save_png:
        fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    return base


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Write JSON with UTF-8 encoding and parent-directory creation."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)
    return out_path
