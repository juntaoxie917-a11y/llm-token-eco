from __future__ import annotations

import sys
from pathlib import Path


def project_root_from_file(file_path: str) -> Path:
    """Resolve repository root from an experiment script path."""
    return Path(file_path).resolve().parents[1]


def ensure_project_root_on_path(file_path: str) -> Path:
    """Prepend repository root to sys.path for direct script execution."""
    root = project_root_from_file(file_path)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
