from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict


def normalize_path(path: str) -> str:
    """Normalize path for cross-platform compatibility."""
    return os.path.normpath(path)


def ensure_dir(path: str) -> None:
    """Ensure directory exists, creating it if necessary."""
    normalized_path = os.path.normpath(path)
    os.makedirs(normalized_path, exist_ok=True)


def make_run_dir(base_output_dir: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Ensure cross-platform path handling
    normalized_base = normalize_path(base_output_dir)
    run_dir = os.path.join(normalized_base, ts)
    ensure_dir(run_dir)
    return run_dir


def write_json(obj: Dict[str, Any], output_path: str) -> str:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return output_path


