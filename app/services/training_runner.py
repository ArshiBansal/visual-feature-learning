"""Launch and inspect a baseline training process from the dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def start_training(project_root: Path, log_path: Path) -> subprocess.Popen:
    """Start the project baseline trainer and redirect output to its log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        return subprocess.Popen(
            [sys.executable, "-u", str(project_root / "scripts" / "train.py")],
            cwd=project_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )


def read_training_log(log_path: Path, *, max_characters: int = 12000) -> str:
    """Return a bounded tail of the training log for the live console panel."""
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")
    return log_path.read_text(encoding="utf-8", errors="replace")[-max_characters:]