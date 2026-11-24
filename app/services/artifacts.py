"""Resolve and summarize artifacts produced by CIFAR-10 workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    """Project paths used by dashboard pages."""

    config: Path
    training_history: Path
    training_log: Path
    checkpoint: Path
    metrics: Path
    ablation_results: Path


def artifact_paths(project_root: Path) -> ArtifactPaths:
    """Build configured locations for model outputs and inputs."""
    return ArtifactPaths(
        config=project_root / "configs" / "cifar10_baseline.json",
        training_history=project_root
        / "outputs"
        / "cifar10_baseline"
        / "training_history.json",
        training_log=project_root / "outputs" / "cifar10_baseline" / "training.log",
        checkpoint=project_root
        / "checkpoints"
        / "cifar10_baseline"
        / "cifar10_cnn_latest.npz",
        metrics=project_root / "outputs" / "cifar10_baseline" / "final_metrics.json",
        ablation_results=project_root / "outputs" / "ablation_studies" / "results.json",
    )


def metrics_are_current(paths: ArtifactPaths) -> bool:
    """Check that saved evaluation metrics are at least as new as the checkpoint."""
    if not paths.metrics.is_file() or not paths.checkpoint.is_file():
        return False
    return paths.metrics.stat().st_mtime_ns >= paths.checkpoint.stat().st_mtime_ns