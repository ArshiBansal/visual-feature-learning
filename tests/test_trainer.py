from __future__ import annotations

import numpy as np
import pytest

from training.optimizer import SGD
from training.trainer import Trainer


class TinyModel:
    """Small deterministic model for testing Trainer.evaluate."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        values = x[:, 0]
        return np.column_stack((values, -values))


def make_trainer(tmp_path) -> Trainer:
    return Trainer(
        model=TinyModel(),
        optimizer=SGD(),
        num_classes=2,
        checkpoint_dir=tmp_path / "checkpoints",
        history_path=tmp_path / "history.json",
    )


def test_evaluate_rejects_empty_data(tmp_path):
    trainer = make_trainer(tmp_path)

    with pytest.raises(ValueError, match="must not be empty"):
        trainer.evaluate(
            x=np.empty((0, 1), dtype=np.float32),
            y=np.empty((0,), dtype=np.int64),
            batch_size=2,
        )


def test_evaluate_computes_loss_accuracy_and_confusion_matrix(tmp_path):
    trainer = make_trainer(tmp_path)
    x = np.array([[1.0], [-1.0], [2.0]], dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int64)

    result = trainer.evaluate(x, y, batch_size=2)

    assert np.isfinite(result["loss"])
    assert result["accuracy"] == 1.0
    np.testing.assert_array_equal(
        result["confusion_matrix"],
        np.array([[2, 0], [0, 1]], dtype=np.int64),
    )


def test_evaluate_rejects_mismatched_sample_counts(tmp_path):
    trainer = make_trainer(tmp_path)

    with pytest.raises(ValueError, match="same number of samples"):
        trainer.evaluate(
            x=np.ones((2, 1), dtype=np.float32),
            y=np.array([0], dtype=np.int64),
            batch_size=2,
        )