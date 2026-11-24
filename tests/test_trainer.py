from __future__ import annotations

import numpy as np
import pytest

from data.preprocessing import preprocess_cifar10_nchw
from evaluation.metrics import evaluate_model
from training.optimizer import SGD
from training.trainer import Trainer


class TinyModel:
    """Small deterministic model for testing Trainer.evaluate."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        values = x[:, 0]
        return np.column_stack((values, -values))


class RecordingModel:
    def __init__(self):
        self.batches = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.batches.append(x.copy())
        return np.zeros((len(x), 2), dtype=np.float32)


class TrainingRecordingModel(RecordingModel):
    conv1 = object()
    conv2 = object()
    dense = object()

    def backward(self, d_logits):
        return {
            "conv1": (None, None),
            "conv2": (None, None),
            "dense": (None, None),
        }


class NoOpOptimizer:
    def step(self, layer, d_weights, d_bias):
        pass


def make_trainer(tmp_path) -> Trainer:
    return Trainer(
        model=TinyModel(),
        optimizer=SGD(),
        num_classes=2,
        checkpoint_dir=tmp_path / "checkpoints",
        history_path=tmp_path / "history.json",
    )


def test_train_one_epoch_preprocesses_each_raw_cifar10_batch(tmp_path):
    model = TrainingRecordingModel()
    trainer = Trainer(
        model=model,
        optimizer=NoOpOptimizer(),
        num_classes=2,
        checkpoint_dir=tmp_path / "checkpoints",
        history_path=tmp_path / "history.json",
    )
    x = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    y = np.zeros(3, dtype=np.int64)

    trainer.train_one_epoch(x, y, batch_size=2, rng=np.random.default_rng(42))

    assert [len(batch) for batch in model.batches] == [2, 1]
    for batch in model.batches:
        assert batch.shape[1:] == (3, 32, 32)
        assert batch.dtype == np.float32


def test_trainer_evaluate_preprocesses_each_raw_cifar10_batch(tmp_path):
    model = RecordingModel()
    trainer = Trainer(
        model=model,
        optimizer=SGD(),
        num_classes=2,
        checkpoint_dir=tmp_path / "checkpoints",
        history_path=tmp_path / "history.json",
    )
    x = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    y = np.zeros(5, dtype=np.int64)

    trainer.evaluate(x, y, batch_size=2)

    assert [len(batch) for batch in model.batches] == [2, 2, 1]
    for batch in model.batches:
        assert batch.shape[1:] == (3, 32, 32)
        assert batch.dtype == np.float32
    np.testing.assert_array_equal(x, np.zeros_like(x))


def test_evaluate_model_preprocesses_each_raw_cifar10_batch():
    model = RecordingModel()
    x = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    y = np.zeros(3, dtype=np.int64)

    evaluate_model(model, x, y, batch_size=2, num_classes=2)

    assert [len(batch) for batch in model.batches] == [2, 1]
    np.testing.assert_array_equal(model.batches[0], preprocess_cifar10_nchw(x[:2]))
    np.testing.assert_array_equal(model.batches[1], preprocess_cifar10_nchw(x[2:]))


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