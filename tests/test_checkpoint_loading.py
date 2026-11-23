from __future__ import annotations

import numpy as np
import pytest

from models.cnn import CIFAR10CNN
from scripts.evaluate import load_checkpoint


def _checkpoint_arrays(flat_convolutions: bool) -> dict[str, np.ndarray]:
    conv1 = np.arange(16 * 3 * 3 * 3, dtype=np.float64).reshape(16, 3, 3, 3)
    conv2 = np.arange(32 * 16 * 3 * 3, dtype=np.float64).reshape(32, 16, 3, 3)

    if flat_convolutions:
        conv1 = conv1.reshape(16, -1)
        conv2 = conv2.reshape(32, -1)

    return {
        "conv1_weights": conv1,
        "conv1_bias": np.arange(16, dtype=np.float64),
        "conv2_weights": conv2,
        "conv2_bias": np.arange(32, dtype=np.float64),
        "dense_weights": np.arange(10 * 2048, dtype=np.float64).reshape(10, 2048),
        "dense_bias": np.arange(10, dtype=np.float64),
    }


@pytest.mark.parametrize("flat_convolutions", [False, True])
def test_loads_current_and_legacy_checkpoint(tmp_path, flat_convolutions):
    checkpoint_path = tmp_path / "checkpoint.npz"
    arrays = _checkpoint_arrays(flat_convolutions)
    np.savez(checkpoint_path, **arrays)

    model = CIFAR10CNN(np.random.default_rng(42))
    load_checkpoint(model, checkpoint_path)

    np.testing.assert_array_equal(
        model.conv1.weights,
        arrays["conv1_weights"].reshape(16, 3, 3, 3),
    )
    np.testing.assert_array_equal(
        model.conv2.weights,
        arrays["conv2_weights"].reshape(32, 16, 3, 3),
    )
    np.testing.assert_array_equal(model.dense.weights, arrays["dense_weights"])
    assert model.conv1.weights.dtype == np.float32
    assert model.conv2.weights.dtype == np.float32


def test_rejects_wrong_checkpoint_shape(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.npz"
    arrays = _checkpoint_arrays(flat_convolutions=True)
    arrays["conv1_weights"] = np.zeros((16, 26), dtype=np.float32)
    np.savez(checkpoint_path, **arrays)

    model = CIFAR10CNN(np.random.default_rng(42))

    with pytest.raises(ValueError, match="conv1_weights"):
        load_checkpoint(model, checkpoint_path)


def test_rejects_missing_checkpoint_array(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.npz"
    arrays = _checkpoint_arrays(flat_convolutions=False)
    del arrays["conv2_bias"]
    np.savez(checkpoint_path, **arrays)

    model = CIFAR10CNN(np.random.default_rng(42))

    with pytest.raises(ValueError, match="conv2_bias"):
        load_checkpoint(model, checkpoint_path)