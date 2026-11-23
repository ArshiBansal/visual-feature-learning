from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from models.config import build_cifar10_cnn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "cifar10_baseline.json"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def test_builds_model_from_baseline_config():
    model = build_cifar10_cnn(
        load_config(),
        np.random.default_rng(42),
    )

    assert model.conv1.weights.shape == (16, 3, 3, 3)
    assert model.conv2.weights.shape == (32, 16, 3, 3)
    assert model.dense.weights.shape == (10, 2048)
    assert model.flattened_features == 2048


def test_rejects_incorrect_dense_input_size():
    config = load_config()
    dense = next(
        layer
        for layer in config["model"]["layers"]
        if layer["type"] == "dense"
    )
    dense["in_features"] = 512

    with pytest.raises(ValueError, match="dense.in_features must be 2048"):
        build_cifar10_cnn(config, np.random.default_rng(42))


def test_rejects_wrong_layer_order():
    config = load_config()
    config["model"]["layers"][1], config["model"]["layers"][2] = (
        config["model"]["layers"][2],
        config["model"]["layers"][1],
    )

    with pytest.raises(ValueError, match="must be ordered"):
        build_cifar10_cnn(config, np.random.default_rng(42))


def test_rejects_unsupported_layer():
    config = load_config()
    config["model"]["layers"][1]["type"] = "dropout"

    with pytest.raises(ValueError, match="must be ordered"):
        build_cifar10_cnn(config, np.random.default_rng(42))


def test_rejects_mismatched_convolution_channels():
    config = load_config()
    config["model"]["layers"][3]["in_channels"] = 8

    with pytest.raises(ValueError, match="conv2.in_channels"):
        build_cifar10_cnn(config, np.random.default_rng(42))