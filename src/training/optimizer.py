from __future__ import annotations

from typing import Protocol

import numpy as np


class ParameterizedLayer(Protocol):
    weights: np.ndarray
    bias: np.ndarray
    velocity_weights: np.ndarray
    velocity_bias: np.ndarray


class SGD:
    """SGD with momentum and weight decay for notebook-style CNN layers."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0005):
        if learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if momentum < 0:
            raise ValueError("momentum must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self, layer: ParameterizedLayer, d_weights: np.ndarray, d_bias: np.ndarray) -> None:
        layer.velocity_weights *= self.momentum
        layer.velocity_weights += d_weights + self.weight_decay * layer.weights
        layer.weights -= self.learning_rate * layer.velocity_weights

        layer.velocity_bias *= self.momentum
        layer.velocity_bias += d_bias
        layer.bias -= self.learning_rate * layer.velocity_bias

    def set_learning_rate(self, learning_rate: float) -> None:
        if learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        self.learning_rate = learning_rate