"""CIFAR-10 convolutional neural network built from NumPy primitives."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from layers.conv2d import Conv2D
from layers.activations import ReLU
from layers.pooling import MaxPool2D
from layers.dense import Dense


class CIFAR10CNN:
    """Two-stage convolutional network for 32x32 CIFAR-10 images."""

    def __init__(
        self,
        rng,
        num_classes: int = 10,
        filters1: int = 16,
        filters2: int = 32,
        kernel_size: int = 3,
    ):
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.num_classes = num_classes
        self.conv1 = Conv2D(3, filters1, kernel_size, rng, padding="same")
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2, 2)

        self.conv2 = Conv2D(
            filters1,
            filters2,
            kernel_size,
            rng,
            padding="same",
        )
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2, 2)

        # 32x32 -> pool -> 16x16 -> pool -> 8x8.
        self.flattened_features = filters2 * 8 * 8
        self.flatten_shape: tuple[int, int, int, int] | None = None
        self.dense = Dense(self.flattened_features, num_classes, rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, channels, height, width)")
        if x.shape[1:] != (3, 32, 32):
            raise ValueError("x must have shape (batch, 3, 32, 32)")

        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        self.flatten_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        return self.dense.forward(x)

    def backward(
        self,
        d_logits: np.ndarray,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        d_x, d_weights, d_bias = self.dense.backward(d_logits)
        dense_gradients = (d_weights, d_bias)

        if self.flatten_shape is None:
            raise RuntimeError("CIFAR10CNN.backward called before forward")
        d_x = d_x.reshape(self.flatten_shape)

        d_x = self.pool2.backward(d_x)
        d_x = self.relu2.backward(d_x)
        d_x, conv2_weights, conv2_bias = self.conv2.backward(d_x)

        d_x = self.pool1.backward(d_x)
        d_x = self.relu1.backward(d_x)
        _, conv1_weights, conv1_bias = self.conv1.backward(d_x)

        return {
            "conv1": (conv1_weights, conv1_bias),
            "conv2": (conv2_weights, conv2_bias),
            "dense": dense_gradients,
        }

    def update(
        self,
        gradients: Mapping[str, tuple[np.ndarray, np.ndarray]],
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        self.conv1.update(
            *gradients["conv1"], learning_rate, momentum, weight_decay
        )
        self.conv2.update(
            *gradients["conv2"], learning_rate, momentum, weight_decay
        )
        self.dense.update(
            *gradients["dense"], learning_rate, momentum, weight_decay
        )

    def parameter_count(self) -> int:
        return sum(
            layer.weights.size + layer.bias.size
            for layer in (self.conv1, self.conv2, self.dense)
        )


def cross_entropy_with_logits(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute mean cross-entropy loss and logits gradient."""
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels)

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, num_classes)")
    if labels.shape != (logits.shape[0],):
        raise ValueError("labels must have shape (batch,")
    if labels.dtype.kind not in "iu":
        raise TypeError("labels must contain integer class indices")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("labels contain an out-of-range class index")

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    batch_indices = np.arange(labels.shape[0])
    loss = -np.log(
        probabilities[batch_indices, labels] + np.float32(1e-12)
    ).mean()

    d_logits = probabilities
    d_logits[batch_indices, labels] -= np.float32(1.0)
    d_logits /= labels.shape[0]
    return float(loss), d_logits