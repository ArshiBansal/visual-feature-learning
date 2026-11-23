"""CIFAR-10 convolutional neural network built from NumPy primitives."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from layers.conv2d import conv2d_backward, conv2d_forward
from layers.max_pool2d import max_pool2d_backward, max_pool2d_forward


class Conv2D:
    """NumPy Conv2D layer with forward, backward, and SGD update support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rng: np.random.Generator,
        stride: int = 1,
        padding: str | int | tuple[int, int] = "valid",
    ) -> None:
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if isinstance(kernel_size, bool) or kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = (
            rng.standard_normal(
                (out_channels, in_channels, kernel_size, kernel_size)
            ).astype(np.float32)
            * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        )
        self.bias = np.zeros(out_channels, dtype=np.float32)
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_bias = np.zeros_like(self.bias)
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        output, self.cache = conv2d_forward(
            x,
            self.weights,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return output.astype(np.float32)

    def backward(
        self,
        d_output: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.cache is None:
            raise RuntimeError("Conv2D.backward called before forward")

        d_x, d_weights, d_bias = conv2d_backward(
            d_output,
            self.cache,
        )
        return (
            d_x.astype(np.float32),
            d_weights.astype(np.float32),
            d_bias.astype(np.float32),
        )

    def update(
        self,
        d_weights: np.ndarray,
        d_bias: np.ndarray,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        d_weights = np.asarray(d_weights, dtype=np.float32)
        d_bias = np.asarray(d_bias, dtype=np.float32)

        self.velocity_weights *= momentum
        self.velocity_weights += d_weights + weight_decay * self.weights
        self.weights -= learning_rate * self.velocity_weights

        self.velocity_bias *= momentum
        self.velocity_bias += d_bias
        self.bias -= learning_rate * self.velocity_bias


class MaxPool2D:
    """NumPy max-pooling layer."""

    def __init__(self, pool_size: int = 2, stride: int | None = None) -> None:
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        output, self.cache = max_pool2d_forward(
            x,
            pool_size=self.pool_size,
            stride=self.stride,
        )
        return output.astype(np.float32)

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("MaxPool2D.backward called before forward")

        d_x = max_pool2d_backward(d_output, self.cache)
        return d_x.astype(np.float32)


class ReLU:
    """Rectified linear activation with a cached gradient mask."""

    def __init__(self) -> None:
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("ReLU.backward called before forward")
        if d_output.shape != self.mask.shape:
            raise ValueError("d_output has an incompatible shape")
        return d_output * self.mask


class Dense:
    """Fully connected layer for rank-2 inputs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
    ) -> None:
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if out_features <= 0:
            raise ValueError("out_features must be positive")

        self.weights = (
            rng.standard_normal((out_features, in_features)).astype(np.float32)
            * np.sqrt(2.0 / in_features)
        )
        self.bias = np.zeros(out_features, dtype=np.float32)
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_bias = np.zeros_like(self.bias)
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.weights.shape[1]:
            raise ValueError(
                f"x must have shape (batch, {self.weights.shape[1]})"
            )

        self.input = x
        return x @ self.weights.T + self.bias

    def backward(
        self,
        d_output: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.input is None:
            raise RuntimeError("Dense.backward called before forward")

        d_output = np.asarray(d_output, dtype=np.float32)
        expected_shape = (self.input.shape[0], self.weights.shape[0])
        if d_output.shape != expected_shape:
            raise ValueError(f"d_output must have shape {expected_shape}")

        d_weights = d_output.T @ self.input
        d_bias = d_output.sum(axis=0)
        d_x = d_output @ self.weights
        return d_x, d_weights, d_bias

    def update(
        self,
        d_weights: np.ndarray,
        d_bias: np.ndarray,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        d_weights = np.asarray(d_weights, dtype=np.float32)
        d_bias = np.asarray(d_bias, dtype=np.float32)

        self.velocity_weights *= momentum
        self.velocity_weights += d_weights + weight_decay * self.weights
        self.weights -= learning_rate * self.velocity_weights

        self.velocity_bias *= momentum
        self.velocity_bias += d_bias
        self.bias -= learning_rate * self.velocity_bias


class CIFAR10CNN:
    """Two-stage convolutional network for 32x32 CIFAR-10 images."""

    def __init__(
        self,
        rng: np.random.Generator,
        num_classes: int = 10,
        filters1: int = 16,
        filters2: int = 32,
        kernel_size: int = 3,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if filters1 <= 0:
            raise ValueError("filters1 must be positive")
        if filters2 <= 0:
            raise ValueError("filters2 must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")

        self.num_classes = num_classes
        self.conv1 = Conv2D(
            3,
            filters1,
            kernel_size,
            rng,
            padding="same",
        )
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
        d_x, conv1_weights, conv1_bias = self.conv1.backward(d_x)

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
            *gradients["conv1"],
            learning_rate,
            momentum,
            weight_decay,
        )
        self.conv2.update(
            *gradients["conv2"],
            learning_rate,
            momentum,
            weight_decay,
        )
        self.dense.update(
            *gradients["dense"],
            learning_rate,
            momentum,
            weight_decay,
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
        raise ValueError("labels must have shape (batch,)")
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