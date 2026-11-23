import numpy as np


class Dense:
    """Fully connected layer for rank-2 (batch, features) inputs."""

    def __init__(self, in_features, out_features, rng):
        if (
            not isinstance(in_features, int)
            or isinstance(in_features, bool)
            or in_features <= 0
        ):
            raise ValueError("in_features must be a positive integer")
        if (
            not isinstance(out_features, int)
            or isinstance(out_features, bool)
            or out_features <= 0
        ):
            raise ValueError("out_features must be a positive integer")

        self.weights = (
            rng.standard_normal((out_features, in_features)).astype(np.float32)
            * np.sqrt(2.0 / in_features)
        )
        self.bias = np.zeros(out_features, dtype=np.float32)
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_bias = np.zeros_like(self.bias)
        self.input = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.weights.shape[1]:
            raise ValueError(
                f"x must have shape (batch, {self.weights.shape[1]})"
            )

        self.input = x
        return x @ self.weights.T + self.bias

    def backward(self, d_output):
        if self.input is None:
            raise RuntimeError("Dense.backward called before Dense.forward")

        d_output = np.asarray(d_output, dtype=np.float32)
        expected_shape = (self.input.shape[0], self.weights.shape[0])
        if d_output.shape != expected_shape:
            raise ValueError(f"d_output must have shape {expected_shape}")

        d_weights = d_output.T @ self.input
        d_bias = d_output.sum(axis=0)
        d_x = d_output @ self.weights
        return d_x, d_weights, d_bias

    def update(self, d_weights, d_bias, learning_rate, momentum, weight_decay):
        d_weights = np.asarray(d_weights, dtype=np.float32)
        d_bias = np.asarray(d_bias, dtype=np.float32)
        if d_weights.shape != self.weights.shape:
            raise ValueError("d_weights has an incompatible shape")
        if d_bias.shape != self.bias.shape:
            raise ValueError("d_bias has an incompatible shape")

        self.velocity_weights *= momentum
        self.velocity_weights += d_weights + weight_decay * self.weights
        self.weights -= learning_rate * self.velocity_weights

        self.velocity_bias *= momentum
        self.velocity_bias += d_bias
        self.bias -= learning_rate * self.velocity_bias