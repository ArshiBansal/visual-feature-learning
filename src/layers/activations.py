import numpy as np


class ReLU:
    """Rectified linear activation with a cached gradient mask."""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        x = np.asarray(x)
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, d_output):
        if self.mask is None:
            raise RuntimeError("ReLU.backward called before ReLU.forward")

        d_output = np.asarray(d_output)
        if d_output.shape != self.mask.shape:
            raise ValueError("d_output has an incompatible shape")
        return d_output * self.mask