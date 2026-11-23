import numpy as np


def _pair(value, name):
    """Convert an integer or a pair of integers to a validated pair."""
    if isinstance(value, int) and not isinstance(value, bool):
        pair = (value, value)
    elif isinstance(value, tuple) and len(value) == 2:
        pair = value
    else:
        raise TypeError(f"{name} must be an int or a pair of ints")

    if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in pair):
        raise ValueError(f"{name} values must be positive integers")
    return pair


def max_pool2d_forward(x, pool_size=2, stride=None):
    """Apply valid max-pooling to an NCHW tensor.

    For tied maxima, the backward pass distributes the gradient evenly among
    all tied maximum positions in the window.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError("x must be a rank-4 NCHW tensor")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")

    pool_height, pool_width = _pair(pool_size, "pool_size")
    if stride is None:
        stride = pool_height
    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        raise ValueError("stride must be a positive integer")

    batch_size, channels, height, width = x.shape
    if min(batch_size, channels, height, width) <= 0:
        raise ValueError("input dimensions must be positive")
    output_height = (height - pool_height) // stride + 1
    output_width = (width - pool_width) // stride + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Pooling window is larger than the input")

    output = np.empty(
        (batch_size, channels, output_height, output_width), dtype=np.float64
    )
    max_mask = np.zeros(
        (batch_size, channels, output_height, output_width, pool_height, pool_width),
        dtype=bool,
    )

    for out_y in range(output_height):
        y_start = out_y * stride
        y_end = y_start + pool_height
        for out_x in range(output_width):
            x_start = out_x * stride
            x_end = x_start + pool_width
            window = x[:, :, y_start:y_end, x_start:x_end]
            maxima = np.max(window, axis=(2, 3), keepdims=True)
            output[:, :, out_y, out_x] = maxima[:, :, 0, 0]
            max_mask[:, :, out_y, out_x] = window == maxima

    cache = {
        "input_shape": x.shape,
        "max_mask": max_mask,
        "pool_size": (pool_height, pool_width),
        "stride": stride,
        "output_shape": output.shape,
    }
    return output, cache


def max_pool2d_backward(d_output, cache):
    """Backpropagate through max_pool2d_forward."""
    d_output = np.asarray(d_output, dtype=np.float64)
    if not np.all(np.isfinite(d_output)):
        raise ValueError("d_output must contain only finite values")
    if d_output.shape != cache["output_shape"]:
        raise ValueError("d_output has an incompatible shape")

    input_shape = cache["input_shape"]
    max_mask = cache["max_mask"]
    pool_height, pool_width = cache["pool_size"]
    stride = cache["stride"]

    batch_size, channels, _, _ = input_shape
    _, _, output_height, output_width = d_output.shape
    d_x = np.zeros(input_shape, dtype=np.float64)

    for out_y in range(output_height):
        y_start = out_y * stride
        y_end = y_start + pool_height
        for out_x in range(output_width):
            x_start = out_x * stride
            x_end = x_start + pool_width
            mask = max_mask[:, :, out_y, out_x]
            counts = np.sum(mask, axis=(2, 3), keepdims=True)
            window_grad = (
                mask * d_output[:, :, out_y, out_x, None, None] / counts
            )
            d_x[:, :, y_start:y_end, x_start:x_end] += window_grad

    return d_x