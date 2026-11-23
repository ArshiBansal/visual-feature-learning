import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _pair(value, name):
    """Convert an integer or a pair of integers to a validated pair."""
    if isinstance(value, int) and not isinstance(value, bool):
        pair = (value, value)
    elif isinstance(value, tuple) and len(value) == 2:
        pair = value
    else:
        raise TypeError(f"{name} must be an int or a pair of ints")

    if any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in pair
    ):
        raise ValueError(f"{name} values must be positive integers")
    return pair


def _padding_for_conv(height, width, kernel_height, kernel_width, stride, padding):
    """Return top, bottom, left, and right convolution padding."""
    if isinstance(padding, str):
        if padding == "valid":
            return 0, 0, 0, 0
        if padding != "same":
            raise ValueError("padding must be 'valid', 'same', an int, or a pair")

        output_height = (height + stride - 1) // stride
        output_width = (width + stride - 1) // stride
        total_height = max(
            (output_height - 1) * stride + kernel_height - height,
            0,
        )
        total_width = max(
            (output_width - 1) * stride + kernel_width - width,
            0,
        )
        pad_top = total_height // 2
        pad_left = total_width // 2
        return (
            pad_top,
            total_height - pad_top,
            pad_left,
            total_width - pad_left,
        )

    if isinstance(padding, int) and not isinstance(padding, bool):
        if padding < 0:
            raise ValueError("padding cannot be negative")
        return padding, padding, padding, padding

    if isinstance(padding, tuple) and len(padding) == 2:
        pad_height, pad_width = padding
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in padding
        ):
            raise ValueError("padding values must be non-negative integers")
        return pad_height, pad_height, pad_width, pad_width

    raise TypeError("padding must be 'valid', 'same', an int, or a pair")


def conv2d_forward(x, weights, bias, stride=1, padding="valid"):
    """Apply multi-channel 2D cross-correlation to an NCHW tensor.

    Args:
        x: Input tensor shaped (N, C_in, H, W).
        weights: Filters shaped (C_out, C_in, K_h, K_w).
        bias: Bias vector shaped (C_out,).
        stride: Positive integer stride.
        padding: 'valid', 'same', a non-negative integer, or (height, width).

    Returns:
        The output tensor and a cache for conv2d_backward.
    """
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)

    if x.ndim != 4 or weights.ndim != 4:
        raise ValueError("x and weights must both be rank-4 tensors")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        raise ValueError("stride must be a positive integer")

    batch_size, in_channels, height, width = x.shape
    out_channels, weight_in_channels, kernel_height, kernel_width = weights.shape

    if min(
        in_channels,
        height,
        width,
        out_channels,
        kernel_height,
        kernel_width,
    ) <= 0:
        raise ValueError("input and filter dimensions must be positive")
    if in_channels != weight_in_channels:
        raise ValueError("Input channels must match the weight input channels")
    if bias.shape != (out_channels,):
        raise ValueError(f"bias must have shape ({out_channels},)")

    pad_top, pad_bottom, pad_left, pad_right = _padding_for_conv(
        height,
        width,
        kernel_height,
        kernel_width,
        stride,
        padding,
    )
    x_padded = np.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right),
        ),
    )

    padded_height, padded_width = x_padded.shape[2:]
    output_height = (padded_height - kernel_height) // stride + 1
    output_width = (padded_width - kernel_width) // stride + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Kernel is larger than the padded input")

    windows = sliding_window_view(
        x_padded,
        (kernel_height, kernel_width),
        axis=(2, 3),
    )
    windows = windows[:, :, ::stride, ::stride, :, :]

    # Order each patch as (C_in, K_h, K_w), matching weights.reshape below.
    patches = windows.transpose(0, 2, 3, 1, 4, 5).reshape(
        batch_size * output_height * output_width,
        in_channels * kernel_height * kernel_width,
    )
    flattened_weights = weights.reshape(out_channels, -1)

    output = patches @ flattened_weights.T + bias
    output = output.reshape(
        batch_size,
        output_height,
        output_width,
        out_channels,
    ).transpose(0, 3, 1, 2)

    cache = {
        "patches": patches,
        "weights": weights,
        "input_shape": x.shape,
        "stride": stride,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "output_shape": output.shape,
    }
    return output, cache


def conv2d_backward(d_output, cache):
    """Backpropagate through conv2d_forward.

    Returns gradients for the input, weights, and bias.
    """
    d_output = np.asarray(d_output, dtype=np.float64)
    if d_output.shape != cache["output_shape"]:
        raise ValueError("d_output has an incompatible shape")

    patches = cache["patches"]
    weights = cache["weights"]
    input_shape = cache["input_shape"]
    stride = cache["stride"]
    pad_top = cache["pad_top"]
    pad_left = cache["pad_left"]

    batch_size, in_channels, height, width = input_shape
    out_channels, _, kernel_height, kernel_width = weights.shape
    _, _, output_height, output_width = d_output.shape

    flattened_d_output = d_output.transpose(0, 2, 3, 1).reshape(
        batch_size * output_height * output_width,
        out_channels,
    )

    d_weights = (
        flattened_d_output.T @ patches
    ).reshape(weights.shape)
    d_bias = flattened_d_output.sum(axis=0)

    flattened_d_patches = flattened_d_output @ weights.reshape(out_channels, -1)
    d_patches = flattened_d_patches.reshape(
        batch_size,
        output_height,
        output_width,
        in_channels,
        kernel_height,
        kernel_width,
    )

    padded_height = height + 2 * pad_top
    padded_width = width + 2 * pad_left
    d_x_padded = np.zeros(
        (batch_size, in_channels, padded_height, padded_width),
        dtype=np.float64,
    )

    for kernel_y in range(kernel_height):
        y_end = kernel_y + output_height * stride
        for kernel_x in range(kernel_width):
            x_end = kernel_x + output_width * stride
            patch_gradient = d_patches[:, :, :, :, kernel_y, kernel_x]
            patch_gradient = patch_gradient.transpose(0, 3, 1, 2)

            d_x_padded[
                :,
                :,
                kernel_y:y_end:stride,
                kernel_x:x_end:stride,
            ] += patch_gradient

    d_x = d_x_padded[
        :,
        :,
        pad_top : pad_top + height,
        pad_left : pad_left + width,
    ]
    return d_x, d_weights, d_bias