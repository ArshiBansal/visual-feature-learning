"""Plot learned convolution filters and their channel strengths."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _validate_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights)
    if weights.ndim != 4:
        raise ValueError("weights must have shape (out_channels, in_channels, height, width)")
    if any(size <= 0 for size in weights.shape):
        raise ValueError("weights dimensions must be positive")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")
    return weights


def plot_first_layer_filters(
    weights: np.ndarray,
    *,
    columns: int = 4,
    title: str = "Learned first-layer RGB filters",
) -> tuple[Figure, np.ndarray]:
    """Display each output filter for a three-channel convolution as RGB."""
    weights = _validate_weights(weights)
    if weights.shape[1] != 3:
        raise ValueError("RGB filter plotting requires exactly three input channels")
    if columns <= 0:
        raise ValueError("columns must be positive")

    count = weights.shape[0]
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.5 * columns, 2.5 * rows), squeeze=False)

    for index, ax in enumerate(axes.flat):
        if index < count:
            rgb_filter = weights[index].transpose(1, 2, 0)
            low = float(rgb_filter.min())
            high = float(rgb_filter.max())
            display_filter = (rgb_filter - low) / (high - low + 1e-8)
            ax.imshow(display_filter)
            ax.set_title(f"Filter {index}")
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_filter_channel_strength(
    weights: np.ndarray,
    *,
    title: str = "Convolution filter strength by input channel",
) -> tuple[Figure, Axes]:
    """Plot RMS weight strength for each output-filter/input-channel pair."""
    weights = _validate_weights(weights)
    channel_strength = np.sqrt(np.mean(np.square(weights), axis=(2, 3)))

    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(channel_strength, aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, label="RMS filter weight")
    ax.set(
        xlabel="Input feature-map channel",
        ylabel="Output filter",
        title=title,
    )
    fig.tight_layout()
    return fig, ax


def plot_filter_kernel_bank(
    weights: np.ndarray,
    output_filter_index: int = 0,
    *,
    columns: int = 4,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Display one output filter's spatial kernel for each input channel."""
    weights = _validate_weights(weights)
    if not 0 <= output_filter_index < weights.shape[0]:
        raise ValueError("output_filter_index is out of range")
    if columns <= 0:
        raise ValueError("columns must be positive")

    kernel_bank = weights[output_filter_index]
    count = kernel_bank.shape[0]
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.5 * columns, 2.5 * rows), squeeze=False)
    limit = float(np.max(np.abs(kernel_bank)))
    if limit == 0:
        limit = 1.0

    for channel_index, ax in enumerate(axes.flat):
        if channel_index < count:
            ax.imshow(
                kernel_bank[channel_index],
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            ax.set_title(f"Input {channel_index}")
        ax.axis("off")

    fig.suptitle(title or f"Kernels for output filter {output_filter_index}")
    fig.tight_layout()
    return fig, axes