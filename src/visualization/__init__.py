"""Visualization helpers for learned filters and activations."""

from .activations_3d import plot_activation_surface, plot_activation_volume
from .filters import (
    plot_filter_channel_strength,
    plot_filter_kernel_bank,
    plot_first_layer_filters,
)

__all__ = [
    "plot_activation_surface",
    "plot_activation_volume",
    "plot_filter_channel_strength",
    "plot_filter_kernel_bank",
    "plot_first_layer_filters",
]