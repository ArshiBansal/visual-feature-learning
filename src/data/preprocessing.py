"""CIFAR-10 preprocessing utilities."""

from __future__ import annotations

from typing import Final

import numpy as np

CIFAR10_MEAN: Final = np.asarray(
    [0.4914, 0.4822, 0.4465],
    dtype=np.float32,
)

CIFAR10_STD: Final = np.asarray(
    [0.2470, 0.2430, 0.2610],
    dtype=np.float32,
)


def scale_to_float32(images: np.ndarray) -> np.ndarray:
    """Scale uint8 images from [0, 255] to float32 [0, 1]."""
    if images.dtype != np.uint8:
        raise TypeError("CIFAR-10 images must use uint8 pixel values")
    if not np.all((images >= 0) & (images <= 255)):
        raise ValueError("uint8 CIFAR-10 values must be in [0, 255]")
    return images.astype(np.float32) / np.float32(255.0)


def normalize_nhwc(
    images: np.ndarray,
    mean: np.ndarray = CIFAR10_MEAN,
    std: np.ndarray = CIFAR10_STD,
) -> np.ndarray:
    """Channel-normalize float32 NHWC images without copying if possible."""
    if images.ndim != 4:
        raise ValueError("Expected NHWC images with shape (N, H, W, C)")
    if images.dtype != np.float32:
        images = images.astype(np.float32)
    if images.shape[-1] != mean.size:
        raise ValueError("Channel count must match the mean and standard deviation")
    if np.any(np.asarray(mean) == 0) or np.any(np.asarray(std) <= 0):
        raise ValueError("mean and std must contain valid normalization values")

    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, -1)
    return (images - mean) / std


def nhwc_to_nchw(images: np.ndarray) -> np.ndarray:
    """Convert NHWC arrays to contiguous NCHW arrays."""
    if images.ndim != 4:
        raise ValueError("Expected NHWC images with shape (N, H, W, C)")
    if images.shape[-1] != 3:
        raise ValueError("CIFAR-10 images must have exactly three channels")
    return np.ascontiguousarray(images.transpose(0, 3, 1, 2))


def preprocess_cifar10_nchw(
    images: np.ndarray,
    mean: np.ndarray = CIFAR10_MEAN,
    std: np.ndarray = CIFAR10_STD,
) -> np.ndarray:
    """Scale, normalize, and convert CIFAR-10 NHWC images to NCHW."""
    scaled = scale_to_float32(images)
    normalized = normalize_nhwc(scaled, mean=mean, std=std)
    nchw = nhwc_to_nchw(normalized)
    return nchw


def compute_channel_statistics(
    images: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute float32 channel means and standard deviations.

    Pass only the training images. Never calculate this from test data.
    """
    if images.dtype == np.uint8:
        values = images.astype(np.float32) / np.float32(255.0)
    else:
        values = np.asarray(images, dtype=np.float32)

    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("Expected NHWC CIFAR-10 images with shape (N, 32, 32, 3)")
    if len(values) == 0:
        raise ValueError("Cannot compute statistics from an empty dataset")

    mean = values.mean(axis=(0, 1, 2), dtype=np.float32)
    std = values.std(axis=(0, 1, 2), dtype=np.float32)
    return mean, std