"""CIFAR-10 image scaling, normalization, and layout conversion."""

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
    """Scale uint8 image values from [0, 255] to float32 [0, 1]."""
    images = np.asarray(images)
    if images.dtype != np.uint8:
        raise TypeError("Expected uint8 images with pixel values in [0, 255]")
    return images.astype(np.float32) / np.float32(255.0)


def normalize_nhwc(
    images: np.ndarray,
    mean: np.ndarray = CIFAR10_MEAN,
    std: np.ndarray = CIFAR10_STD,
) -> np.ndarray:
    """Normalize float NHWC images using channel-wise means and standard deviations."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 4:
        raise ValueError("Expected NHWC images shaped (N, H, W, C)")
    if not np.all(np.isfinite(images)):
        raise ValueError("Images must contain finite values")

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
        raise ValueError("mean and std must be one-dimensional arrays of equal size")
    if images.shape[-1] != mean.size:
        raise ValueError(
            f"Image has {images.shape[-1]} channels but normalization "
            f"statistics have {mean.size}"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("mean and std must contain finite values")
    if np.any(std <= 0):
        raise ValueError("std values must be greater than zero")

    mean = mean.reshape(1, 1, 1, -1)
    std = std.reshape(1, 1, 1, -1)
    return (images - mean) / std


def nhwc_to_nchw(images: np.ndarray) -> np.ndarray:
    """Convert a rank-4 NHWC image batch to contiguous NCHW layout."""
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError("Expected NHWC images shaped (N, H, W, C)")
    if images.shape[-1] != 3:
        raise ValueError("CIFAR-10 images must have three color channels")
    return np.ascontiguousarray(images.transpose(0, 3, 1, 2))


def preprocess_cifar10_nchw(
    images: np.ndarray,
    mean: np.ndarray = CIFAR10_MEAN,
    std: np.ndarray = CIFAR10_STD,
) -> np.ndarray:
    """Scale, normalize, and convert CIFAR-10 uint8 images to float32 NCHW."""
    scaled = scale_to_float32(images)
    normalized = normalize_nhwc(scaled, mean=mean, std=std)
    return nhwc_to_nchw(normalized)


def prepare_cifar10_batch(
    images: np.ndarray,
    mean: np.ndarray = CIFAR10_MEAN,
    std: np.ndarray = CIFAR10_STD,
) -> np.ndarray:
    """Preprocess raw CIFAR-10 batches while preserving model-ready inputs."""
    images = np.asarray(images)
    if images.dtype == np.uint8 and images.ndim == 4 and images.shape[-1] == 3:
        return preprocess_cifar10_nchw(images, mean=mean, std=std)
    return images


def compute_channel_statistics(
    images: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and standard deviation from NHWC images.

    Use training images only, not the held-out test split.
    """
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Expected RGB NHWC images shaped (N, H, W, 3)")
    if images.shape[0] == 0:
        raise ValueError("Cannot calculate statistics for an empty dataset")

    if images.dtype == np.uint8:
        values = scale_to_float32(images)
    else:
        values = images.astype(np.float32, copy=False)

    if not np.all(np.isfinite(values)):
        raise ValueError("Images must contain finite values")

    mean = values.mean(axis=(0, 1, 2), dtype=np.float32)
    std = values.std(axis=(0, 1, 2), dtype=np.float32)
    return mean, std