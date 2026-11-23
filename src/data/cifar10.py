"""CIFAR-10 loading and validation."""

from __future__ import annotations

from typing import Final

import numpy as np

CIFAR10_TRAIN_SAMPLES: Final = 50_000
CIFAR10_TEST_SAMPLES: Final = 10_000
CIFAR10_IMAGE_SHAPE: Final = (32, 32, 3)
CIFAR10_NUM_CLASSES: Final = 10
CIFAR10_CLASS_NAMES: Final = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_cifar10() -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Fetch and validate the complete CIFAR-10 split at runtime.

    Returns:
        ((x_train, y_train), (x_test, y_test)), with images in uint8 NHWC
        layout and labels as one-dimensional int64 arrays.
    """
    try:
        from tensorflow.keras.datasets import cifar10
    except ImportError as error:
        raise ImportError(
            'TensorFlow is required to fetch CIFAR-10. Install it with '
            '`pip install -e ".[datasets]"`.'
        ) from error

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = np.asarray(y_train).reshape(-1).astype(np.int64)
    y_test = np.asarray(y_test).reshape(-1).astype(np.int64)

    validate_cifar10_split(x_train, y_train, x_test, y_test)
    return (x_train, y_train), (x_test, y_test)


def validate_cifar10_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Raise an error if arrays don't match the complete official split."""
    expected_train_images = (CIFAR10_TRAIN_SAMPLES, *CIFAR10_IMAGE_SHAPE)
    expected_test_images = (CIFAR10_TEST_SAMPLES, *CIFAR10_IMAGE_SHAPE)

    if x_train.shape != expected_train_images:
        raise ValueError(
            f"Expected training images shaped {expected_train_images}; "
            f"got {x_train.shape}"
        )
    if x_test.shape != expected_test_images:
        raise ValueError(
            f"Expected test images shaped {expected_test_images}; "
            f"got {x_test.shape}"
        )
    if y_train.shape != (CIFAR10_TRAIN_SAMPLES,):
        raise ValueError(
            f"Expected {CIFAR10_TRAIN_SAMPLES} training labels; "
            f"got {y_train.shape}"
        )
    if y_test.shape != (CIFAR10_TEST_SAMPLES,):
        raise ValueError(
            f"Expected {CIFAR10_TEST_SAMPLES} test labels; "
            f"got {y_test.shape}"
        )
    if x_train.dtype != np.uint8 or x_test.dtype != np.uint8:
        raise TypeError("CIFAR-10 images must have dtype uint8")

    _validate_labels(y_train, "training")
    _validate_labels(y_test, "test")


def _validate_labels(labels: np.ndarray, split_name: str) -> None:
    if labels.dtype.kind not in "iu":
        raise TypeError(f"{split_name} labels must be integers")
    if np.any((labels < 0) | (labels >= CIFAR10_NUM_CLASSES)):
        raise ValueError(
            f"{split_name} labels must be in "
            f"[0, {CIFAR10_NUM_CLASSES - 1}]"
        )