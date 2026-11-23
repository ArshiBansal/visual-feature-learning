"""CIFAR-10 loading utilities."""

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
    """Load the complete official CIFAR-10 split at runtime.

    The images remain in NHWC layout:
    `(samples, height, width, channels)`.

    Returns:
        ((x_train, y_train), (x_test, y_test))
    """
    try:
        from tensorflow.keras.datasets import cifar10
    except ImportError as error:
        raise ImportError(
            'TensorFlow is required to load CIFAR-10. Install the '
            'datasets extra with `pip install -e ".[datasets]"`.'
        ) from error

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = y_train.reshape(-1).astype(np.int64)
    y_test = y_test.reshape(-1).astype(np.int64)

    validate_cifar10_split(x_train, y_train, x_test, y_test)
    return (x_train, y_train), (x_test, y_test)


def validate_cifar10_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Validate the complete official CIFAR-10 split."""
    if x_train.shape != (CIFAR10_TRAIN_SAMPLES, *CIFAR10_IMAGE_SHAPE):
        raise ValueError(
            f"Expected training images "
            f"{(CIFAR10_TRAIN_SAMPLES, *CIFAR10_IMAGE_SHAPE)}, "
            f"received {x_train.shape}"
        )
    if x_test.shape != (CIFAR10_TEST_SAMPLES, *CIFAR10_IMAGE_SHAPE):
        raise ValueError(
            f"Expected test images "
            f"{(CIFAR10_TEST_SAMPLES, *CIFAR10_IMAGE_SHAPE)}, "
            f"received {x_test.shape}"
        )
    if y_train.shape != (CIFAR10_TRAIN_SAMPLES,):
        raise ValueError(
            f"Expected {CIFAR10_TRAIN_SAMPLES} training labels, "
            f"received {y_train.shape}"
        )
    if y_test.shape != (CIFAR10_TEST_SAMPLES,):
        raise ValueError(
            f"Expected {CIFAR10_TEST_SAMPLES} test labels, "
            f"received {y_test.shape}"
        )
    if x_train.dtype != np.uint8 or x_test.dtype != np.uint8:
        raise TypeError("CIFAR-10 images must use uint8 pixel values")

    validate_labels(y_train, "training")
    validate_labels(y_test, "test")


def validate_labels(labels: np.ndarray, split_name: str) -> None:
    """Check that labels are integers in the 0–9 range."""
    if labels.dtype.kind not in "iu":
        raise TypeError(f"{split_name} labels must contain integers")
    if not np.all((labels >= 0) & (labels < CIFAR10_NUM_CLASSES)):
        raise ValueError(
            f"{split_name} labels must be between 0 and "
            f"{CIFAR10_NUM_CLASSES - 1}"
        )