"""CIFAR-10 loading and preprocessing utilities."""

from .cifar10 import (
    CIFAR10_CLASS_NAMES,
    CIFAR10_IMAGE_SHAPE,
    CIFAR10_NUM_CLASSES,
    CIFAR10_TEST_SAMPLES,
    CIFAR10_TRAIN_SAMPLES,
    load_cifar10,
    validate_cifar10_split,
)
from .preprocessing import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    compute_channel_statistics,
    nhwc_to_nchw,
    normalize_nhwc,
    preprocess_cifar10_nchw,
    scale_to_float32,
)

__all__ = [
    "CIFAR10_CLASS_NAMES",
    "CIFAR10_IMAGE_SHAPE",
    "CIFAR10_MEAN",
    "CIFAR10_NUM_CLASSES",
    "CIFAR10_STD",
    "CIFAR10_TEST_SAMPLES",
    "CIFAR10_TRAIN_SAMPLES",
    "compute_channel_statistics",
    "load_cifar10",
    "nhwc_to_nchw",
    "normalize_nhwc",
    "preprocess_cifar10_nchw",
    "scale_to_float32",
    "validate_cifar10_split",
]