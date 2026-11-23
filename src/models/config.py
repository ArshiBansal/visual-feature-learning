"""Build supported CNN models from project configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .cnn import CIFAR10CNN


def _positive_int(section: Mapping[str, Any], key: str, context: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer")
    return value


def build_cifar10_cnn(
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> CIFAR10CNN:
    """Build the supported two-convolution CIFAR-10 model from its config."""
    try:
        dataset = config["dataset"]
        model_config = config["model"]
        input_shape = tuple(dataset["input_shape"])
        num_classes = _positive_int(dataset, "num_classes", "dataset")
        layers = model_config["layers"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing or invalid model configuration: {error}") from error

    if input_shape != (3, 32, 32):
        raise ValueError(
            f"Model supports input_shape (3, 32, 32), got {input_shape}"
        )

    class_names = dataset.get("class_names")
    if class_names is not None and len(class_names) != num_classes:
        raise ValueError("class_names length must match dataset num_classes")

    expected_layer_types = [
        "conv2d",
        "relu",
        "max_pool2d",
        "conv2d",
        "relu",
        "max_pool2d",
        "flatten",
        "dense",
    ]
    if not isinstance(layers, list):
        raise ValueError("model.layers must be a list")

    actual_layer_types = [
        layer.get("type") if isinstance(layer, Mapping) else None
        for layer in layers
    ]
    if actual_layer_types != expected_layer_types:
        raise ValueError(
            "model.layers must be ordered as conv2d, relu, max_pool2d, "
            "conv2d, relu, max_pool2d, flatten, dense"
        )

    convolution = model_config.get("convolution")
    if not isinstance(convolution, Mapping):
        raise ValueError("model.convolution must be an object")
    if (
        convolution.get("operation") != "cross_correlation"
        or convolution.get("stride") != 1
        or convolution.get("padding") != "same"
    ):
        raise ValueError(
            "model.convolution must specify cross_correlation, "
            "stride 1, and same padding"
        )

    conv1 = layers[0]
    pool1 = layers[2]
    conv2 = layers[3]
    pool2 = layers[5]
    dense = layers[7]

    for index, conv in enumerate((conv1, conv2), start=1):
        if (
            _positive_int(conv, "stride", f"model.layers[{index}]") != 1
            or conv.get("padding") != "same"
        ):
            raise ValueError(
                f"Convolution layer {index} must use stride 1 and same padding"
            )

    input_channels = input_shape[0]
    conv1_inputs = _positive_int(conv1, "in_channels", "conv1")
    filters1 = _positive_int(conv1, "out_channels", "conv1")
    conv1_kernel = _positive_int(conv1, "kernel_size", "conv1")

    conv2_inputs = _positive_int(conv2, "in_channels", "conv2")
    filters2 = _positive_int(conv2, "out_channels", "conv2")
    conv2_kernel = _positive_int(conv2, "kernel_size", "conv2")

    if conv1_inputs != input_channels:
        raise ValueError("conv1.in_channels must match dataset.input_shape")
    if conv2_inputs != filters1:
        raise ValueError("conv2.in_channels must match conv1.out_channels")
    if conv1_kernel != conv2_kernel:
        raise ValueError("Both convolution layers must use the same kernel size")

    for index, pool in enumerate((pool1, pool2), start=1):
        if (
            _positive_int(pool, "pool_size", f"pool{index}") != 2
            or _positive_int(pool, "stride", f"pool{index}") != 2
        ):
            raise ValueError(
                f"pool{index} must use pool_size 2 and stride 2"
            )

    expected_dense_inputs = filters2 * 8 * 8
    dense_inputs = _positive_int(dense, "in_features", "dense")
    dense_outputs = _positive_int(dense, "out_features", "dense")

    if dense_inputs != expected_dense_inputs:
        raise ValueError(
            f"dense.in_features must be {expected_dense_inputs}, "
            f"got {dense_inputs}"
        )
    if dense_outputs != num_classes:
        raise ValueError("dense.out_features must match dataset.num_classes")

    output_config = model_config.get("output")
    if not isinstance(output_config, Mapping) or (
        output_config.get("type") != "logits"
        or output_config.get("loss") != "cross_entropy_with_logits"
    ):
        raise ValueError(
            "model.output must specify logits and cross_entropy_with_logits"
        )

    return CIFAR10CNN(
        rng=rng,
        num_classes=num_classes,
        filters1=filters1,
        filters2=filters2,
        kernel_size=conv1_kernel,
    )