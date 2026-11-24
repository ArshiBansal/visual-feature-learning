"""Evaluate a trained CIFAR-10 NumPy CNN checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cifar10 import load_cifar10
from evaluation.metrics import evaluate_model
from models.cnn import CIFAR10CNN
from models.config import build_cifar10_cnn

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cifar10_baseline.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 NumPy CNN")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object")
    return config


def _require_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _checkpoint_array(
    checkpoint: np.lib.npyio.NpzFile,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if name not in checkpoint.files:
        raise ValueError(f"Checkpoint is missing {name!r}")

    array = np.asarray(checkpoint[name])
    if array.dtype.kind not in "iuf":
        raise TypeError(f"Checkpoint array {name!r} must contain numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Checkpoint array {name!r} must contain only finite values")
    if array.shape == expected_shape:
        return array.copy()

    # Older training notebooks saved convolution weights as flattened filters.
    if (
        len(expected_shape) == 4
        and array.ndim == 2
        and array.size == int(np.prod(expected_shape))
    ):
        return array.reshape(expected_shape).copy()

    raise ValueError(
        f"Checkpoint array {name!r} has shape {array.shape}; "
        f"expected {expected_shape}"
    )


def load_checkpoint(model: CIFAR10CNN, path: Path) -> None:
    path = path if path.is_absolute() else PROJECT_ROOT / path

    with np.load(path, allow_pickle=False) as checkpoint:
        model.conv1.weights = _checkpoint_array(
            checkpoint,
            "conv1_weights",
            model.conv1.weights.shape,
        ).astype(np.float32)
        model.conv1.bias = _checkpoint_array(
            checkpoint,
            "conv1_bias",
            model.conv1.bias.shape,
        ).astype(np.float32)
        model.conv2.weights = _checkpoint_array(
            checkpoint,
            "conv2_weights",
            model.conv2.weights.shape,
        ).astype(np.float32)
        model.conv2.bias = _checkpoint_array(
            checkpoint,
            "conv2_bias",
            model.conv2.bias.shape,
        ).astype(np.float32)
        model.dense.weights = _checkpoint_array(
            checkpoint,
            "dense_weights",
            model.dense.weights.shape,
        ).astype(np.float32)
        model.dense.bias = _checkpoint_array(
            checkpoint,
            "dense_bias",
            model.dense.bias.shape,
        ).astype(np.float32)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = _require_int(config["project"]["seed"], "project.seed")
    dataset_config = config["dataset"]
    normalization_config = dataset_config["normalization"]
    training_config = config["training"]
    class_names = dataset_config["class_names"]
    num_classes = _require_int(
        dataset_config["num_classes"],
        "dataset.num_classes",
    )
    expected_train_samples = _require_int(
        dataset_config["expected_train_samples"],
        "dataset.expected_train_samples",
    )
    expected_test_samples = _require_int(
        dataset_config["expected_test_samples"],
        "dataset.expected_test_samples",
    )
    epochs = _require_int(training_config["epochs"], "training.epochs")
    batch_size = _require_int(training_config["batch_size"], "training.batch_size")
    normalization_mean = np.asarray(
        normalization_config["mean"],
        dtype=np.float32,
    )
    normalization_std = np.asarray(
        normalization_config["std"],
        dtype=np.float32,
    )

    if len(class_names) != num_classes:
        raise ValueError("class_names length must match dataset num_classes")
    if expected_train_samples <= 0:
        raise ValueError("dataset.expected_train_samples must be positive")
    if expected_test_samples <= 0:
        raise ValueError("dataset.expected_test_samples must be positive")
    if epochs <= 0:
        raise ValueError("training.epochs must be positive")
    if batch_size <= 0:
        raise ValueError("training.batch_size must be positive")

    (_, _), (x_test, y_test) = load_cifar10()
    if len(x_test) != expected_test_samples:
        raise ValueError("Unexpected CIFAR-10 test split size")

    model = build_cifar10_cnn(
        config,
        np.random.default_rng(seed),
    )
    checkpoint_dir = PROJECT_ROOT / config["artifacts"]["checkpoint_dir"]
    checkpoint = args.checkpoint or checkpoint_dir / "cifar10_cnn_latest.npz"
    load_checkpoint(model, checkpoint)

    result: dict[str, Any] = evaluate_model(
        model=model,
        x=x_test,
        y=y_test,
        batch_size=batch_size,
        num_classes=num_classes,
        mean=normalization_mean,
        std=normalization_std,
    )

    print(f"Test loss:     {result['loss']:.4f}")
    print(f"Test accuracy: {result['accuracy']:.2%}")
    print("\nPer-class test accuracy:")
    for name, class_accuracy in zip(
        class_names,
        result["per_class_accuracy"],
        strict=False,
    ):
        print(f"  {name:>10}: {class_accuracy:.2%}")

    output_dir = PROJECT_ROOT / config["artifacts"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "test_loss": result["loss"],
        "test_accuracy": result["accuracy"],
        "per_class_accuracy": {
            name: float(class_accuracy)
            for name, class_accuracy in zip(
                class_names,
                result["per_class_accuracy"],
                strict=False,
            )
        },
        "confusion_matrix": result["confusion_matrix"].tolist(),
        "train_samples": expected_train_samples,
        "test_samples": len(x_test),
        "epochs": epochs,
        "batch_size": batch_size,
        "parameter_count": model.parameter_count(),
    }

    metrics_path = output_dir / "final_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nSaved metrics to:", metrics_path)


if __name__ == "__main__":
    main()