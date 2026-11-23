"""Evaluate a trained CIFAR-10 NumPy CNN checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cifar10 import load_cifar10
from data.preprocessing import preprocess_cifar10_nchw
from evaluation.metrics import evaluate_model
from models.cnn import CIFAR10CNN


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cifar10_baseline.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 NumPy CNN")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_checkpoint(model: CIFAR10CNN, path: Path) -> None:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    with np.load(path) as checkpoint:
        model.conv1.weights = np.asarray(checkpoint["conv1_weights"]).copy()
        model.conv1.bias = np.asarray(checkpoint["conv1_bias"]).copy()
        model.conv2.weights = np.asarray(checkpoint["conv2_weights"]).copy()
        model.conv2.bias = np.asarray(checkpoint["conv2_bias"]).copy()
        model.dense.weights = np.asarray(checkpoint["dense_weights"]).copy()
        model.dense.bias = np.asarray(checkpoint["dense_bias"]).copy()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config["project"]["seed"])
    dataset_config = config["dataset"]
    training_config = config["training"]

    (_, _), (x_test_raw, y_test) = load_cifar10()
    x_test = preprocess_cifar10_nchw(x_test_raw)

    if len(x_test) != dataset_config["expected_test_samples"]:
        raise ValueError("Unexpected CIFAR-10 test split size")

    rng = np.random.default_rng(seed)
    model = CIFAR10CNN(rng)

    checkpoint_dir = PROJECT_ROOT / config["artifacts"]["checkpoint_dir"]
    checkpoint = args.checkpoint or checkpoint_dir / "cifar10_cnn_latest.npz"
    load_checkpoint(model, checkpoint)

    batch_size = int(training_config["batch_size"])
    num_classes = int(dataset_config["num_classes"])
    result = evaluate_model(
        model=model,
        x=x_test,
        y=y_test,
        batch_size=batch_size,
        num_classes=num_classes,
    )

    confusion = result["confusion_matrix"]
    per_class = result["per_class_accuracy"]
    class_names = dataset_config["class_names"]

    print(f"Test loss:     {result['loss']:.4f}")
    print(f"Test accuracy: {result['accuracy']:.2%}")
    print("\nPer-class test accuracy:")
    for name, class_accuracy in zip(class_names, per_class):
        print(f"  {name:>10}: {class_accuracy:.2%}")

    output_dir = PROJECT_ROOT / config["artifacts"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "test_loss": result["loss"],
        "test_accuracy": result["accuracy"],
        "per_class_accuracy": {
            name: float(class_accuracy)
            for name, class_accuracy in zip(class_names, per_class)
        },
        "confusion_matrix": confusion.tolist(),
        "test_samples": len(x_test),
        "batch_size": batch_size,
        "parameter_count": model.parameter_count(),
    }

    metrics_path = output_dir / "final_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print("\nSaved metrics to:", metrics_path)


if __name__ == "__main__":
    main()