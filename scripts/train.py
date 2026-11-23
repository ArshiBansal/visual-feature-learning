"""Train the CIFAR-10 NumPy CNN baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cifar10 import load_cifar10
from data.preprocessing import preprocess_cifar10_nchw
from models.cnn import CIFAR10CNN, cross_entropy_with_logits


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cifar10_baseline.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 NumPy CNN")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    with path.open("encoding="utf-8") as file:
        return json.load(file)


def save_checkpoint(model: CIFAR10CNN, path: Path) -> None:
    np.savez(
        path,
        conv1_weights=model.conv1.weights,
        conv1_bias=model.conv1.bias,
        conv2_weights=model.conv2.weights,
        conv2_bias=model.conv2.bias,
        dense_weights=model.dense.weights,
        dense_bias=model.dense.bias,
    )


def train_one_epoch(
    model: CIFAR10CNN,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    rng: np.random.Generator,
    horizontal_flip: bool,
    log_every: int,
) -> dict[str, float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of samples")

    order = rng.permutation(len(x))
    total_loss = 0.0
    total_correct = 0
    batch_count = (len(x) + batch_size - 1) // batch_size
    started_at = time.time()

    for batch_number, start in enumerate(range(0, len(x), batch_size), start=1):
        indices = order[start : start + batch_size]
        batch_x = x[indices].copy()
        batch_y = y[indices]

        if horizontal_flip:
            flip_mask = rng.random(len(batch_x)) < 0.5
            batch_x[flip_mask] = batch_x[flip_mask, :, :, ::-1]

        logits = model.forward(batch_x)
        loss, d_logits = cross_entropy_with_logits(logits, batch_y)
        gradients = model.backward(d_logits)
        model.update(
            gradients,
            learning_rate,
            momentum,
            weight_decay,
        )

        total_loss += loss * len(batch_y)
        total_correct += int(np.sum(logits.argmax(axis=1) == batch_y))

        if log_every and batch_number % log_every == 0:
            samples_seen = start + len(batch_y)
            print(
                f"  batch {batch_number:>4}/{batch_count} "
                f"- loss {total_loss / samples_seen:.4f}"
            )

    return {
        "loss": total_loss / len(x),
        "accuracy": total_correct / len(x),
        "seconds": time.time() - started_at,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config["project"]["seed"])
    dataset_config = config["dataset"]
    training_config = config["training"]
    optimizer_config = training_config["optimizer"]
    schedule_config = training_config["learning_rate_schedule"]

    (x_train_raw, y_train), (x_test_raw, y_test) = load_cifar10()

    if len(x_train_raw) != dataset_config["expected_train_samples"]:
        raise ValueError("Unexpected CIFAR-10 training split size")
    if len(x_test_raw) != dataset_config["expected_test_samples"]:
        raise ValueError("Unexpected CIFAR-10 test split size")

    x_train = preprocess_cifar10_nchw(x_train_raw)
    x_test = preprocess_cifar10_nchw(x_test_raw)

    rng = np.random.default_rng(seed)
    model = CIFAR10CNN(rng)

    if model.flattened_features != 32 * 8 * 8:
        raise RuntimeError("CNN flattened feature size is inconsistent")

    epochs = int(training_config["epochs"])
    batch_size = int(training_config["batch_size"])
    learning_rate = float(optimizer_config["learning_rate"])
    momentum = float(optimizer_config["momentum"])
    weight_decay = float(optimizer_config["weight_decay"])
    milestones = set(schedule_config["milestones"])
    schedule_gamma = float(schedule_config["gamma"])
    horizontal_flip = bool(dataset_config["augmentation"]["horizontal_flip"])
    log_every = int(training_config["log_every_n_batches"])

    output_dir = PROJECT_ROOT / config["artifacts"]["output_dir"]
    checkpoint_dir = PROJECT_ROOT / config["artifacts"]["checkpoint_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "learning_rate": [],
        "seconds": [],
    }

    print(f"Training samples per epoch: {len(x_train):,}")
    print(f"Total epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Parameters: {model.parameter_count():,}")

    for epoch in range(1, epochs + 1):
        result = train_one_epoch(
            model=model,
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            rng=rng,
            horizontal_flip=horizontal_flip,
            log_every=log_every,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(result["loss"])
        history["train_accuracy"].append(result["accuracy"])
        history["learning_rate"].append(learning_rate)
        history["seconds"].append(result["seconds"])

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss {result['loss']:.4f} | "
            f"accuracy {result['accuracy']:.2%} | "
            f"lr {learning_rate:g} | "
            f"{result['seconds'] / 60:.1f} min"
        )

        save_checkpoint(model, checkpoint_dir / "cifar10_cnn_latest.npz")
        (output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        if epoch in milestones:
            learning_rate *= schedule_gamma

    del x_train, x_test, x_train_raw, x_test_raw
    print("Training complete.")


if __name__ == "__main__":
    main()