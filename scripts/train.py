"""Train the CIFAR-10 NumPy CNN baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cifar10 import load_cifar10
from data.preprocessing import preprocess_cifar10_nchw
from models.config import build_cifar10_cnn
from models.cnn import CIFAR10CNN, cross_entropy_with_logits


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cifar10_baseline.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 NumPy CNN")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def save_checkpoint(model: CIFAR10CNN, path: Path) -> None:
    """Atomically replace a checkpoint so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".npz",
        dir=path.parent,
    )

    try:
        with os.fdopen(file_descriptor, "wb") as checkpoint_file:
            np.savez(
                checkpoint_file,
                conv1_weights=model.conv1.weights,
                conv1_bias=model.conv1.bias,
                conv2_weights=model.conv2.weights,
                conv2_bias=model.conv2.bias,
                dense_weights=model.dense.weights,
                dense_bias=model.dense.bias,
            )
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())

        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


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
    shuffle_each_epoch: bool,
    drop_last_batch: bool,
) -> dict[str, float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(x) == 0:
        raise ValueError("training data must not be empty")
    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of samples")

    sample_count = len(x)
    if drop_last_batch:
        sample_count = (sample_count // batch_size) * batch_size
        if sample_count == 0:
            raise ValueError(
                "drop_last_batch would discard the entire training dataset"
            )

    order = (
        rng.permutation(len(x))
        if shuffle_each_epoch
        else np.arange(len(x))
    )
    order = order[:sample_count]

    total_loss = 0.0
    total_correct = 0
    batch_count = (sample_count + batch_size - 1) // batch_size
    started_at = time.time()

    for batch_number, start in enumerate(
        range(0, sample_count, batch_size),
        start=1,
    ):
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
        "loss": total_loss / sample_count,
        "accuracy": total_correct / sample_count,
        "seconds": time.time() - started_at,
    }


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config["project"]["seed"])
    dataset_config = config["dataset"]
    training_config = config["training"]
    optimizer_config = training_config["optimizer"]
    schedule_config = training_config["learning_rate_schedule"]
    artifact_config = config["artifacts"]

    if dataset_config.get("allow_subsampling", False):
        raise ValueError("Training currently supports only the complete CIFAR-10 split")
    if optimizer_config.get("name") != "sgd":
        raise ValueError("Only the SGD optimizer is supported")
    if schedule_config.get("name") != "step_decay":
        raise ValueError("Only the step_decay learning-rate schedule is supported")

    epochs = int(training_config["epochs"])
    batch_size = int(training_config["batch_size"])
    if epochs <= 0:
        raise ValueError("training.epochs must be positive")
    if batch_size <= 0:
        raise ValueError("training.batch_size must be positive")

    shuffle_each_epoch = _require_bool(
        training_config["shuffle_each_epoch"],
        "training.shuffle_each_epoch",
    )
    drop_last_batch = _require_bool(
        training_config["drop_last_batch"],
        "training.drop_last_batch",
    )
    checkpoint_every_epoch = _require_bool(
        training_config["checkpoint_every_epoch"],
        "training.checkpoint_every_epoch",
    )
    horizontal_flip = _require_bool(
        dataset_config["augmentation"]["horizontal_flip"],
        "dataset.augmentation.horizontal_flip",
    )

    (x_train_raw, y_train), (_, _) = load_cifar10()
    x_train = preprocess_cifar10_nchw(x_train_raw)

    if len(x_train) != dataset_config["expected_train_samples"]:
        raise ValueError("Unexpected CIFAR-10 training split size")

    rng = np.random.default_rng(seed)
    model = build_cifar10_cnn(config, rng)

    learning_rate = float(optimizer_config["learning_rate"])
    momentum = float(optimizer_config["momentum"])
    weight_decay = float(optimizer_config["weight_decay"])
    milestones = {int(milestone) for milestone in schedule_config["milestones"]}
    schedule_gamma = float(schedule_config["gamma"])
    log_every = int(training_config["log_every_n_batches"])

    if not np.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("optimizer.learning_rate must be finite and non-negative")
    if not np.isfinite(momentum) or momentum < 0:
        raise ValueError("optimizer.momentum must be finite and non-negative")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("optimizer.weight_decay must be finite and non-negative")
    if not np.isfinite(schedule_gamma) or schedule_gamma < 0:
        raise ValueError("learning_rate_schedule.gamma must be finite and non-negative")
    if any(milestone <= 0 for milestone in milestones):
        raise ValueError("learning_rate_schedule.milestones must be positive")
    if log_every < 0:
        raise ValueError("training.log_every_n_batches must be non-negative")

    output_dir = PROJECT_ROOT / artifact_config["output_dir"]
    checkpoint_dir = PROJECT_ROOT / artifact_config["checkpoint_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "cifar10_cnn_latest.npz"
    metrics_path = output_dir / "final_metrics.json"
    save_history = _require_bool(
        artifact_config["save_training_history"],
        "artifacts.save_training_history",
    )

    history: dict[str, list[float] | list[int]] = {
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
            shuffle_each_epoch=shuffle_each_epoch,
            drop_last_batch=drop_last_batch,
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

        if checkpoint_every_epoch:
            save_checkpoint(model, checkpoint_path)
            if metrics_path.is_file():
                metrics_path.unlink()

        if save_history:
            (output_dir / "training_history.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )

        if epoch in milestones:
            learning_rate *= schedule_gamma

    if not checkpoint_every_epoch:
        save_checkpoint(model, checkpoint_path)
        if metrics_path.is_file():
            metrics_path.unlink()

    print("Training complete. Run scripts/evaluate.py to refresh test metrics.")


if __name__ == "__main__":
    main()