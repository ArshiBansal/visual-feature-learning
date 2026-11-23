from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .loss import cross_entropy_with_logits
from .optimizer import SGD


class Trainer:
    def __init__(
        self,
        model: Any,
        optimizer: SGD,
        num_classes: int,
        checkpoint_dir: str | Path,
        history_path: str | Path,
    ):
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.history_path = Path(history_path)

    def train_one_epoch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
        horizontal_flip: bool = False,
        log_every: int = 0,
    ) -> dict[str, float]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(x) == 0:
            raise ValueError("training data must not be empty")
        if len(x) != len(y):
            raise ValueError("x and y must contain the same number of samples")

        order = rng.permutation(len(x))
        total_loss = 0.0
        total_correct = 0
        batch_count = (len(x) + batch_size - 1) // batch_size
        start_time = time.time()

        for batch_number, start in enumerate(range(0, len(x), batch_size), start=1):
            indices = order[start : start + batch_size]
            batch_x = np.asarray(x[indices]).copy()
            batch_y = np.asarray(y[indices])

            if horizontal_flip:
                flip_mask = rng.random(len(batch_x)) < 0.5
                batch_x[flip_mask] = batch_x[flip_mask, :, :, ::-1]

            logits = self.model.forward(batch_x)
            loss, d_logits = cross_entropy_with_logits(logits, batch_y)
            gradients = self.model.backward(d_logits)

            self.optimizer.step(self.model.conv1, *gradients["conv1"])
            self.optimizer.step(self.model.conv2, *gradients["conv2"])
            self.optimizer.step(self.model.dense, *gradients["dense"])

            total_loss += loss * len(batch_y)
            predictions = logits.argmax(axis=1)
            total_correct += int(np.sum(predictions == batch_y))

            if log_every and batch_number % log_every == 0:
                samples_seen = start + len(batch_y)
                print(
                    f"  batch {batch_number:>4}/{batch_count} "
                    f"- loss {total_loss / samples_seen:.4f}"
                )

        return {
            "loss": total_loss / len(x),
            "accuracy": total_correct / len(x),
            "seconds": time.time() - start_time,
        }

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
    ) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(x) != len(y):
            raise ValueError("x and y must contain the same number of samples")

        total_loss = 0.0
        total_correct = 0
        confusion = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

        for start in range(0, len(x), batch_size):
            batch_x = np.asarray(x[start : start + batch_size])
            batch_y = np.asarray(y[start : start + batch_size])
            logits = self.model.forward(batch_x)
            loss, _ = cross_entropy_with_logits(logits.copy(), batch_y)
            predictions = logits.argmax(axis=1)

            total_loss += loss * len(batch_y)
            total_correct += int(np.sum(predictions == batch_y))
            confusion += np.bincount(
                batch_y * self.num_classes + predictions,
                minlength=self.num_classes * self.num_classes,
            ).reshape(self.num_classes, self.num_classes)

        sample_count = max(len(x), 1)
        return {
            "loss": total_loss / sample_count,
            "accuracy": total_correct / sample_count,
            "confusion_matrix": confusion,
        }

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        rng: np.random.Generator,
        horizontal_flip: bool = False,
        log_every: int = 0,
        milestones: list[int] | set[int] | None = None,
        schedule_gamma: float = 1.0,
    ) -> dict[str, list[Any]]:
        milestones = set(milestones or [])
        history: dict[str, list[Any]] = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "learning_rate": [],
            "seconds": [],
        }

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            result = self.train_one_epoch(
                x=x,
                y=y,
                batch_size=batch_size,
                rng=rng,
                horizontal_flip=horizontal_flip,
                log_every=log_every,
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(result["loss"])
            history["train_accuracy"].append(result["accuracy"])
            history["learning_rate"].append(self.optimizer.learning_rate)
            history["seconds"].append(result["seconds"])

            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"loss {result['loss']:.4f} | "
                f"accuracy {result['accuracy']:.2%} | "
                f"lr {self.optimizer.learning_rate:g} | "
                f"{result['seconds'] / 60:.1f} min"
            )

            self.save_checkpoint()
            self.history_path.write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

            if epoch in milestones:
                self.optimizer.set_learning_rate(
                    self.optimizer.learning_rate * schedule_gamma
                )

        return history

    def save_checkpoint(self, name: str = "cifar10_cnn_latest.npz") -> Path:
        path = self.checkpoint_dir / name
        np.savez(
            path,
            conv1_weights=self.model.conv1.weights,
            conv1_bias=self.model.conv1.bias,
            conv2_weights=self.model.conv2.weights,
            conv2_bias=self.model.conv2.bias,
            dense_weights=self.model.dense.weights,
            dense_bias=self.model.dense.bias,
        )
        return path