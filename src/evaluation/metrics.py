"""Model evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from training.loss import cross_entropy_with_logits


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Return classification accuracy as a fraction in [0, 1]."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if predictions.ndim != 1:
        raise ValueError("predictions must be a one-dimensional array")
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have equal shapes")
    if labels.size == 0:
        raise ValueError("labels and predictions must not be empty")

    return float(np.mean(labels == predictions))


def per_class_accuracy(confusion: np.ndarray) -> np.ndarray:
    """Return recall for each true class from a square confusion matrix."""
    confusion = np.asarray(confusion)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square two-dimensional array")

    true_totals = confusion.sum(axis=1)
    result = np.zeros(confusion.shape[0], dtype=np.float64)
    np.divide(
        np.diag(confusion),
        true_totals,
        out=result,
        where=true_totals != 0,
    )
    return result


def evaluate_model(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_classes: int,
) -> dict[str, Any]:
    """Evaluate loss, accuracy, and a row-major true/predicted confusion matrix."""
    x = np.asarray(x)
    y = np.asarray(y)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of samples")
    if len(x) == 0:
        raise ValueError("evaluation data must not be empty")

    total_loss = 0.0
    all_predictions: list[np.ndarray] = []
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for start in range(0, len(x), batch_size):
        batch_x = x[start : start + batch_size]
        batch_y = y[start : start + batch_size]
        logits = model.forward(batch_x)
        loss, _ = cross_entropy_with_logits(logits.copy(), batch_y)
        predictions = logits.argmax(axis=1)

        total_loss += loss * len(batch_y)
        all_predictions.append(predictions)
        confusion += np.bincount(
            batch_y * num_classes + predictions,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    predictions = np.concatenate(all_predictions)

    return {
        "loss": total_loss / len(x),
        "accuracy": accuracy(y, predictions),
        "confusion_matrix": confusion,
        "per_class_accuracy": per_class_accuracy(confusion),
    }