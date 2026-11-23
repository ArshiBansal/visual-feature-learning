"""Confusion-matrix construction utilities."""

from __future__ import annotations

import numpy as np


def build_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Build a row=true-class, column=predicted-class confusion matrix."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError("labels and predictions must be one-dimensional arrays")
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have equal shapes")
    if labels.dtype.kind not in "iu" or predictions.dtype.kind not in "iu":
        raise TypeError("labels and predictions must contain integer class indices")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if labels.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("labels contain an out-of-range class index")
    if np.any(predictions < 0) or np.any(predictions >= num_classes):
        raise ValueError("predictions contain an out-of-range class index")

    return np.bincount(
        labels * num_classes + predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def normalized_confusion_matrix(confusion: np.ndarray) -> np.ndarray:
    """Normalize each row of a confusion matrix by its true-class total."""
    confusion = np.asarray(confusion)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square two-dimensional array")

    normalized = np.zeros(confusion.shape, dtype=np.float64)
    row_totals = confusion.sum(axis=1, keepdims=True)
    np.divide(
        confusion,
        row_totals,
        out=normalized,
        where=row_totals != 0,
    )
    return normalized