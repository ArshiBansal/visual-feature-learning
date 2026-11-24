"""Single-image prediction using a configured CIFAR-10 CNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from data.preprocessing import preprocess_cifar10_nchw


@dataclass(frozen=True)
class Prediction:
    """Top prediction and normalized class probabilities for an image."""

    predicted_class: str
    confidence: float
    probabilities: np.ndarray
    class_names: tuple[str, ...]


def predict_cifar10_image(
    model: Any,
    image: np.ndarray,
    class_names: list[str] | tuple[str, ...],
) -> Prediction:
    """Normalize a raw RGB image and return stable softmax probabilities."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be an RGB image shaped (height, width, 3)")
    if image.dtype != np.uint8:
        raise TypeError("image must use uint8 CIFAR-10 pixel values")
    if not class_names:
        raise ValueError("class_names must not be empty")

    logits = np.asarray(model.forward(preprocess_cifar10_nchw(image[None, ...])))
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(class_names):
        raise ValueError("model output must contain one logit per class")
    if not np.all(np.isfinite(logits)):
        raise ValueError("model output must contain finite logits")

    shifted = logits[0] - logits[0].max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    predicted_index = int(probabilities.argmax())
    return Prediction(
        predicted_class=class_names[predicted_index],
        confidence=float(probabilities[predicted_index]),
        probabilities=probabilities,
        class_names=tuple(class_names),
    )