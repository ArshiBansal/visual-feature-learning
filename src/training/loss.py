from __future__ import annotations

import numpy as np


def cross_entropy_with_logits(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels)

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, num_classes)")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape (batch,")
    if labels.dtype.kind not in "iu":
        raise TypeError("labels must contain integer class indices")
    if logits.shape[0] == 0:
        raise ValueError("batch must not be empty")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("labels contain an out-of-range class index")

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    batch_indices = np.arange(labels.shape[0])
    loss = -np.log(probabilities[batch_indices, labels] + np.float32(1e-12)).mean()

    d_logits = probabilities.copy()
    d_logits[batch_indices, labels] -= np.float32(1.0)
    d_logits /= labels.shape[0]
    return float(loss), d_logits