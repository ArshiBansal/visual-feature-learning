"""CIFAR-10 model implementations."""

from .cnn import CIFAR10CNN, cross_entropy_with_logits

__all__ = [
    "CIFAR10CNN",
    "cross_entropy_with_logits",
]