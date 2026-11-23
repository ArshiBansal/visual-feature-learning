import numpy as np
import pytest

from models.cnn import CIFAR10CNN, cross_entropy_with_logits


def test_cifar10_cnn_forward_shape():
    rng = np.random.default_rng(42)
    model = CIFAR10CNN(rng)
    x = np.random.default_rng(1).standard_normal((2, 3, 32, 32)).astype(np.float32)

    logits = model.forward(x)

    assert logits.shape == (2, 10)
    assert logits.dtype == np.float32
    assert np.all(np.isfinite(logits))


def test_cifar10_cnn_forward_backward_smoke():
    rng = np.random.default_rng(42)
    model = CIFAR10CNN(rng)
    x = np.random.default_rng(1).standard_normal((2, 3, 32, 32)).astype(np.float32)
    y = np.array([1, 7], dtype=np.int64)

    logits = model.forward(x)
    _, d_logits = cross_entropy_with_logits(logits, y)
    gradients = model.backward(d_logits)

    assert set(gradients) == {"conv1", "conv2", "dense"}
    assert gradients["conv1"][0].shape == model.conv1.weights.shape
    assert gradients["conv1"][1].shape == model.conv1.bias.shape
    assert gradients["conv2"][0].shape == model.conv2.weights.shape
    assert gradients["conv2"][1].shape == model.conv2.bias.shape
    assert gradients["dense"][0].shape == model.dense.weights.shape
    assert gradients["dense"][1].shape == model.dense.bias.shape

    for gradient_pair in gradients.values():
        for gradient in gradient_pair:
            assert np.all(np.isfinite(gradient))


def test_cifar10_cnn_rejects_wrong_spatial_shape():
    rng = np.random.default_rng(42)
    model = CIFAR10CNN(rng)

    with pytest.raises(ValueError, match="3, 32, 32"):
        model.forward(np.zeros((2, 3, 28, 28), dtype=np.float32))


def test_cifar10_cnn_rejects_rank_invalid_input():
    rng = np.random.default_rng(42)
    model = CIFAR10CNN(rng)

    with pytest.raises(ValueError, match="rank-4|shape"):
        model.forward(np.zeros((3, 32, 32), dtype=np.float32))


def test_cifar10_cnn_parameter_count():
    rng = np.random.default_rng(42)
    model = CIFAR10CNN(rng)

    expected = (
        model.conv1.weights.size
        + model.conv1.bias.size
        + model.conv2.weights.size
        + model.conv2.bias.size
        + model.dense.weights.size
        + model.dense.bias.size
    )

    assert model.parameter_count() == expected


def test_cross_entropy_loss_and_gradient():
    logits = np.array([[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)

    loss, d_logits = cross_entropy_with_logits(logits, labels)

    assert np.isfinite(loss)
    assert loss >= 0.0
    assert d_logits.shape == logits.shape
    assert np.allclose(d_logits.sum(axis=1), 0.0, atol=1e-6)


@pytest.mark.parametrize("labels", [np.array([0]), np.array([-1, 0]), np.array([0, 10])])
def test_cross_entropy_rejects_invalid_label_shapes(labels):
    logits = np.zeros((2, 10), dtype=np.float32)

    with pytest.raises((TypeError, ValueError)):
        cross_entropy_with_logits(logits, labels)