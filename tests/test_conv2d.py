import numpy as np
import pytest

from layers.conv2d import conv2d_backward, conv2d_forward
from models.cnn import Conv2D


def test_valid_convolution_output_shape_and_values():
    x = np.arange(1, 1 + 2 * 4 * 4, dtype=np.float64).reshape(1, 2, 4, 4)
    weights = np.ones((3, 2, 3, 3), dtype=np.float64)
    bias = np.zeros(3, dtype=np.float64)

    output, _ = conv2d_forward(x, weights, bias, stride=1, padding="valid")

    assert output.shape == (1, 3, 2, 2)
    np.testing.assert_allclose(output[:, 0], output[:, 1])
    np.testing.assert_allclose(output[:, 1], output[:, 2])


def test_same_convolution_output_shape():
    x = np.arange(1, 33, dtype=np.float64).reshape(1, 2, 4, 4)
    weights = np.ones((3, 2, 3, 3), dtype=np.float64)
    bias = np.zeros(3, dtype=np.float64)

    output, _ = conv2d_forward(x, weights, bias, stride=1, padding="same")

    assert output.shape == (1, 3, 4, 4)


@pytest.mark.parametrize("padding", [(-1,), -1, "invalid"])
def test_invalid_padding_raises(padding):
    x = np.ones((1, 1, 4, 4), dtype=np.float64)
    weights = np.ones((1, 1, 3, 3), dtype=np.float64)
    bias = np.zeros(1, dtype=np.float64)

    with pytest.raises((TypeError, ValueError)):
        conv2d_forward(x, weights, bias, padding=padding)


def test_channel_mismatch_raises():
    x = np.ones((1, 2, 4, 4), dtype=np.float64)
    weights = np.ones((3, 1, 3, 3), dtype=np.float64)
    bias = np.zeros(3, dtype=np.float64)

    with pytest.raises(ValueError, match="channels"):
        conv2d_forward(x, weights, bias)


def test_bias_shape_mismatch_raises():
    x = np.ones((1, 1, 4, 4), dtype=np.float64)
    weights = np.ones((2, 1, 3, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="bias"):
        conv2d_forward(x, weights, np.zeros(1, dtype=np.float64))


def test_backward_rejects_incompatible_upstream_shape():
    x = np.ones((1, 1, 4, 4), dtype=np.float64)
    weights = np.ones((1, 1, 3, 3), dtype=np.float64)
    bias = np.zeros(1, dtype=np.float64)
    _, cache = conv2d_forward(x, weights, bias, padding="same")

    with pytest.raises(ValueError, match="d_output"):
        conv2d_backward(np.ones((1, 1, 3, 3), dtype=np.float64), cache)


def test_conv2d_layer_matches_functional_forward_and_backward():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(2, 2, 5, 5)).astype(np.float32)
    upstream = rng.normal(size=(2, 3, 5, 5)).astype(np.float32)

    layer = Conv2D(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        rng=rng,
        padding="same",
    )
    weights = layer.weights.copy()
    bias = layer.bias.copy()

    expected_output, cache = conv2d_forward(
        x,
        weights,
        bias,
        padding="same",
    )
    actual_output = layer.forward(x)
    actual_d_x, actual_d_weights, actual_d_bias = layer.backward(upstream)
    expected_d_x, expected_d_weights, expected_d_bias = conv2d_backward(
        upstream,
        cache,
    )

    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_d_x, expected_d_x, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        actual_d_weights,
        expected_d_weights,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        actual_d_bias,
        expected_d_bias,
        rtol=1e-5,
        atol=1e-5,
    )


def test_conv2d_backward_requires_forward():
    layer = Conv2D(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        rng=np.random.default_rng(1),
        padding="same",
    )

    with pytest.raises(RuntimeError, match="before forward"):
        layer.backward(np.ones((1, 1, 4, 4), dtype=np.float32))


def test_asymmetric_padding_backward_matches_numerical_gradient():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(1, 1, 3, 3))
    weights = rng.normal(size=(1, 1, 2, 2))
    bias = rng.normal(size=1)
    output, cache = conv2d_forward(
        x,
        weights,
        bias,
        stride=2,
        padding=(1, 0),
    )
    upstream = rng.normal(size=output.shape)
    d_x, d_weights, d_bias = conv2d_backward(upstream, cache)

    def objective():
        result, _ = conv2d_forward(
            x,
            weights,
            bias,
            stride=2,
            padding=(1, 0),
        )
        return np.sum(result * upstream)

    epsilon = 1e-5
    numeric_weights = np.zeros_like(weights)
    for index in np.ndindex(weights.shape):
        original = weights[index]
        weights[index] = original + epsilon
        plus = objective()
        weights[index] = original - epsilon
        minus = objective()
        weights[index] = original
        numeric_weights[index] = (plus - minus) / (2 * epsilon)

    np.testing.assert_allclose(d_weights, numeric_weights, rtol=1e-7, atol=1e-7)
    assert d_x.shape == x.shape
    assert d_bias.shape == bias.shape