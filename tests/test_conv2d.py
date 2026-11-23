import numpy as np
import pytest

from layers.conv2d import conv2d_backward, conv2d_forward


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