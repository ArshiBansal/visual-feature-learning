import numpy as np
import pytest

from layers.max_pool2d import max_pool2d_backward, max_pool2d_forward


def test_max_pool_output_and_overlapping_backward():
    x = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, 9.0, 6.0], [7.0, 8.0, 5.0]]]],
        dtype=np.float64,
    )

    pooled, cache = max_pool2d_forward(x, pool_size=2, stride=1)

    np.testing.assert_allclose(pooled, np.array([[[[9.0, 9.0], [9.0, 9.0]]]]))

    upstream = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float64)
    d_x = max_pool2d_backward(upstream, cache)

    expected = np.zeros_like(x)
    expected[0, 0, 1, 1] = 10.0
    np.testing.assert_allclose(d_x, expected)


def test_pool_output_shape():
    x = np.arange(64, dtype=np.float64).reshape(1, 1, 8, 8)

    pooled, cache = max_pool2d_forward(x, pool_size=2, stride=2)

    assert pooled.shape == (1, 1, 4, 4)
    assert cache["output_shape"] == (1, 1, 4, 4)


def test_pool_size_must_be_positive():
    x = np.ones((1, 1, 4, 4), dtype=np.float64)

    with pytest.raises((TypeError, ValueError)):
        max_pool2d_forward(x, pool_size=0)


def test_pool_window_must_fit_input():
    x = np.ones((1, 1, 2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="larger"):
        max_pool2d_forward(x, pool_size=3)


def test_tied_maximum_split_gradient():
    x = np.ones((1, 1, 2, 2), dtype=np.float64)

    pooled, cache = max_pool2d_forward(x, pool_size=2, stride=2)
    d_x = max_pool2d_backward(np.array([[[[4.0]]]], dtype=np.float64), cache)

    np.testing.assert_allclose(d_x, np.ones_like(x), rtol=0, atol=1e-12)


def test_backward_rejects_incompatible_upstream_shape():
    x = np.ones((1, 1, 4, 4), dtype=np.float64)
    _, cache = max_pool2d_forward(x, pool_size=2, stride=2)

    with pytest.raises(ValueError, match="d_output"):
        max_pool2d_backward(np.ones((1, 1, 1, 1), dtype=np.float64), cache)