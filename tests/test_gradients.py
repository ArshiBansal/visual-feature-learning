import numpy as np
import pytest

from layers.conv2d import conv2d_backward, conv2d_forward
from layers.max_pool2d import max_pool2d_backward, max_pool2d_forward


def relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    analytical = np.asarray(analytical)
    numerical = np.asarray(numerical)
    numerator = np.linalg.norm(analytical - numerical)
    denominator = max(
        1e-12,
        np.linalg.norm(analytical) + np.linalg.norm(numerical),
    )
    return float(numerator / denominator)


def numerical_gradient(
    array: np.ndarray,
    objective,
    epsilon: float = 1e-5,
) -> np.ndarray:
    gradient = np.zeros_like(array, dtype=np.float64)
    iterator = np.nditer(
        array,
        flags=["multi_index"],
        op_flags=["readwrite"],
    )

    while not iterator.finished:
        index = iterator.multi_index
        original = array[index]

        array[index] = original + epsilon
        objective_plus = objective()

        array[index] = original - epsilon
        objective_minus = objective()

        array[index] = original
        gradient[index] = (
            objective_plus - objective_minus
        ) / (2 * epsilon)
        iterator.iternext()

    return gradient


@pytest.mark.parametrize(
    ("stride", "padding"),
    [
        (1, "same"),
        (2, "same"),
        (1, 1),
        (1, (1, 2)),
        (1, (1, 2, 3, 4)),
        (2, (1, 2, 3, 4)),
    ],
)
def test_conv2d_input_weight_and_bias_gradients(stride, padding):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1, 2, 4, 5))
    weights = rng.normal(size=(2, 2, 2, 3))
    bias = rng.normal(size=2)

    output, cache = conv2d_forward(
        x,
        weights,
        bias,
        stride=stride,
        padding=padding,
    )
    upstream = rng.normal(size=output.shape)

    d_x, d_weights, d_bias = conv2d_backward(upstream, cache)

    def objective():
        result, _ = conv2d_forward(
            x,
            weights,
            bias,
            stride=stride,
            padding=padding,
        )
        return np.sum(result * upstream)

    numeric_d_x = numerical_gradient(x, objective)
    numeric_d_weights = numerical_gradient(weights, objective)
    numeric_d_bias = numerical_gradient(bias, objective)

    assert relative_error(d_x, numeric_d_x) < 1e-7
    assert relative_error(d_weights, numeric_d_weights) < 1e-7
    assert relative_error(d_bias, numeric_d_bias) < 1e-7


def test_even_kernel_same_padding_gradient():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(1, 2, 5, 5))
    weights = rng.normal(size=(3, 2, 2, 2))
    bias = rng.normal(size=3)

    output, cache = conv2d_forward(
        x,
        weights,
        bias,
        stride=2,
        padding="same",
    )
    upstream = rng.normal(size=output.shape)
    d_x, d_weights, d_bias = conv2d_backward(upstream, cache)

    def objective():
        result, _ = conv2d_forward(
            x,
            weights,
            bias,
            stride=2,
            padding="same",
        )
        return np.sum(result * upstream)

    numeric_d_x = numerical_gradient(x, objective)
    numeric_d_weights = numerical_gradient(weights, objective)
    numeric_d_bias = numerical_gradient(bias, objective)

    assert relative_error(d_x, numeric_d_x) < 1e-7
    assert relative_error(d_weights, numeric_d_weights) < 1e-7
    assert relative_error(d_bias, numeric_d_bias) < 1e-7


def test_max_pool2d_input_gradient():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(1, 2, 4, 4))
    upstream = rng.normal(size=(1, 2, 2, 2))

    _, cache = max_pool2d_forward(
        x,
        pool_size=2,
        stride=2,
    )

    d_x = max_pool2d_backward(upstream, cache)

    def objective():
        result, _ = max_pool2d_forward(
            x,
            pool_size=2,
            stride=2,
        )
        return np.sum(result * upstream)

    numeric_d_x = numerical_gradient(x, objective)

    assert relative_error(d_x, numeric_d_x) < 1e-7


def test_overlapping_pool_input_gradient():
    x = np.array(
        [[[[1.0, 2.0, 3.0],
           [4.0, 9.0, 6.0],
           [7.0, 8.0, 5.0]]]],
        dtype=np.float64,
    )
    upstream = np.array(
        [[[[1.0, 2.0],
           [3.0, 4.0]]]],
        dtype=np.float64,
    )

    _, cache = max_pool2d_forward(
        x,
        pool_size=2,
        stride=1,
    )

    d_x = max_pool2d_backward(upstream, cache)

    def objective():
        result, _ = max_pool2d_forward(
            x,
            pool_size=2,
            stride=1,
        )
        return np.sum(result * upstream)

    numeric_d_x = numerical_gradient(x, objective)

    assert relative_error(d_x, numeric_d_x) < 1e-7