import sys
import types

import numpy as np
import pytest

from data.cifar10 import (
    CIFAR10_IMAGE_SHAPE,
    CIFAR10_NUM_CLASSES,
    CIFAR10_TEST_SAMPLES,
    CIFAR10_TRAIN_SAMPLES,
    load_cifar10,
    validate_cifar10_split,
)
from data.preprocessing import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    compute_channel_statistics,
    nhwc_to_nchw,
    normalize_nhwc,
    preprocess_cifar10_nchw,
    scale_to_float32,
)


def test_cifar10_constants():
    assert CIFAR10_IMAGE_SHAPE == (32, 32, 3)
    assert CIFAR10_NUM_CLASSES == 10
    assert CIFAR10_TRAIN_SAMPLES == 50_000
    assert CIFAR10_TEST_SAMPLES == 10_000


def test_scale_to_float32():
    images = np.array([[[[0, 127, 255]]]], dtype=np.uint8)

    scaled = scale_to_float32(images)

    assert scaled.dtype == np.float32
    np.testing.assert_allclose(scaled[0, 0, 0], [0.0, 127 / 255, 1.0])


def test_scale_rejects_non_uint8():
    with pytest.raises(TypeError):
        scale_to_float32(np.zeros((1, 32, 32, 3), dtype=np.float32))


def test_nhwc_to_nchw():
    images = np.arange(1 * 2 * 3 * 3, dtype=np.float32).reshape(1, 2, 3, 3)

    converted = nhwc_to_nchw(images)

    assert converted.shape == (1, 3, 2, 3)
    np.testing.assert_array_equal(converted[0, 0], images[0, :, :, 0])


def test_normalize_nhwc():
    images = np.ones((2, 32, 32, 3), dtype=np.float32)

    normalized = normalize_nhwc(images)

    assert normalized.shape == images.shape
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, (1.0 - CIFAR10_MEAN) / CIFAR10_STD)


def test_normalize_rejects_bad_shape():
    with pytest.raises(ValueError, match="NHWC"):
        normalize_nhwc(np.ones((3, 32, 32), dtype=np.float32))


def test_compute_channel_statistics():
    images = np.concatenate(
        [
            np.zeros((1, 2, 2, 3), dtype=np.uint8),
            np.full((1, 2, 2, 3), 255, dtype=np.uint8),
        ],
        axis=0,
    )

    mean, std = compute_channel_statistics(images)

    np.testing.assert_allclose(mean, np.full(3, 0.5), atol=1e-6)
    np.testing.assert_allclose(std, np.full(3, 0.5), atol=1e-6)


def test_preprocess_cifar10_nchw():
    images = np.zeros((2, 32, 32, 3), dtype=np.uint8)

    preprocessed = preprocess_cifar10_nchw(images)

    assert preprocessed.shape == (2, 3, 32, 32)
    assert preprocessed.dtype == np.float32
    np.testing.assert_allclose(preprocessed, -CIFAR10_MEAN[:, None, None] / CIFAR10_STD[:, None, None])


def test_validate_cifar10_split_accepts_official_shapes():
    x_train = np.zeros((CIFAR10_TRAIN_SAMPLES, 32, 32, 3), dtype=np.uint8)
    y_train = np.zeros(CIFAR10_TRAIN_SAMPLES, dtype=np.int64)
    x_test = np.zeros((CIFAR10_TEST_SAMPLES, 32, 32, 3), dtype=np.uint8)
    y_test = np.zeros(CIFAR10_TEST_SAMPLES, dtype=np.int64)

    validate_cifar10_split(x_train, y_train, x_test, y_test)


@pytest.mark.parametrize(
    ("train_size", "test_size"),
    [(49_999, 10_000), (50_000, 9_999)],
)
def test_validate_cifar10_split_rejects_wrong_split_size(train_size, test_size):
    x_train = np.zeros((train_size, 32, 32, 3), dtype=np.uint8)
    y_train = np.zeros(train_size, dtype=np.int64)
    x_test = np.zeros((test_size, 32, 32, 3), dtype=np.uint8)
    y_test = np.zeros(test_size, dtype=np.int64)

    with pytest.raises(ValueError):
        validate_cifar10_split(x_train, y_train, x_test, y_test)


def test_load_cifar10_with_mock_tensorflow(monkeypatch):
    x_train = np.zeros((CIFAR10_TRAIN_SAMPLES, 32, 32, 3), dtype=np.uint8)
    y_train = np.zeros((CIFAR10_TRAIN_SAMPLES, 1), dtype=np.int64)
    x_test = np.zeros((CIFAR10_TEST_SAMPLES, 32, 32, 3), dtype=np.uint8)
    y_test = np.zeros((CIFAR10_TEST_SAMPLES, 1), dtype=np.int64)

    cifar10_module = types.ModuleType("tensorflow.keras.datasets.cifar10")
    cifar10_module.load_data = lambda: (
        (x_train, y_train),
        (x_test, y_test),
    )

    tensorflow = types.ModuleType("tensorflow")
    keras = types.ModuleType("tensorflow.keras")
    datasets = types.ModuleType("tensorflow.keras.datasets")

    datasets.cifar10 = cifar10_module
    keras.datasets = datasets
    tensorflow.keras = keras

    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", keras)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.datasets", datasets)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.datasets.cifar10", cifar10_module)

    (loaded_x_train, loaded_y_train), (loaded_x_test, loaded_y_test) = load_cifar10()

    assert loaded_x_train.shape == x_train.shape
    assert loaded_x_test.shape == x_test.shape
    assert loaded_y_train.shape == (CIFAR10_TRAIN_SAMPLES,)
    assert loaded_y_test.shape == (CIFAR10_TEST_SAMPLES,)
    assert loaded_y_train.dtype == np.int64
    assert loaded_y_test.dtype == np.int64