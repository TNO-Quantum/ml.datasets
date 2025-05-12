"""This module contains test function for the datasets package."""

from __future__ import annotations

from collections.abc import Callable
from math import ceil
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tno.quantum.ml.datasets._datasets import (
    _pre_process_data,
    get_anomalous_spiky_time_series_dataset,
    get_blobs_clustering_dataset,
    get_circles_dataset,
    get_iris_dataset,
    get_linearly_separable_dataset,
    get_moons_dataset,
    get_wine_dataset,
)


@pytest.mark.parametrize("n_features", [1, 2, 4])
@pytest.mark.parametrize("n_classes", [1, 2, 3])
def test_get_wine_dataset(n_features: int, n_classes: int) -> None:
    """Test if n_features and n_classes gives expected output.

    Args:
        n_features: See `n_features` of ``get_wine_dataset``.
        n_classes: See `n_classes` of ``get_wine_dataset``.
    """
    X_train, y_train, X_val, y_val = get_wine_dataset(
        n_features=n_features, n_classes=n_classes, random_seed=0
    )

    assert X_train.shape[1] == n_features
    assert X_val.shape[1] == n_features
    assert not np.array_equal(X_train, X_val)

    assert len(np.unique(y_train)) == n_classes
    assert len(np.unique(y_val)) == n_classes
    assert not np.array_equal(y_train, y_val)


@pytest.mark.parametrize("n_features", [1, 2, 4])
@pytest.mark.parametrize("n_classes", [1, 2, 3])
def test_get_iris_dataset(n_features: int, n_classes: int) -> None:
    """Test if n_features and n_classes gives expected output.

    Args:
        n_features: See `n_features` of ``get_iris_dataset``.
        n_classes: See `n_classes` of ``get_iris_dataset``.
    """
    X_train, y_train, X_val, y_val = get_iris_dataset(
        n_features=n_features, n_classes=n_classes, random_seed=0
    )

    assert X_train.shape[1] == n_features
    assert X_val.shape[1] == n_features
    assert not np.array_equal(X_train, X_val)

    assert len(np.unique(y_train)) == n_classes
    assert len(np.unique(y_val)) == n_classes
    assert not np.array_equal(y_train, y_val)


@pytest.mark.parametrize("n_samples", [100, 500, 255])
@pytest.mark.parametrize("n_features", [1, 2, 4])
@pytest.mark.parametrize("n_times", [100, 155, 200])
def test_get_anomalous_spiky_time_series_dataset(
    n_samples: int, n_features: int, n_times: int
) -> None:
    """Test if n_series, n_features, and n_classes give expected output.

    Args:
        n_samples: See `n_samples` of ``get_anomalous_spiky_time_series_dataset``.
        n_features: See `n_features` of ``get_anomalous_spiky_time_series_dataset``.
        n_classes: See `n_times` of ``get_anomalous_spiky_time_series_dataset``.
    """
    test_size = 0.4
    anomaly_proportion = 0.5
    X_train, y_train, X_val, y_val = get_anomalous_spiky_time_series_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_times=n_times,
        anomaly_proportion=anomaly_proportion,
        test_size=test_size,
        random_seed=42,
    )

    assert X_train.shape[0] == int(n_samples * (1 - test_size))
    assert X_train.shape[1] == n_features
    assert X_train.shape[2] == n_times

    assert X_val.shape[0] == int(n_samples * test_size)
    assert X_val.shape[1] == n_features
    assert X_val.shape[2] == n_times
    assert not np.array_equal(X_train, X_val)

    expected_n_labels = 2
    assert len(np.unique(y_train)) == expected_n_labels
    assert len(np.unique(y_val)) == expected_n_labels
    assert not np.array_equal(y_train, y_val)

    assert X_val.shape[0] == y_val.shape[0]
    assert X_train.shape[0] + X_val.shape[0] == n_samples


@pytest.mark.parametrize(
    ("method_to_test", "args"),
    [
        (get_circles_dataset, []),
        (get_iris_dataset, []),
        (get_linearly_separable_dataset, []),
        (get_moons_dataset, []),
        (get_wine_dataset, []),
        (get_blobs_clustering_dataset, [10, 10, 10]),
        (get_anomalous_spiky_time_series_dataset, [100, 2, 10, 0.4]),
    ],
)
def test_seed(method_to_test: Any, args: list[Any]) -> None:
    """Test if the random_seed gives expected output.

    In other words, two dataset should be equal if the seed is equal.
    If the seeds are different, then the datasets should be different.

    Args:
        method_to_test: Method to test the `random_seed` argument for.
        args: Additional arguments of the method to test.
    """
    data_a = method_to_test(*args, random_seed=0)
    data_b = method_to_test(*args, random_seed=0)
    data_c = method_to_test(*args, random_seed=1)

    for el_a, el_b in zip(data_a, data_b):
        assert np.array_equal(el_a, el_b)

    for el_a, el_c in zip(data_a, data_c):
        assert not np.array_equal(el_a, el_c)


def test_pre_process_data() -> None:
    """Test if _preprocess_data gives the expected output.

    _pre_process_data should slice X, and y to have the correct number of classes and
    number of features. Furthermore, the datatype of y is set to the datatype of X.
    """
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
    y = np.array([1, 0], dtype=float)

    assert X.dtype != y.dtype
    X, y = _pre_process_data(X, y, 2, 1)

    assert X.dtype == y.dtype
    assert X.shape == (1, 2)
    assert y.shape == (1,)


@pytest.mark.parametrize(
    "get_dataset",
    [
        get_wine_dataset,
        get_iris_dataset,
        get_moons_dataset,
        get_circles_dataset,
        get_linearly_separable_dataset,
        get_anomalous_spiky_time_series_dataset,
    ],
)
@pytest.mark.parametrize("test_size", [None, 0, 0.3, 0.33, 0.38, 0.5, 1, 10, 15])
def test_test_size(
    get_dataset: Callable[
        [Any],
        tuple[
            NDArray[np.float64], NDArray[np.int_], NDArray[np.float64], NDArray[np.int_]
        ],
    ],
    test_size: float | int | None,
) -> None:
    """Test if test_size argument splits the dataset accordingly."""
    X_train, y_train, X_val, y_val = get_dataset(test_size=test_size)  # type: ignore[call-arg]
    n_samples = X_train.shape[0] + X_val.shape[0]

    if test_size is None:
        test_size = 0.25  # default argument if test_size is None

    if isinstance(test_size, float):
        expected_test_size = ceil(test_size * n_samples)
    elif isinstance(test_size, int):
        expected_test_size = test_size

    assert y_val.shape[0] == expected_test_size
    assert y_train.shape[0] == n_samples - expected_test_size
