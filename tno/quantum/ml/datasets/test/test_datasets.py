"""This module contains test function for the datasets package."""
from typing import Any, List

import numpy as np
import pytest

from tno.quantum.ml.datasets._datasets import (
    _pre_process_data,
    get_blobs_clustering_dataset,
    get_circles_dataset,
    get_iris_dataset,
    get_linearly_separables_dataset,
    get_moons_dataset,
    get_wine_dataset,
)

# pylint: disable=invalid-name


@pytest.mark.parametrize("n_features", [1, 2, 4])
@pytest.mark.parametrize("n_classes", [1, 2, 3])
def test_get_wine_dataset(n_features: int, n_classes: int) -> None:
    """Test if n_features and n_classes gives expected output.

    Args:
        n_features: See `n_features` of ``get_wine_dataset``.
        n_classes: See `n_classes` of ``get_wine_dataset``.
    """
    X_training, y_training, X_validation, y_validation = get_wine_dataset(
        n_features=n_features, n_classes=n_classes, random_seed=0
    )

    assert X_training.shape[1] == n_features
    assert X_validation.shape[1] == n_features
    assert not np.array_equal(X_training, X_validation)

    assert len(np.unique(y_training)) == n_classes
    assert len(np.unique(y_validation)) == n_classes
    assert not np.array_equal(y_training, y_validation)


@pytest.mark.parametrize("n_features", [1, 2, 4])
@pytest.mark.parametrize("n_classes", [1, 2, 3])
def test_get_iris_dataset(n_features: int, n_classes: int) -> None:
    """Test if n_features and n_classes gives expected output.

    Args:
        n_features: See `n_features` of ``get_iris_dataset``.
        n_classes: See `n_classes` of ``get_iris_dataset``.
    """
    X_training, y_training, X_validation, y_validation = get_iris_dataset(
        n_features=n_features, n_classes=n_classes, random_seed=0
    )

    assert X_training.shape[1] == n_features
    assert X_validation.shape[1] == n_features
    assert not np.array_equal(X_training, X_validation)

    assert len(np.unique(y_training)) == n_classes
    assert len(np.unique(y_validation)) == n_classes
    assert not np.array_equal(y_training, y_validation)


@pytest.mark.parametrize(
    "method_to_test, args",
    [
        (get_circles_dataset, []),
        (get_iris_dataset, []),
        (get_linearly_separables_dataset, []),
        (get_moons_dataset, []),
        (get_wine_dataset, []),
        (get_blobs_clustering_dataset, [10, 10, 10]),
    ],
)
def test_seed(method_to_test: Any, args: List[Any]) -> None:
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
