"""This module contains datasets.

The datasets in this module can be used to test classifiers and clustering algorithms.
"""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.model_selection import train_test_split

from tno.quantum.utils.validation import check_int, check_real


def _pre_process_data(
    X: NDArray[Any], y: NDArray[Any], n_features: int, n_classes: int
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Slice `X` and `y` and cast the dtype of `y` to the dtype of `X`.

    Args:
        X: Feature matrix to slice.
        y: Target samples to slice and cast.
        n_features: Number of features.
        n_classes: Number of classes

    Returns:
        ``Tuple`` of `X` and `y`, where `X`, and `y` are sliced to have the correct
        number of classes and number of features. Furthermore, the datatype of `y` is
        set to the datatype of `X`.
    """
    # Validate input
    n_features = check_int(n_features, name="n_features", l_bound=1)
    n_classes = check_int(n_classes, name="n_classes", l_bound=1)

    y = y.astype(X.dtype)
    ind = y < n_classes
    X = X[ind, :n_features]
    y = y[ind]
    return X, y


def get_wine_dataset(
    n_features: int = 13,
    n_classes: int = 3,
    random_seed: int = 0,
    test_size: float | int | None = None,
) -> tuple[
    NDArray[np.float64], NDArray[np.int_], NDArray[np.float64], NDArray[np.int_]
]:
    r"""Load wine dataset.

    This function wraps :func:`~sklearn.datasets.load_wine` of :mod:`sklearn.datasets`.
    The dataset is loaded and split into training and validation data.

    Example usage::

        >>> from tno.quantum.ml.datasets import get_wine_dataset
        >>> X_train, y_train, X_val, y_val = get_wine_dataset()
        >>> print(f"{X_train.shape=}\n{y_train.shape=}\n{X_val.shape=}\n{y_val.shape=}")
        X_train.shape=(133, 13)
        y_train.shape=(133,)
        X_val.shape=(45, 13)
        y_val.shape=(45,)

    Args:
        n_features: Number of features. Defaults to 13.
        n_classes: Number of classes, must be 1, 2 or 3. Defaults to 3.
        random_seed: Seed to give to the random number generator. Defaults to 0.
        test_size: The proportion of the dataset that is included in the test-split.
            Either represented by a percentage in the range [0.0, 1.0) or as absolute
            number of test samples in the range [1, inf). Defaults to 0.25.

    Returns:
        A tuple containing ``X_training``, ``y_training``, ``X_validation`` and
        ``y_validation`` of the wine dataset.
    """
    # Validate input
    n_features = check_int(n_features, name="n_features", l_bound=1)
    n_classes = check_int(n_classes, name="n_classes", l_bound=1)
    random_seed = check_int(random_seed, name="random_seed", l_bound=0)

    # Load data and take subset
    X, y = datasets.load_wine(return_X_y=True)
    X, y = _pre_process_data(X, y, n_features, n_classes)

    # Split into training and validation data sets
    if test_size == 0:
        return (
            X,
            y,
            np.empty(shape=(0, n_features), dtype=np.float64),
            np.empty(shape=(0, 1), dtype=np.int_),
        )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    return X_train, y_train, X_val, y_val


def get_iris_dataset(
    n_features: int = 4,
    n_classes: int = 3,
    random_seed: int = 0,
    test_size: float | int | None = None,
) -> tuple[
    NDArray[np.float64], NDArray[np.int_], NDArray[np.float64], NDArray[np.int_]
]:
    r"""Generate the iris dataset.

    This function wraps :func:`~sklearn.datasets.load_iris` of :mod:`sklearn.datasets`.
    The dataset is loaded and split into training and validation data.

    Example usage:

        >>> from tno.quantum.ml.datasets import get_iris_dataset
        >>> X_train, y_train, X_val, y_val = get_iris_dataset()
        >>> print(f"{X_train.shape=}\n{y_train.shape=}\n{X_val.shape=}\n{y_val.shape=}")
        X_train.shape=(112, 4)
        y_train.shape=(112,)
        X_val.shape=(38, 4)
        y_val.shape=(38,)

    Args:
        n_features: Number of features. Defaults to 4.
        n_classes: Number of classes, must be 1, 2 or 3. Defaults to 3.
        random_seed: Seed to give to the random number generator. Defaults to 0.
        test_size: The proportion of the dataset that is included in the test-split.
            Either represented by a percentage in the range [0.0, 1.0) or as absolute
            number of test samples in the range [1, inf). Defaults to 0.25.

    Returns:
        A tuple containing ``X_training``, ``y_training``, ``X_validation`` and
        ``y_validation`` of the iris dataset.
    """
    # Validate input
    n_features = check_int(n_features, name="n_features", l_bound=1)
    n_classes = check_int(n_classes, name="n_classes", l_bound=1)
    random_seed = check_int(random_seed, name="random_seed", l_bound=0)

    # Load data and take subset
    X, y = datasets.load_iris(return_X_y=True)
    X, y = _pre_process_data(X, y, n_features, n_classes)

    # Split into training and validation data sets
    if test_size == 0:
        return (
            X,
            y,
            np.empty(shape=(0, n_features), dtype=np.float64),
            np.empty(shape=(0, 1), dtype=np.int_),
        )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    return X_train, y_train, X_val, y_val


def get_moons_dataset(
    n_samples: int = 100,
    random_seed: int = 0,
    test_size: int | float | None = None,
) -> tuple[
    NDArray[np.float64], NDArray[np.int_], NDArray[np.float64], NDArray[np.int_]
]:
    r"""Generate a random dataset with a moon shape.

    This function wraps :func:`~sklearn.datasets.make_moons` of :mod:`sklearn.datasets`
    with a fixed noise factor of 0.3. Furthermore, the data is split into training and
    validation data, where 60% of the data is training and 40% is validation.

    Example usage:

        >>> from tno.quantum.ml.datasets import get_moons_dataset
        >>> X_train, y_train, X_val, y_val = get_moons_dataset()
        >>> print(f"{X_train.shape=}\n{y_train.shape=}\n{X_val.shape=}\n{y_val.shape=}")
        X_train.shape=(75, 2)
        y_train.shape=(75,)
        X_val.shape=(25, 2)
        y_val.shape=(25,)

    Args:
        n_samples: Number of samples
        random_seed: Seed to give to the random number generator. Defaults to 0.
        test_size: The proportion of the dataset that is included in the test-split.
            Either represented by a percentage in the range [0.0, 1.0) or as absolute
            number of test samples in the range [1, inf). Defaults to 0.25.

    Returns:
        A tuple containing ``X_training``, ``y_training``, ``X_validation`` and
        ``y_validation`` of a moon shaped dataset.
    """
    # Validate input
    n_samples = check_int(n_samples, name="n_samples", l_bound=1)
    random_seed = check_int(random_seed, name="random_seed", l_bound=0)

    X, y = datasets.make_moons(n_samples=n_samples, noise=0.3, random_state=random_seed)

    # Split into training and validation data sets
    if test_size == 0:
        return (
            X,
            y,
            np.empty(shape=(0, 2), dtype=np.float64),
            np.empty(shape=(0, 1), dtype=np.int_),
        )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    return X_train, y_train, X_val, y_val


def get_circles_dataset(
    n_samples: int = 100,
    random_seed: int = 0,
    test_size: int | float | None = None,
) -> tuple[
    NDArray[np.float64], NDArray[np.int_], NDArray[np.float64], NDArray[np.int_]
]:
    r"""Generate a random dataset with the shape of two circles.

    This function wraps :func:`~sklearn.datasets.make_circles` of
    :mod:`sklearn.datasets` with a fixed noise factor of `0.2` and factor of `0.5`.

    Example usage:

        >>> from tno.quantum.ml.datasets import get_circles_dataset
        >>> X_train, y_train, X_val, y_val = get_circles_dataset(n_samples=100, test_size=0.4)
        >>> print(f"{X_train.shape=}\n{y_train.shape=}\n{X_val.shape=}\n{y_val.shape=}")
        X_train.shape=(60, 2)
        y_train.shape=(60,)
        X_val.shape=(40, 2)
        y_val.shape=(40,)

    Args:
        n_samples: Total number of generated data samples.
        random_seed: Seed to give to the random number generator. Defaults to 0.
        test_size: The proportion of the dataset that is included in the test-split.
            Either represented by a percentage in the range [0.0, 1.0) or as absolute
            number of test samples in the range [1, inf). Defaults to 0.25.

    Returns:
        A tuple containing ``X_training``, ``y_training``, ``X_validation`` and
        ``y_validation`` of a dataset with two circles.
    """  # noqa: E501
    # Validate input
    n_samples = check_int(n_samples, name="n_samples", l_bound=1)
    random_seed = check_int(random_seed, name="random_seed", l_bound=0)

    X, y = datasets.make_circles(
        n_samples=n_samples, noise=0.2, factor=0.5, random_state=random_seed
    )

    # Split into training and validation data sets
    if test_size == 0:
        return (
            X,
            y,
            np.empty(shape=(0, 2), dtype=np.float64),
            np.empty(shape=(0, 1), dtype=np.int_),
        )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    return X_train, y_train, X_val, y_val


def get_linearly_separable_dataset(
    n_samples: int = 100,
    random_seed: int = 0,
    test_size: float | int | None = None,
) -> tuple[
    NDArray[np.float64], NDArray[np.int_], NDArray[np.float64], NDArray[np.int_]
]:
    r"""Generate a random dataset that is linearly separable.

    This function wraps :func:`~sklearn.datasets.make_classification` of
    :mod:`sklearn.datasets` with the following fixed arguments: `n_features=2`,
    `n_redundant=0`, `n_informative=2` and `n_clusters_per_class=1`. Afterwards,
    uniformly distributed noise is added.

    Example usage:

        >>> from tno.quantum.ml.datasets import get_linearly_separable_dataset
        >>> X_train, y_train, X_val, y_val = get_linearly_separable_dataset()
        >>> print(f"{X_train.shape=}\n{y_train.shape=}\n{X_val.shape=}\n{y_val.shape=}")
        X_train.shape=(75, 2)
        y_train.shape=(75,)
        X_val.shape=(25, 2)
        y_val.shape=(25,)

    Args:
        n_samples: Total number of generated data samples.
        random_seed: Seed to give to the random number generator. Defaults to 0.
        test_size: The proportion of the dataset that is included in the test-split.
            Either represented by a percentage in the range [0.0, 1.0) or as absolute
            number of test samples in the range [1, inf). Defaults to 0.25.

    Returns:
        A tuple containing ``X_training``, ``y_training``, ``X_validation`` and
        ``y_validation`` of a dataset that is linearly separable.
    """
    # Validate input
    n_samples = check_int(n_samples, name="n_samples", l_bound=1)
    random_seed = check_int(random_seed, name="random_seed", l_bound=0)

    X, y = datasets.make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_seed,
        n_clusters_per_class=1,
    )

    rng = np.random.default_rng(random_seed)
    X += 2 * rng.uniform(size=X.shape)

    # Split into training and validation data sets
    if test_size == 0:
        return (
            X,
            y,
            np.empty(shape=(0, 2), dtype=np.float64),
            np.empty(shape=(0, 1), dtype=np.int_),
        )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    return X_train, y_train, X_val, y_val


def get_blobs_clustering_dataset(
    n_samples: int, n_features: int, n_centers: int, random_seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Load a blobs clustering dataset.

    This function wraps :func:`~sklearn.datasets.make_blobs` of :mod:`sklearn.datasets`
    with a fixed cluster standard deviation of 0.1.

    Example usage:

        >>> from tno.quantum.ml.datasets import get_blobs_clustering_dataset
        >>> X, true_labels = get_blobs_clustering_dataset(100, 3, 2)
        >>> print(f"{X.shape=}\n{true_labels.shape=}")
        X.shape=(100, 3)
        true_labels.shape=(100,)

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        n_centers: Number of centers.
        random_seed: Seed to give to the random number generator. Defaults to `42`.

    Returns:
        A tuple containing ``X`` and ``true_labels`` of a blobs clustering dataset.
    """
    # Validate input
    n_samples = check_int(n_samples, name="n_classes", l_bound=1)
    n_features = check_int(n_features, name="n_features", l_bound=1)
    n_centers = check_int(n_centers, name="n_centers", l_bound=1)
    random_seed = check_int(random_seed, name="random_seed", l_bound=0)

    centers = np.array(
        [[i] + [(f + i) % 2 for f in range(n_features - 1)] for i in range(n_centers)]
    )

    X, true_labels = datasets.make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=0.1, random_state=random_seed
    )
    return X, true_labels


def get_anomalous_spiky_time_series_dataset(  # noqa: PLR0913
    n_samples: int = 100,
    n_features: int = 4,
    n_times: int = 200,
    anomaly_proportion: float = 0.5,
    random_seed: int = 42,
    test_size: float | int | None = None,
) -> tuple[
    NDArray[np.float64], NDArray[np.int_], NDArray[np.float64], NDArray[np.int_]
]:
    r"""Create a time series dataset.

    This uses normally distributed spikes centered around zero.

    This function generates non-anomalous time series' with spikes for each feature
    which come from distributions with random standard deviations between `0` and `0.5`.

    Anomalous time series' have alternating intervals of small spikes and large spikes.
    These intervals are of length `10` time units. The small spikes for each feature
    come from distributions with random standard deviations between `0` and `0.3` and
    the large spikes for each features come from distributions with random standard
    deviations between `0.8` and `1.6`. There is an `80%` chance of a small spike and a
    `20%` chance of a large spike at each time for each feature.

    Example usage:

        >>> from tno.quantum.ml.datasets import get_anomalous_spiky_time_series_dataset
        >>> X_train, y_train, X_val, y_val = get_anomalous_spiky_time_series_dataset(
        ...     n_samples=100, n_features=2, n_times=100, anomaly_proportion=0.5
        ... )
        >>> print(f"{X_train.shape=}\n{y_train.shape=}\n{X_val.shape=}\n{y_val.shape=}")
        X_train.shape=(75, 2, 100)
        y_train.shape=(75,)
        X_val.shape=(25, 2, 100)
        y_val.shape=(25,)

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_times: Number of evenly spaced times.
        anomaly_proportion: Percentage of the dataset that contains anomalies.
        random_seed: Seed to give to the random number generator. Defaults to `42`.
        test_size: The proportion of the dataset that is included in the test-split.
            Either represented by a percentage in the range [0.0, 1.0) or as absolute
            number of test samples in the range [1, inf). Defaults to 0.25.

    Returns:
        A tuple containing ``X_training``, ``y_training``, `X_validation`` and
        ``y_validation``.
    """
    rng = np.random.default_rng(random_seed)

    # Validate input
    n_samples = check_int(n_samples, "n_samples", l_bound=1)
    n_features = check_int(n_features, "n_features", l_bound=1)
    n_times = check_int(n_times, "n_times", l_bound=1)
    anomaly_proportion = check_real(
        anomaly_proportion, "anomaly_proportion", l_bound=0, u_bound=1
    )

    # Generate non-anomalous time series.

    n_samples_anomalous = int(anomaly_proportion * n_samples)
    n_samples_non_anomalous = n_samples - n_samples_anomalous

    # Define noise levels
    noise_levels_non_anomalous = rng.uniform(low=0, high=0.5, size=n_features)
    low_noise_levels = rng.uniform(low=0, high=0.3, size=n_features)
    high_noise_levels = rng.uniform(low=0.8, high=1.6, size=n_features)

    # Generate non-anomalous time series
    X_non_anomalous = np.zeros(shape=(n_samples_non_anomalous, n_features, n_times))
    y_non_anomalous = np.zeros(shape=(n_samples_non_anomalous), dtype=np.int_)

    for feature, noise_level in enumerate(noise_levels_non_anomalous):
        X_non_anomalous[:, feature, :] = rng.normal(
            0, noise_level, (n_samples_non_anomalous, n_times)
        )

    # Generate anomalous time series.
    X_anomalous = np.zeros(shape=(n_samples_anomalous, n_features, n_times))
    y_anomalous = np.ones(shape=(n_samples_anomalous), dtype=np.int_)
    probability_large_spike = 0.2

    spike_types = rng.binomial(n=1, p=probability_large_spike, size=ceil(n_times / 10))
    for sample in range(n_samples_anomalous):
        for feature in range(n_features):
            low_noise_level = low_noise_levels[feature]
            high_noise_level = high_noise_levels[feature]

            X_anomalous[sample, feature, :] = np.array(
                [
                    rng.normal(
                        0,
                        high_noise_level if spike_type else low_noise_level,
                        size=10,
                    )
                    for spike_type in spike_types
                ]
            ).flatten()[:n_times]

    X = np.concatenate((X_non_anomalous, X_anomalous), axis=0)
    y = np.concatenate((y_non_anomalous, y_anomalous), axis=0)

    # Split into training and validation data sets
    if test_size == 0:
        return (
            X,
            y,
            np.empty(shape=(0, n_features, n_times), dtype=np.float64),
            np.empty(shape=(0, 1), dtype=np.int_),
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    return X_train, y_train, X_val, y_val
