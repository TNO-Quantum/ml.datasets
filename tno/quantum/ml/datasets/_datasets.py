"""This module contains datasets.

The datasets in this module can be used to test classifiers and clustering algorithms.
"""
from typing import Any, Tuple

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.model_selection import train_test_split

# pylint: disable=invalid-name


def _pre_process_data(
    X: NDArray[Any], y: NDArray[Any], n_features: int, n_classes: int
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """Slice `X` and `y` and cast the dtype of `y` to the dtype of `X`.

    Args:
        X: Feature matrix to slice.
        y: Target samples to slice and cast.
        n_features: Number of features.
        n_classes: Number of classes, must be 1, 2 or 3.

    Returns:
        ``Tuple`` of `X` and `y`, where `X`, and `y` are sliced to have the correct
        number of classes and number of features. Furthermore, the datatype of `y` is
        set to the datatype of `X`.
    """
    y = y.astype(X.dtype)
    ind = y < n_classes
    X = X[ind, :n_features]
    y = y[ind]
    return X, y


def get_wine_dataset(
    n_features: int = 13, n_classes: int = 3, random_seed: int = 0
) -> Tuple[NDArray[np.float_], NDArray[np.int_], NDArray[np.float_], NDArray[np.int_]]:
    # pylint: disable=line-too-long
    r"""Load the `wine <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine>`_ dataset.

    The dataset is loaded and split into training and validation data, with a ratio of 3
    to 1 (75% of the data is training and 25% is validation).

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
        n_classes: Nuber of classes, must be 1, 2 or 3. Defaults to 3.
        random_seed: Seed to give to the random number generator. Defaults to 0.

    Returns:
        A tuple containing ``X_training``, ``y_training``, ``X_validation`` and
        ``y_validation`` of the wine dataset.
    """
    # pylint: enable=line-too-long
    # Load data and take subset
    X, y = datasets.load_wine(return_X_y=True)
    X, y = _pre_process_data(X, y, n_features, n_classes)

    # Split into training and validation data sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=random_seed
    )

    return X_train, y_train, X_val, y_val


def get_iris_dataset(
    n_features: int = 4, n_classes: int = 3, random_seed: int = 0
) -> Tuple[NDArray[np.float_], NDArray[np.int_], NDArray[np.float_], NDArray[np.int_]]:
    # pylint: disable=line-too-long
    r"""Load the `iris <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris>`_ dataset.

    The dataset is loaded and split into training and validation data, with a ratio of 3
    to 1 (75% of the data is training and 25% is validation).

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
        n_classes: Nuber of classes, must be 1, 2 or 3. Defaults to 3.
        random_seed: Seed to give to the random number generator. Defaults to 0.

    Returns:
        A tuple containing ``X_training``, ``y_training``, ``X_validation`` and
        ``y_validation`` of the iris dataset.
    """
    # pylint: enable=line-too-long
    # Load data and take subset
    X, y = datasets.load_iris(return_X_y=True)
    X, y = _pre_process_data(X, y, n_features, n_classes)

    # Split into training and validation data sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=random_seed
    )

    return X_train, y_train, X_val, y_val


def get_moons_dataset(
    random_seed: int = 0,
) -> Tuple[NDArray[np.float_], NDArray[np.int_], NDArray[np.float_], NDArray[np.int_]]:
    # pylint: disable=line-too-long
    r"""Generate a random dataset with a moon shape.

    This function wraps the `make_moons <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons>`_
    method of ``sklearn.datasets`` with a fixed noise factor of 0.3. Furthermore, the
    data is split into training and validation data, where 60% of the data is training
    and 40% is validation.

    Example usage:

        >>> from tno.quantum.ml.datasets import get_moons_dataset
        >>> X_train, y_train, X_val, y_val = get_moons_dataset()
        >>> print(f"{X_train.shape=}\n{y_train.shape=}\n{X_val.shape=}\n{y_val.shape=}")
        X_train.shape=(60, 2)
        y_train.shape=(60,)
        X_val.shape=(40, 2)
        y_val.shape=(40,)

    Args:
        random_seed: Seed to give to the random number generator. Defaults to 0.

    Returns:
        A tuple containing ``X_training``, ``y_training``, ``X_validation`` and
        ``y_validation`` of a moon shaped dataset.
    """
    # pylint: enable=line-too-long
    X, y = datasets.make_moons(noise=0.3, random_state=random_seed)

    # Split into training and validation data sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.4, random_state=random_seed
    )

    return X_train, y_train, X_val, y_val


def get_circles_dataset(
    random_seed: int = 0,
) -> Tuple[NDArray[np.float_], NDArray[np.int_], NDArray[np.float_], NDArray[np.int_]]:
    # pylint: disable=line-too-long
    r"""Generate a random dataset with the shape of two circles.

    This function wraps the `make_circles <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles>`_
    method of ``sklearn.datasets`` with a fixed noise factor of 0.2 and factor of 0.5.
    Furthermore, the data is split into training and validation data, where 60% of the
    data is training and 40% is validation.

    Example usage:

        >>> from tno.quantum.ml.datasets import get_circles_dataset
        >>> X_train, y_train, X_val, y_val = get_circles_dataset()
        >>> print(f"{X_train.shape=}\n{y_train.shape=}\n{X_val.shape=}\n{y_val.shape=}")
        X_train.shape=(60, 2)
        y_train.shape=(60,)
        X_val.shape=(40, 2)
        y_val.shape=(40,)

    Args:
        random_seed: Seed to give to the random number generator. Defaults to 0.

    Returns:
        A tuple containing ``X_training``, ``y_training``, ``X_validation`` and
        ``y_validation`` of a dataset with two circles.
    """
    # pylint: enable=line-too-long
    X, y = datasets.make_circles(noise=0.2, factor=0.5, random_state=random_seed)

    # Split into training and validation data sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.4, random_state=random_seed
    )

    return X_train, y_train, X_val, y_val


def get_linearly_separables_dataset(
    random_seed: int = 0,
) -> Tuple[NDArray[np.float_], NDArray[np.int_], NDArray[np.float_], NDArray[np.int_]]:
    # pylint: disable=line-too-long
    r"""Generate a random dataset that is linearly separable.

    This function wraps the `make_classification <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification>`_
    method of ``sklearn.datasets`` with the following fixed arguments:
    `n_features=2`, `n_redundant=0`, `n_informative=2` and `n_clusters_per_class=1`.
    Afterwards, uniformly distributed noise is added. Lastly, the data is split into
    training and validation data, where 60% of the data is training and 40% is
    validation.

    Example usage:

        >>> from tno.quantum.ml.datasets import get_linearly_separables_dataset
        >>> X_train, y_train, X_val, y_val = get_linearly_separables_dataset()
        >>> print(f"{X_train.shape=}\n{y_train.shape=}\n{X_val.shape=}\n{y_val.shape=}")
        X_train.shape=(60, 2)
        y_train.shape=(60,)
        X_val.shape=(40, 2)
        y_val.shape=(40,)

    Args:
        random_seed: Seed to give to the random number generator. Defaults to 0.

    Returns:
        A tuple containing ``X_training``, ``y_training``, ``X_validation`` and
        ``y_validation`` of a dataset that is linearly separable.
    """
    # pylint: enable=line-too-long
    X, y = datasets.make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_seed,
        n_clusters_per_class=1,
    )

    rng = RandomState(random_seed)
    X += 2 * rng.uniform(size=X.shape)

    # Split into training and validation data sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.4, random_state=random_seed
    )

    return X_train, y_train, X_val, y_val


def get_blobs_clustering_dataset(
    n_samples: int, n_features: int, n_centers: int, random_seed: int = 42
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    # pylint: disable=line-too-long
    r"""Load a blobs clustering dataset.

    This function wraps the `make_blobs <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs>`_
    method of ``sklearn.datasets`` with a fixed cluster standard deviation of 0.1

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
        random_seed: Seed to give to the random number generator. Defaults to 42.

    Returns:
        A tuple containing ``X`` and ``true_labels`` of a blobs clustering dataset.
    """
    # pylint: enable=line-too-long
    centers = np.array(
        [[i] + [(f + i) % 2 for f in range(n_features - 1)] for i in range(n_centers)]
    )

    # pylint: disable-next=unbalanced-tuple-unpacking
    X, true_labels = datasets.make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=0.1, random_state=random_seed
    )
    return X, true_labels
