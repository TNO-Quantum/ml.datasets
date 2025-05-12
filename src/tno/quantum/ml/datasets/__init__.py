"""This package contains wrapper functions around :mod:`sklearn.datasets`.

The :mod:`~tno.quantum.ml.datasets` package only wraps some of the functionality of the
:mod:`sklearn.datasets`. This package is used for testing the :mod:`tno.quantum.ml`
classifiers and clustering algorithms in an easy, reproducible and consistent way.
"""

from tno.quantum.ml.datasets._datasets import (
    get_anomalous_spiky_time_series_dataset,
    get_blobs_clustering_dataset,
    get_circles_dataset,
    get_iris_dataset,
    get_linearly_separable_dataset,
    get_moons_dataset,
    get_wine_dataset,
)

__all__ = [
    "get_anomalous_spiky_time_series_dataset",
    "get_blobs_clustering_dataset",
    "get_circles_dataset",
    "get_iris_dataset",
    "get_linearly_separable_dataset",
    "get_moons_dataset",
    "get_wine_dataset",
]

__version__ = "2.0.1"
