"""This package contains wrapper functions around ``sklearn.datasets``.

The ``tno.quantum.ml.datasets`` package only wraps some of the functionality of the
``sklearn.datasets``. This package is used for testing the ``tno.quantum.ml``
classifiers and clustering algorithms in an easy, reproducible and consistent way.
"""

# Explicit re-export of all functionalities, such that they can be imported properly.
# Following https://www.python.org/dev/peps/pep-0484/#stub-files and
# https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-no-implicit-reexport


from tno.quantum.ml.datasets._datasets import (
    get_blobs_clustering_dataset,
    get_circles_dataset,
    get_iris_dataset,
    get_linearly_separables_dataset,
    get_moons_dataset,
    get_wine_dataset,
)

__all__ = [
    "get_blobs_clustering_dataset",
    "get_circles_dataset",
    "get_iris_dataset",
    "get_linearly_separables_dataset",
    "get_moons_dataset",
    "get_wine_dataset",
]

__version__ = "1.2.1"
