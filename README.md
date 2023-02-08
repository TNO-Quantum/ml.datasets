# TNO Quantum: Datasets

TNO Quantum provides generic software components aimed at facilitating the development of quantum applications.

The ``tno.quantum.ml.datasets`` package wraps some of the functionality of the [sklearn.datasets](https://scikit-learn.org/stable/datasets.html).
This package is used for testing the ``tno.quantum.ml`` classifiers and clustering algorithms in an easy, reproducible and consistent way.

*Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws.*

## Documentation

Documentation of the `tno.quantum.ml.datasets` package can be found [here](https://tno-quantum.github.io/ml.datasets/).


## Install

Easily install the `tno.quantum.ml.datasets` package using pip:

```console
$ python -m pip install tno.quantum.ml.datasets
```

If you wish to run the tests you can use:
```console
$ python -m pip install 'tno.quantum.ml.datasets[tests]'
```

## Usage

Here's an example of how the ``datasets`` package can be used to load an iris dataset.

```python
from tno.quantum.ml.datasets import get_iris_dataset
X_train, y_train, X_val, y_val = get_iris_dataset()
```