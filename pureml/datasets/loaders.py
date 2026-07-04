"""Centralized dataset loaders.

The Iris and MNIST loading logic previously lived inline in several
``experiments/*.py`` scripts. It is gathered here so the TUI (and any future
consumer) can share a single implementation.
"""

from gzip import open as gzip_open
from pathlib import Path
from struct import unpack

import numpy as np

# Repo-root-relative default location of the bundled datasets.
DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"


def load_iris(
    path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the Iris dataset as ``(X, y, class_names)``.

    ``X`` is an ``(n_samples, 4)`` float array, ``y`` holds integer class ids,
    and ``class_names`` maps those ids back to their string labels.
    """
    if path is None:
        path = DATASETS_DIR / "iris" / "iris.data"

    raw_data = np.genfromtxt(path, delimiter=",", dtype=str)
    raw_data = raw_data[raw_data[:, -1] != ""]

    X = raw_data[:, :4].astype(float)
    labels = raw_data[:, 4]

    class_names = sorted(str(label) for label in set(labels))
    label_to_id = {label: index for index, label in enumerate(class_names)}
    y = np.array([label_to_id[label] for label in labels])

    return X, y, class_names


def _load_mnist_images(path: Path, max_images: int | None = None) -> np.ndarray:
    with gzip_open(path, "rb") as f:
        magic, n_images, n_rows, n_cols = unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        if max_images is not None:
            n_images = min(max_images, n_images)
        data = np.frombuffer(f.read(n_images * n_rows * n_cols), dtype=np.uint8)

    images = data.reshape(n_images, n_rows * n_cols)
    return images.astype(float) / 255.0


def _load_mnist_labels(path: Path, max_labels: int | None = None) -> np.ndarray:
    with gzip_open(path, "rb") as f:
        magic, n_labels = unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        if max_labels is not None:
            n_labels = min(max_labels, n_labels)
        labels = np.frombuffer(f.read(n_labels), dtype=np.uint8)

    if len(labels) != n_labels:
        raise ValueError(f"Expected {n_labels} labels, got {len(labels)}")
    return labels.astype(int)


def load_mnist(
    directory: Path | None = None,
    max_train: int | None = None,
    max_test: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load MNIST as ``(X_train, y_train, X_test, y_test, class_names)``.

    Images are flattened to ``(n, 784)`` floats in ``[0, 1]``. ``max_train`` and
    ``max_test`` cap the split sizes so a TUI run stays responsive.
    """
    if directory is None:
        directory = DATASETS_DIR / "mnist"

    X_train = _load_mnist_images(directory / "train-images-idx3-ubyte.gz", max_train)
    y_train = _load_mnist_labels(directory / "train-labels-idx1-ubyte.gz", max_train)
    X_test = _load_mnist_images(directory / "t10k-images-idx3-ubyte.gz", max_test)
    y_test = _load_mnist_labels(directory / "t10k-labels-idx1-ubyte.gz", max_test)

    class_names = [str(digit) for digit in range(10)]
    return X_train, y_train, X_test, y_test, class_names


def standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean / unit-variance scale using train statistics only.

    A ``std`` of 0 (constant feature) is treated as 1 to avoid dividing by zero.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std
