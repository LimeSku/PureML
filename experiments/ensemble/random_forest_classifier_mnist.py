from gzip import open as gzip_open
from pathlib import Path
from struct import unpack
from time import perf_counter

import numpy as np

from pureml.ensemble.random_forest_classifier import RandomForestClassifier


def load_mnist_images(path: Path) -> np.ndarray:
    with gzip_open(path, "rb") as f:
        magic, n_images, n_rows, n_cols = unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")

        data = np.frombuffer(f.read(), dtype=np.uint8)

    images = data.reshape(n_images, n_rows * n_cols)
    return images.astype(float) / 255.0


def load_mnist_labels(path: Path) -> np.ndarray:
    with gzip_open(path, "rb") as f:
        magic, n_labels = unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    if len(labels) != n_labels:
        raise ValueError(f"Expected {n_labels} labels, got {len(labels)}")

    return labels.astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def print_feature_importance_summary(importances: np.ndarray, top_k: int = 10) -> None:
    image_importances = importances.reshape(28, 28)
    max_importance = np.max(image_importances)
    chars = " .:-=+*#%@"

    print("Feature importance heatmap:")
    if max_importance == 0:
        for _ in range(28):
            print(" " * 28)
    else:
        scaled = image_importances / max_importance
        for row in scaled:
            line = "".join(chars[int(value * (len(chars) - 1))] for value in row)
            print(line)

    top_indices = np.argsort(importances)[-top_k:][::-1]
    print(f"Top {top_k} pixels:")
    for index in top_indices:
        row, col = divmod(int(index), 28)
        print(f"- row={row:02d}, col={col:02d}, importance={importances[index]:.4f}")


def main() -> None:
    dataset_dir = Path("datasets/mnist")
    X_train = load_mnist_images(dataset_dir / "train-images-idx3-ubyte.gz")
    y_train = load_mnist_labels(dataset_dir / "train-labels-idx1-ubyte.gz")
    X_test = load_mnist_images(dataset_dir / "t10k-images-idx3-ubyte.gz")
    y_test = load_mnist_labels(dataset_dir / "t10k-labels-idx1-ubyte.gz")

    train_size = 200
    test_size = 100
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_split=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
    )
    start = perf_counter()
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    test_proba = model.predict_proba(X_test)

    elapsed = perf_counter() - start

    print("Dataset: MNIST")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Trees: {model.n_estimators}")
    print(f"Max depth: {model.max_depth}")
    print(f"OOB score: {model.oob_score_}")
    print(f"Train accuracy: {accuracy(y_train, train_predictions):.3f}")
    print(f"Test accuracy: {accuracy(y_test, test_predictions):.3f}")
    # print(f"Test probas: \n{test_proba}")
    print(f"Elapsed: {elapsed:.3f}s")
    print_feature_importance_summary(model.feature_importances_)


if __name__ == "__main__":
    main()
