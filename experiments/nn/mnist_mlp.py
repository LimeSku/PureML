from gzip import open as gzip_open
from pathlib import Path
from struct import unpack

import numpy as np

from ml_models.neural_networks.mlp.classifier import MLPClassifier


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


def main() -> None:
    np.random.seed(0)

    dataset_dir = Path("datasets/mnist")
    X_train = load_mnist_images(dataset_dir / "train-images-idx3-ubyte.gz")
    y_train = load_mnist_labels(dataset_dir / "train-labels-idx1-ubyte.gz")
    X_test = load_mnist_images(dataset_dir / "t10k-images-idx3-ubyte.gz")
    y_test = load_mnist_labels(dataset_dir / "t10k-labels-idx1-ubyte.gz")

    train_size = 5_000
    test_size = 1_000
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]

    model = MLPClassifier(
        input_dim=784,
        hidden_dims=[128, 64],
        num_classes=10,
        init_std=0.05,
    )

    losses = model.fit(
        X=X_train,
        y=y_train,
        epochs=100,
        learning_rate=0.05,
        log_every=10,
        batch_size=64,
        X_val=X_test,  # just for test purpose, should not use test set
        y_val=y_test,
    )

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print()
    print("Dataset: MNIST")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Train accuracy: {accuracy(y_train, train_predictions):.3f}")
    print(f"Test accuracy: {accuracy(y_test, test_predictions):.3f}")


if __name__ == "__main__":
    main()
