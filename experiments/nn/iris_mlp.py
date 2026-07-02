from pathlib import Path

import numpy as np

from ml_models.model_selection.train_test_split import train_test_split
from ml_models.neural_networks.mlp.classifier import MLPClassifier


def load_iris(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw_data = np.genfromtxt(path, delimiter=",", dtype=str)
    raw_data = raw_data[raw_data[:, -1] != ""]

    X = raw_data[:, :4].astype(float)
    labels = raw_data[:, 4]

    class_names = sorted(str(label) for label in set(labels))
    label_to_id = {label: index for index, label in enumerate(class_names)}
    y = np.array([label_to_id[label] for label in labels])

    return X, y, class_names


def standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    return (X_train - mean) / std, (X_test - mean) / std


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def main() -> None:
    np.random.seed(0)

    dataset_path = Path("datasets/iris/iris.data")
    X, y, class_names = load_iris(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    X_train, X_test = standardize(X_train, X_test)

    model = MLPClassifier(
        input_dim=4,
        hidden_dim=16,
        num_classes=len(class_names),
        init_std=0.1,
    )

    losses = model.fit(
        X=X_train,
        y=y_train,
        epochs=1000,
        learning_rate=0.1,
        log_every=100,
        batch_size=8,
    )

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print()
    print("Dataset: Iris")
    print(f"Samples: {len(X)}")
    print(f"Classes: {class_names}")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Train accuracy: {accuracy(y_train, train_predictions):.3f}")
    print(f"Test accuracy: {accuracy(y_test, test_predictions):.3f}")


if __name__ == "__main__":
    main()
