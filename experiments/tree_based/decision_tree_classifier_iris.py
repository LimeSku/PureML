from pathlib import Path

import numpy as np

from pureml.model_selection.train_test_split import train_test_split
from pureml.supervised.tree_based.decision_tree_classifier import (
    DecisionTreeClassifier,
)


def load_iris(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw_data = np.genfromtxt(path, delimiter=",", dtype=str)
    raw_data = raw_data[raw_data[:, -1] != ""]

    X = raw_data[:, :4].astype(float)
    labels = raw_data[:, 4]

    class_names = sorted(str(label) for label in set(labels))
    label_to_id = {label: index for index, label in enumerate(class_names)}
    y = np.array([label_to_id[label] for label in labels])

    return X, y, class_names


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def main() -> None:
    X, y, class_names = load_iris(Path("datasets/iris/iris.data"))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print("Dataset: Iris")
    print(f"Classes: {class_names}")
    print(f"Train accuracy: {accuracy(y_train, train_predictions):.3f}")
    print(f"Test accuracy: {accuracy(y_test, test_predictions):.3f}")
    print("Test predictions:", test_predictions)
    print("Test targets:", y_test)


if __name__ == "__main__":
    main()
