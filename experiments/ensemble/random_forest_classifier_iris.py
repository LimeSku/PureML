from pathlib import Path
from time import perf_counter

import numpy as np

from pureml.ensemble.random_forest_classifier import RandomForestClassifier
from pureml.model_selection.train_test_split import train_test_split


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


def format_error_markers(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    markers = [
        "^" if predicted != actual else " " for predicted, actual in zip(y_pred, y_true)
    ]
    return "[" + " ".join(markers) + "]"


def print_aligned_results(y_pred: np.ndarray, y_true: np.ndarray) -> None:
    first_label = "Test predictions"
    second_label = "Test targets"
    error_label = "Errors"
    label_width = max(len(first_label), len(second_label))
    print(f"{first_label:<{label_width}}: {y_pred}")
    print(f"{second_label:<{label_width}}: {y_true}")
    print(f"{error_label:<{label_width}}: {format_error_markers(y_true, y_pred)}")


def main() -> None:
    X, y, class_names = load_iris(Path("datasets/iris/iris.data"))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_split=2,
        bootstrap=True,
        random_state=42,
    )
    start = perf_counter()
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    elapsed = perf_counter() - start

    print("Dataset: Iris")
    print(f"Classes: {class_names}")
    print(f"Trees: {model.n_estimators}")
    print(f"Bootstrap: {model.bootstrap}")
    print(f"Train accuracy: {accuracy(y_train, train_predictions):.3f}")
    print(f"Test accuracy: {accuracy(y_test, test_predictions):.3f}")
    print(f"Elapsed: {elapsed:.3f}s")
    print_aligned_results(test_predictions, y_test)
    print(model.feature_importances_)


if __name__ == "__main__":
    main()
