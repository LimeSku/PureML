import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from pureml.datasets import load_iris, load_mnist
from pureml.ensemble.gradient_boosting_classifier import GradientBoostingClassifier
from pureml.model_selection.train_test_split import train_test_split

IRIS_FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def format_error_markers(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    markers = [
        "^" if predicted != actual else " " for predicted, actual in zip(y_pred, y_true)
    ]
    return "[" + " ".join(markers) + "]"


def print_aligned_results(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    max_items: int = 40,
) -> None:
    y_pred = y_pred[:max_items]
    y_true = y_true[:max_items]
    first_label = "Test predictions"
    second_label = "Test targets"
    error_label = "Errors"
    label_width = max(len(first_label), len(second_label))
    print(f"{first_label:<{label_width}}: {y_pred}")
    print(f"{second_label:<{label_width}}: {y_true}")
    print(f"{error_label:<{label_width}}: {format_error_markers(y_true, y_pred)}")


def print_tabular_feature_importances(
    feature_names: list[str],
    importances: np.ndarray,
) -> None:
    print("Feature importances:")
    for feature, importance in sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"- {feature:<16} {importance:.4f}")


def print_mnist_feature_importance_summary(
    importances: np.ndarray,
    top_k: int = 10,
) -> None:
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
            print("".join(chars[int(value * (len(chars) - 1))] for value in row))

    top_indices = np.argsort(importances)[-top_k:][::-1]
    print(f"Top {top_k} pixels:")
    for index in top_indices:
        row, col = divmod(int(index), 28)
        print(f"- row={row:02d}, col={col:02d}, importance={importances[index]:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=["iris", "mnist"],
        nargs="?",
        default="iris",
    )
    parser.add_argument("--max-train", type=int, default=200)
    parser.add_argument("--max-test", type=int, default=100)
    return parser.parse_args()


def load_dataset(
    name: str,
    max_train: int,
    max_test: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    if name == "iris":
        X, y, class_names = load_iris(Path("datasets/iris/iris.data"))
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.25,
            random_state=7,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test, class_names

    X_train_full, y_train_full, X_test, y_test, class_names = load_mnist(
        Path("datasets/mnist"),
        max_train=max_train,
        max_test=max_test,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=7,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names


def make_model(dataset: str) -> GradientBoostingClassifier:
    if dataset == "mnist":
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=3,
            max_thresholds=8,
            random_state=42,
        )
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=5,
        random_state=42,
    )


def main() -> None:
    args = parse_args()
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dataset(
        args.dataset,
        args.max_train,
        args.max_test,
    )

    model = make_model(args.dataset)
    start = perf_counter()
    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=20,
    )

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    elapsed = perf_counter() - start

    print(f"Dataset: {args.dataset.upper()}")
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Trees: {model.n_estimators}")
    print(f"Best validation round: {model.best_iteration_}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Train accuracy: {accuracy(y_train, train_predictions):.3f}")
    print(f"Test accuracy: {accuracy(y_test, test_predictions):.3f}")
    print(f"Final train log loss: {model.training_loss_[-1]:.4f}")
    print(f"Final validation log loss: {model.validation_loss_[-1]:.4f}")
    print(f"Elapsed: {elapsed:.3f}s")
    print_aligned_results(test_predictions, y_test)
    if args.dataset == "iris":
        print_tabular_feature_importances(
            IRIS_FEATURE_NAMES, model.feature_importances_
        )
    else:
        print_mnist_feature_importance_summary(model.feature_importances_)


if __name__ == "__main__":
    main()
