import argparse
from pathlib import Path

import numpy as np

from pureml.ensemble.xgboost_regressor import XGBoostRegressor
from pureml.metrics.regression import root_mean_squared_error
from pureml.model_selection.train_test_split import train_test_split

concrete_feature_names = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]


def make_toy_regression(
    n_samples: int = 160,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-3.0, 3.0, size=(n_samples, 1))
    noise = rng.normal(0.0, 0.15, size=n_samples)
    y = np.sin(2 * X[:, 0]) + 0.3 * X[:, 0] ** 2 + noise
    return X, y


def load_concrete(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)

    X = np.column_stack([data[name] for name in concrete_feature_names])
    y = data["compressive_strength"]
    return X, y


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "toy":
        return make_toy_regression()
    if name == "concrete":
        return load_concrete(Path("datasets/concrete/concrete.csv"))
    raise ValueError(f"Unknown dataset: {name}")


def print_score(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rmse = root_mean_squared_error(y_true, y_pred)
    relative_rmse = rmse / np.mean(np.abs(y_true))
    print(f"{name:<24} RMSE: {rmse:7.3f} | rel RMSE: {relative_rmse:6.1%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=["toy", "concrete"],
        nargs="?",
        default="toy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y = load_dataset(args.dataset)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=7,
    )

    print(f"Dataset: {args.dataset}")
    print(
        f"Train rows: {len(y_train)}, "
        f"validation rows: {len(y_val)}, "
        f"test rows: {len(y_test)}"
    )
    print()

    baseline_prediction = np.full(len(y_test), np.mean(y_train))
    print_score("Mean baseline", y_test, baseline_prediction)

    for n_estimators in [1, 200]:
        model = XGBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=5,
            min_samples_leaf=5,
            subsample=0.9,
        )
        model.fit(X_train, y_train)
        print_score(
            f"XGB-like {n_estimators:>2} trees",
            y_test,
            model.predict(X_test),
        )

    model = XGBoostRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        min_samples_leaf=5,
        subsample=0.9,
    )
    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=20,
    )
    best_iteration = model.best_iteration_
    best_test_prediction = list(model.staged_predict(X_test))[best_iteration - 1]
    print()
    print(f"Best validation round: {best_iteration} trees")
    print_score("XGB-like best stage", y_test, best_test_prediction)
    print_score(
        f"XGB-like stopped {len(model.trees):>3} trees",
        y_test,
        model.predict(X_test),
    )

    if args.dataset == "concrete":
        for feature, importance in zip(
            concrete_feature_names, model.feature_importances_
        ):
            print(f"{feature:20}\t{importance}")


if __name__ == "__main__":
    main()
