"""Declarative catalogue of datasets and models the TUI can train.

Each model declares its editable hyperparameters (surfaced as inputs in the
sidebar) plus a factory that turns a dict of those values into a trainer. Adding
a model is a single ``MODEL_SPECS`` entry (and, for a new family, a
``MODEL_FAMILIES`` line) plus a trainer adapter in :mod:`pureml.tui.trainers`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pureml.datasets import load_iris, load_mnist
from pureml.model_selection.train_test_split import train_test_split
from pureml.tui.trainers import (
    Dataset,
    ForestTrainer,
    GradientBoostingTrainer,
    MlpTrainer,
    TreeTrainer,
)

# Default row cap for the sidebar "Max rows" control.
DEFAULT_MAX_ROWS = 1500


@dataclass
class HyperParam:
    """One editable hyperparameter: a trainer kwarg with a UI label + default."""

    key: str
    label: str
    default: float
    is_int: bool = True


@dataclass
class ModelSpec:
    params: list[HyperParam]
    # Builds a trainer from a dict of {param.key: value}.
    factory: Callable[[dict], object]


MODEL_SPECS: dict[str, ModelSpec] = {
    "Decision Tree": ModelSpec(
        params=[HyperParam("max_depth", "Max depth", 8)],
        factory=lambda v: TreeTrainer(max_depth=v["max_depth"]),
    ),
    "Random Forest": ModelSpec(
        params=[
            HyperParam("n_estimators", "Trees", 60),
            HyperParam("max_depth", "Max depth", 8),
        ],
        factory=lambda v: ForestTrainer(
            n_estimators=v["n_estimators"], max_depth=v["max_depth"]
        ),
    ),
    "Gradient Boosting": ModelSpec(
        params=[
            HyperParam("n_estimators", "Rounds", 30),
            HyperParam("learning_rate", "Learning rate", 0.1, is_int=False),
            HyperParam("max_depth", "Max depth", 3),
        ],
        factory=lambda v: GradientBoostingTrainer(
            n_estimators=v["n_estimators"],
            learning_rate=v["learning_rate"],
            max_depth=v["max_depth"],
        ),
    ),
    "MLP": ModelSpec(
        params=[
            HyperParam("epochs", "Epochs", 50),
            HyperParam("learning_rate", "Learning rate", 0.1, is_int=False),
            HyperParam("batch_size", "Batch size", 32),
        ],
        factory=lambda v: MlpTrainer(
            hidden_dims=[128, 64],
            epochs=v["epochs"],
            learning_rate=v["learning_rate"],
            batch_size=v["batch_size"],
            init_std=0.05,
        ),
    ),
}

# Models grouped by family for the sidebar tree (insertion order is display order).
MODEL_FAMILIES: dict[str, list[str]] = {
    "Tree-based": ["Decision Tree", "Random Forest", "Gradient Boosting"],
    "Neural Networks": ["MLP"],
}


def _load_iris_dataset(max_rows: int) -> Dataset:
    X, y, class_names = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, y_train = X_train[:max_rows], y_train[:max_rows]
    return Dataset("Iris", X_train, y_train, X_test, y_test, class_names)


def _load_mnist_dataset(max_rows: int) -> Dataset:
    # Scale the test split with the train cap so evaluation stays proportional.
    test_cap = min(max(1, max_rows // 3), 1000)
    X_train, y_train, X_test, y_test, class_names = load_mnist(
        max_train=max_rows, max_test=test_cap
    )
    return Dataset("MNIST", X_train, y_train, X_test, y_test, class_names)


DATASETS: dict[str, Callable[[int], Dataset]] = {
    "Iris": _load_iris_dataset,
    "MNIST": _load_mnist_dataset,
}


def model_params(model_name: str) -> list[HyperParam]:
    return MODEL_SPECS[model_name].params


def default_params(model_name: str) -> dict:
    return {p.key: p.default for p in MODEL_SPECS[model_name].params}


def load_dataset(name: str, max_rows: int = DEFAULT_MAX_ROWS) -> Dataset:
    return DATASETS[name](max_rows)


def make_trainer(model_name: str, values: dict):
    return MODEL_SPECS[model_name].factory(values)
