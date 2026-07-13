"""Model training adapters.

Each trainer owns exactly one model's training + evaluation and reports progress
through a plain ``emit`` callback. There are **no Textual imports here** so the
trainers can be exercised in a headless unit test without a running app.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from pureml.datasets import standardize
from pureml.ensemble.random_forest_classifier import RandomForestClassifier
from pureml.metrics.classification import accuracy_score, confusion_matrix
from pureml.nn.mlp.classifier import MLPClassifier
from pureml.supervised.tree_based.decision_tree_classifier import DecisionTreeClassifier


@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]


@dataclass
class Progress:
    """A single live update emitted during training."""

    step: int
    total: int
    unit: str  # "epoch" or "tree"
    loss: float | None = None
    train_acc: float | None = None


@dataclass
class Result:
    """Final outcome shown once training finishes."""

    train_acc: float
    test_acc: float
    elapsed: float
    class_names: list[str]
    loss_history: list[float] = field(default_factory=list)
    oob_score: float | None = None
    confusion: np.ndarray | None = None


Emit = Callable[[Progress], None]


class TreeTrainer:
    """DecisionTreeClassifier — recursive build, so a single fit step."""

    unit = "step"

    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.total = 1

    def run(self, data: Dataset, emit: Emit) -> Result:
        start = time.perf_counter()
        emit(Progress(step=0, total=1, unit=self.unit))
        model = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
        model.fit(data.X_train, data.y_train)
        emit(Progress(step=1, total=1, unit=self.unit))
        return _evaluate(model, data, start)


class ForestTrainer:
    """RandomForestClassifier — streams progress tree-by-tree."""

    unit = "tree"

    def __init__(self, n_estimators: int, max_depth: int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.total = n_estimators

    def run(self, data: Dataset, emit: Emit) -> Result:
        start = time.perf_counter()
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
        )
        model.fit(
            data.X_train,
            data.y_train,
            progress_callback=lambda done, total: emit(
                Progress(step=done, total=total, unit=self.unit)
            ),
        )
        result = _evaluate(model, data, start)
        result.oob_score = model.oob_score_
        return result


class MlpTrainer:
    """MLPClassifier — one epoch per fit call, streaming loss + train accuracy."""

    unit = "epoch"

    def __init__(
        self,
        hidden_dims: list[int],
        epochs: int,
        learning_rate: float,
        batch_size: int,
        init_std: float,
    ):
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.init_std = init_std
        self.total = epochs

    def run(self, data: Dataset, emit: Emit) -> Result:
        start = time.perf_counter()
        # MLPs need standardized inputs; scale using train statistics only.
        X_train, X_test = standardize(data.X_train, data.X_test)
        scaled = Dataset(
            name=data.name,
            X_train=X_train,
            y_train=data.y_train,
            X_test=X_test,
            y_test=data.y_test,
            class_names=data.class_names,
        )

        model = MLPClassifier(
            input_dim=X_train.shape[1],
            hidden_dims=self.hidden_dims,
            num_classes=len(data.class_names),
            init_std=self.init_std,
        )

        loss_history: list[float] = []
        for epoch in range(1, self.epochs + 1):
            # fit does not re-init weights, so each call continues training.
            epoch_losses = model.fit(
                X=X_train,
                y=data.y_train,
                epochs=1,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
            )
            loss = float(epoch_losses[-1])
            loss_history.append(loss)
            train_acc = accuracy_score(data.y_train, model.predict(X_train))
            emit(
                Progress(
                    step=epoch,
                    total=self.epochs,
                    unit=self.unit,
                    loss=loss,
                    train_acc=train_acc,
                )
            )

        result = _evaluate(model, scaled, start)
        result.loss_history = loss_history
        return result


def _evaluate(model, data: Dataset, start: float) -> Result:
    train_pred = model.predict(data.X_train)
    test_pred = model.predict(data.X_test)
    labels = np.arange(len(data.class_names))
    return Result(
        train_acc=accuracy_score(data.y_train, train_pred),
        test_acc=accuracy_score(data.y_test, test_pred),
        elapsed=time.perf_counter() - start,
        class_names=data.class_names,
        confusion=confusion_matrix(data.y_test, test_pred, labels=labels),
    )
