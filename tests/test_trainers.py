"""Headless trainer tests — no Textual app required."""

import numpy as np
import pytest

from pureml.tui.registry import default_params, load_dataset, make_trainer
from pureml.tui.trainers import Progress


@pytest.fixture(scope="module")
def iris():
    return load_dataset("Iris")


def _make(name):
    return make_trainer(name, default_params(name))


def _run(trainer, dataset):
    events: list[Progress] = []
    result = trainer.run(dataset, events.append)
    return events, result


def test_decision_tree_trains_on_iris(iris):
    events, result = _run(_make("Decision Tree"), iris)
    assert events[-1].step == events[-1].total
    assert result.test_acc >= 0.8


def test_random_forest_emits_one_event_per_tree(iris):
    trainer = _make("Random Forest")
    events, result = _run(trainer, iris)
    assert len(events) == trainer.n_estimators
    # progress is monotonically increasing, one per tree.
    assert [event.step for event in events] == list(range(1, trainer.n_estimators + 1))
    assert result.oob_score is not None
    assert result.test_acc >= 0.8


def test_gradient_boosting_streams_loss(iris):
    trainer = _make("Gradient Boosting")
    events, result = _run(trainer, iris)
    assert len(events) == trainer.n_estimators
    assert all(event.loss is not None for event in events)
    assert len(result.loss_history) == trainer.n_estimators
    assert result.test_acc >= 0.8


def test_mlp_streams_loss_and_improves(iris):
    trainer = _make("MLP")
    events, result = _run(trainer, iris)
    assert len(events) == trainer.epochs
    assert all(event.loss is not None for event in events)
    assert len(result.loss_history) == trainer.epochs
    # training should reduce the loss from first to last epoch.
    assert result.loss_history[-1] < result.loss_history[0]
    assert result.test_acc >= 0.8


def test_confusion_matrix_is_square_over_classes(iris):
    _, result = _run(_make("Decision Tree"), iris)
    n = len(result.class_names)
    assert result.confusion.shape == (n, n)
    # total counts equal the number of test samples.
    assert int(np.sum(result.confusion)) == len(iris.y_test)
