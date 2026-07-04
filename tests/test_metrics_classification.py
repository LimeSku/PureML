import numpy as np

from pureml.metrics.classification import accuracy_score, confusion_matrix


def test_accuracy_score_matches_mean():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    assert accuracy_score(y_true, y_pred) == float(np.mean(y_true == y_pred))
    assert accuracy_score(y_true, y_pred) == 0.8


def test_accuracy_score_empty():
    assert accuracy_score(np.array([]), np.array([])) == 0.0


def test_confusion_matrix_counts():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    matrix = confusion_matrix(y_true, y_pred, labels=np.array([0, 1]))
    assert matrix.tolist() == [[1, 1], [0, 2]]
