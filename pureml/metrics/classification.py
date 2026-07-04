import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that match the ground truth labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Counts of true (rows) vs predicted (columns) labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_index = {label: index for index, label in enumerate(labels)}

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[label_to_index[true_label], label_to_index[predicted_label]] += 1
    return matrix
