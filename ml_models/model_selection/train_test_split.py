import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle: bool = True):
    """
    Returns X train, X test, y train, y test
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
    n_test = int(n_samples * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
