import numpy as np

from pureml.datasets import load_iris, load_mnist, standardize


def test_load_iris_shape_and_classes():
    X, y, class_names = load_iris()
    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert len(class_names) == 3


def test_standardize_uses_train_stats():
    X_train = np.array([[0.0], [2.0], [4.0]])
    X_test = np.array([[2.0]])
    scaled_train, scaled_test = standardize(X_train, X_test)
    assert np.isclose(scaled_train.mean(), 0.0)
    assert np.isclose(scaled_train.std(), 1.0)
    # test point equals the train mean, so it standardizes to 0.
    assert np.isclose(scaled_test[0, 0], 0.0)


def test_standardize_constant_feature_no_nan():
    X_train = np.array([[5.0], [5.0]])
    X_test = np.array([[5.0]])
    scaled_train, scaled_test = standardize(X_train, X_test)
    assert not np.isnan(scaled_train).any()
    assert not np.isnan(scaled_test).any()


def test_load_mnist_caps():
    X_train, y_train, X_test, y_test, class_names = load_mnist(
        max_train=100, max_test=50
    )
    assert X_train.shape == (100, 784)
    assert y_train.shape == (100,)
    assert X_test.shape == (50, 784)
    assert len(class_names) == 10
