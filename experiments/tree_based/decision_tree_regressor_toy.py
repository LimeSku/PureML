import numpy as np

from pureml.metrics.regression import mean_squared_error
from pureml.supervised.tree_based.decision_tree_regressor import (
    DecisionTreeRegressor,
)

# X = np.array([[0], [1], [2], [3], [4], [5]])
# y = np.array([0, 0, 1, 1, 2, 2])
X = np.array([[0], [1], [2], [3], [4], [5], [6], [7]])
y = np.array([0.1, 0.0, 1.2, 0.9, 2.1, 2.0, 2.8, 3.2])
model = DecisionTreeRegressor(max_depth=3)
model.fit(X, y)

preds = model.predict(X)

print("Predictions:", preds)
print("MSE:", mean_squared_error(y, preds))
