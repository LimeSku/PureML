import numpy as np

from pureml.metrics.regression import root_mean_squared_error
from pureml.supervised.tree_based.decision_tree_regressor import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 5,
        subsample: float = 1.0,
        max_thresholds: int | None = 32,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_thresholds = max_thresholds
        self.random_state = random_state

        self.trees = []

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        early_stopping_rounds: int | None = None,
    ):
        X = np.asarray(X)
        y = np.asarray(y)

        rng = np.random.default_rng(self.random_state)

        self.trees = []
        self.training_loss_ = []
        self.init_prediction_ = float(np.mean(y))
        current_prediction = np.full(len(y), self.init_prediction_)
        if X_val is not None:
            val_pred = np.full(len(y_val), self.init_prediction_)
            self.validation_loss_ = []
            self.best_validation_loss = float("inf")
            self.best_iteration_ = None
            rounds_without_improvement = 0
        for estimator_index in range(self.n_estimators):
            residual = y - current_prediction
            if self.subsample < 1.0:
                sample_size = max(1, int(len(y) * self.subsample))
                sample_indices = rng.choice(len(y), size=sample_size, replace=False)
            else:
                sample_indices = np.arange(len(y))
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_thresholds=self.max_thresholds,
            )
            tree.fit(X[sample_indices], residual[sample_indices])
            current_prediction += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            loss = float(root_mean_squared_error(y, current_prediction))
            self.training_loss_.append(loss)
            if X_val is not None:
                val_pred += self.learning_rate * tree.predict(X_val)
                val_loss = float(root_mean_squared_error(y_val, val_pred))
                self.validation_loss_.append(val_loss)
                if val_loss < self.best_validation_loss:
                    self.best_validation_loss = val_loss
                    self.best_iteration_ = estimator_index + 1
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                if early_stopping_rounds is not None:
                    if rounds_without_improvement >= early_stopping_rounds:
                        self.trees = self.trees[: self.best_iteration_]
                        break
        self.feature_importances_ = self._compute_feature_importances(X.shape[1])
        return self

    def predict(self, X):
        X = np.asarray(X)
        prediction = np.full(X.shape[0], self.init_prediction_)

        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)
        return prediction

    def staged_predict(self, X):
        X = np.asarray(X)
        prediction = np.full(X.shape[0], self.init_prediction_)

        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)
            yield prediction.copy()

    def _compute_feature_importances(self, n_features: int) -> np.ndarray:
        if not self.trees:
            return np.zeros(n_features)
        importances = np.array([
            tree.feature_importances(n_features) for tree in self.trees
        ])
        importances = np.mean(importances, axis=0)
        total = np.sum(importances)
        if total == 0:
            return importances
        return importances / total
