from collections.abc import Callable

import numpy as np

from pureml.supervised.tree_based.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_features: int | str | None = "sqrt",
        bootstrap: bool = True,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []

    def fit(
        self,
        X,
        y,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        X = np.asarray(X)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.oob_votes_ = [[] for _ in range(n_samples)]
        tree_max_features = self._resolve_max_features(n_features)
        self.classes_ = np.unique(y)

        for i in range(self.n_estimators):
            if self.bootstrap:
                X_sample, y_sample, sampled_indices = self._bootstrap_sample(
                    X, y, rng=rng
                )

            else:
                X_sample, y_sample = X, y
                sampled_indices = np.arange(n_samples)
            tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=tree_max_features,
                random_state=tree_seed,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            if progress_callback is not None:
                progress_callback(i + 1, self.n_estimators)
            if self.bootstrap:
                all_indices = np.arange(n_samples)
                oob_indices = np.setdiff1d(all_indices, sampled_indices)
                if len(oob_indices) > 0:
                    oob_predictions = tree.predict(X[oob_indices])
                    for sampled_index, prediction in zip(oob_indices, oob_predictions):
                        self.oob_votes_[sampled_index].append(prediction)
        self.oob_score_ = self._compute_oob_score(y) if self.bootstrap else None
        self.feature_importances_ = self._compute_feature_importances(X.shape[1])

        return self

    def predict(self, X):
        if not self.trees:
            raise ValueError("Model MUST be fitted before calling predict")
        X = np.asarray(X)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = []
        for sample_index in range(X.shape[0]):
            sample_predictions = tree_predictions[:, sample_index]
            predictions.append(self._most_common_class(sample_predictions))
        return np.array(predictions)

    def predict_proba(self, X):
        if not self.trees:
            raise ValueError("Model must be fitted before calling predict_proba")
        X = np.asarray(X)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        proba = np.zeros((X.shape[0], len(self.classes_)))

        for sample_index in range(X.shape[0]):
            sample_predictions = tree_predictions[:, sample_index]
            for class_index, class_label in enumerate(self.classes_):
                proba[sample_index, class_index] = np.mean(
                    sample_predictions == class_label
                )
        return proba

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        n_samples = X.shape[0]
        indices = rng.integers(0, n_samples, size=n_samples)
        return X[indices], y[indices], indices

    def _compute_oob_score(self, y):
        oob_predictions = []
        oob_targets = []
        for sample_index, votes in enumerate(self.oob_votes_):
            if not votes:
                continue
            oob_predictions.append(self._most_common_class(np.array(votes)))
            oob_targets.append(y[sample_index])
        if not oob_predictions:
            return None
        return float(np.mean(np.array(oob_predictions) == np.array(oob_targets)))

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return None
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, int):
            if self.max_features <= 0:
                raise ValueError("max_features must be positive")
            return min(self.max_features, n_features)
        raise ValueError("max_features must be None, int, 'sqrt' or 'log2'")

    def _most_common_class(self, y: np.ndarray):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def _compute_feature_importances(self, n_features: int):
        if not self.trees:
            return np.zeros(n_features)
        importances = np.array([
            tree.feature_importances(n_features) for tree in self.trees
        ])
        forest_importances = np.mean(importances, axis=0)
        total = np.sum(forest_importances)
        if total == 0:
            return forest_importances
        return forest_importances / total
