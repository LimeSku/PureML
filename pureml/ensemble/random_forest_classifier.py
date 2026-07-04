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

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        n_features = X.shape[1]
        tree_max_features = self._resolve_max_features(n_features)

        for i in range(self.n_estimators):
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y, rng=rng)
            else:
                X_sample, y_sample = X, y
            tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=tree_max_features,
                random_state=tree_seed,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = []
        for sample_index in range(X.shape[0]):
            sample_predictions = tree_predictions[:, sample_index]
            predictions.append(self._most_common_class(sample_predictions))
        return np.array(predictions)

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        n_samples = X.shape[0]
        indices = rng.integers(0, n_samples, size=n_samples)
        return X[indices], y[indices]

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

    def _majority_vote(self, predictions):
        pass

    def _average_predictions(self, predictions):
        pass

    def _most_common_class(self, y: np.ndarray):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]
