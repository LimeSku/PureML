import numpy as np

from pureml.supervised.tree_based.decision_tree_regressor import DecisionTreeRegressor


class GradientBoostingClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 5,
        max_features: int | str | None = "sqrt",
        subsample: float = 1.0,
        max_thresholds: int | None = 32,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
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
        progress_callback=None,
    ):
        X = np.asarray(X)
        y = np.asarray(y)

        rng = np.random.default_rng(self.random_state)

        self.classes_, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        Y = np.eye(n_classes)[y_indices]
        eps = 1e-12
        class_prior = np.mean(Y, axis=0)
        class_prior = np.clip(class_prior, eps, 1)

        self.init_score_ = np.log(class_prior)

        scores = np.tile(self.init_score_, (len(y), 1))
        self.trees = []
        self.training_loss_ = []

        if X_val is not None:
            val_indices = np.searchsorted(self.classes_, y_val)
            if not np.all(np.isin(y_val, self.classes_)):
                raise ValueError("y_val contains classes not seen during training")
            Y_val = np.eye(n_classes)[val_indices]
            val_scores = np.tile(self.init_score_, (len(y_val), 1))
            self.validation_loss_ = []
            self.best_validation_loss = float("inf")
            self.best_iteration_ = None
            rounds_without_improvement = 0
        for estimator_index in range(self.n_estimators):
            p = self._softmax(scores)
            residual = Y - p
            if self.subsample < 1.0:
                sample_size = max(1, int(len(y) * self.subsample))
                sample_indices = rng.choice(len(Y), size=sample_size, replace=False)
            else:
                sample_indices = np.arange(len(Y))
            class_trees = []
            max_features_ = self._resolve_max_features(n_features=X.shape[1])
            for k in range(n_classes):
                tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                tree_k = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_thresholds=self.max_thresholds,
                    max_features=max_features_,
                    random_state=tree_seed,
                )
                tree_k.fit(X[sample_indices], residual[sample_indices, k])
                class_trees.append(tree_k)
            for k, tree in enumerate(class_trees):
                scores[:, k] += self.learning_rate * tree.predict(X)

            self.trees.append(class_trees)
            proba = self._softmax(scores)
            self.training_loss_.append(self._cross_entropy(Y, proba))
            if progress_callback is not None:
                progress_callback(
                    estimator_index + 1,
                    self.n_estimators,
                    self.training_loss_[-1],
                )

            if X_val is not None:
                for k, tree in enumerate(class_trees):
                    val_scores[:, k] += self.learning_rate * tree.predict(X_val)
                val_proba = self._softmax(val_scores)
                val_loss = self._cross_entropy(Y_val, val_proba)
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
                        self.training_loss_ = self.training_loss_[
                            : self.best_iteration_
                        ]
                        self.validation_loss_ = self.validation_loss_[
                            : self.best_iteration_
                        ]
                        break
        self.feature_importances_ = self._compute_feature_importances(X.shape[1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        scores = np.tile(self.init_score_, (X.shape[0], 1))
        for class_trees in self.trees:
            for class_index, tree in enumerate(class_trees):
                scores[:, class_index] += self.learning_rate * tree.predict(X)
        return self._softmax(scores)

    def predict(self, X):
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]

    def _compute_feature_importances(self, n_features: int) -> np.ndarray:
        if not self.trees:
            return np.zeros(n_features)
        importances = np.array([
            tree.feature_importances(n_features)
            for class_trees in self.trees
            for tree in class_trees
        ])
        importances = np.mean(importances, axis=0)
        total = np.sum(importances)
        if total == 0:
            return importances
        return importances / total

    def _softmax(self, scores):
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _cross_entropy(self, Y, proba):
        eps = 1e-12
        proba = np.clip(proba, eps, 1 - eps)
        return -np.mean(np.sum(Y * np.log(proba), axis=1))

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
