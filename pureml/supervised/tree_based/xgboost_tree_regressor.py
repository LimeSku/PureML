import numpy as np


class XGBoostNode:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        gain=0.0,
    ):
        # index of the feature used for the split
        # feature_index = 2 => split using X[:, 2]
        self.feature_index = feature_index
        # threshold used for the current split: (X[:, feature_index] <= threshold) to left
        self.threshold = threshold

        # contains samples sent to left
        self.left = left
        # contains samples sent to right
        self.right = right

        # prediction value for a leaf node
        # not null only when node is a leaf
        self.value = value
        self.gain = gain

    def is_leaf(self):
        return self.value is not None


class XGBoostTreeRegressor:
    def __init__(
        self,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        min_child_weight=1.0,
        reg_lambda=1.0,
        gamma=0.0,
        max_features: int | None = None,
        random_state: int | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.max_features = max_features
        self.random_state = random_state
        self.rng = None
        self.root = None

    def fit(self, X, gradients, hessians):
        X = np.asarray(X)
        gradients = np.asarray(gradients)
        hessians = np.asarray(hessians)
        self.rng = np.random.default_rng(self.random_state)
        self.root = self._build_tree(X, gradients, hessians, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.root) for x in X])

    def feature_importances(self, n_features: int) -> np.ndarray:
        importances = np.zeros(n_features)
        self._collect_feature_importances(self.root, importances)
        total = np.sum(importances)
        if total == 0:
            return importances
        return importances / total

    def _collect_feature_importances(self, node: XGBoostNode, importances: np.ndarray):
        if node is None or node.is_leaf():
            return
        importances[node.feature_index] += node.gain
        self._collect_feature_importances(node.left, importances)
        self._collect_feature_importances(node.right, importances)

    def _build_tree(self, X, gradients, hessians, depth):
        n_samples = X.shape[0]

        ### stopping flags
        max_depth_reached = depth >= self.max_depth
        not_enough_samples = n_samples < self.min_samples_split
        min_weight = np.sum(hessians) < self.min_child_weight
        if max_depth_reached or not_enough_samples or min_weight:
            return XGBoostNode(value=self._leaf_value(gradients, hessians))

        best_split = self._find_best_split(X, gradients, hessians)

        # if no valid split, create a leaf
        if best_split is None:
            return XGBoostNode(value=self._leaf_value(gradients, hessians))

        # boolean mask for left and right child nodes
        left_indices = X[:, best_split["feature_index"]] <= best_split["threshold"]
        right_indices = ~left_indices

        # recursively build left and right subtrees
        left = self._build_tree(
            X[left_indices], gradients[left_indices], hessians[left_indices], depth + 1
        )
        right = self._build_tree(
            X[right_indices],
            gradients[right_indices],
            hessians[right_indices],
            depth + 1,
        )
        return XGBoostNode(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left,
            right=right,
            gain=best_split["gain"],
        )

    def _find_best_split(self, X, gradients, hessians):
        """
        Find the split with the largest regularized gain.
        """
        _, n_features = X.shape

        parent_G = np.sum(gradients)
        parent_H = np.sum(hessians)
        best_gain = 0.0
        best_split = None

        n_candidate_features = self._resolve_max_features(n_features)
        candidate_features = self.rng.choice(
            n_features,
            size=n_candidate_features,
            replace=False,
        )

        for feature_index in candidate_features:
            split = self._find_best_split_for_feature(
                X,
                gradients,
                hessians,
                feature_index,
                parent_G,
                parent_H,
            )
            if split is not None and split["gain"] > best_gain:
                best_gain = split["gain"]
                best_split = split
        return best_split

    def _find_best_split_for_feature(
        self, X, gradients, hessians, feature_index, parent_G, parent_H
    ):
        order = np.argsort(X[:, feature_index])
        feature_values = X[order, feature_index]
        sorted_gradients = gradients[order]
        sorted_hessians = hessians[order]

        G_left = 0.0
        H_left = 0.0
        best_gain = 0.0
        best_split = None
        n_samples = X.shape[0]
        for i in range(n_samples - 1):
            G_left += sorted_gradients[i]
            H_left += sorted_hessians[i]
            if feature_values[i] == feature_values[i + 1]:
                continue
            left_count = i + 1
            right_count = n_samples - left_count

            if left_count < self.min_samples_leaf:
                continue
            if right_count < self.min_samples_leaf:
                continue
            G_right = parent_G - G_left
            H_right = parent_H - H_left

            if H_left < self.min_child_weight:
                continue
            if H_right < self.min_child_weight:
                continue
            threshold = (feature_values[i] + feature_values[i + 1]) / 2
            gain = self._split_gain(
                G_left, H_left, G_right, H_right, parent_G, parent_H
            )
            if gain > best_gain:
                best_gain = gain
                best_split = {
                    "feature_index": feature_index,
                    "threshold": threshold,
                    "gain": gain,
                }
        return best_split

    def _predict_one(self, x, node: XGBoostNode):
        """
        predict a single sample by traversing the tree.
        """
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def _resolve_max_features(self, n_features: int):
        if self.max_features is None:
            return n_features
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")
        return min(self.max_features, n_features)

    def _score(self, G: float, H: float):
        return G**2 / (H + self.reg_lambda)

    def _split_gain(self, G_left, H_left, G_right, H_right, G_parent, H_parent):
        return (
            0.5
            * (
                self._score(G_left, H_left)
                + self._score(G_right, H_right)
                - self._score(G_parent, H_parent)
            )
            - self.gamma
        )

    def _leaf_value(self, gradients, hessians):
        G = np.sum(gradients)
        H = np.sum(hessians)
        return -G / (H + self.reg_lambda)
