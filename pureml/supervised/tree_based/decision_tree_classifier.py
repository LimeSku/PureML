"""
- classification only atm
- gini impurity only
- no missing values handling
- numeric features only
- max_depth, min_samples_split
- prediction = most common class in the leaf
"""

import numpy as np


class Node:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, value=None
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

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth=3, min_samples_split=2):
        # maximum depth allowed for the tree
        # higher depth => more complex trees but more prone to overfitting
        self.max_depth = max_depth
        # min number of samples needed to try a new split
        # if a node has fewer samples than this, it becomes a leaf
        self.min_samples_split = min_samples_split
        # root node of the tree, created at fit() time
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.root) for x in X])

    def _build_tree(self, X, y, depth):
        """
        Recursively build the tree.

        Internal nodes store a feature and threshold.
        Leaf nodes store the majority class of the samples that reached the leaf.
        """
        n_samples = X.shape[0]

        # stopping flags
        max_depth_reached = depth >= self.max_depth
        not_enough_samples = n_samples < self.min_samples_split
        # pure node: all labels are the same, so there is nothing left to split
        pure_node = np.all(y == y[0])

        if max_depth_reached or not_enough_samples or pure_node:
            # classification leaf prediction is the most frequent class
            return Node(value=self._most_common_class(y))

        best_split = self._find_best_split(X, y)

        # if no valid split, create a leaf
        if best_split is None:
            return Node(value=self._most_common_class(y))

        # boolean mask for left and right child nodes
        left_indices = X[:, best_split["feature_index"]] <= best_split["threshold"]
        right_indices = ~left_indices

        # recursively build left and right subtrees
        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left,
            right=right,
        )

    def _find_best_split(self, X, y):
        """
        A split is defined by:
        - feature index
        - threshold value for this feature

        Best split is the one minimizing weighted Gini impurity.
        """
        n_samples, n_features = X.shape
        best_gini = float("inf")
        best_split = None

        for feature_index in range(n_features):
            # candidate thresholds (for the split value) are the unique values of that feature
            # => simple and NOT optimized
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                # one side should not be empty: skip invalid splits
                if left_indices.sum() == 0 or right_indices.sum() == 0:
                    continue

                gini = self._weighted_gini(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                    }
        return best_split

    def _weighted_gini(self, y_left, y_right):
        """
        Compute the impurity of a split as the size-weighted Gini impurity
        of its left and right child nodes.
        """
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right
        gini_left = self._gini(y_left)
        gini_right = self._gini(y_right)
        wgini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
        return wgini

    def _gini(self, y):
        """
        Gini impurity measures how mixed the classes are inside one node.

        A pure node has Gini 0. Higher values mean classes are more mixed.
        """
        _, counts = np.unique(y, return_counts=True)
        gini = 1 - sum((counts / len(y)) ** 2)
        return gini

    def _predict_one(self, x, node: Node):
        """
        predict a single sample by traversing the tree.
        """
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def _most_common_class(self, y):
        """
        Return the majority class among samples that reached a leaf.
        """
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]
