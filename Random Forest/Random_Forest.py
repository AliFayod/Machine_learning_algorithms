import numpy as np
from collections import Counter


def entropy(y):
    """
    Calculate the entropy of a dataset.

    Parameters:
    y (array-like): The target values (labels).

    Returns:
    float: The entropy value.
    """
    n_classes = np.bincount(y)
    prop = n_classes / len(y)
    return -np.sum([p * np.log2(p) for p in prop if p > 0])


class Node:
    """
    A Node in the decision tree.

    Attributes:
    - feature (int): Index of the feature used for splitting.
    - threshold (float): Threshold value for splitting.
    - left (Node): Left child node.
    - right (Node): Right child node.
    - value (int): Class label for a leaf node.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Check if the node is a leaf node.

        Returns:
        bool: True if the node is a leaf node, False otherwise.
        """
        return self.value is not None


class DecisionTree:
    """
    Decision tree classifier for Random Forest.

    Attributes:
    - min_samples_split (int): The minimum number of samples required to split a node.
    - max_depth (int): The maximum depth of the tree.
    - n_feats (int): Number of features to consider for splitting.
    - root (Node): The root node of the tree.
    """

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        """
        Build the decision tree classifier.

        Parameters:
        X (array-like): Feature dataset.
        y (array-like): Target values.
        """
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the tree.

        Parameters:
        X (array-like): Feature dataset.
        y (array-like): Target values.
        depth (int): The current depth of the tree.

        Returns:
        Node: The root node of the subtree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping conditions for recursion
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # Find the best split
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        # Grow left and right children
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        """
        Find the best splitting criteria.

        Parameters:
        X (array-like): Feature dataset.
        y (array-like): Target values.
        feat_idxs (array-like): Indices of features to consider.

        Returns:
        tuple: The index of the best feature and the best threshold.
        """
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        """
        Calculate information gain.

        Parameters:
        y (array-like): Target values.
        X_column (array-like): Feature values.
        split_thresh (float): Threshold for splitting.

        Returns:
        float: The information gain.
        """
        parent_entropy = entropy(y)

        # Generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate weighted average child entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Information gain
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        """
        Split the dataset based on the split threshold.

        Parameters:
        X_column (array-like): Feature values.
        split_thresh (float): Threshold for splitting.

        Returns:
        tuple: Indices of samples in the left and right splits.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        """
        Find the most common label in the target array.

        Parameters:
        y (array-like): Target values.

        Returns:
        int: The most common label.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X (array-like): Feature dataset.

        Returns:
        array: Predicted class labels.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to predict the class label of a sample.

        Parameters:
        x (array-like): A single sample.
        node (Node): The current node.

        Returns:
        int: Predicted class label.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest:
    """
    Random Forest classifier - an ensemble of decision trees.

    Attributes:
    - n_trees (int): Number of trees in the forest.
    - min_samples_split (int): Minimum samples required to split a node.
    - max_depth (int): Maximum depth of each tree.
    - n_feats (int): Number of features to consider at each split.
    - trees (list): List of decision trees.
    """

    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        """
        Initialize the Random Forest classifier.

        Parameters:
        n_trees (int): Number of trees in the forest.
        min_samples_split (int): Minimum samples required to split a node.
        max_depth (int): Maximum depth of each tree.
        n_feats (int): Number of features to consider at each split.
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        """
        Build the Random Forest by training multiple decision trees.

        Parameters:
        X (array-like): Training features.
        y (array-like): Training labels.
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample of the dataset.

        Parameters:
        X (array-like): Feature dataset.
        y (array-like): Target values.

        Returns:
        tuple: Bootstrap sampled features and labels.
        """
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        """
        Predict class labels for samples in X using majority voting.

        Parameters:
        X (array-like): Feature dataset.

        Returns:
        array: Predicted class labels.
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Transpose to get predictions for each sample
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        # Use majority voting
        y_pred = [self._most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

    def _most_common_label(self, y):
        """
        Find the most common label using majority voting.

        Parameters:
        y (array-like): Array of predictions from different trees.

        Returns:
        int: The most common label.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


# Example usage:
# rf = RandomForest(n_trees=20, max_depth=10)
# rf.fit(X_train, y_train)
# predictions = rf.predict(X_test)
