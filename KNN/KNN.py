import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        """
        Initializes the KNN classifier with a specified number of neighbors.

        :param k: Number of neighbors to consider (default is 3).
        """
        self.k = k

    def fit(self, X, y):
        """
        Fits the classifier with training data and labels.

        :param X: Training data.
        :param y: Training labels.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts the labels for given data.

        :param X: Data for which labels are to be predicted.
        :return: Predicted labels for the input data.
        """
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        """
        Helper method to predict the label for a single sample.

        :param x: A single data sample.
        :return: Predicted label for the sample.
        """
        # Compute distances between x and all samples in the training set.
        distances = [self._distance(x, x_train) for x_train in self.X_train]

        # Sort the distances and return indices of the first k neighbors.
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbors.
        k_labels = [self.y_train[i] for i in k_indices]

        # Determine the most common label among the nearest neighbors.
        most_common_label = Counter(k_labels).most_common(1)[0][0]
        return most_common_label

    def _distance(self, x1, x2):
        """
        Computes the Euclidean distance between two data points.

        :param x1: First data point.
        :param x2: Second data point.
        :return: Euclidean distance between x1 and x2.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

# Example usage:
# knn = KNN(k=3)
# knn.fit(X_train, y_train)
# predictions = knn.predict(X_test)
