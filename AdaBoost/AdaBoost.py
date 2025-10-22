import numpy as np

class DecisionStump:
    """
    Simple decision stump used as weak classifier in AdaBoost.
    A decision stump is a one-level decision tree.
    """
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        """Make predictions using the decision stump."""
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions


class AdaBoost:
    """
    AdaBoost (Adaptive Boosting) classifier implementation from scratch.

    AdaBoost is an ensemble learning method that combines multiple weak classifiers
    to create a strong classifier. It adaptively adjusts the weights of training
    samples, giving more weight to misclassified samples in subsequent iterations.

    Parameters:
    -----------
    n_clf : int, default=5
        Number of weak classifiers (decision stumps) to train

    Attributes:
    -----------
    clfs : list
        List of trained decision stumps

    Methods:
    --------
    fit(X, y)
        Train the AdaBoost classifier
    predict(X)
        Make predictions on new data
    """

    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        """
        Train the AdaBoost classifier.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features)
        y : numpy.ndarray
            Target values of shape (n_samples,)
            Values should be -1 or 1
        """
        n_samples, n_features = X.shape

        # Initialize weights uniformly
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []

        # Train n_clf weak classifiers
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            # Find the best decision stump by trying all features and thresholds
            for feature_idx in range(n_features):
                X_column = X[:, feature_idx]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    # Try both polarities
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        if polarity == 1:
                            predictions[X_column < threshold] = -1
                        else:
                            predictions[X_column > threshold] = -1

                        # Calculate weighted error
                        misclassified = w[y != predictions]
                        error = sum(misclassified)

                        # Save the best configuration
                        if error < min_error:
                            min_error = error
                            clf.polarity = polarity
                            clf.threshold = threshold
                            clf.feature_idx = feature_idx

            # Calculate classifier weight (alpha)
            # Add small epsilon to avoid division by zero
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # Make predictions with current classifier
            predictions = clf.predict(X)

            # Update weights
            w *= np.exp(-clf.alpha * y * predictions)
            # Normalize weights
            w /= np.sum(w)

            # Save classifier
            self.clfs.append(clf)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        -----------
        X : numpy.ndarray
            Data to predict of shape (n_samples, n_features)

        Returns:
        --------
        y_pred : numpy.ndarray
            Predicted class labels (-1 or 1) of shape (n_samples,)
        """
        clf_preds = np.array([clf.alpha * clf.predict(X) for clf in self.clfs])
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred
