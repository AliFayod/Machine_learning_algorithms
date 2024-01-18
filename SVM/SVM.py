import numpy as np

class SVM:
    def __init__(self, lr=0.001, _lambda=0.01, n_iter=1000):
        """
        Constructor for the SVM class.
        :param lr: Learning rate for the gradient descent.
        :param _lambda: Regularization parameter.
        :param n_iter: Number of iterations for training.
        """
        self.lr = lr
        self._lambda = _lambda
        self.n_iter = n_iter
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the SVM model to the training data.
        :param X: Feature matrix for training data.
        :param y: Labels for training data.
        """
        # Convert binary labels into -1 and 1 and get dataset dimensions.
        _y = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        # Initialize weights and bias.
        self.weight = np.zeros(n_features)
        self.bias = 0

        # Gradient descent optimization.
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                if _y[idx] * (np.dot(x_i, self.weight) - self.bias) >= 1:
                    self.weight -= self.lr * (2 * self._lambda * self.weight)
                else:
                    dw = 2 * self._lambda * self.weight - np.dot(_y[idx], x_i)
                    db = _y[idx]
                    self.weight -= self.lr * dw
                    self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict class labels for samples in X.
        :param X: Feature matrix for data to be predicted.
        :return: Predicted class labels.
        """
        linear_output = np.dot(X, self.weight) - self.bias
        return np.sign(linear_output)

# Example usage:
# svm = SVM()
# svm.fit(X_train, y_train)
# predictions = svm.predict(X_test)
