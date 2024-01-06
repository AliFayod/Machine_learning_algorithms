import numpy as np


class Reg:
    def __init__(self, lr=0.001, n_iters=1000):
        """
        Base regression class.

        Parameters:
        lr (float): Learning rate for gradient descent.
        n_iters (int): Number of iterations for the optimization algorithm.
        """
        self.lr = lr
        self.n_iter = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fits the regression model to the training data.

        Parameters:
        X (numpy.ndarray): Training features.
        y (numpy.ndarray): Training labels.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iter):
            y_predict = self._approximation(X, self.weights, self.bias)
            dw = (1 / n_samples) * np.dot(X.T, (y_predict - y))
            db = (1 / n_samples) * np.sum(y_predict - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def _approximation(self, x, w, b):
        """
        Approximation function to be implemented in subclasses.
        """
        raise NotImplementedError


class LogisticRegression(Reg):
    def _approximation(self, x, w, b):
        """
        Applies logistic regression model to compute predictions.

        Parameters:
        x (numpy.ndarray): Input data.
        w (numpy.ndarray): Weights of the model.
        b (float): Bias of the model.

        Returns:
        numpy.ndarray: Predicted values after applying the sigmoid function.
        """
        linear_model = np.dot(x, w) + b
        return self._sigmoid(linear_model)

    def _predict(self, x):
        """
        Predicts class labels for the given input data.

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        list: Predicted class labels.
        """
        linear_model = np.dot(x, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        predicted_classes = [1 if i > 0.5 else 0 for i in y_predicted]
        return predicted_classes

    def _sigmoid(self, x):
        """
        Sigmoid function.

        Parameters:
        x (numpy.ndarray or float): Input value.

        Returns:
        numpy.ndarray or float: Sigmoid of x.
        """
        return 1 / (1 + np.exp(-x))


class LinearRegression(Reg):
    def _approximation(self, x, w, b):
        """
        Applies linear regression model to compute predictions.

        Parameters:
        x (numpy.ndarray): Input data.
        w (numpy.ndarray): Weights of the model.
        b (float): Bias of the model.

        Returns:
        numpy.ndarray: Predicted values.
        """
        return np.dot(x, w) + b

    def prediction(self, x):
        """
        Predicts output for the given input data using the trained model.

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Predicted values.
        """
        return np.dot(x, self.weights) + self.bias

# Example usage:
# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# predictions = lr._predict(X_test)
