import numpy as np

class RidgeRegression:
    """
    Ridge Regression implementation from scratch.

    Ridge Regression is a regularized version of Linear Regression that adds
    an L2 penalty term to the cost function. This helps prevent overfitting
    by constraining the model weights, making it particularly useful when
    dealing with multicollinearity or when the number of features is large.

    The Ridge cost function is:
    J(w) = MSE(w) + alpha * ||w||^2

    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength. Must be a positive float.
        Larger values specify stronger regularization.

    learning_rate : float, default=0.01
        Learning rate for gradient descent optimization

    n_iterations : int, default=1000
        Number of iterations for gradient descent

    Attributes:
    -----------
    weights : numpy.ndarray
        Model weights after fitting

    bias : float
        Model bias term after fitting

    Methods:
    --------
    fit(X, y)
        Train the Ridge Regression model using gradient descent

    predict(X)
        Make predictions on new data
    """

    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the Ridge Regression model.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features)
        y : numpy.ndarray
            Target values of shape (n_samples,)
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent optimization
        for _ in range(self.n_iterations):
            # Predictions
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate gradients with L2 regularization
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + (self.alpha / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

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
            Predicted values of shape (n_samples,)
        """
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

    def get_params(self):
        """
        Get the model parameters.

        Returns:
        --------
        dict
            Dictionary containing weights and bias
        """
        return {
            'weights': self.weights,
            'bias': self.bias,
            'alpha': self.alpha
        }


class RidgeRegressionClosed:
    """
    Ridge Regression using closed-form solution (Normal Equation).

    This implementation uses the analytical solution to Ridge Regression:
    w = (X^T * X + alpha * I)^-1 * X^T * y

    This is faster than gradient descent but may be computationally expensive
    for very large datasets.

    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength. Must be a positive float.

    Attributes:
    -----------
    weights : numpy.ndarray
        Model weights after fitting

    bias : float
        Model bias term after fitting

    Methods:
    --------
    fit(X, y)
        Train the Ridge Regression model using closed-form solution

    predict(X)
        Make predictions on new data
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the Ridge Regression model using closed-form solution.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features)
        y : numpy.ndarray
            Target values of shape (n_samples,)
        """
        n_samples, n_features = X.shape

        # Add bias column to X
        X_b = np.c_[np.ones((n_samples, 1)), X]

        # Create identity matrix (with 0 for bias term)
        I = np.eye(n_features + 1)
        I[0, 0] = 0  # Don't regularize the bias term

        # Closed-form solution: w = (X^T * X + alpha * I)^-1 * X^T * y
        weights_b = np.linalg.inv(X_b.T.dot(X_b) + self.alpha * I).dot(X_b.T).dot(y)

        self.bias = weights_b[0]
        self.weights = weights_b[1:]

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
            Predicted values of shape (n_samples,)
        """
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

    def get_params(self):
        """
        Get the model parameters.

        Returns:
        --------
        dict
            Dictionary containing weights and bias
        """
        return {
            'weights': self.weights,
            'bias': self.bias,
            'alpha': self.alpha
        }
