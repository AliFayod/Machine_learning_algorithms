import numpy as np

class Perceptron:
    def __init__(self, lr=0.001, n_iter=1000):
        """
        Initialize the Perceptron classifier.

        Parameters:
        lr (float): Learning rate (between 0.0 and 1.0).
        n_iter (int): Number of passes over the training dataset.
        """
        self.lr = lr
        self.n_iter = n_iter
        self.activation_function = self.unit_step
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters:
        X (array-like): Training vectors, where n_samples is the number of samples and
                        n_features is the number of features.
        y (array-like): Target values.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights to zero
        self.bias = 0                        # Initialize bias to zero

        y_true = np.array([1 if i > 0 else 0 for i in y])  # Convert target values to binary (0 or 1)

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_function(linear_output)
                updates = self.lr * (y_true[idx] - y_pred)
                self.weights += updates * x_i
                self.bias += updates

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X (array-like): Test data.

        Returns:
        array-like: Predicted class labels.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)

    def unit_step(self, x):
        """
        Unit step activation function.

        Parameters:
        x (float): Value to be transformed by the unit step function.

        Returns:
        int: Transformed value (0 or 1).
        """
        return np.where(x >= 0, 1, 0)

# Example usage:
# perceptron = Perceptron(lr=0.01, n_iter=1000)
# perceptron.fit(X_train, y_train)
# predictions = perceptron.predict(X_test)
