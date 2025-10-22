import numpy as np

class NeuralNetwork:
    """
    Multi-Layer Perceptron (Neural Network) implementation from scratch.

    A simple feedforward neural network with one hidden layer, implemented using
    backpropagation for training. This implementation demonstrates the fundamental
    concepts of deep learning including forward propagation, loss calculation,
    and backpropagation.

    Parameters:
    -----------
    input_size : int
        Number of input features

    hidden_size : int, default=10
        Number of neurons in the hidden layer

    output_size : int, default=1
        Number of output neurons

    learning_rate : float, default=0.01
        Learning rate for gradient descent optimization

    n_iterations : int, default=1000
        Number of training iterations

    activation : str, default='relu'
        Activation function for hidden layer ('relu', 'sigmoid', or 'tanh')

    Attributes:
    -----------
    W1 : numpy.ndarray
        Weights for input to hidden layer

    b1 : numpy.ndarray
        Biases for hidden layer

    W2 : numpy.ndarray
        Weights for hidden to output layer

    b2 : numpy.ndarray
        Biases for output layer

    Methods:
    --------
    fit(X, y)
        Train the neural network

    predict(X)
        Make predictions on new data

    predict_proba(X)
        Get probability predictions (for binary classification)
    """

    def __init__(self, input_size, hidden_size=10, output_size=1,
                 learning_rate=0.01, n_iterations=1000, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation_name = activation

        # Initialize weights with small random values
        np.random.seed(42)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

        self.losses = []

    def _relu(self, Z):
        """ReLU activation function."""
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        """Derivative of ReLU."""
        return (Z > 0).astype(float)

    def _sigmoid(self, Z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

    def _sigmoid_derivative(self, Z):
        """Derivative of sigmoid."""
        s = self._sigmoid(Z)
        return s * (1 - s)

    def _tanh(self, Z):
        """Tanh activation function."""
        return np.tanh(Z)

    def _tanh_derivative(self, Z):
        """Derivative of tanh."""
        return 1 - np.tanh(Z) ** 2

    def _activate(self, Z):
        """Apply the selected activation function."""
        if self.activation_name == 'relu':
            return self._relu(Z)
        elif self.activation_name == 'sigmoid':
            return self._sigmoid(Z)
        elif self.activation_name == 'tanh':
            return self._tanh(Z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_name}")

    def _activate_derivative(self, Z):
        """Apply the derivative of the selected activation function."""
        if self.activation_name == 'relu':
            return self._relu_derivative(Z)
        elif self.activation_name == 'sigmoid':
            return self._sigmoid_derivative(Z)
        elif self.activation_name == 'tanh':
            return self._tanh_derivative(Z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_name}")

    def _forward_propagation(self, X):
        """
        Perform forward propagation.

        Returns:
        --------
        A2 : numpy.ndarray
            Output of the network
        cache : dict
            Dictionary containing Z1, A1, Z2, A2 for backpropagation
        """
        # Hidden layer
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._activate(Z1)

        # Output layer
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._sigmoid(Z2)  # Always use sigmoid for output

        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2, cache

    def _compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss."""
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def _backward_propagation(self, X, y, cache):
        """
        Perform backward propagation.

        Returns:
        --------
        gradients : dict
            Dictionary containing gradients for all parameters
        """
        m = X.shape[0]
        Z1, A1, Z2, A2 = cache['Z1'], cache['A1'], cache['Z2'], cache['A2']

        # Backpropagation
        dZ2 = A2 - y
        dW2 = (1 / m) * np.dot(A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._activate_derivative(Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return gradients

    def fit(self, X, y):
        """
        Train the neural network.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features)
        y : numpy.ndarray
            Target values of shape (n_samples,) or (n_samples, 1)
        """
        # Reshape y if necessary
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.losses = []

        # Training loop
        for i in range(self.n_iterations):
            # Forward propagation
            y_pred, cache = self._forward_propagation(X)

            # Compute loss
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)

            # Backward propagation
            gradients = self._backward_propagation(X, y, cache)

            # Update parameters
            self.W1 -= self.learning_rate * gradients['dW1']
            self.b1 -= self.learning_rate * gradients['db1']
            self.W2 -= self.learning_rate * gradients['dW2']
            self.b2 -= self.learning_rate * gradients['db2']

            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.n_iterations}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        """
        Get probability predictions.

        Parameters:
        -----------
        X : numpy.ndarray
            Data to predict of shape (n_samples, n_features)

        Returns:
        --------
        probabilities : numpy.ndarray
            Predicted probabilities of shape (n_samples, 1)
        """
        y_pred, _ = self._forward_propagation(X)
        return y_pred

    def predict(self, X):
        """
        Make binary predictions.

        Parameters:
        -----------
        X : numpy.ndarray
            Data to predict of shape (n_samples, n_features)

        Returns:
        --------
        predictions : numpy.ndarray
            Predicted class labels (0 or 1) of shape (n_samples,)
        """
        y_pred = self.predict_proba(X)
        return (y_pred > 0.5).astype(int).flatten()

    def get_params(self):
        """
        Get the model parameters.

        Returns:
        --------
        dict
            Dictionary containing all weights and biases
        """
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
