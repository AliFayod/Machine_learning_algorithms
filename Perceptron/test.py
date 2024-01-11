import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from Perceptron import Perceptron

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions against true labels.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    float: Accuracy as a proportion of correct predictions.
    """
    return np.sum(y_true == y_pred) / len(y_true)

# Generate synthetic dataset
X, y = datasets.make_blobs(n_samples=800, n_features=2, centers=2, cluster_std=1.05, random_state=2)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Initialize and train Perceptron
P = Perceptron(lr=0.01, n_iter=1000)
P.fit(X_train, y_train)

# Predict on test set and calculate accuracy
pred = P.predict(X_test)
print("Accuracy:", accuracy(y_test, pred))

# Visualize the training data and decision boundary
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

# Calculate decision boundary line
x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])
x1_1 = (-P.weights[0] * x0_1 - P.bias) / P.weights[1]
x1_2 = (-P.weights[0] * x0_2 - P.bias) / P.weights[1]
ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

# Set plot limits
ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin-3, ymax+3])

plt.show()
