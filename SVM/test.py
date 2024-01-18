import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from SVM import SVM  # Assuming SVM.py contains the definition of the SVM class

# Generate a synthetic dataset
X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.50, random_state=40)
y = np.where(y == 0, -1, 1)  # Transform labels to -1 and 1 for SVM

# Initialize and train the SVM classifier
clf = SVM(lr=0.001)
clf.fit(X, y)

# Print out the learned weights and bias
print(clf.weight, clf.bias)

# Function to visualize the SVM decision boundaries
def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        """Calculate the hyperplane value for plotting."""
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)  # Plot data points

    # Determine the x-values for plotting the hyperplanes
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    # Calculate y-values for the decision boundary and margins
    x1_1 = get_hyperplane_value(x0_1, clf.weight, clf.bias, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.weight, clf.bias, 0)

    x1_1_m = get_hyperplane_value(x0_1, clf.weight, clf.bias, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.weight, clf.bias, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.weight, clf.bias, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.weight, clf.bias, 1)

    # Plot the decision boundary and margins
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")  # Decision boundary
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")  # Lower margin
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")  # Upper margin

    # Set the limits of y-axis for better visualization
    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()

# Call the function to visualize the SVM decision boundaries
visualize_svm()
