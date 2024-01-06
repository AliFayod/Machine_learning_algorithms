import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pylab as plt
from Naive_Bayes import NaiveBayes

def accuracy(y_true, y_pred):
    """
    Calculates the accuracy of predictions against the true labels.

    Parameters:
    y_true (numpy.ndarray): True labels.
    y_pred (numpy.ndarray): Predicted labels.

    Returns:
    float: Accuracy of the predictions.
    """
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Generate a synthetic dataset
X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=1234)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Initialize the Naive Bayes classifier
NB = NaiveBayes()

# Train the classifier
NB.fit(X_train, y_train)

# Make predictions on the test set
predictions = NB.predict(X_test)

# Calculate and print the accuracy
print("Accuracy:", accuracy(y_test, predictions))
