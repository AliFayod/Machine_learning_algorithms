# Importing necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Decision_Tree import DecisionTree

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    float: Accuracy score.
    """
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Load the breast cancer dataset from sklearn datasets
data = datasets.load_breast_cancer()
X = data.data  # Feature matrix
y = data.target  # Target vector

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Initializing and training the Decision Tree classifier
clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)

# Predicting the labels on the test set
y_pred = clf.predict(X_test)

# Calculating the accuracy of the predictions
acc = accuracy(y_test, y_pred)

# Output the accuracy
print("Accuracy = ", acc)
