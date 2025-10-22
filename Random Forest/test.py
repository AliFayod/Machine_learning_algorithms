import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Random_Forest import RandomForest


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


# Load dataset
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Train Random Forest
print("Training Random Forest...")
rf = RandomForest(n_trees=20, max_depth=10)
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)

# Calculate accuracy
acc = accuracy(y_test, predictions)
print(f"Random Forest Accuracy: {acc:.4f}")
