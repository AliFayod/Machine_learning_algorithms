import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from AdaBoost import AdaBoost

def accuracy(y_true, y_pred):
    """Calculate accuracy score."""
    return np.sum(y_true == y_pred) / len(y_true)

# Load dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Convert labels to -1 and 1 for AdaBoost
y[y == 0] = -1

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train AdaBoost classifier
print("Training AdaBoost classifier...")
clf = AdaBoost(n_clf=10)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
acc = accuracy(y_test, y_pred)
print(f"AdaBoost Classification Accuracy: {acc:.4f}")

# Additional test with different number of classifiers
print("\nTesting with different numbers of weak classifiers:")
for n_clf in [5, 10, 20, 30]:
    clf = AdaBoost(n_clf=n_clf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(f"n_clf={n_clf:2d}: Accuracy = {acc:.4f}")
