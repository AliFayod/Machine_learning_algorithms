import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from KNN import KNN


def train_and_evaluate_knn(X_train, y_train, X_test, y_test, k):
    """
    Trains the KNN classifier and evaluates its performance.

    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Testing data
    :param y_test: Testing labels
    :param k: Number of neighbors for KNN
    :return: Accuracy of the classifier
    """
    # Initialize and train KNN classifier
    clf = KNN(k=k)
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)

    # Calculate accuracy
    accuracy = np.sum(predictions == y_test) / len(y_test)
    return accuracy


def main():
    # Load Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Train the classifier and evaluate its performance
    k = 3
    accuracy = train_and_evaluate_knn(X_train, y_train, X_test, y_test, k)
    print(f"Accuracy for k={k}: {accuracy}")


if __name__ == "__main__":
    main()
