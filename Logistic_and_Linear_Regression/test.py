from sklearn import datasets
from sklearn.model_selection import train_test_split
from LG import LogisticRegression
import numpy as np

# Load the breast cancer dataset
bc = datasets.load_breast_cancer()
X, Y = bc.data, bc.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# Initialize the Logistic Regression model
lg = LogisticRegression(lr=0.0001, n_iters=1000)

# Fit the model on the training data
lg.fit(x_train, y_train)

# Make predictions on the test data
y_predict = lg._predict(x_test)

# Calculate the accuracy
acc = np.sum(y_test == y_predict) / len(y_test)

print(f"Accuracy: {acc}")
