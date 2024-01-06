import numpy as np
from collections import Counter

def distance(x1,x2):
    return np.sqrt(np.sum((x1 - x2) **2)) #Euclidean method

class KNN:
    def __init__(self, k=3):
        self.k = k #Number of neighbours

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self,x):
        Distance = [distance(x,x_train) for x_train in self.X_train]
        k_idx = np.argsort(Distance)[:self.k]
        k_label = [self.y_train[i] for i in k_idx]
        most_common_label = Counter(k_label).most_common(1)
        return most_common_label[0][0]



