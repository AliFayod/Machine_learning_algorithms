# Machine Learning Algorithms from Scratch

A comprehensive collection of machine learning algorithms implemented from scratch using Python and NumPy. This repository provides clean, well-documented implementations to help understand the mathematical foundations and inner workings of popular ML algorithms.

## Table of Contents

- [Overview](#overview)
- [Implemented Algorithms](#implemented-algorithms)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
  - [Dimensionality Reduction](#dimensionality-reduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains from-scratch implementations of fundamental machine learning algorithms. Each algorithm is built using only NumPy for numerical computations, avoiding high-level ML libraries to provide clear insight into how these algorithms work under the hood.

The implementations focus on:
- **Clarity**: Clean, readable code with extensive documentation
- **Educational value**: Understanding the mathematical foundations
- **Practical usage**: Working examples and test cases for each algorithm

## Implemented Algorithms

### Supervised Learning

#### 1. [K-Nearest Neighbors (KNN)](./KNN)
A simple, instance-based learning algorithm that classifies new data points based on the majority class of their k-nearest neighbors.

**Key Features:**
- Euclidean distance metric
- Configurable number of neighbors
- Non-parametric classifier

**Use Cases:** Classification tasks, pattern recognition, recommendation systems

---

#### 2. [Linear & Logistic Regression](./Logistic_and_Linear_Regression)
Fundamental regression algorithms implemented with gradient descent optimization.

**Linear Regression:**
- Predicts continuous values
- Uses gradient descent for optimization
- Configurable learning rate and iterations

**Logistic Regression:**
- Binary classification using sigmoid activation
- Gradient descent optimization
- Probabilistic predictions

**Use Cases:** Prediction, trend analysis, binary classification

---

#### 3. [Naive Bayes](./Naive%20Bayes)
A probabilistic classifier based on Bayes' theorem with strong independence assumptions.

**Key Features:**
- Gaussian Naive Bayes implementation
- Efficient for high-dimensional data
- Calculates class priors and likelihoods

**Use Cases:** Text classification, spam filtering, sentiment analysis

---

#### 4. [Perceptron](./Perceptron)
A simple linear classifier and the foundation of neural networks.

**Key Features:**
- Binary classification
- Unit step activation function
- Online learning with weight updates

**Use Cases:** Linearly separable classification problems, foundational neural network understanding

---

#### 5. [Support Vector Machine (SVM)](./SVM)
A powerful classifier that finds the optimal hyperplane to separate classes.

**Key Features:**
- Linear SVM with hinge loss
- L2 regularization
- Gradient descent optimization
- Maximum margin classifier

**Use Cases:** Binary classification, text categorization, image classification

---

#### 6. [Decision Tree](./Decision%20Tree)
A tree-based model that makes decisions by splitting data based on feature values.

**Key Features:**
- Information gain (entropy) based splitting
- Configurable maximum depth
- Handles both categorical and continuous features
- Recursive tree building

**Use Cases:** Classification, feature importance analysis, interpretable models

---

#### 7. [Random Forest](./Random%20Forest)
An ensemble learning method that combines multiple decision trees for improved accuracy and robustness.

**Key Features:**
- Bootstrap aggregating (bagging)
- Random feature selection
- Majority voting for predictions
- Reduces overfitting compared to single decision trees

**Use Cases:** Classification with high accuracy, feature importance ranking, handling complex datasets

---

### Unsupervised Learning

#### 8. [K-Means Clustering](./K-Means)
A clustering algorithm that partitions data into K distinct clusters based on feature similarity.

**Key Features:**
- Iterative centroid-based clustering
- Euclidean distance metric
- Convergence detection
- Optional visualization of clustering process

**Use Cases:** Customer segmentation, image compression, pattern discovery, anomaly detection

---

### Dimensionality Reduction

#### 9. [Principal Component Analysis (PCA)](./PCA)
A statistical technique for reducing data dimensionality while preserving maximum variance.

**Key Features:**
- Eigenvalue decomposition of covariance matrix
- Configurable number of components
- Explained variance ratio calculation
- Inverse transformation for reconstruction

**Use Cases:** Feature reduction, data visualization, noise filtering, compression

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AliFayod/Machine_learning_algorithms.git
cd Machine_learning_algorithms
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Each algorithm is contained in its own directory with a main implementation file and a test file demonstrating usage.

### Basic Example - K-Nearest Neighbors

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from KNN.KNN import KNN

# Load data
data = load_iris()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
knn = KNN(k=3)
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Calculate accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {accuracy:.2f}")
```

### Running Tests

Each algorithm includes a test file. To run tests:

```bash
# Example: Test KNN
cd KNN
python test.py

# Example: Test Random Forest
cd "Random Forest"
python test.py

# Example: Test K-Means
cd K-Means
python test.py
```

## Project Structure

```
Machine_learning_algorithms/
│
├── KNN/
│   ├── KNN.py              # K-Nearest Neighbors implementation
│   └── test.py             # Test and usage example
│
├── Logistic_and_Linear_Regression/
│   ├── Base_RG.py          # Linear & Logistic Regression
│   └── test.py             # Test and usage example
│
├── Naive Bayes/
│   ├── Naive_Bayes.py      # Naive Bayes implementation
│   └── test.py             # Test and usage example
│
├── Perceptron/
│   ├── Perceptron.py       # Perceptron implementation
│   └── test.py             # Test and usage example
│
├── SVM/
│   ├── SVM.py              # Support Vector Machine
│   └── test.py             # Test and usage example
│
├── Decision Tree/
│   ├── Decision_Tree.py    # Decision Tree implementation
│   └── test.py             # Test and usage example
│
├── Random Forest/
│   ├── Random_Forest.py    # Random Forest implementation
│   └── test.py             # Test and usage example
│
├── K-Means/
│   ├── K_Means.py          # K-Means Clustering
│   └── test.py             # Test and usage example
│
├── PCA/
│   ├── PCA.py              # Principal Component Analysis
│   └── test.py             # Test and usage example
│
└── README.md               # This file
```

## Testing

All implementations have been tested with standard datasets from scikit-learn:
- **Classification**: Iris, Breast Cancer, Wine datasets
- **Clustering**: Synthetic blob data
- **Regression**: Boston Housing, Diabetes datasets

Test files demonstrate:
- Data loading and preprocessing
- Model training
- Prediction
- Performance evaluation

## Requirements

- Python 3.7+
- NumPy >= 1.19.0
- scikit-learn >= 0.24.0 (for datasets and evaluation only)
- matplotlib >= 3.3.0 (for visualization in some tests)

Create a `requirements.txt` file:
```
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
```

## Algorithm Comparison

| Algorithm | Type | Training Complexity | Prediction Complexity | Parametric |
|-----------|------|--------------------|-----------------------|------------|
| KNN | Supervised | O(1) | O(n·d) | No |
| Linear Regression | Supervised | O(n·d·i) | O(d) | Yes |
| Logistic Regression | Supervised | O(n·d·i) | O(d) | Yes |
| Naive Bayes | Supervised | O(n·d) | O(c·d) | Yes |
| Perceptron | Supervised | O(n·d·i) | O(d) | Yes |
| SVM | Supervised | O(n·d·i) | O(d) | Yes |
| Decision Tree | Supervised | O(n·d·log n) | O(log n) | No |
| Random Forest | Supervised | O(t·n·d·log n) | O(t·log n) | No |
| K-Means | Unsupervised | O(k·n·d·i) | O(k·d) | Yes |
| PCA | Dim. Reduction | O(d²·n + d³) | O(d·k) | Yes |

*where n=samples, d=features, i=iterations, c=classes, k=clusters/components, t=trees*

## Contributing

Contributions are welcome! If you'd like to add new algorithms or improve existing implementations:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Commit your changes (`git commit -m 'Add new algorithm'`)
4. Push to the branch (`git push origin feature/new-algorithm`)
5. Open a Pull Request

Please ensure your code:
- Follows the existing code style
- Includes comprehensive docstrings
- Includes a test file with usage examples
- Is well-commented and easy to understand

## License

This project is created and maintained by [Ali Fayod](https://github.com/AliFayod).

## Acknowledgments

These implementations are educational in nature and inspired by various ML resources and textbooks. For production use, consider using optimized libraries like scikit-learn, TensorFlow, or PyTorch.

## Contact

For questions, suggestions, or issues, please open an issue on the [GitHub repository](https://github.com/AliFayod/Machine_learning_algorithms/issues).

---

**Note**: These implementations prioritize clarity and educational value over performance. For production applications, use established ML libraries that are optimized and battle-tested.
