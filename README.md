# Machine Learning Algorithms from Scratch

A comprehensive collection of fundamental machine learning algorithms implemented from scratch using Python and NumPy. This repository demonstrates the mathematical foundations and inner workings of popular ML algorithms through clean, well-documented code.

## Table of Contents

- [Overview](#overview)
- [Implemented Algorithms](#implemented-algorithms)
  - [Supervised Learning](#supervised-learning)
  - [Ensemble Methods](#ensemble-methods)
  - [Unsupervised Learning](#unsupervised-learning)
  - [Dimensionality Reduction](#dimensionality-reduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Algorithm Comparison](#algorithm-comparison)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository provides educational implementations of fundamental machine learning algorithms built using only NumPy for numerical computations. By avoiding high-level ML libraries, these implementations offer clear insight into how algorithms work under the hood.

**Key Features:**
- **Educational Focus**: Clear, readable code with extensive documentation
- **Mathematical Foundation**: Understanding core concepts through implementation
- **Practical Examples**: Working test cases and real-world dataset applications
- **No Black Boxes**: See exactly how each algorithm makes decisions

## Implemented Algorithms

### Supervised Learning

#### 1. [K-Nearest Neighbors (KNN)](./KNN)
Instance-based learning algorithm that classifies data points based on the majority class of their k-nearest neighbors.

**Implementation Features:**
- Euclidean distance metric
- Configurable number of neighbors
- Non-parametric approach
- Lazy learning (no explicit training phase)

**Complexity**: O(1) training, O(n·d) prediction

**Use Cases**: Image recognition, recommender systems, pattern classification

---

#### 2. [Linear Regression](./Logistic_and_Linear_Regression)
Predicts continuous values by fitting a linear model to the data using gradient descent optimization.

**Implementation Features:**
- Gradient descent optimization
- Configurable learning rate and iterations
- Mean squared error loss function
- Closed-form solution available

**Complexity**: O(n·d·i) training, O(d) prediction

**Use Cases**: Price prediction, trend analysis, forecasting

---

#### 3. [Logistic Regression](./Logistic_and_Linear_Regression)
Binary classification algorithm using the sigmoid function for probabilistic predictions.

**Implementation Features:**
- Sigmoid activation function
- Binary cross-entropy loss
- Gradient descent optimization
- Probability outputs for classification confidence

**Complexity**: O(n·d·i) training, O(d) prediction

**Use Cases**: Spam detection, medical diagnosis, customer churn prediction

---

#### 4. [Ridge Regression](./Ridge%20Regression)
Regularized linear regression with L2 penalty to prevent overfitting and handle multicollinearity.

**Implementation Features:**
- L2 regularization (penalty on weight magnitudes)
- Both gradient descent and closed-form solutions
- Configurable regularization strength (alpha)
- Improved generalization over standard linear regression

**Complexity**: O(n·d·i) training (gradient descent), O(d³) (closed-form)

**Use Cases**: High-dimensional data, multicollinear features, overfitting prevention

---

#### 5. [Naive Bayes](./Naive%20Bayes)
Probabilistic classifier based on Bayes' theorem with strong independence assumptions between features.

**Implementation Features:**
- Gaussian Naive Bayes for continuous features
- Efficient training and prediction
- Probability-based classification
- Handles high-dimensional data well

**Complexity**: O(n·d) training, O(c·d) prediction

**Use Cases**: Text classification, spam filtering, sentiment analysis, document categorization

---

#### 6. [Perceptron](./Perceptron)
The simplest form of a neural network - a linear binary classifier that forms the foundation of deep learning.

**Implementation Features:**
- Binary linear classification
- Unit step activation function
- Online learning with iterative weight updates
- Historical significance in AI development

**Complexity**: O(n·d·i) training, O(d) prediction

**Use Cases**: Linearly separable problems, learning algorithm fundamentals, simple binary classification

---

#### 7. [Support Vector Machine (SVM)](./SVM)
Finds the optimal hyperplane that maximizes the margin between classes for robust classification.

**Implementation Features:**
- Linear SVM with hinge loss
- L2 regularization for margin maximization
- Gradient descent optimization
- Maximum margin principle

**Complexity**: O(n·d·i) training, O(d) prediction

**Use Cases**: Text categorization, image classification, bioinformatics, face detection

---

#### 8. [Neural Network (MLP)](./Neural%20Network)
Multi-layer perceptron with one hidden layer demonstrating fundamental deep learning concepts.

**Implementation Features:**
- Feedforward architecture with backpropagation
- Multiple activation functions (ReLU, sigmoid, tanh)
- Configurable hidden layer size
- Binary cross-entropy loss
- Gradient descent optimization

**Complexity**: O(n·d·h·i) training, O(d·h) prediction

**Use Cases**: Pattern recognition, complex non-linear relationships, feature learning, image classification

---

#### 9. [Decision Tree](./Decision%20Tree)
Tree-based model that recursively splits data based on feature values to make decisions.

**Implementation Features:**
- Information gain (entropy-based) splitting criterion
- Configurable maximum depth to control overfitting
- Handles both categorical and continuous features
- Recursive tree building algorithm
- Interpretable decision rules

**Complexity**: O(n·d·log n) training, O(log n) prediction

**Use Cases**: Credit scoring, medical diagnosis, feature importance analysis, rule extraction

---

### Ensemble Methods

#### 10. [Random Forest](./Random%20Forest)
Ensemble of decision trees using bootstrap aggregating and random feature selection for improved accuracy.

**Implementation Features:**
- Bootstrap aggregating (bagging)
- Random feature subset selection at each split
- Majority voting for final predictions
- Reduced overfitting compared to single trees
- Implicit feature importance ranking

**Complexity**: O(t·n·d·log n) training, O(t·log n) prediction

**Use Cases**: High-accuracy classification, feature ranking, robust predictions, handling imbalanced data

---

#### 11. [AdaBoost](./AdaBoost)
Adaptive boosting algorithm that combines weak classifiers (decision stumps) into a strong classifier.

**Implementation Features:**
- Sequential training of weak learners
- Adaptive sample weighting
- Focus on misclassified examples
- Weighted voting for predictions
- Automatic feature selection

**Complexity**: O(n·d·t) training, O(t·d) prediction

**Use Cases**: Face detection, text classification, improving weak classifiers, feature selection

---

### Unsupervised Learning

#### 12. [K-Means Clustering](./K-Means)
Partitions data into K distinct clusters based on feature similarity using iterative centroid refinement.

**Implementation Features:**
- Iterative centroid-based clustering
- Euclidean distance metric
- Convergence detection
- Optional visualization of clustering process
- Random initialization strategies

**Complexity**: O(k·n·d·i) training, O(k·d) prediction

**Use Cases**: Customer segmentation, image compression, document clustering, anomaly detection

---

### Dimensionality Reduction

#### 13. [Principal Component Analysis (PCA)](./PCA)
Reduces data dimensionality by projecting onto principal components that capture maximum variance.

**Implementation Features:**
- Eigenvalue decomposition of covariance matrix
- Configurable number of components
- Explained variance ratio calculation
- Inverse transformation for reconstruction
- Data standardization for better results

**Complexity**: O(d²·n + d³) training, O(d·k) prediction

**Use Cases**: Feature reduction, data visualization, noise filtering, compression, exploratory data analysis

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/AliFayod/Machine_learning_algorithms.git
cd Machine_learning_algorithms
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy scikit-learn matplotlib
```

## Quick Start

Here's a simple example using K-Nearest Neighbors:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from KNN.KNN import KNN

# Load dataset
X, y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = KNN(k=3)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {accuracy:.2%}")
```

## Usage Examples

### Classification: Neural Network

```python
from Neural_Network.Neural_Network import NeuralNetwork
from sklearn.preprocessing import StandardScaler

# Prepare data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train model
nn = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden_size=20,
    output_size=1,
    learning_rate=0.1,
    n_iterations=1000,
    activation='relu'
)
nn.fit(X_train, y_train)

# Predict
predictions = nn.predict(X_test)
probabilities = nn.predict_proba(X_test)
```

### Regression: Ridge Regression

```python
from Ridge_Regression.Ridge_Regression import RidgeRegression
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model with regularization
model = RidgeRegression(alpha=1.0, learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

### Ensemble: AdaBoost

```python
from AdaBoost.AdaBoost import AdaBoost

# Ensure labels are -1 and 1
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1

# Train ensemble
model = AdaBoost(n_clf=10)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

### Clustering: K-Means

```python
from K_Means.K_Means import KMeans

# Cluster data
kmeans = KMeans(K=3, max_iters=100)
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.predict(X)
centroids = kmeans.centroids
```

### Running Tests

Each algorithm includes a comprehensive test file:

```bash
# Test Neural Network
cd "Neural Network"
python test.py

# Test AdaBoost
cd AdaBoost
python test.py

# Test Ridge Regression
cd "Ridge Regression"
python test.py

# Test any algorithm
cd <algorithm_directory>
python test.py
```

## Project Structure

```
Machine_learning_algorithms/
│
├── KNN/
│   ├── KNN.py                          # K-Nearest Neighbors
│   └── test.py
│
├── Logistic_and_Linear_Regression/
│   ├── Base_RG.py                      # Linear & Logistic Regression
│   └── test.py
│
├── Ridge Regression/
│   ├── Ridge_Regression.py             # Ridge Regression (L2)
│   └── test.py
│
├── Naive Bayes/
│   ├── Naive_Bayes.py                  # Gaussian Naive Bayes
│   └── test.py
│
├── Perceptron/
│   ├── Perceptron.py                   # Perceptron Classifier
│   └── test.py
│
├── SVM/
│   ├── SVM.py                          # Support Vector Machine
│   └── test.py
│
├── Neural Network/
│   ├── Neural_Network.py               # Multi-Layer Perceptron
│   └── test.py
│
├── Decision Tree/
│   ├── Decision_Tree.py                # Decision Tree Classifier
│   └── test.py
│
├── Random Forest/
│   ├── Random_Forest.py                # Random Forest Ensemble
│   └── test.py
│
├── AdaBoost/
│   ├── AdaBoost.py                     # AdaBoost Ensemble
│   └── test.py
│
├── K-Means/
│   ├── K_Means.py                      # K-Means Clustering
│   └── test.py
│
├── PCA/
│   ├── PCA.py                          # Principal Component Analysis
│   └── test.py
│
└── README.md
```

## Algorithm Comparison

### By Type

| Algorithm | Type | Training | Prediction | Parametric | Handles Non-linear |
|-----------|------|----------|------------|------------|-------------------|
| KNN | Supervised | O(1) | O(n·d) | No | Yes |
| Linear Regression | Supervised | O(n·d·i) | O(d) | Yes | No |
| Logistic Regression | Supervised | O(n·d·i) | O(d) | Yes | No |
| Ridge Regression | Supervised | O(n·d·i) | O(d) | Yes | No |
| Naive Bayes | Supervised | O(n·d) | O(c·d) | Yes | No |
| Perceptron | Supervised | O(n·d·i) | O(d) | Yes | No |
| SVM | Supervised | O(n·d·i) | O(d) | Yes | No* |
| Neural Network | Supervised | O(n·d·h·i) | O(d·h) | Yes | Yes |
| Decision Tree | Supervised | O(n·d·log n) | O(log n) | No | Yes |
| Random Forest | Ensemble | O(t·n·d·log n) | O(t·log n) | No | Yes |
| AdaBoost | Ensemble | O(n·d·t) | O(t·d) | No | Yes |
| K-Means | Unsupervised | O(k·n·d·i) | O(k·d) | Yes | No |
| PCA | Dim. Reduction | O(d²·n + d³) | O(d·k) | Yes | No |

*Linear SVM; kernel trick enables non-linearity*

**Legend:**
- n = number of samples
- d = number of features
- i = number of iterations
- c = number of classes
- k = number of clusters/components
- t = number of trees/weak learners
- h = hidden layer size

### When to Use Each Algorithm

**For High Accuracy:**
- Random Forest, AdaBoost, Neural Networks

**For Interpretability:**
- Decision Tree, Linear Regression, Logistic Regression

**For Speed:**
- Naive Bayes, Perceptron, KNN (training)

**For Small Datasets:**
- KNN, Naive Bayes, Logistic Regression

**For Large Datasets:**
- Neural Networks, Ridge Regression, Linear Models

**For Non-linear Relationships:**
- Neural Networks, Decision Trees, Random Forest, KNN

**For Feature Selection:**
- Ridge Regression (regularization), Random Forest (importance), AdaBoost

## Requirements

- **Python**: 3.7 or higher
- **NumPy**: >= 1.19.0 (core numerical computations)
- **scikit-learn**: >= 0.24.0 (datasets and evaluation metrics only)
- **matplotlib**: >= 3.3.0 (visualization in test files)

### Installing Requirements

Create a `requirements.txt` file:
```
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
```

Then install:
```bash
pip install -r requirements.txt
```

## Testing

All implementations have been validated using standard datasets from scikit-learn:

**Classification Datasets:**
- Iris (multi-class)
- Breast Cancer (binary)
- Wine (multi-class)

**Regression Datasets:**
- Diabetes
- California Housing

**Clustering:**
- Synthetic blob data
- Iris (for validation)

Each test file demonstrates:
1. Data loading and preprocessing
2. Model initialization and training
3. Prediction on test data
4. Performance evaluation
5. Comparison of hyperparameters

## Contributing

Contributions are welcome! Here's how you can help:

### Adding New Algorithms

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Implement the algorithm following the existing structure:
   - Create a new directory: `Algorithm_Name/`
   - Add implementation: `Algorithm_Name.py`
   - Add test file: `test.py`
4. Ensure your code includes:
   - Comprehensive docstrings (numpy style)
   - Clear variable names
   - Comments explaining key steps
   - A working test with a real dataset
5. Update the README.md with algorithm details
6. Commit your changes (`git commit -m 'Add [Algorithm Name]'`)
7. Push to the branch (`git push origin feature/new-algorithm`)
8. Open a Pull Request

### Code Style Guidelines

- Follow PEP 8 conventions
- Use descriptive variable names
- Add type hints where appropriate
- Include docstrings for all classes and methods
- Comment complex mathematical operations
- Keep functions focused and modular

### Improving Existing Code

- Bug fixes are always welcome
- Performance improvements
- Better documentation
- Additional test cases
- Visualization improvements

## License

This project is open source and maintained by [Ali Fayod](https://github.com/AliFayod).

Feel free to use this code for learning, teaching, or any educational purposes. For production applications, please consider using established libraries like scikit-learn, TensorFlow, or PyTorch.

## Acknowledgments

These implementations are inspired by foundational machine learning textbooks and research papers:

- *Pattern Recognition and Machine Learning* by Christopher Bishop
- *The Elements of Statistical Learning* by Hastie, Tibshirani, and Friedman
- *Machine Learning: A Probabilistic Perspective* by Kevin Murphy
- Various research papers and academic resources

The goal is to demystify machine learning algorithms and make their inner workings accessible to learners at all levels.

## Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/AliFayod/Machine_learning_algorithms/issues)
- **Pull Requests**: Contributions are welcome
- **Questions**: Open an issue for discussion

## Future Additions

Potential algorithms to be added:
- Lasso Regression (L1 regularization)
- Elastic Net
- Gradient Boosting
- DBSCAN clustering
- Hierarchical Clustering
- Linear Discriminant Analysis (LDA)
- Gaussian Mixture Models

---

**Educational Note**: These implementations prioritize clarity and understanding over performance. They are designed to help you learn how algorithms work internally. For production use, leverage optimized libraries that have been thoroughly tested and optimized for performance.

**Star this repository** if you find it helpful for learning machine learning! ⭐
