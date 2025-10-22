import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from Neural_Network import NeuralNetwork

def accuracy(y_true, y_pred):
    """Calculate accuracy score."""
    return np.sum(y_true == y_pred) / len(y_true)

# Load dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("=" * 60)
print("Neural Network Training")
print("=" * 60)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print()

# Train neural network
nn = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden_size=20,
    output_size=1,
    learning_rate=0.1,
    n_iterations=1000,
    activation='relu'
)

print("Training Neural Network...")
nn.fit(X_train, y_train)

# Make predictions
y_pred_train = nn.predict(X_train)
y_pred_test = nn.predict(X_test)

# Calculate accuracy
train_acc = accuracy(y_train, y_pred_train)
test_acc = accuracy(y_test, y_pred_test)

print()
print("=" * 60)
print("Results")
print("=" * 60)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Test with different configurations
print()
print("=" * 60)
print("Testing Different Configurations")
print("=" * 60)
print(f"{'Hidden Size':<15} {'Activation':<15} {'Test Accuracy':<15}")
print("-" * 45)

configurations = [
    (10, 'relu'),
    (20, 'relu'),
    (30, 'relu'),
    (20, 'sigmoid'),
    (20, 'tanh'),
]

for hidden_size, activation in configurations:
    nn_test = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        output_size=1,
        learning_rate=0.1,
        n_iterations=500,
        activation=activation
    )
    nn_test.fit(X_train, y_train)
    y_pred = nn_test.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(f"{hidden_size:<15} {activation:<15} {acc:<15.4f}")

# Plot training loss
print()
print("Generating loss curve...")
plt.figure(figsize=(10, 6))
plt.plot(nn.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Neural Network Training Loss')
plt.grid(True)
plt.savefig('neural_network_loss.png')
print("Loss curve saved as 'neural_network_loss.png'")

# Show some predictions
print()
print("=" * 60)
print("Sample Predictions")
print("=" * 60)
print(f"{'Predicted':<15} {'Actual':<15} {'Correct':<15}")
print("-" * 45)
for i in range(min(15, len(y_test))):
    correct = "✓" if y_pred_test[i] == y_test[i] else "✗"
    print(f"{y_pred_test[i]:<15} {y_test[i]:<15} {correct:<15}")
