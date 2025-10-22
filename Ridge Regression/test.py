import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Ridge_Regression import RidgeRegression, RidgeRegressionClosed

def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """Calculate R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Load dataset
X, y = datasets.load_diabetes(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("=" * 60)
print("Ridge Regression with Gradient Descent")
print("=" * 60)

# Test different alpha values
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
print("\nTesting different alpha values:")
print(f"{'Alpha':<10} {'MSE':<15} {'R² Score':<15}")
print("-" * 40)

for alpha in alphas:
    model = RidgeRegression(alpha=alpha, learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{alpha:<10.2f} {mse:<15.2f} {r2:<15.4f}")

print("\n" + "=" * 60)
print("Ridge Regression with Closed-Form Solution")
print("=" * 60)

print("\nTesting different alpha values:")
print(f"{'Alpha':<10} {'MSE':<15} {'R² Score':<15}")
print("-" * 40)

for alpha in alphas:
    model = RidgeRegressionClosed(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{alpha:<10.2f} {mse:<15.2f} {r2:<15.4f}")

# Detailed test with optimal alpha
print("\n" + "=" * 60)
print("Detailed Results with Alpha = 1.0")
print("=" * 60)

model_gd = RidgeRegression(alpha=1.0, learning_rate=0.1, n_iterations=1000)
model_gd.fit(X_train, y_train)
y_pred_gd = model_gd.predict(X_test)

model_closed = RidgeRegressionClosed(alpha=1.0)
model_closed.fit(X_train, y_train)
y_pred_closed = model_closed.predict(X_test)

print(f"\nGradient Descent:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_gd):.2f}")
print(f"  R² Score: {r2_score(y_test, y_pred_gd):.4f}")

print(f"\nClosed-Form Solution:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_closed):.2f}")
print(f"  R² Score: {r2_score(y_test, y_pred_closed):.4f}")

print(f"\nFirst 10 predictions vs actual values (Closed-Form):")
print(f"{'Predicted':<12} {'Actual':<12} {'Error':<12}")
print("-" * 36)
for i in range(10):
    error = abs(y_pred_closed[i] - y_test[i])
    print(f"{y_pred_closed[i]:<12.2f} {y_test[i]:<12.2f} {error:<12.2f}")
