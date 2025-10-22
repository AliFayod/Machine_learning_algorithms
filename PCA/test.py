import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from PCA import PCA


# Load the Iris dataset
data = datasets.load_iris()
X = data.data
y = data.target

print(f"Original data shape: {X.shape}")

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)

print(f"Transformed data shape: {X_projected.shape}")

# Get explained variance ratio
variance_ratio = pca.explained_variance_ratio()
print(f"\nExplained variance ratio by component:")
for i, ratio in enumerate(variance_ratio):
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
print(f"Total variance explained: {np.sum(variance_ratio):.4f} ({np.sum(variance_ratio)*100:.2f}%)")

# Visualize the results
plt.figure(figsize=(12, 5))

# Original data (first 2 features)
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50, alpha=0.7)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Original Data (First 2 Features)')
plt.colorbar(label='Class')

# PCA transformed data
plt.subplot(1, 2, 2)
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, cmap='viridis', edgecolor='k', s=50, alpha=0.7)
plt.xlabel(f'PC1 ({variance_ratio[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({variance_ratio[1]*100:.1f}% variance)')
plt.title('PCA Transformed Data (2 Components)')
plt.colorbar(label='Class')

plt.tight_layout()
plt.savefig('pca_results.png')
print("\nVisualization saved to pca_results.png")

# Test inverse transform
X_reconstructed = pca.inverse_transform(X_projected)
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print(f"\nReconstruction error (MSE): {reconstruction_error:.6f}")
