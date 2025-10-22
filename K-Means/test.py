import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from K_Means import KMeans


# Generate synthetic data
X, y = make_blobs(
    centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
)

print("Data shape:", X.shape)

# Cluster the data
clusters = len(np.unique(y))
print(f"Number of clusters: {clusters}")

k = KMeans(K=clusters, max_iters=150, plot_steps=False)
k.fit(X)

# Get cluster labels
labels = k.get_cluster_labels()

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot original data
ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.6)
ax1.set_title('Original Data with True Labels')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# Plot clustered data
ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)

# Plot centroids
centroids = np.array(k.centroids)
ax2.scatter(
    centroids[:, 0], centroids[:, 1],
    marker='x', s=300, linewidths=3,
    color='red', label='Centroids'
)
ax2.set_title('K-Means Clustering Results')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.legend()

plt.tight_layout()
plt.savefig('kmeans_results.png')
print("Results saved to kmeans_results.png")
print(f"K-Means clustering completed with {clusters} clusters")
