import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    """
    K-Means clustering algorithm.

    Attributes:
    - K (int): Number of clusters.
    - max_iters (int): Maximum number of iterations.
    - plot_steps (bool): Whether to plot intermediate steps.
    - clusters (list): List of sample indices for each cluster.
    - centroids (array): Cluster centroids.
    """

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        """
        Initialize K-Means clustering.

        Parameters:
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        plot_steps (bool): Whether to visualize intermediate steps.
        """
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # List of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # Mean feature vector for each cluster
        self.centroids = []

    def fit(self, X):
        """
        Fit the K-Means model to the data.

        Parameters:
        X (array-like): Training data of shape (n_samples, n_features).
        """
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize centroids randomly
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimization loop
        for i in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Check for convergence
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

    def _create_clusters(self, centroids):
        """
        Assign samples to the closest centroid to create clusters.

        Parameters:
        centroids (list): Current centroid positions.

        Returns:
        list: List of clusters, where each cluster is a list of sample indices.
        """
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        """
        Find the index of the closest centroid to a sample.

        Parameters:
        sample (array-like): A data sample.
        centroids (list): List of centroids.

        Returns:
        int: Index of the closest centroid.
        """
        distances = [self._euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        """
        Calculate new centroids as the mean of samples in each cluster.

        Parameters:
        clusters (list): List of clusters.

        Returns:
        list: New centroid positions.
        """
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        """
        Check if centroids have converged.

        Parameters:
        centroids_old (list): Previous centroid positions.
        centroids (list): Current centroid positions.

        Returns:
        bool: True if converged, False otherwise.
        """
        distances = [
            self._euclidean_distance(centroids_old[i], centroids[i])
            for i in range(self.K)
        ]
        return sum(distances) == 0

    def _euclidean_distance(self, x1, x2):
        """
        Calculate Euclidean distance between two points.

        Parameters:
        x1 (array-like): First point.
        x2 (array-like): Second point.

        Returns:
        float: Euclidean distance.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """
        Predict cluster labels for samples.

        Parameters:
        X (array-like): Data samples.

        Returns:
        array: Cluster labels for each sample.
        """
        labels = np.zeros(X.shape[0])
        for idx, sample in enumerate(X):
            labels[idx] = self._closest_centroid(sample, self.centroids)
        return labels

    def get_cluster_labels(self):
        """
        Get cluster labels for the training data.

        Returns:
        array: Cluster labels.
        """
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def plot(self):
        """
        Visualize the clusters and centroids (for 2D data).
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, cluster in enumerate(self.clusters):
            point = self.X[cluster].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2, s=200)

        plt.show()


# Example usage:
# kmeans = KMeans(K=3, max_iters=150)
# kmeans.fit(X)
# labels = kmeans.get_cluster_labels()
