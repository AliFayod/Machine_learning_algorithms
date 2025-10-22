import numpy as np


class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.

    PCA transforms data to a new coordinate system where the greatest
    variance by any projection lies on the first coordinate (first principal
    component), the second greatest variance on the second coordinate, and so on.

    Attributes:
    - n_components (int): Number of principal components to keep.
    - components (array): Principal components (eigenvectors).
    - mean (array): Mean of training data.
    - explained_variance (array): Variance explained by each component.
    """

    def __init__(self, n_components):
        """
        Initialize PCA.

        Parameters:
        n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X):
        """
        Fit PCA to the training data.

        Parameters:
        X (array-like): Training data of shape (n_samples, n_features).
        """
        # Center the data by subtracting the mean
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Calculate covariance matrix
        # Note: using (n_samples - 1) for unbiased estimate
        cov_matrix = np.cov(X_centered.T)

        # Calculate eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        # Eigenvectors are columns, so we transpose
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Store the first n_components eigenvectors
        self.components = eigenvectors[:self.n_components]

        # Store explained variance
        self.explained_variance = eigenvalues[:self.n_components]

    def transform(self, X):
        """
        Transform data to the principal component space.

        Parameters:
        X (array-like): Data to transform.

        Returns:
        array: Transformed data in principal component space.
        """
        # Center the data
        X_centered = X - self.mean

        # Project data onto principal components
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X):
        """
        Fit PCA and transform the data in one step.

        Parameters:
        X (array-like): Training data.

        Returns:
        array: Transformed data in principal component space.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Transform data back to original space.

        Parameters:
        X_transformed (array-like): Data in principal component space.

        Returns:
        array: Data transformed back to original space.
        """
        return np.dot(X_transformed, self.components) + self.mean

    def explained_variance_ratio(self):
        """
        Calculate the proportion of variance explained by each component.

        Returns:
        array: Ratio of variance explained by each principal component.
        """
        total_variance = np.sum(self.explained_variance)
        return self.explained_variance / total_variance

    def get_components(self):
        """
        Get the principal components.

        Returns:
        array: Principal components (eigenvectors).
        """
        return self.components


# Example usage:
# pca = PCA(n_components=2)
# X_transformed = pca.fit_transform(X)
# variance_ratio = pca.explained_variance_ratio()
# print(f"Variance explained: {variance_ratio}")
