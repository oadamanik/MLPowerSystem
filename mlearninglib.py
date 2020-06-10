import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self, k=6, max_iteration=100):
        # Initialize the number of clusters
        self.k = k
        # Initialize the maximum number of the iteration
        self.max_iteration = max_iteration

        # Creating list of clusters based on the number of clusters
        self.clusters = [[] for _ in range(self.k)]
        # Initialize the centroids
        self.centroids = []

    def predict(self, x_set):
        # Data samples in form of array
        self.x_set = x_set
        # Get the number of the samples and the features
        self.n_samples, self.n_features = self.x_set.shape

        # Choosing k numbers different random centroids
        random_sample_indices = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.x_set[idx] for idx in random_sample_indices]

        # Do optimization
        for i in range(self.max_iteration):
            # Update clusters
            self.clusters = self._create_clusters(self.centroids)

            # Update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Check if converged
            if self._is_converged(centroids_old, self.centroids):
                break

        # Return cluster labels
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # Assign label for each cluster
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Clustering based on the current centroid
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.x_set):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        # Find the index with the minimum distance
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        # Initialize centroids
        centroids = np.zeros((self.k, self.n_features))  # this will be tuples
        # Get the centroid based on the current cluster
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.x_set[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # Function to check the convergence
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0


def accuracy(y_true, y_pred):
    # Function to calculate accuracy of the prediction
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x_test):
        y_pred = [self._predict(x) for x in x_test]
        return np.array(y_pred)

    def _predict(self, x):
        # Calculate distances between x and all samples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.x_train]
        # Find the indices of the first k neighbors and sort them by distance
        k_idx = np.argsort(distances)[:self.k]
        # Get the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # Find and return the most common label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]