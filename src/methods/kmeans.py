import numpy as np

class KMeans(object):
    def __init__(self, max_iters=500):
        self.max_iters = max_iters
        self.centroids = None
        self.cluster_assignments = None

    @staticmethod
    def compute_distance(data, centers):
        return np.sqrt(((data[:, np.newaxis] - centers) ** 2).sum(axis=2))

    @staticmethod
    def find_closest_cluster(distances):
        return np.argmin(distances, axis=1)

    @staticmethod
    def compute_centers(data, cluster_assignments, K, old_centroids=None):
        centers = np.zeros((K, data.shape[1]))
        for k in range(K):
            mask = (cluster_assignments == k)
            if np.sum(mask) > 0:
                centers[k] = np.mean(data[mask], axis=0)
            else:
                # Handle empty cluster: use farthest point from largest cluster
                largest_cluster = np.argmax(np.bincount(cluster_assignments))
                cluster_data = data[cluster_assignments == largest_cluster]
                distances = np.linalg.norm(cluster_data - centers[largest_cluster], axis=1)
                centers[k] = cluster_data[np.argmax(distances)]
        return centers

    def fit(self, training_data, K):
        # k-means initialization
        K = 5
        self.centroids = [training_data[np.random.choice(training_data.shape[0])]]
        for i in range(1, K):
            distances = self.compute_distance(training_data, np.array(self.centroids))
            min_distances = np.min(distances, axis=1)
            prob = min_distances / min_distances.sum()
            new_idx = np.random.choice(training_data.shape[0], p=prob)
            self.centroids.append(training_data[new_idx])
        self.centroids = np.array(self.centroids)

        for i in range(self.max_iters):
            old_centroids = self.centroids.copy()
            distances = self.compute_distance(training_data, self.centroids)
            self.cluster_assignments = self.find_closest_cluster(distances)
            self.centroids = self.compute_centers(training_data, self.cluster_assignments, K, old_centroids)
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged after {i+1} iterations")
                break
        return self.predict(training_data)

    def predict(self, test_data):
        distances = self.compute_distance(test_data, self.centroids)
        return self.find_closest_cluster(distances)