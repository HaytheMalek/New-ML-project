import numpy as np

class KMeans(object):
    """
    K-Means clustering object.
    """
    
    def __init__(self, max_iters=500):
        """
        Initialize K-Means clustering.
        
        Arguments:
            max_iters: maximum number of iterations
        """
        self.max_iters = max_iters
        self.centroids = None
        self.cluster_assignments = None
        self.K = None

    @staticmethod
    def compute_distance(data, centers):
        """
        Compute the euclidean distance between each datapoint and each center.
        """
        N = data.shape[0]
        K = centers.shape[0]
        D = data.shape[1]
        distances = np.zeros((N, K))
        
        for i in range(N):
            for j in range(K):
                distances[i, j] = np.sqrt(np.sum((data[i] - centers[j])**2))
        
        return distances

    @staticmethod
    def find_closest_cluster(distances):
        """
        Assign datapoints to the closest clusters.
        """
        return np.argmin(distances, axis=1)

    @staticmethod
    def compute_centers(data, cluster_assignments, K):
        """
        Compute the center of each cluster based on the assigned points.
        """
        N, D = data.shape
        centers = np.zeros((K, D))
        
        for k in range(K):
            cluster_points = data[cluster_assignments == k]
            if len(cluster_points) > 0:
                centers[k] = np.mean(cluster_points, axis=0)
            else:
                centers[k] = data[np.random.randint(N)]
        
        return centers

    def fit(self, training_data, K=None):
        """
        Trains the K-Means model.
        
        Arguments:
            training_data: training data of shape (N,D)
            K: number of clusters (optional)
        """
        if K is None:
            # Default to square root rule if K not specified
            self.K = int(np.sqrt(training_data.shape[0]/2))
        else:
            self.K = K
            
        # Initialize centroids
        random_indices = np.random.choice(len(training_data), self.K, replace=False)
        self.centroids = training_data[random_indices]
        
        for i in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Compute distances and assignments
            distances = self.compute_distance(training_data, self.centroids)
            self.cluster_assignments = self.find_closest_cluster(distances)
            
            # Update centroids
            self.centroids = self.compute_centers(training_data, self.cluster_assignments, self.K)
            
            # Check for convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged after {i+1} iterations")
                break
                
        return self

    def predict(self, test_data):
        """
        Predict cluster assignments for new data.
        
        Arguments:
            test_data: test data of shape (N,D)
        Returns:
            cluster assignments of shape (N,)
        """
        if self.centroids is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        distances = self.compute_distance(test_data, self.centroids)
        return self.find_closest_cluster(distances)