import numpy as np

def euclidean_dist(example, training_examples):
    """Compute the Euclidean distance between a single example
    vector and all training_examples.

    Inputs:
        example: shape (D,)
        training_examples: shape (NxD) 
    Outputs:
        euclidean distances: shape (N,)
    """
    # WRITE YOUR CODE HERE
    return np.sqrt(np.sum((example - training_examples) ** 2, axis=1))

def find_k_nearest_neighbors(k, distances):
    """ Find the indices of the k smallest distances from a list of distances.
        Tip: use np.argsort()

    Inputs:
        k: integer
        distances: shape (N,) 
    Outputs:
        indices of the k nearest neighbors: shape (k,)
    """
    # WRITE YOUR CODE HERE
    indices = np.argsort(distances)[:k]
    return indices

def predict_label(neighbor_labels):
    """Return the most frequent label in the neighbors'.

    Inputs:
        neighbor_labels: shape (N,) 
    Outputs:
        most frequent label
    """
    # WRITE YOUR CODE HERE
    return np.argmax(np.bincount(neighbor_labels.astype(int)))

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.X_train = None
        self.Y_train = None

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.X_train = training_data
        self.Y_train = training_labels
        return self.predict(training_data)

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        N = test_data.shape[0]
        test_labels = np.zeros(N, dtype=int)

        for i, x in enumerate(test_data):
            distances = euclidean_dist(x, self.X_train)
            nn_indices = find_k_nearest_neighbors(self.k, distances)
            neighbor_labels = self.Y_train[nn_indices]
            test_labels[i] = predict_label(neighbor_labels)

        return test_labels