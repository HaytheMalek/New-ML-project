import numpy as np
from ..utils import macrof1_fn


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
    
    def KFold_cross_validation_KNN(self, xtrain, ytrain, candidate_ks, K):
        """
        Searches for the best k using K-fold cross-validation.

        Inputs:
            xtrain (np.array): training features, shape (N, D)
            ytrain (np.array): training labels, shape (N,)
            candidate_ks (list): list of k values to try
            K (int): number of folds for cross-validation

        Returns:
            best_k (int): k with highest average validation accuracy
        """
        N = xtrain.shape[0]
        all_indices = np.arange(N)
        split_size = N // K

        best_k = None
        best_avg_acc = 0

        acc_per_k = []
        f1_per_k = []

        for k in candidate_ks:
            accuracies = []
            f1s = []

            for fold in range(K):
                val_start = fold * split_size
                val_end = (fold + 1) * split_size
                val_indices = all_indices[val_start:val_end]
                train_indices = np.setdiff1d(all_indices, val_indices)

                X_train_fold = xtrain[train_indices]
                Y_train_fold = ytrain[train_indices]
                X_val_fold = xtrain[val_indices]
                Y_val_fold = ytrain[val_indices]

                knn = KNN(k=k)
                knn.fit(X_train_fold, Y_train_fold)
                Y_val_pred = knn.predict(X_val_fold)

                acc = np.mean(Y_val_pred == Y_val_fold)
                f1 = macrof1_fn(Y_val_pred, Y_val_fold)
                
                accuracies.append(acc)
                f1s.append(f1)


            avg_acc = np.mean(accuracies)
            avg_f1 = np.mean(f1s)
            
            acc_per_k.append(avg_acc)

            f1_per_k.append(avg_f1)
            print(f"k = {k}, average validation accuracy = {avg_acc:.4f}")

            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_k = k

        print(f"Best k selected by CV: {best_k} (avg val acc = {best_avg_acc:.4f})")
        return best_k, acc_per_k, f1_per_k