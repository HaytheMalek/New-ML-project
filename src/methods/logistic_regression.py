import numpy as np
from ..utils import get_n_classes, label_to_onehot, onehot_to_label
from src import utils

def f_softmax(data, W):
    """
    Softmax function for multi-class logistic regression.
    
    Args:
        data (array): Input data of shape (N, D)
        W (array): Weights of shape (D, C) where C is the number of classes
    Returns:
        array of shape (N, C): Probability array where each value is in the
            range [0, 1] and each row sums to 1.
            The row i corresponds to the prediction of the ith data sample, and 
            the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
    """

    scores = np.dot(data, W)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    sum_exp = np.sum(exp_scores, axis=1, keepdims=True)
    return exp_scores / sum_exp

def loss_logistic_multi(data, labels, w):
    """ 
    Loss function for multi class logistic regression, i.e., multi-class entropy.
    
    Args:
        data (array): Input data of shape (N, D)
        labels (array): Labels of shape  (N, C)  (in one-hot representation)
        w (array): Weights of shape (D, C)
    Returns:
        float: Loss value 
    """

    softmax = np.clip(f_softmax(data, w), 1e-15, 1.0)
    log_softmax = np.log(softmax)
    loss = -np.sum(labels * log_softmax)
    return loss

def gradient_logistic_multi(data, labels, W):
    """
    Compute the gradient of the entropy for multi-class logistic regression.
    
    Args:
        data (array): Input data of shape (N, D)
        labels (array): Labels of shape (N, C) (in one-hot representation)
        W (array): Weights of shape (D, C)
    Returns:
        grad (np.array): Gradients of shape (D, C)
    """
    return np.dot(data.T, (f_softmax(data, W) - labels))

class LogisticRegression(object):
    def __init__(self, lr=1e-5, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """

        self.lr = lr
        self.max_iters = max_iters
        self.weights = None

    def fit(self, data, labels, lr=0.001):
        """
        Training function for multi class logistic regression.
        
        Args:
            data (array): Dataset of shape (N, D).

            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)


            max_iters (int): Maximum number of iterations. Default: 10
            lr (int): The learning rate of  the gradient step. Default: 0.001
            print_period (int): Number of iterations to print current loss. 
                If 0, never printed.
            plot_period (int): Number of iterations to plot current predictions.
                If 0, never plotted.
        Returns:
            weights (array): weights of the logistic regression model, of shape(D, C)
        """
        D = data.shape[1]
        C = 5
        # Set a random seed for reproducibility
        np.random.seed(42)
        self.weights = np.random.normal(0, 0.1, (D, C))

        for it in range(self.max_iters):
            gradient = gradient_logistic_multi(data, utils.label_to_onehot(labels), self.weights)
            self.weights = self.weights - gradient * lr
            predictions = self.predict(data)
            if utils.accuracy_fn(predictions, labels) == 100:
                break
        return predictions

    def predict(self, test_data):
        """
        Runs prediction on the test data.
        
        Args:
            test_data (array): test data of shape (N, D)
            weights (array): trained weights of shape (D, C)
        Returns:
            pred_labels (array): predicted labels of shape (N,)
        """
        probas = f_softmax(test_data, self.weights)
        predictions = np.argmax(probas, axis=1)
        return predictions.astype(int)

    def cross_validate(self, data, labels, k_folds):
        N = data.shape[0]
        all_indices = np.arange(N)
        split_size = N // k_folds

        accuracies = []
        for fold in range(k_folds):
            val_start = fold * split_size
            val_end = (fold + 1) * split_size
            val_indices = all_indices[val_start:val_end]
            train_indices = np.setdiff1d(all_indices, val_indices)

            X_train_fold = data[train_indices]
            Y_train_fold = labels[train_indices]
            X_val_fold = data[val_indices]
            Y_val_fold = labels[val_indices]

            self.fit(X_train_fold, Y_train_fold, lr=self.lr)
            Y_val_pred = self.predict(X_val_fold)
            acc = np.mean(Y_val_pred == Y_val_fold)
            accuracies.append(acc)

        return np.mean(accuracies)

    def tune_hyperparameters(self, data, labels, lr_values, max_iters_values, k_folds=5):
        accuracies = []
        best_lr, best_max_iters = None, None
        best_accuracy = 0
        hyperparam_pairs = []

        for lr in lr_values:
            for max_iters in max_iters_values:
                self.lr = lr
                self.max_iters = max_iters
                hyperparam_pairs.append((lr, max_iters))
                avg_acc = self.cross_validate(data, labels, k_folds)
                accuracies.append(avg_acc)
                print(f"lr = {lr}, max_iters = {max_iters}, avg val accuracy = {avg_acc:.4f}")
                if avg_acc > best_accuracy:
                    best_accuracy = avg_acc
                    best_lr, best_max_iters = lr, max_iters

        print("\nHyperparameter pairs and accuracies:")
        for pair, acc in zip(hyperparam_pairs, accuracies):
            print(f"Pair: {pair}, Accuracy: {acc:.4f}")

        print(f"Best hyperparameters: lr = {best_lr}, max_iters = {best_max_iters} (avg val acc = {best_accuracy:.4f})")
        return best_lr, best_max_iters, accuracies, hyperparam_pairs