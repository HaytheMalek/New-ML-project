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
    """
    Logistic regression classifier.
    """

    def __init__(self, lr= 1e-5, max_iters=100):
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

   

    def fit(self, data, labels, lr=0.001, print_period=10):
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
        # Initialize the weights randomly according to a Gaussian distribution
        
        D = data.shape[1]  # number of features
        C = 5  # number of classes
        
        # Random initialization of the weights
        self.weights = np.random.normal(0, 0.1, (D, C))

        for it in range(self.max_iters):
            gradient = gradient_logistic_multi(data, utils.label_to_onehot(labels), self.weights)
            self.weights = self.weights - gradient * lr

            predictions = self.predict(data)
            if utils.accuracy_fn(predictions, labels) == 100:
                break
            #logging and plotting
            #if print_period and it % print_period == 0:
            #   print('loss at iteration', it, ":", loss_logistic_multi(data, utils.label_to_onehot(labels), self.weights))     

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

        probas = f_softmax(test_data, self.weights) #size (N, C)

        predictions = np.argmax(probas, axis=1)  

        return predictions.astype(int)