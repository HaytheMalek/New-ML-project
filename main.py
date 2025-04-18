import argparse

import numpy as np

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data

    # EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load("features.npz", allow_pickle=True)
        xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
        ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]

    # ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, "dog-small-64")
        xtrain, xtest, ytrain, ytest = load_data(data_dir)

    ## 2. Then we must prepare it. This is where you can create a validation set, normalize, add bias, etc.
    # Make a validation set (it can overwrite xtest, ytest)
    fullxtrain = xtrain.copy()
    fullytrain = ytrain.copy()

    if not args.test:
        ### WRITE YOUR CODE HERE
        validation_size = int(0.2 * len(xtrain))  # 20% of the data

        # Create validation set from the end of the training set
        xtest = xtrain[-validation_size:]
        ytest = ytrain[-validation_size:]

        # Remaining 80% goes to the training set
        xtrain = xtrain[:-validation_size]
        ytrain = ytrain[:-validation_size]


    ### WRITE YOUR CODE HERE to do any other data processing
    mean_val = np.mean(xtrain, axis=0, keepdims=True)
    std_val = np.std(xtrain, axis=0, keepdims=True)

    normalizedXtrain = normalize_fn(xtrain, mean_val, std_val)
    xtrain = normalizedXtrain

    mean_val2 = np.mean(xtest, axis=0, keepdims=True)
    std_val2 = np.std(xtest, axis=0, keepdims=True)

    normalizedXtest = normalize_fn(xtest, mean_val2, std_val2)
    xtest = normalizedXtest

    mean_val3 = np.mean(fullxtrain, axis=0, keepdims=True)
    std_val3 = np.mean(fullxtrain, axis=0, keepdims=True)

    fullxtrain = normalize_fn(fullxtrain, mean_val3, std_val3)

    ## 3. Initialize the method you want to use.
    def KFold_cross_validation_KNN(xtrain, ytrain, candidate_ks, K=5):
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

        for k in candidate_ks:
            accuracies = []

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
                accuracies.append(acc)

            avg_acc = np.mean(accuracies)
            print(f"k = {k}, average validation accuracy = {avg_acc:.4f}")

            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_k = k

        print(f"Best k selected by CV: {best_k} (avg val acc = {best_avg_acc:.4f})")
        return best_k

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    if args.method == "knn":
        candidate_ks = [1, 3, 5, 7, 9]
        best_k = KFold_cross_validation_KNN(fullxtrain, fullytrain, candidate_ks, K=5)
        method_obj = KNN(k=best_k)
        
    if args.method == "logistic_regression":
        method_obj = LogisticRegression()

    elif ...:  ### WRITE YOUR CODE HERE
        pass

    ## 4. Train and evaluate the method
    # Fit (:=train) the method on the training data for classification task
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    # Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / kmeans / nn (MS2)",
    )
    parser.add_argument(
        "--data_path", default="data", type=str, help="path to your dataset"
    )
    parser.add_argument(
        "--data_type", default="features", type=str, help="features/original(MS2)"
    )
    parser.add_argument(
        "--K", type=int, default=1, help="number of neighboring datapoints used for knn"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, otherwise use a validation set",
    )

    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument(
        "--nn_type",
        default="cnn",
        help="which network to use, can be 'Transformer' or 'cnn'",
    )
    parser.add_argument(
        "--nn_batch_size", type=int, default=64, help="batch size for NN training"
    )

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
