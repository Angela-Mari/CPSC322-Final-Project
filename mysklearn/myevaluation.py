import mysklearn.myutils as myutils
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import math

def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap the element and i with 
        rand_index =np.random.randint(0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None: 
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting
    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        # seed your random number generator
        np.random.seed(random_state)
    
    if shuffle: 
        # shuffle the rows in X and y before splitting
        randomize_in_place(X,y)
        
    num_instances = len(X) 
    if isinstance(test_size, float):
        # proportion
        test_size = math.ceil(num_instances * test_size) # ceil(8 * 0.33) = 3
    split_index = num_instances - test_size 
    
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold
    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    X_train_folds = []
    X_test_folds = []
    for i in range(n_splits):
        X_train_folds.append([])
        X_test_folds.append([])
    
    # split data into bins
    index = 0
    for i in range(len(X)):
        X_test_folds[index].append(i)
        index = (index + 1) % n_splits
    
    # combine bins into train sets
    for i in range(len(X_test_folds)):
        for j in range(len(X_test_folds)):
            if j != i:
                X_train_folds[i].extend(X_test_folds[j])

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.
    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # first group by y lables
    
    _, group_subtables = myutils.group_by(X,y)
    
    X_train_folds = []
    X_test_folds = []
    for i in range(n_splits):
        X_train_folds.append([])
        X_test_folds.append([])
    
    # split data into bins
    index = 0
    for row in group_subtables:
        for item in row:
            X_test_folds[index].append(item)
            index = (index + 1) % n_splits
    
    # combine bins into train sets
    for i in range(len(X_test_folds)):
        for j in range(len(X_test_folds)):
            if j != i:
                X_train_folds[i].extend(X_test_folds[j])

    

    return X_train_folds, X_test_folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # create matrix shape
    my_matrix = []
    length = len(labels)
    for _ in range(length):
        my_matrix.append([0 for j in range(length)])
            
    
    for i in range(len(y_true)):
        # convert to int (might be converting in place??)
        if not isinstance(y_true[i], int):
            y_true[i] = labels.index(y_true[i])

            y_pred[i] = labels.index(y_pred[i])
        my_matrix[y_true[i]][y_pred[i]] += 1

    return my_matrix 