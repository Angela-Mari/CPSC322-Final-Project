import mysklearn.myutils as myutils
import numpy as np
import math 
import operator
import random
import mysklearn.myevaluation as myevaluation


class MySimpleLinearRegressor:
    """Represents a simple linear regressor.
    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b
    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.
        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        total = 0
        for row in X_train:
            for item in row:
                total += item
        
        mean_x = total/len(X_train)
        mean_y = np.mean(y_train) 
        m = sum([(X_train[i][0] - mean_x) * (y_train[i] - mean_y) for i in range(len(X_train))]) \
        / sum([(X_train[i][0] - mean_x) ** 2 for i in range(len(X_train))])
        # y = mx + b => y - mx
        b = mean_y - m * mean_x

        self.slope = m
        self.intercept = b

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for row in X_test:
            for item in row:
                y_predicted.append(self.slope*item + self.intercept)
        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        for item in X_test:
            item_distances = []             
            for row in self.X_train:
                #assert len(item) == len(row) #check that X_test and each instance of X_train are the same len
                row_distance = []
                for i in range(len(row)):
                    # check if values are categorical
                    if isinstance(row[i], str):
                        if row[i] == X_test[i]:
                            row_distance.append(0)
                        else:
                            row_distance.append(1)
                    else:
                        row_distance.append((row[i] - item[i]) ** 2)
                item_distances.append(np.sqrt(sum(row_distance)))
            distances.append(item_distances)
        
        # combine lists
        distances_sorted = []
        for i in range(len(distances)): # there are two rows
            new_row = []
            for j in range(len(distances[i])):
                new_row.append([distances[i][j],j])
            distances_sorted.append(new_row)
        top_k = []
        for row in distances_sorted: 
            row = sorted(row, key=operator.itemgetter(0))
            top_k.append(row[:self.n_neighbors])
        
        final_dist = []
        final_idx = []
        for row in top_k:
            for item in row:
                new_row_dist = []
                new_row_i = []
                for instance in row:
                    new_row_dist.append(instance[0])
                    new_row_i.append(instance[-1])
            final_dist.append(new_row_dist)
            final_idx.append(new_row_i)

        return final_dist, final_idx

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        distances, indices = self.kneighbors(X_test)
        lables = []
        for row in indices:
            new_row = []
            for index in row:
                new_row.append(self.y_train[index])
            lables.append(new_row)
        
        lables.sort()
        keys = []
        counts = []
        for row in lables:
            new_keys = []
            new_counts = []
            for item in row:
                if item not in new_keys:
                    new_keys.append(item)
                    new_counts.append(1)
                    continue
                if item in new_keys:
                    idx = new_keys.index(item)
                    new_counts[idx] += 1
            keys.append(new_keys)
            counts.append(new_counts)

        key_idxs = []
        for item in counts:
            key_idxs.append(item.index(max(item)))

        y_predicted = []
        for i in range(len(keys)):
            y_predicted.append(keys[i][key_idxs[i]])
        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train
        # create "headers"
        header = []
        for i in range(len(X_train[0])):
            header.append("att"+str(i))
        
        results = []
        result_totals = []
        
        # find priors
        for item in y_train:
            if item not in results:
                results.append(item)
                result_totals.append(1)
            else: 
                index = results.index(item)
                result_totals[index] += 1
        
        priors = {}
        for i in range(len(results)):
            priors[results[i]] = (result_totals[i]/len(y_train))
        self.priors = priors

        # find posteriors
        posteriors = {}
        for i in range(len(header)):
            attribute = header[i]
            # create line in dic for each attribute
            posteriors[attribute] = {}
            # for each attribute get that column
            col = []
            for row in X_train:
                col.append(row[i])
            # for that attribute find the categories
            categories = []
            for i in range(len(col)):
                if col[i] in posteriors[attribute].keys():
                    result = y_train[i]
                    posteriors[attribute][col[i]][result] += 1
                else:
                    categories.append(col[i])
                    col_results = {}
                    for item in results:
                        col_results[item] = 0
                    posteriors[attribute].update({col[i]:col_results})
                    result = y_train[i]
                    posteriors[attribute][col[i]][result] += 1

            #devide by total for category
            for j in range(len(categories)):
                for k in range(len(results)):
                    posteriors[attribute][categories[j]][results[k]] /= result_totals[k]

        self.posteriors = posteriors

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """        
        header = []
        for i in range(len(X_test[0])):
            header.append("att"+str(i))

        possibilities = []
        for item in self.y_train:
            if item not in possibilities:
                possibilities.append(item)

        # [[[yes data],[no data]]]
        y_predicted = []
        for _ in X_test:
            my_row = []
            for item in possibilities:
                my_row.append([])
            y_predicted.append(my_row)
        
        for i in range(len(X_test)): # i is the number of test instances
            for k in range(len(possibilities)): # devide into two possibilities 
                y_predicted[i][k].append(self.priors[possibilities[k]]) # add prior to each possibility
                for j in range(len(header)): # j corresponds with attribute and header, multiply by all attributes
                    y_predicted[i][k].append(self.posteriors[header[j]][X_test[i][j]][possibilities[k]])
                # multiply numbers to get the prediction
                predict = 1
                for m in range(len(y_predicted[i][k])):
                    predict *= y_predicted[i][k][m]
                y_predicted[i][k] = predict
            # get the max
            result_index = y_predicted[i].index(max(y_predicted[i]))
            result = possibilities[result_index]
            y_predicted[i] = result

        return y_predicted

class MyZeroRClassifier():
    """Represents a Zero Rules Classifier.

    Attributes:
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
    """
    def __init__(self):
        """Initializer for MyZeroRClassifier.
        Args:
            y_train(list of objects): Lazy classifier will just store this in fit()
        """
        self.y_train = None
    
    def fit(self, y_train):
        """Fits a Zero Rules classifier to y_train.
        Args:
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero Rules only looks at most common lable, it does not need X-train. This method just stores y_train.
        """
        self.y_train = y_train
    
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """    
        possibilities = []
        counts = []
        for item in self.y_train:
            if item not in possibilities:
                possibilities.append(item)
                counts.append(1)
            else:
                index = possibilities.index(item)
                counts[index] += 1
        
        prediction = possibilities[counts.index(max(counts))]

        y_predicted = []
        for item in X_test:
            y_predicted.append(prediction)

        return y_predicted

class MyRandomClassifier():
    """Represents a Random Classifier.

    Attributes:
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
    """
    def __init__(self):
        """Initializer for MyZeroRClassifier.
        Args:
            y_train(list of objects): Lazy classifier will just store this in fit()
        """
        self.y_train = None
        self.random_state = None
    
    def fit(self, y_train, random_state=None):
        """Fits a Random classifier to y_train.
        Args:
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
            random_state(int): integer used for seeding a random number generator for reproducible results

        Notes:
            Since Random only chooses a random lable, it does not need X-train. This method just stores y_train.
        """
        self.y_train = y_train
        if random_state is not None:
        # store seed 
            self.random_state = random_state
    
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """    

        if self.random_state is not None:
        # seed your random number generator
            np.random.seed(self.random_state)

        possibilities = []
        counts = []
        randomize = []
        for item in self.y_train:
            if item not in possibilities:
                possibilities.append(item)
                counts.append(1)
            else:
                index = possibilities.index(item)
                counts[index] += 1
        for i in range(len(counts)):
            for _ in range(counts[i]):
                randomize.append(i)
                
        y_predicted = []
        for item in X_test:
            prediction = possibilities[random.choice(randomize)]
            y_predicted.append(prediction)

        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        ##fit() accepts X_train and y_train
        # # TODO: calculate the attribute domains dictionary
        # # TODO: calculate a header (e.g. ["att0", "att1", ...])
        # # my advice: stitch together X_train and y_train
        # train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # available_attributes = header.copy() # recall: Python is pass
        # # by object reference
        # # initial tdidt() call
        self.X_train = X_train
        self.y_train = y_train
        train = myutils.stitch_x_and_y_trains(X_train, y_train)
        available_attributes = myutils.get_generic_header(X_train) # TODO: check that this is used for only X_trains
        attribute_domains = myutils.calculate_attribute_domains(X_train) # TODO: think about if this should be X_train or "train"
        
        self.tree = myutils.tdidt(train, available_attributes, attribute_domains, None)
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
    
        train = myutils.stitch_x_and_y_trains(self.X_train, self.y_train)
        header = myutils.get_generic_header(train)
        y_predicted = []
        for item in X_test:
            y_predicted.append(myutils.tdidt_predict(header, self.tree, item))
        
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        myutils.tdit_print_decision_rules(self.tree, attribute_names, class_name, "")

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        forest (nested list): The extracted list of tree models in the forest.
        M (int): a the size subset of the most accurate trees in the forest
        N (int): the total number of trees in the forest
        F (int): the size of the subset of available attributes 
        random_state (int): integer used for seeding a random number generator for reproducible results

    """
    def __init__(self):
        """Initializer for MyRandomForestClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.forest = []
        self.M = None
        self.N = None
        self.F = None
        self.random_state = None

    def fit(self, X_train, y_train, user_F, user_N, user_M, random_state=None):
        """Fits a random forest classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if random_state is not None:
        # store seed 
            self.random_state = random_state
            np.random.seed(self.random_state)
        self.X_train = X_train
        self.y_train = y_train
        self.F = user_F
        self.N = user_N
        self.M = user_M
        stratified_test, stratified_remainder = myevaluation.random_stratified_test_remainder_set(X_train, y_train)
        train = myutils.stitch_x_and_y_trains(X_train, y_train)
        attribute_domains = myutils.calculate_attribute_domains(train) # TODO: think about if this should be X_train or "train"
        N_forest = []
        for _ in range(self.N): 
            bootstrapped_table = myutils.bootstrap(stratified_remainder)
            available_attributes = myutils.get_generic_header(bootstrapped_table) # TODO: check that this is used for only X_trains
            tree = myutils.tdidt(bootstrapped_table, available_attributes, attribute_domains, self.F)
            N_forest.append(tree)
        header = myutils.get_generic_header(stratified_remainder)
        header.append("y")
        y_predicted = []
        y_true = []
        all_accuracies = []
        # testing accuracy of N_forest trees to find the top M accuracies 
        for tree in N_forest:
            y_predicted_row = []
            for item in stratified_test:
                y_predicted_row.append(myutils.tdidt_predict(header, tree, item[:-1])) 
            y_predicted.append(y_predicted_row)
    
        y_true = myutils.get_column(stratified_test, header, "y")
        for predicted_sublist in y_predicted:
            accuracy, _ = myutils.accuracy_errorrate(predicted_sublist, y_true)
            all_accuracies.append(accuracy)
        
        for _ in range(self.M):
            max_ind = all_accuracies.index(max(all_accuracies))
            self.forest.append(N_forest[max_ind])
            all_accuracies[max_ind] = -1
    
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        train = myutils.stitch_x_and_y_trains(self.X_train, self.y_train)
        header = myutils.get_generic_header(train)
        
        y_predicted = []
        for test in X_test:
            tree_predictions = []
            for tree in self.forest:
                tree_predictions.append(myutils.tdidt_predict(header, tree, test))
            y_predicted.append(myutils.majority_vote_predictions(tree_predictions))

        return y_predicted

