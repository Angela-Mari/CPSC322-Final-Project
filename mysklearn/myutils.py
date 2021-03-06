import mysklearn.mypytable as mypytable
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

def group_by_value(table, header, group_by_col_name):
    """Returns the data that is grouped together by the passed in column
    
    Args:
        table (list): list of lists representing data that the column will be read from
        header (list): list of column headings
        group_by_col_name (string): name of column that the data is to be grouped by 
    
    Returns: 
        group_names (list): list of strings to label the group names
        group_subtables (list): parallel lists of lists of their subtables
    """
    col = get_column(table, header, group_by_col_name)
    col_index = header.index(group_by_col_name)
    
    group_names = sorted(list(set(col))) 
    group_subtables = [[] for _ in group_names] 
    for row in table:
        group_by_value = row[col_index]
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(row.copy()) # shallow copy
    
    return group_names, group_subtables

def get_column(table, header, col_name):
    """Returns a column using the header and it's column name
    
    Args:
        table (list): list of lists representing data that the column will be read from
        header (list): list of column headings
        col_name (string): name of column that is to be returned 
    
    Returns: 
        double: appropriate column from the table 
    """
    col_index = header.index(col_name)
    col = []

    for row in table: 
        if row[col_index] != "NA":
            col.append(row[col_index])
    return col

def group_by(X, y):
    """Groups rows into subtables based on a single column
    
    Args:
        table(MyPytable): the table to sort into groups
        group_by_col_name(string): which column will dictate categories
        
    Returns:
        group_names(list), group_subtables(list of lists)
    """
    
    # get a list of unique values for the column
    group_names = []
    for item in y:
        if item in group_names:
            continue
        else:
            group_names.append(item)
    group_subtables = [[] for _ in group_names] # [[], [], []]

    # walk through each row and assign it to the appropriate group
    for i in range(len(X)):
        group_value = y[i]
        # which group_subtable??
        index = group_names.index(group_value)
        group_subtables[index].append(i) # shallow copy

    return group_names, group_subtables

def print_prediction(title, instances, predictions, test):
    """Prints for steps 1 & 2
    """
    print("===========================================")
    print(title)
    print("===========================================")
    for i in range(len(instances)):
        print("instance: " + str(instances[i]))
        print("class: " + str(predictions[i]) + ", actual: " + str(test[i]))     

def normalize(col_names, table):
    """takes list of cols that need to be normalized in a table
    Args:
        col_names(list): the columns needing normalization
        table(MyPyTable): table of data
    Note: 
        Normalizes in place for better or for worse
    """
    mins = []
    maxes = []
    for item in col_names:
        col = table.get_column(item)
        mins.append(min(col))
        maxes.append(max(col)-mins[-1])

    for row in table.data:
        for i in range(len(col_names)):
            try:
                col_index = table.column_names.index(col_names[i])
            except ValueError:
                return
            row[col_index] = (row[col_index]-mins[i])/maxes[i]

def accuracy_errorrate(predicted, actual):
    """calcualte accuracy over a whole data set
    Args:
        predicted (list): list of predicted values
        actual(list): list of real values (parallel)
    Returns:
        accuracy as a decmial and error as a decimal
    """
    tp = 0
    for i in range(len(predicted)):
        if predicted[i]==actual[i]:
            tp += 1
    
    return tp/len(predicted), (len(predicted)-tp)/len(predicted)

def convert_vals_into_cutoffs(table, header, col_name, cutoffs, lables):
    """converts values into cutoffs
    Args:
        values(list): list of values needing to be converted
        cutoffs(list): list of cutoffs
        lables(list): runs paralell to cutoffs
    Notes:
        converts in place!
    """
    col_index = header.index(col_name)
    for i in range(len(table)):
        if table[i][col_index] >= cutoffs[-1]:
            table[i][col_index] = lables[-1]
            continue
        if table[i][col_index] <= cutoffs[0]:
            table[i][col_index] = lables[0]
            continue
        j = i
        for j in range(len(cutoffs) - 1):
            if cutoffs[j] <= table[i][col_index] < cutoffs[j + 1]:
                table[i][col_index] = lables[j]

def get_x_and_y_trains(table):
    """splits a table into an X_train and y_train

        Args:
            table (list of lists): the table which needs to be split
        Retuns: X_train (list of lists) and y_train (list of obj)
        Notes:
            rows are at least 2 elements long or else the function returns empty tables
    """
    X_train = []
    y_train = []
    if len(table[0]) >= 2:
        for row in table:
            X_train.append(row[0:-1]) # everything but the last element [...]
            y_train.append(row[-1]) # the last element
    return X_train, y_train

def stitch_x_and_y_trains(X_train, y_train):
    """stitches the table back together
    Args: 
        X_train (list of lists): is the X_train
        y_train (list of obj): is the the y_train
    Returns: 
        stitched_table which is the two trains combined back together
    """
    stitched_table = []
    for i in range(len(X_train)):
        stitched_table.append(X_train[i].copy())
        stitched_table[i].append(y_train[i])
    return stitched_table

def all_same_class(instances):
    """helper function to check if instances in a subtree could make a leaf node by having the same lable

    Args:
        instances(list of lists): the instances in the subtree branch
    
    Returns:
        True or False if they have the same label
    Notes:
        - This should not be an empty list 
        - Assumes the training data has been stitched back to together
        - Class label is at index -1
    """
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True # if we get here, all instance labels matched the first label


def partition_instances(instances, split_attribute, attribute_domains):
    """group data by attribute domains (creates pairwise disjoint partitions)
        Args: 
            instances (list of lists): instances to group
            split_attribute (obj): lable to split the groups by
            attribute_domains: from the initial table, all possible values for each att
        Returns: partitions (dictionary of lists)
    """
    # comments refer to split_attribute "level"
    header = get_generic_header(instances)

    # we can't do this because partition instances won't fufil the full domain
    attribute_domain = attribute_domains[split_attribute] # ["Senior", "Mid", "Junior"]
    attribute_index = header.index(split_attribute) # 0
   
    # creates a dictionary
    partitions = {} # key (attribute value): value (partition)
    for attribute_value in attribute_domain:
        # creates a list of instances that match that key
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)

    return partitions 
# 2. append subtree to values_subtree and to tree appropriately
# 3. work on CASE 1, then CASE 2, then CASE 3 (write helper functions!!)

def majority_vote(parition):
    """picks the majority choice 
        Args: partiton(list of list): current items that need a majority vote
        Returns: the majority choice
    """
    options = []
    votes = []
    for item in parition:
        if item[-1] in options:
            votes[options.index(item[-1])] += 1
        else:
            options.append(item[-1])
            votes.append(1)
    
    return options[votes.index(max(votes))]


# 4. finish the TODOs in fit_starter_code()
def calculate_attribute_domains(table):
    """creates a dictionary of domains of each attribute from stitched table
    Args:
        table (list of lists): must have X_train and y_train combined, assumes all possible values are in column
    Returns: 
        nested dictionary of possible values for each attribute
    """
    header = get_generic_header(table)
    attribute_domains = {}
    for item in header:
        attribute_domains[item] = [] # add empty dicts for each item in header
    for i in range(len(table)):
        for j in range(len(table[i])-1): # -1 because we don't care about domain range for the result
            if table[i][j] not in attribute_domains[header[j]]:
                #print(header[j])
                attribute_domains[header[j]].append(table[i][j])
                #print(attribute_domains[header[j]])
    return attribute_domains

def get_generic_header(train):
    """
    Notes: does not include the last element
    """
    header = []
    for i in range(len(train[0])-1):
        header.append("att" + str(i))
    return header

def calcualte_entropy(instances, available_attributes):
    """goes through availble_attributes in the table 
     calcualtes which index in availble_attributes has least entropy

    Args: 
        availble_attributes (list): sublist of header which are aviliable for the subtree
        instances (list of lists): subtable of table requiered to calculate entropy
    Returns: index in availble_attributes to use for the next split
    """
    enews_attribute = []
    header = get_generic_header(instances)
    for attribute in available_attributes: # go through attributes remaining
        # split into subtables by domain values
        split_col = []
        att_index = header.index(attribute)
        split_col = [row[att_index] for row in instances]
        domain_names, domain_table_indexes = group_by(instances, split_col)
        enews_domain = []
        for i in range(len(domain_names)): # go through subtable of domain values
            # calculate the true vs false
            results = []
            totals = []
            for index in domain_table_indexes[i]:
                if (instances[index][-1]) not in results:
                    results.append(instances[index][-1])
                    totals.append(1)
                else:
                    totals[results.index(instances[index][-1])] += 1
            enew = 0
            for j in range(len(totals)):
                enew -= ((totals[j]/len(domain_table_indexes[i]))*math.log((totals[j]/len(domain_table_indexes[i])),2))
                weight = len(domain_table_indexes[i])
            if len(totals) == 1:
                enew = 0
            enews_domain.append([enew, weight])
        enew_att = 0
        for item in enews_domain:
            enew_att += (item[0] * (item[1]/len(instances)))
        enews_attribute.append(enew_att)
    entropy_index = enews_attribute.index(min(enews_attribute))
    return available_attributes[entropy_index]
    
def tdidt(current_instances, available_attributes, attribute_domains, F):
    """recursive call to build the tree
        Args:
            current_instances(list of lists): should be the full list (X_train and y_train combined)
            availble_attributes(list of str): remaining options to split on
            attribute_domains (dict of list): all possible items to branch on for each att
        Returns: 
            tree(list of list)
    """
    if F != None:
        available_attributes = random_attribute_subset(available_attributes, F)
    # basic approach (uses recursion!!):
    # select an attribute to split on based on entropy, 
    split_attribute = calcualte_entropy(current_instances, available_attributes)
    # remove split attribute from available attributes
    # because, we can't split on the same attribute twice in a branch
    available_attributes.remove(split_attribute) # Python is pass by object reference!!
    tree = ["Attribute", split_attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attribute_domains)
    # for each partition, repeat unless one of the following occurs (base case)

    for attribute_value, partition in sorted(partitions.items()):
        values_subtree = ["Value", attribute_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            values_subtree.append(["Leaf", partition[0][-1], len(partition), len(current_instances)])
        #   CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            majority = majority_vote(partition)
            values_subtree.append(["Leaf", majority, len(partition), len(current_instances)])
        #   CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            # get a list of all instances in partions 
            total_instances = []
            for key, value in partitions.items():
                if value == []:
                    continue
                else:
                    for item in value:
                        total_instances.append(item)
            majority = majority_vote(total_instances)
            tree = ["Leaf", majority, len(partition), len(current_instances)]
            break
        else: # all base cases are false, recurse!!
            subtree = tdidt(partition, available_attributes.copy(), attribute_domains, F)
            values_subtree.append(subtree)

        if tree != values_subtree: # if there are new values to add
            tree.append(values_subtree)
    return tree

# 5. replace random w/entropy (compare tree w/interview_tree)
# 6. move over starter code to PA6 OOP w/unit test fit()
# 7. move on to predict()...

def tdidt_predict(header, tree, instance):
    """ recursive call to find prediciton
        Args:
            header(list of str): the header for the tree
            tree(list of list): the tree to parse
            instance(list): the item which we are preicting the result for
        Returns: prediction 
    """
    # returns "True" or "False" if a leaf node is hit
    # None otherwise 
    info_type = tree[0]
    if info_type == "Attribute":
        # get the value of this attribute for the instance
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # recurse, we have a match!!
                return tdidt_predict(header, value_list[2], instance)
    else: # Leaf
        return tree[1] # label

def tdit_print_decision_rules(tree, attribute_names, class_name, rule):
    """ recursive call to create decision rules
        Args: 
            tree(list of lists): the tree to parse
            attribute_names(list of str): names of the attributes
            class_name(str): the result name
            rule(str): string of the rule which is built recursivly 
    """
    for i in range(2, len(tree)):
        
        if tree[0] == "Attribute":
            temp = "IF " + str(attribute_names[int(tree[1][-1])]) + " == "
            if temp not in rule:
                rule += temp
            tdit_print_decision_rules(tree[i], attribute_names, class_name, rule)
        
        if tree[0] == "Value":
            temp = str(tree[1]) + " "
            rule += temp
            if tree[i][0] != "Leaf":
                temp = " AND " # add and if its not a leaf
                rule += temp
            return tdit_print_decision_rules(tree[i], attribute_names, class_name, rule)

        if tree[0] == "Leaf":
            temp = " THEN " + class_name + " == " + str(tree[1])
            rule += temp
            print(rule)
            return

def bootstrap(table, random_state):
    if random_state is not None:
        # store seed 
        random_state = random_state
        np.random.seed(random_state)
    
    n = len(table)
    sample = []
    for _ in range(n):
        rand_index = np.random.randint(0, n)
        sample.append(table[rand_index])
    return sample
    
def random_attribute_subset(attributes, F):
    # shuffle and pick first F
    shuffled = attributes[:] # make a copy
    random.shuffle(shuffled)
    return shuffled[:F]

def majority_vote_predictions(predicitons):
    """picks the majority choice 
        Args: partiton(list of list): current items that need a majority vote
        Returns: the majority choice
    """
    options = []
    votes = []
    for item in predicitons:
        if item in options:
            votes[options.index(item)] += 1
        else:
            options.append(item)
            votes.append(1)
    
    return options[votes.index(max(votes))]

def bar_chart(x, y):
    """Displays bar chart of data passed in
    
    Args:
        x: list of categorical data for x axis
        y: the cateogories frequencies for y axis
    
    """
    if len(x) > 12:
        plt.figure(figsize=(18,5))
    else: 
        plt.figure()
    plt.bar(x, y, width=.5)
    plt.xticks(x, rotation=45, horizontalalignment="right", size='small')
    plt.show()

def get_frequencies(table, header, col_name):
    """Returns the frequencies of the data from a column of a table that is passed in
    
    Args:
        table (list): list of lists representing data that the column will be read from
        header (list): list of column headings
        col_name (string): name of column that is to be returned 
    
    Returns: 
        values (list): list of things that were counted 
        counts (list): parallel lists of their counts
    """
    col = get_column(table, header, col_name)
    for i in range(len(col)):
        item = str(col[i])
        col[i] = item
    col.sort()
    values = []
    counts = []

    for value in col:
        if value not in values:
            values.append(value)
            counts.append(1)
        else:
            counts[-1] += 1 # list is sorted
    return values, counts

def replace_values_in_column(value1, value2, table, header, col_name):
    col_index = header.index(col_name)
    for i in range(len(table)):
        if table[i][col_index] == value1:
            table[i][col_index] = value2
    pass

def convert_bool_to_numerical_value(col, attributes):
    '''Converts attributes with only positive and negative in their domain to numerical values of 1 for positive and 0 for negative

    Args:
        col (list) : column of attribute that is to be converted
        attributes (list) : list of attribute domain with positive value in index 0 and negative value in index 1
    
    returns:
        converted_col (list) : new column with converted values
    '''
    converted_col = []
    for item in col:
        if item == attributes[0]:
            converted_col.append(1)
        else:
            converted_col.append(0)
    
    return converted_col

def calculate_linear_regression(x, y):
    """Returns slope and y intercept of data passed in
    
    Args:
        x (list): list of numerical data from the x axis
        y (list): list of numerical data from y axis 
    
    Returns: 
        m (float): slope of data
        b (float): y intercept of data
    """
    num = 0
    x_avg = sum(x)/len(x)
    y_avg = sum(y)/len(y)
    
    m = sum([(x[i] - x_avg) * (y[i] - y_avg) for i in range(len(x))]) / sum([(x[i] - x_avg) ** 2 for i in range(len(x))])
    b = y_avg - m*x_avg

    return m, b

def scatter_plot(x, y, m, b, xlabel, ylabel):
    """Displays scatter plot and corresponding linear regression of data passed in
    
    Args:
        x (list): list of data for x axis
        y (list): list of data for y axis
        m (float): slope of data
        b (float): y intercept of data
        xlabel (string): label for x axis
        ylabel (string): label for y axis
    
    """
    x_avg = sum(x)/len(x)
    y_avg = sum(y)/len(y)
    
    covariance = (sum([(x[i] - x_avg) * (y[i] - y_avg) for i in range(len(x))])) / (len(x)-1) 
    correlation = (sum([(x[i] - x_avg) * (y[i] - y_avg) for i in range(len(x))])) / (math.sqrt(sum([(x[i] - x_avg) **2 for i in range(len(x))])) * math.sqrt(sum([(y[i] - y_avg)**2 for i in range(len(y))])))
    plt.figure()
    plt.plot([min(x), max(x)], [m*min(x) + b, m*max(x)+b], c="r", lw=5)
    plt.scatter(x, y, label=f'Correlation ={correlation}\n Covariance ={covariance}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def pie_chart(labs, data):
    """Displays pie chart of data passed in
    
    Args:
        lab: list of labels for the data
        y: list of lists of data that is to be read into pie chart
    
    """
    plt.figure()
    plt.pie(data, labels=labs, autopct="%1.1f%%")
    plt.show()

def histogram(data):
    """Displays histogram chart of data passed in
    
    Args:
        data: 1D list of data values
    """
    plt.figure()
    plt.hist(data, bins=10)
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.show()

def box_plot(distributions, labels):
    """Displays box plot of distributions passed in
    
    Args:
        distributions (list): list of distributions to be plotted
        labels (list): list of labels to label to label each x plot
    
    """
    plt.figure(figsize= (18, 5))
    plt.boxplot(distributions)
    plt.xticks(list(range(1, len(labels) + 1)), labels, rotation=45, horizontalalignment="right")
    plt.show()