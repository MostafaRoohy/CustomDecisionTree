import numpy as np
from collections import Counter



class Node:


    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        
        self.feature   = feature   # numpy array
        self.threshold = threshold # float
        self.left      = left      # Node
        self.right     = right     # Node
        self.value     = value     # label
    #


    def is_leaf_node(self):

        # A leaf node is a node in a tree data structure that has no children.

        return (self.value is not None)
    #
#



class DecisionTree:


    def __init__(self, criterion='entropy', min_samples_split=2, max_depth=100, n_features=None):

        self.criterion         = criterion
        self.min_samples_split = min_samples_split # The minimum number of data points that a node must have in order to be considered for splitting.
        self.max_depth         = max_depth         # The maximum depth of the tree.
        self.n_features        = n_features        # The number of features to consider when splitting a node. Used to randomly select a subset of features to consider when splitting a node.
        self.root              = None              # The root node of the tree.
        self.default_value     = 0                 # For empty nodes
    #

    ################################################################################
    ################################################################################
    ################################################################################

    def fit(self, X, y):

        # If n_features is not specified, it sets it to the number of features in the dataset.
        if (not self.n_features): 

            self.n_features = X.shape[1]
        #
        else:

            self.n_features = min(X.shape[1], self.n_features)
        #


        # It calls the _grow_tree method to grow the tree recursively.
        self.root = self._grow_tree(X, y) 
    #

    def _grow_tree(self, X, y, depth=0):

        n_samples = X.shape[0]
        n_feats   = X.shape[1]
        n_labels  = len(np.unique(y))


        # check the stopping criteria
        if (n_labels==1 or depth>=self.max_depth or n_samples<self.min_samples_split):

            leaf_value = self._most_common_label(y)

            return (Node(value=leaf_value))
        #


        # Randomly select a subset of features from the total number of features in the dataset.
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if (best_feature is None or best_thresh is None):

            return (Node(value=self._most_common_label(y)))
        #

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left  = self._grow_tree(X[left_idxs, :] , y[left_idxs] , depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)


        return (Node(best_feature, best_thresh, left, right))
    #

    def _best_split(self, X, y, feat_idxs):

        # The goal of this method is to find the split that results in the largest reduction in impurity (i.e., the largest information gain).

        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:

            X_column   = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:

                left_idxs, right_idxs = self._split(X_column, thr)

                if (len(left_idxs) == 0 or len(right_idxs) == 0):

                    continue
                #
                    

                gain = (self._information_gain(y, X_column, thr)) if (self.criterion=='entropy') else (self._gini_gain(y, X_column, thr))

                if (gain > best_gain):

                    best_gain       = gain
                    split_idx       = feat_idx
                    split_threshold = thr
                #
            #
        #


        return (split_idx, split_threshold)
    #

    ################################################################################

    def _information_gain(self, y, X_column, threshold):

        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        if (len(left_idxs) == 0 or len(right_idxs) == 0):

            return (0)
        #

        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy


        return (information_gain)
    #

    def _entropy(self, y):

        hist = np.bincount(y)
        ps   = hist / len(y)


        return (-np.sum([p * np.log(p) for p in ps if p>0]))
    #

    ################################################################################    

    def _gini_gain(self, y, X_column, threshold):

        # Parent Gini impurity
        parent_gini = self._gini(y)

        # Split data
        left_idxs, right_idxs = self._split(X_column, threshold)
        if (len(left_idxs) == 0 or len(right_idxs) == 0):

            return (0)
        #

        # Calculate weighted child Gini
        n   = len(y)
        n_l = len(left_idxs)
        n_r = len(right_idxs)
        gini_left  = self._gini(y[left_idxs])
        gini_right = self._gini(y[right_idxs])
        child_gini = (n_l / n) * gini_left + (n_r / n) * gini_right

        return (parent_gini - child_gini)
    #

    def _gini(self, y):

        counts = np.bincount(y)
        probs  = counts / len(y)
        return (1 - np.sum(probs ** 2))
    #

    ################################################################################

    def _split(self, X_column, split_thresh):

        left_idxs  = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column  > split_thresh).flatten()


        return (left_idxs, right_idxs)
    #

    def _most_common_label(self, y):

        # Returns the most common label in a node.

        if (len(y) == 0):

            return (self.default_value)
        #

        counter = Counter(y)
        value   = counter.most_common(1)[0][0]

        return (value)
    #

    ################################################################################
    ################################################################################
    ################################################################################

    def predict(self, X):

        return (np.array([self._traverse_tree(x, self.root) for x in X]))
    #

    def _traverse_tree(self, x, node):

        if node.is_leaf_node():

            return (node.value)
        #

        if (x[node.feature] <= node.threshold):

            return (self._traverse_tree(x, node.left))
        #


        return (self._traverse_tree(x, node.right))
    #
#