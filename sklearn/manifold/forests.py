"""
Manifold forests
"""

# authors: Fabian Richter <fabrichter@uos.de>, Maxim Schuwalow <mschuwalow@uos.de>
# License: BSD

import numpy as np
import math
from ..base import BaseEstimator
from ..utils import check_random_state
from ..utils.extmath import cartesian

# TODO: See whether using existing methods/classes for density estimation / tree-based methods would be helpful

def _entropy(data):
    """
    continous entropy of multivariate gaussian, simplified for use in information gain
    """
    means = np.mean(data, axis=0)
    normalized = data - means
    cov = np.dot(normalized.T, normalized)
    det = np.linalg.det(cov)
    if det == 0:
        return 0
    entropy = np.log(det)
    return entropy

def _affinity_matrix(X):
    """
    Computes affinity matrix out of tree partition
    using binary affinity
    """
    combinations = cartesian((X, X))
    def binary_affinity(pair):
        a, b = pair
        dist = 1 if a == b else 0
        return dist

    distance = np.apply_along_axis(binary_affinity, 1, combinations)
    k = X.shape[0]
    matrix = distance.reshape((k, k))
    return matrix
    

class _Split:
    """
    Members
    -------
    value: split value of weak learner
    feature: value used for splitting
    """

    def __init__(self, features, num_options=10):
        """
        A node of a tree, marking a split
        
        Parameters
        ----------
        features : array
            Indices of features to consider when splitting data

        num_options : int
            number of options for each feature to evaluate
        """
        self.features = features
        self.num_options = num_options

    def fit(self, X, random_state=0):
        """
        Search and return for the weak learner that gives us the most information.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Samples that should be split at this node

        random_state : RandomState or an int seed (0 by default)
            A random number generator instance to define the state of the
            random permutations generator.
        
        Returns
        -------
        
        split : array, shape (n_samples, )
            Boolean array that specifies the side of the split for each sample
        """
        X = np.asarray(X)
        random_state = check_random_state(random_state)
        choices = np.empty((len(self.features), self.num_options))
        gain = np.empty((len(self.features), self.num_options))
        # evaluate different splits: loop over features, save gain for each option
        for index, feature in enumerate(self.features):
            # simple axis-aligned split, TODO: use linear function?
            if X[:,feature].shape[0] == 0:
                gain[index] = 0
                continue

            choices[index] = random_state.choice(X[:,feature], self.num_options)
            # determine splits
            splits = [X[:,feature] < option for option in choices[index]]
            # get corresponding datasets
            split_l = [X[split] for split in splits]
            inverted_splits = np.logical_not(splits)
            split_r = [X[split] for split in inverted_splits]
            # compute information gain
            total = X.shape[0]
            size_left = [len(split) for split in split_l]
            size_right = [total - size for size in size_left]
            entropy_before = _entropy(X)
            entropy_left = [_entropy(split) for split in split_l]
            entropy_right = [_entropy(split) for split in split_r]
            gain[index] = entropy_before - (np.divide(size_left, total) * entropy_left + np.divide(size_right, total) * entropy_right)

        # find best split & feature, return split
        best = np.argmax(gain)
        feature_idx, value_idx = np.unravel_index(best, gain.shape)
        value = choices[feature_idx, value_idx]
        feature = self.features[feature_idx]
        split = X[:, feature] < value 
        self.feature = feature
        self.value = value
        return split

    def predict(self, X):
        """
        Apply fitted weak learner to dataset
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data
        
        Returns
        -------
        
        result : array, shape (n_samples)
            boolean symbolizing left or right side of split of weak learner
        """
        X = np.asarray(X)
        return X[:,self.feature] < self.value


# NOTE: bagging not implemented, as in reference
class _Tree:
    def __init__(self, depth, num_features, num_options):
        """
        A decision tree of a random forest.
        
        Parameters
        ----------
        depth : int
            Depth of the tree (number of splits).
        num_features : int
            Number of features to consider at each split
        num_options : int
            Number of values for each feature to consider at each split
        """
        self.depth = depth
        self.num_features = num_features
        self.num_options = num_options
        self.splits = [] # binary tree of splits

    def fit(self, X, random_state=0):
        """
        Build tree. Compute splits for the tree to cluster the data correspondingly.
        
        Parameters
        ----------
        X : array (num_samples, num_features)
            Input data.

        random_state : RandomState or an int seed (0 by default)
            A random number generator instance to define the state of the
            random permutations generator.
        """
        X = np.asarray(X)
        random_state = check_random_state(random_state)
        num_nodes = 2**(self.depth+1) - 1
        split_data = []
        for i in range(num_nodes):
            features = random_state.randint(low=0, high=X.shape[1], size=self.num_features)

            if i == 0:
                parent_data = X
            else:
                parent = math.floor((i-1) / 2)
                parent_data = split_data[parent]

            self.splits.append(_Split(features=features, num_options=self.num_options))
            split = self.splits[i].fit(parent_data, random_state=random_state)
            split_data.append(parent_data[split])

    def predict(self, X):
        """
        Sort data into clusters according to fitted tree.

        Parameters
        ----------
        data : array (num_samples, num_features)
            Input data.
        
        Returns
        -------
        
        indices : array, shape (num_samples,)
            Indices of leaf nodes of tree (i.e. associated cluster) for each sample
        """
        X = np.asarray(X)
        indices = np.zeros((X.shape[0],), np.int) # start at root node; save which node is used for prediction at each step
        for i in range(self.depth):
            # predict split using correct node
            split_direction = []
            for index, node in enumerate(np.nditer(indices)):
                datum = X[index]
                split_node = self.splits[node]
                split_direction.append(split_node.predict([datum])[0])

            # go to next node (child): left for predict = False, right = predict = True
            indices = (indices * 2) + 1 + split_direction

        return indices

class ManifoldForest(BaseEstimator):
    """ Manifold forests

    References
    ----------s    "Decision Forests for Computer Vision and Medical Image Analysis, Advances in Computer Vision and Pattern Recognition" Criminisi, A.; Shotton, J.
    
    Springer-Verlag London (2013)

    """
    def __init__(self, num_trees, depth, num_options, num_features):
        """
        Initialize forest for estimating affinity matrix and projecting into 
        
        Parameters
        ----------
        
        num_trees : int
            Number of trees in forest
        depth : int
            Depth of trees in forest
        num_options: int
            Number of considered weak learners for each feature
        num_features: int
            Number of considered features for each split
        """
        self.num_trees = num_trees
        self.depth = depth
        self.num_options = num_options
        self.num_features = num_features
    

    def fit_transform(self, X, y=None, random_state=0):
        """
        Fit the data from X, and return the calculated distances

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data.

        random_state : RandomState or an int seed (0 by default)
            A random number generator instance to define the state of the
            random permutations generator.
        
        Returns
        -------
        W : array, shape (n_samples, n_samples)
                      Affinity matrix of samples.
        """
        X = np.asarray(X)
        random_state = check_random_state(random_state)
        def make_tree():
            return _Tree(depth=self.depth, num_features=self.num_features, num_options=self.num_options)

        self.trees = [make_tree() for i in range(self.num_trees)]
        affinities = np.empty((self.num_trees, X.shape[0], X.shape[0]))
        for index, tree in enumerate(self.trees):
            tree.fit(X, random_state=random_state)
            clusters = tree.predict(X)
            affinities[index] = _affinity_matrix(clusters)

        self.W = np.sum(affinities, axis=0) / self.num_trees
        return self.W

    def fit(self, X, y=None):
        """
        Computes the position of the points in the embedding space

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data.
        """
        self.fit_transform(X)
        return self
        
