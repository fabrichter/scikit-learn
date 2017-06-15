import numpy as np
import math
from ..base import BaseEstimator
from ..utils import check_random_state
from ._utils import _affinity_matrix
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
    entropy = np.log(abs(det))  # due to numerical reasons might be slightly negative -> abs as workaround
    return entropy

class _Split:
    """
    Members
    -------
    value: split value of weak learner
    feature: value used for splitting
    """

    def __init__(self, num_features, remaining_depth, num_options=10):
        """
        A node of a tree, marking a split

        Parameters
        ----------
        features : array
            Indices of features to consider when splitting data
        num_options : int
            number of options for each feature to evaluate
        """
        self.num_features = num_features
        self.num_options = num_options
        self.left = None
        self.right = None
        self.remaining_depth = remaining_depth

    def fit_recursive(self, X, random_state=0):
        self.feature, self.value = self.determine_split(X, random_state=random_state)

        if self.remaining_depth <= 1:
            return

        split = self.predict(X)

        left_split = X[np.logical_not(split)]
        right_split = X[split]

        if len(left_split) > 1:
            self.left = _Split(self.num_features, self.remaining_depth - 1, self.num_options)
            self.left.fit_recursive(left_split, random_state)
        if len(right_split) > 1:
            self.right = _Split(self.num_features, self.remaining_depth - 1, self.num_options)
            self.right.fit_recursive(right_split, random_state)

    def predict_recursive(self, datum, node_id):
        """
        predict one datapoint
        :param datum:
        :return:
        """
        if datum[self.feature] < self.value:
            # left
            id = 2 * node_id + 1
            node = self.left

        else:
            # right
            id = 2 * node_id + 2
            node = self.right

        if node:
            return node.predict_recursive(datum, id)
        else:
            # the child node is not a split node but a leave. return index of that node
            return id

    def determine_split(self, X, random_state=0):
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
        feature_index, split_value
        """
        X = np.asarray(X)
        random_state = check_random_state(random_state)
        choices = np.empty((self.num_features, self.num_options))
        gain = np.empty((self.num_features, self.num_options))
        feature_choices = random_state.choice(X.shape[1], self.num_features, replace=False)
        # evaluate different splits: loop over features, save gain for each option
        for index, feature in enumerate(feature_choices):
            # simple axis-aligned split, TODO: use linear function?
            # if X[:, feature].shape[0] == 0:
            #     gain[index] = 0
            #     continue

            # if we split allong the minimal or the maximal feature we have an empty and completely useless split
            feature_values = X[:, feature]
            inner_values = (feature_values.min() < feature_values) # & (feature_values <= feature_values.max())

            if np.count_nonzero(inner_values) >= self.num_options:
                choices[index] = random_state.choice(feature_values[inner_values], self.num_options, replace=False)
            else:
                # num_options was larger then the feature values in the middle (not min/max)
                # draw with replacement so we don't have to change the gain collection matrix/choice matrix
                choices[index, :] = np.mean(feature_values)
                choices[index, 0:np.count_nonzero(inner_values)] = feature_values[inner_values]
                # random_state.choice(feature_values[inner_values], self.num_options, replace=True)
                # TODO now we ae taking all feature_values and a bunch of completely random ones

            # determine splits
            splits = [X[:, feature] < option for option in choices[index]]
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
            gain[index] = entropy_before - (
            np.divide(size_left, total) * entropy_left + np.divide(size_right, total) * entropy_right)

        # find best split & feature, return split
        best = np.argmax(gain)
        feature_idx, value_idx = np.unravel_index(best, gain.shape)
        value = choices[feature_idx, value_idx]
        feature = feature_choices[feature_idx]
        return feature, value

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
        return X[:, self.feature] < self.value


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

        self.root = _Split(self.num_features, self.depth, self.num_options)

        self.root.fit_recursive(X, random_state=random_state)

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
        indices = np.zeros((X.shape[0],),
                           np.int)  # start at root node; save which node is used for prediction at each step
        for i, datum in enumerate(X):
            indices[i] = self.root.predict_recursive(datum, 0)

        return indices


class AffinityForestNaiveRecursive(BaseEstimator):
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

    def fit_transform(self, X, dim=None, y=None, random_state=0):
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
        raise NotImplementedError("Better not use it")
        self.fit_transform(X)
        return self

