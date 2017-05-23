"""
Manifold forests
"""

# authors: Fabian Richter <fabrichter@uos.de>, Maxim Schuwalow <mschuwalow@uos.de>
# License: BSD

import numpy as np
from ..base import BaseEstimator

class ManifoldForest(BaseEstimator):
    """ Manifold forests

    References
    ----------
    "Decision Forests for Computer Vision and Medical Image Analysis, Advances in Computer Vision and Pattern Recognition" Criminisi, A.; Shotton, J.
    Springer-Verlag London (2013)

    """
    def __init__(self):
        """
        """
        # TODO: get all parameters
        raise NotImplementedError()

    def fit_transform(self, X, y=None):
        """
        Fit the data from X, and returns the embedded coordinates

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data.
        """
        raise NotImplementedError()

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
        
