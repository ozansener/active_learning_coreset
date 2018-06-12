# -*- coding: utf-8 -*-
"""Forward greedy facility location"""

import numpy as np
import warnings

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.metrics import normalized_mutual_info_score
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
import bisect


class ForwardGreedyFacility(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    k-medoids class.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        How many medoids. Must be positive.

    distance_metric : string, optional, default: 'euclidean'
        What distance metric to use.

    clustering : {'pam'}, optional, default: 'pam'
        What clustering mode to use.

    init : {'random', 'heuristic'}, optional, default: 'heuristic'
        Specify medoid initialization.

    max_iter : int, optional, default : 300
        Specify the maximum number of iterations when fitting.

    random_state : int, optional, default: None
        Specify random state for the random number generator.
    """

    # Supported clustering methods
    CLUSTERING_METHODS = ['pam']

    # Supported initialization methods
    INIT_METHODS = ['random', 'heuristic']

    def __init__(self, n_clusters=8, distance_metric='euclidean'):

        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.center_ics_ = None

    def _check_init_args(self):

        # Check n_clusters
        if self.n_clusters is None or self.n_clusters <= 0 or \
                not isinstance(self.n_clusters, int):
            raise ValueError("n_clusters has to be nonnegative integer")

        # Check distance_metric
        if callable(self.distance_metric):
            self.distance_func = self.distance_metric
        elif self.distance_metric in PAIRWISE_DISTANCE_FUNCTIONS:
            self.distance_func = \
                PAIRWISE_DISTANCE_FUNCTIONS[self.distance_metric]
        else:
            raise ValueError("distance_metric needs to be " +
                             "callable or one of the " +
                             "following strings: " +
                             "{}".format(PAIRWISE_DISTANCE_FUNCTIONS.keys()) +
                             ". Instead, '{}' ".format(self.distance_metric) +
                             "was given.")

    def fit(self, X, y=None):
        """Fit K-Medoids to the provided data.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)

        Returns
        -------
        self
        """

        self._check_init_args()

        # Check that the array is good and attempt to convert it to
        # Numpy array if possible
        X = self._check_array(X)

        # Apply distance metric to get the distance matrix
        D = self.distance_func(X)

        num_data = X.shape[0]
        candidate_ids = range(num_data)
        candidate_scores = np.zeros(num_data,)
        subset = []

        k = 0
        while k < self.n_clusters:
          candidate_scores = []
          for i in candidate_ids:
            # push i to subset
            subset.append(i)
            marginal_cost = np.sum(np.min(D[:, subset], axis=1))
            candidate_scores.append(marginal_cost)
            # remove i from subset
            subset.pop()

          # push i_star to subset
          i_star = candidate_ids[np.argmin(candidate_scores)]
          bisect.insort(subset, i_star)
          # remove i_star from candiate indices
          del candidate_ids[bisect.bisect_left(candidate_ids, i_star)]

          k = k + 1

          #print '|S|: %d, F(S): %f' % (k, np.min(candidate_scores))

        # Expose labels_ which are the assignments of
        # the training data to clusters
        self.labels_ = self._get_cluster_ics(D, subset)

        # Expose cluster centers, i.e. medoids
        self.cluster_centers_ = X.take(subset, axis=0)

        # Expose indices of chosen cluster centers
        self.center_ics_ = subset

        return self

    def loss_augmented_fit(self, X, y, loss_mult):
        """Fit K-Medoids to the provided data.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)

        Returns
        -------
        self
        """

        self._check_init_args()

        # Check that the array is good and attempt to convert it to
        # Numpy array if possible
        X = self._check_array(X)

        # Apply distance metric to get the distance matrix
        D = self.distance_func(X)

        num_data = X.shape[0]
        candidate_ids = range(num_data)
        candidate_scores = np.zeros(num_data,)
        subset = []

        k = 0
        while k < self.n_clusters:
          candidate_scores = []
          for i in candidate_ids:
            # push i to subset
            subset.append(i)
            marginal_cost = np.sum(np.min(D[:, subset], axis=1))
            loss = normalized_mutual_info_score(y,self._get_cluster_ics(D, subset))
            candidate_scores.append(marginal_cost - loss_mult*loss)
            # remove i from subset
            subset.pop()

          # push i_star to subset
          i_star = candidate_ids[np.argmin(candidate_scores)]
          bisect.insort(subset, i_star)
          # remove i_star from candiate indices
          del candidate_ids[bisect.bisect_left(candidate_ids, i_star)]

          k = k + 1

          #print '|S|: %d, F(S): %f' % (k, np.min(candidate_scores))

        # Expose labels_ which are the assignments of
        # the training data to clusters
        self.labels_ = self._get_cluster_ics(D, subset)

        # Expose cluster centers, i.e. medoids
        self.cluster_centers_ = X.take(subset, axis=0)

        # Expose indices of chosen cluster centers
        self.center_ics_ = subset

        return self


    def _check_array(self, X):

        X = check_array(X)

        # Check that the number of clusters is less than or equal to
        # the number of samples
        if self.n_clusters > X.shape[0]:
            raise ValueError("The number of medoids " +
                             "({}) ".format(self.n_clusters) +
                             "must be larger than the number " +
                             "of samples ({})".format(X.shape[0]))

        return X

    def _get_cluster_ics(self, D, subset):
        """Returns cluster indices for D and current medoid indices"""

        # Assign data points to clusters based on
        # which cluster assignment yields
        # the smallest distance
        cluster_ics = np.argmin(D[subset, :], axis=0)

        return cluster_ics
