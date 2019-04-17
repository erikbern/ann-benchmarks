from __future__ import absolute_import
import sklearn.neighbors
import sklearn.preprocessing
from ann_benchmarks.algorithms.base import BaseANN


class BallTree(BaseANN):
    def __init__(self, metric, leaf_size=20):
        self._leaf_size = leaf_size
        self._metric = metric
        self.name = 'BallTree(leaf_size=%d)' % self._leaf_size

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._tree = sklearn.neighbors.BallTree(X, leaf_size=self._leaf_size)

    def query(self, v, n):
        if self._metric == 'angular':
            v = sklearn.preprocessing.normalize([v], axis=1, norm='l2')[0]
        dist, ind = self._tree.query([v], k=n)
        return ind[0]
