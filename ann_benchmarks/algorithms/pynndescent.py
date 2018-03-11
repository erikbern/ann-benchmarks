from __future__ import absolute_import
import pynndescent
import numpy as np
from ann_benchmarks.algorithms.base import BaseANN

class PyNNDescent(BaseANN):
    def __init__(self, metric, n_neighbors=10, n_trees=8, leaf_size=40, queue_size=2.0):
        self._n_neighbors = int(n_neighbors)
        self._n_trees = int(n_trees)
        self._leaf_size = int(leaf_size)
        self._queue_size = float(queue_size)
        self._pynnd_metric = {'angular': 'cosine', 'euclidean': 'euclidean'}[metric]
        self.name = 'PyNNDescent(n_neighbors=%d,n_trees=%d,leaf_size=%d,queue_size=%.2f)' % \
                        (self._n_neighbors, self._n_trees, self._leaf_size, self._queue_size)

    def fit(self, X):
        self._index = pynndescent.NNDescent(X,
                        n_neighbors=self._n_neighbors,
                        n_trees=self._n_trees,
                        leaf_size=self._leaf_size,
                        metric=self._pynnd_metric)

    def query(self, v, n):
        ind, dist = self._index.query(np.array([v]), k=n, queue_size=self._queue_size)
        return ind[0]

    def use_threads(self):
        return False
