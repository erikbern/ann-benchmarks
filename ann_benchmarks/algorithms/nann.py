#from __future__ import absolute_import
from nann import nann
from ann_benchmarks.algorithms.base import BaseANN

class Nann(BaseANN):
    def __init__(self, metric):
        self.metric=metric
#        self._n_neighbors = int(n_neighbors)
#        self._n_trees = int(n_trees)
#        self._leaf_size = int(leaf_size)
#        self._queue_size=None

    def fit(self, X):
        self._index = nann.Nann(X)


    def query(self, v, n):
        ind, dist = self._index.query(v, k=n)
        return ind

#    def __str__(self):
#        return 'Nann(n_neighbors=%d, n_trees=%d, leaf_size=%d, queue_size=%.2f)' % (self._n_neighbors,
#                                                                                           self._n_trees,
#                                                                                           self._leaf_size,
#                                                                                           self._queue_size)
