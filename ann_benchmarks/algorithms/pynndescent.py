from __future__ import absolute_import
import pynndescent
from ann_benchmarks.algorithms.base import BaseANN


class PyNNDescent(BaseANN):
    def __init__(self, metric, n_neighbors=10, n_trees=8, leaf_size=20):
        self._n_neighbors = int(n_neighbors)
        self._n_trees = int(n_trees)
        self._leaf_size = int(leaf_size)
        self._queue_size = None
        self._pynnd_metric = {'angular': 'cosine',
                              'euclidean': 'euclidean',
                              'hamming': 'hamming',
                              'jaccard': 'jaccard'}[metric]

    def fit(self, X):
        self._index = pynndescent.NNDescent(X,
                                            n_neighbors=self._n_neighbors,
                                            n_trees=self._n_trees,
                                            leaf_size=self._leaf_size,
                                            metric=self._pynnd_metric)

    def set_query_arguments(self, queue_size):
        self._queue_size = float(queue_size)

    def query(self, v, n):
        ind, dist = self._index.query(
            v.reshape(1, -1).astype('float32'), k=n,
            queue_size=self._queue_size)
        return ind[0]

    def __str__(self):
        str_template = ('PyNNDescent(n_neighbors=%d, n_trees=%d, leaf_size=%d'
                        ', queue_size=%.2f)')
        return str_template % (self._n_neighbors, self._n_trees,
                               self._leaf_size, self._queue_size)
