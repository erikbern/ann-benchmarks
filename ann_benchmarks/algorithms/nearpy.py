from __future__ import absolute_import
import nearpy
from nearpy.filters import NearestFilter
import sklearn.preprocessing
from ann_benchmarks.algorithms.base import BaseANN


class NearPy(BaseANN):
    def __init__(self, metric, n_bits, hash_counts):
        self._n_bits = n_bits
        self._hash_counts = hash_counts
        self._metric = metric
        self._filter = NearestFilter(10)
        self.name = 'NearPy(n_bits=%d, hash_counts=%d)' % (
            self._n_bits, self._hash_counts)

    def fit(self, X):
        hashes = []

        for k in range(self._hash_counts):
            nearpy_rbp = nearpy.hashes.RandomBinaryProjections(
                'rbp_%d' % k, self._n_bits)
            hashes.append(nearpy_rbp)

        if self._metric == 'euclidean':
            dist = nearpy.distances.EuclideanDistance()
            self._nearpy_engine = nearpy.Engine(
                X.shape[1],
                lshashes=hashes,
                distance=dist)
        else:  # Default (angular) = Cosine distance
            self._nearpy_engine = nearpy.Engine(
                X.shape[1],
                lshashes=hashes,
                vector_filters=[self._filter])

        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        for i, x in enumerate(X):
            self._nearpy_engine.store_vector(x, i)

    def query(self, v, n):
        # XXX: This feels like an unpleasant hack, but it's not clear how to do
        # better without making changes to NearPy
        self._filter.N = n
        if self._metric == 'angular':
            v = sklearn.preprocessing.normalize([v], axis=1, norm='l2')[0]
        return [y for x, y, z in self._nearpy_engine.neighbours(v)]
