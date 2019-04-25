from __future__ import absolute_import
import numpy
import sklearn.neighbors
from ann_benchmarks.distance import metrics as pd
from ann_benchmarks.algorithms.base import BaseANN


class BruteForce(BaseANN):
    def __init__(self, metric):
        if metric not in ('angular', 'euclidean', 'hamming'):
            raise NotImplementedError(
                "BruteForce doesn't support metric %s" % metric)
        self._metric = metric
        self.name = 'BruteForce()'

    def fit(self, X):
        metric = {'angular': 'cosine', 'euclidean': 'l2',
                  'hamming': 'hamming'}[self._metric]
        self._nbrs = sklearn.neighbors.NearestNeighbors(
            algorithm='brute', metric=metric)
        self._nbrs.fit(X)

    def query(self, v, n):
        return list(self._nbrs.kneighbors(
            [v], return_distance=False, n_neighbors=n)[0])

    def query_with_distances(self, v, n):
        (distances, positions) = self._nbrs.kneighbors(
            [v], return_distance=True, n_neighbors=n)
        return zip(list(positions[0]), list(distances[0]))


class BruteForceBLAS(BaseANN):
    """kNN search that uses a linear scan = brute force."""

    def __init__(self, metric, precision=numpy.float32):
        if metric not in ('angular', 'euclidean', 'hamming', 'jaccard'):
            raise NotImplementedError(
                "BruteForceBLAS doesn't support metric %s" % metric)
        elif metric == 'hamming' and precision != numpy.bool:
            raise NotImplementedError(
                "BruteForceBLAS doesn't support precision"
                " %s with Hamming distances" % precision)
        self._metric = metric
        self._precision = precision
        self.name = 'BruteForceBLAS()'

    def fit(self, X):
        """Initialize the search index."""
        if self._metric == 'angular':
            # precompute (squared) length of each vector
            lens = (X ** 2).sum(-1)
            # normalize index vectors to unit length
            X /= numpy.sqrt(lens)[..., numpy.newaxis]
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
        elif self._metric == 'hamming':
            # Regarding bitvectors as vectors in l_2 is faster for blas
            X = X.astype(numpy.float32)
            # precompute (squared) length of each vector
            lens = (X ** 2).sum(-1)
            self.index = numpy.ascontiguousarray(X, dtype=numpy.float32)
            self.lengths = numpy.ascontiguousarray(lens, dtype=numpy.float32)
        elif self._metric == 'euclidean':
            # precompute (squared) length of each vector
            lens = (X ** 2).sum(-1)
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
            self.lengths = numpy.ascontiguousarray(lens, dtype=self._precision)
        elif self._metric == 'jaccard':
            self.index = X
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"

    def query(self, v, n):
        return [index for index, _ in self.query_with_distances(v, n)]

    def query_with_distances(self, v, n):
        """Find indices of `n` most similar vectors from the index to query
        vector `v`."""

        if self._metric != 'jaccard':
            # use same precision for query as for index
            v = numpy.ascontiguousarray(v, dtype=self.index.dtype)

        # HACK we ignore query length as that's a constant
        # not affecting the final ordering
        if self._metric == 'angular':
            # argmax_a cossim(a, b) = argmax_a dot(a, b) / |a||b| = argmin_a -dot(a, b)  # noqa
            dists = -numpy.dot(self.index, v)
        elif self._metric == 'euclidean':
            # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab  # noqa
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        elif self._metric == 'hamming':
            # Just compute hamming distance using euclidean distance
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        elif self._metric == 'jaccard':
            dists = [pd[self._metric]['distance'](v, e) for e in self.index]
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"
        # partition-sort by distance, get `n` closest
        nearest_indices = numpy.argpartition(dists, n)[:n]
        indices = [idx for idx in nearest_indices if pd[self._metric]
                   ["distance_valid"](dists[idx])]

        def fix(index):
            ep = self.index[index]
            ev = v
            return (index, pd[self._metric]['distance'](ep, ev))
        return map(fix, indices)
