from __future__ import absolute_import
import numpy
import falconn
from ann_benchmarks.algorithms.base import BaseANN

class FALCONN(BaseANN):
    def __init__(self, metric, num_bits, num_tables, num_probes = None):
        if not num_probes:
            num_probes = num_tables
        self.name = 'FALCONN(K={}, L={}, T={})'.format(num_bits, num_tables, num_probes)
        self._metric = metric
        self._num_bits = num_bits
        self._num_tables = num_tables
        self._num_probes = num_probes
        self._center = None
        self._params = None
        self._index = None
        self._buf = None

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        if self._metric == 'hamming':
            # replace all zeroes by -1
            X[X < 0.5] = -1
        if self._metric == 'angular' or self._metric == 'hamming':
            X /= numpy.linalg.norm(X, axis=1).reshape(-1,  1)
        self._center = numpy.mean(X, axis=0)
        X -= self._center
        self._params = falconn.LSHConstructionParameters()
        self._params.dimension = X.shape[1]
        self._params.distance_function = 'euclidean_squared'
        self._params.lsh_family = 'cross_polytope'
        falconn.compute_number_of_hash_functions(self._num_bits, self._params)
        self._params.l = self._num_tables
        self._params.num_rotations = 1
        self._params.num_setup_threads = 0
        self._params.storage_hash_table = 'flat_hash_table'
        self._params.seed = 95225714
        self._index = falconn.LSHIndex(self._params)
        self._index.setup(X)
        self._index.set_num_probes(self._num_probes)
        self._buf = numpy.zeros((X.shape[1],), dtype=numpy.float32)

    def query(self, v, n):
        numpy.copyto(self._buf, v)
        if self._metric == 'hamming':
            # replace all zeroes by -1
            self._buf[self._buf < 0.5] = -1
        if self._metric == 'angular' or self._metric == 'hamming':
            self._buf /= numpy.linalg.norm(self._buf)
        self._buf -= self._center
        return self._index.find_k_nearest_neighbors(self._buf, n)

    def use_threads(self):
        # See https://github.com/FALCONN-LIB/FALCONN/issues/6
        return False
