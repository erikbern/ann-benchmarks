from __future__ import absolute_import
import numpy
import falconn
from ann_benchmarks.algorithms.base import BaseANN

class FALCONN(BaseANN):
    # See https://github.com/FALCONN-LIB/FALCONN/blob/master/src/examples/glove/glove.py
    def __init__(self, metric, num_bits, num_tables, num_probes = None):
        if not num_probes:
            num_probes = num_tables
        self._metric = metric
        self._num_bits = num_bits
        self._num_tables = num_tables
        self._num_probes = num_probes
        self._center = None
        self._params = None
        self._index = None
        self._buf = None
        self.name = 'FALCONN(K={}, L={}, T={})'.format(self._num_bits, self._num_tables, self._num_probes)

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
        self._params.distance_function = falconn.DistanceFunction.EuclideanSquared
        self._params.lsh_family = falconn.LSHFamily.CrossPolytope
        falconn.compute_number_of_hash_functions(self._num_bits, self._params)
        self._params.l = self._num_tables
        self._params.num_rotations = 1
        self._params.num_setup_threads = 0
        self._params.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        self._params.seed = 95225714
        self._index = falconn.LSHIndex(self._params)
        self._index.setup(X)
        self._query_object = self._index.construct_query_object()
        self._query_object.set_num_probes(self._num_probes)

    def query(self, v, n):
        if self._metric == 'hamming':
            # replace all zeroes by -1
            v[v < 0.5] = -1
        if self._metric == 'angular' or self._metric == 'hamming':
            v /= numpy.linalg.norm(v)
        v -= self._center
        return self._query_object.find_k_nearest_neighbors(v, n)

    def use_threads(self):
        # See https://github.com/FALCONN-LIB/FALCONN/issues/6
        return False
