from __future__ import absolute_import
import n2
from ann_benchmarks.algorithms.base import BaseANN


class N2(BaseANN):
    def __init__(self, metric, max_m0):
        self._max_m0 = max_m0
        self._search_k = None
        self._metric = metric

    def fit(self, X):
        self._n2 = n2.HnswIndex(X.shape[1], self._metric)
        for x in X:
            self._n2.add_data(x.tolist())
        self._n2.build(self._max_m0)

    def set_query_arguments(self, search_k):
        self._search_k = search_k

    def query(self, v, n):
        return self._n2.search_by_vector(v.tolist(), n, self._search_k)

    def __str__(self):
        return 'N2(max_m0=%d, search_k=%d)' % (self._max_m0, self._search_k)
