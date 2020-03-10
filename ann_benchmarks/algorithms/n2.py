from __future__ import absolute_import
import n2
from ann_benchmarks.algorithms.base import BaseANN


class N2(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._m0 = self._m * 2
        self._ef_construction = method_param['efConstruction']
        self._n_threads = 1
        self._ef_search = -1

    def fit(self, X):
        self._n2 = n2.HnswIndex(X.shape[1], self._metric)
        for x in X:
            self._n2.add_data(x)
        self._n2.build(m=self._m, max_m0=self._m0, ef_construction=self._ef_construction, n_threads=self._n_threads, graph_merging='merge_level0')

    def set_query_arguments(self, ef):
        self._ef_search = ef

    def query(self, v, n):
        return self._n2.search_by_vector(v, n, self._ef_search)

    def __str__(self):
        return "N2 (M%d_efCon%d)" % (self._m, self._ef_construction)
