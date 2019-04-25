from __future__ import absolute_import
import annoy
from ann_benchmarks.algorithms.base import BaseANN


class Annoy(BaseANN):
    def __init__(self, metric, n_trees):
        self._n_trees = n_trees
        self._search_k = None
        self._metric = metric

    def fit(self, X):
        self._annoy = annoy.AnnoyIndex(X.shape[1], metric=self._metric)
        for i, x in enumerate(X):
            self._annoy.add_item(i, x.tolist())
        self._annoy.build(self._n_trees)

    def set_query_arguments(self, search_k):
        self._search_k = search_k

    def query(self, v, n):
        return self._annoy.get_nns_by_vector(v.tolist(), n, self._search_k)

    def __str__(self):
        return 'Annoy(n_trees=%d, search_k=%d)' % (self._n_trees,
                                                   self._search_k)
