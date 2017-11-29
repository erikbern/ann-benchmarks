from __future__ import absolute_import
import annoy
from ann_benchmarks.algorithms.base import BaseANN

class Annoy(BaseANN):
    def __init__(self, metric, n_trees, search_k):
        self._n_trees = int(n_trees)
        self._search_k = int(search_k)
        self._metric = metric
        self.name = 'Annoy(n_trees=%d, search_k=%d)' % (self._n_trees, self._search_k)

    def fit(self, X):
        self._annoy = annoy.AnnoyIndex(X.shape[1], metric=self._metric)
        self._annoy.verbose(True)
        for i, x in enumerate(X):
            self._annoy.add_item(i, x.tolist())
        self._annoy.build(self._n_trees)

    def query(self, v, n):
        return self._annoy.get_nns_by_vector(v.tolist(), n, self._search_k)
