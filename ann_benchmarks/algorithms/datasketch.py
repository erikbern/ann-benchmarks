from __future__ import absolute_import
from datasketch import MinHashLSHForest, MinHash
from ann_benchmarks.algorithms.base import BaseANN


class DataSketch(BaseANN):
    def __init__(self, metric, n_perm, n_rep):
        if metric not in ('jaccard'):
            raise NotImplementedError(
                "Datasketch doesn't support metric %s" % metric)
        self._n_perm = n_perm
        self._n_rep = n_rep
        self._metric = metric
        self.name = 'Datasketch(n_perm=%d, n_rep=%d)' % (n_perm, n_rep)

    def fit(self, X):
        self._index = MinHashLSHForest(num_perm=self._n_perm, l=self._n_rep)
        for i, x in enumerate(X):
            m = MinHash(num_perm=self._n_perm)
            for e in x:
                m.update(str(e))
            self._index.add(str(i), m)
        self._index.index()

    def query(self, v, n):
        m = MinHash(num_perm=self._n_perm)
        for e in v:
            m.update(str(e))
        return map(int, self._index.query(m, n))
