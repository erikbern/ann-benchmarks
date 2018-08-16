from __future__ import absolute_import
import hdidx
from ann_benchmarks.algorithms.base import BaseANN

class HDIdx(BaseANN):
    def __init__(self, n_bits, n_subq):
        # TODO other indexers inside hdidx
        self._index = hdidx.indexer.SHIndexer()
        self._index.set_storage()  # defaults to main memory
        self._n_subq = n_subq
        self._n_bits = n_bits

    def fit(self, X):
        self._index.build({'vals': X, 'nbits': self._n_bits, 'nsubq': self._n_subq})
        self._index.add(X)

    def query(self, v, n):
        return self._index.search(v, n)

    def __str__(self):
        return 'HdIdx(n_bits=%d, n_subq=%d)' % (self._n_bits, self._n_subq)
