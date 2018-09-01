from __future__ import absolute_import
import numpy as np
from hdidx.indexer import SHIndexer
from ann_benchmarks.algorithms.base import BaseANN


class HdIdx(BaseANN):
    def __init__(self, index):
        self._index = index
        self._index.set_storage()  # defaults to main memory
        self._params = dict()  # set by sub-classes

    def fit(self, X):
        self._params['vals'] = X
        self._index.build(self._params)
        self._index.add(X)

    def query(self, v, n):
        return self._index.search(np.expand_dims(v, axis=0), n)[0][0].tolist()


class SHIdx(HdIdx):
    def __init__(self, n_bits=256):
        super(SHIdx, self).__init__(SHIndexer())
        self._params['nbits'] = n_bits

    def __str__(self):
        return 'SHIndexer_({})'.format(self._params)
