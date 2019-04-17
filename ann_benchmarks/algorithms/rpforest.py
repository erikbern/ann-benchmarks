from __future__ import absolute_import
import rpforest
import numpy
from ann_benchmarks.algorithms.base import BaseANN


class RPForest(BaseANN):
    def __init__(self, leaf_size, n_trees):
        self.name = 'RPForest(leaf_size=%d, n_trees=%d)' % (leaf_size, n_trees)
        self._model = rpforest.RPForest(leaf_size=leaf_size, no_trees=n_trees)

    def fit(self, X):
        if X.dtype != numpy.double:
            X = numpy.array(X).astype(numpy.double)
        self._model.fit(X)

    def query(self, v, n):
        if v.dtype != numpy.double:
            v = numpy.array(v).astype(numpy.double)
        return self._model.query(v, n)
