from __future__ import absolute_import
import pyflann
import numpy
import sklearn.preprocessing
from ann_benchmarks.algorithms.base import BaseANN


class FLANN(BaseANN):
    def __init__(self, metric, target_precision):
        self._target_precision = target_precision
        self.name = 'FLANN(target_precision=%f)' % self._target_precision
        self._metric = metric

    def fit(self, X):
        self._flann = pyflann.FLANN(
            target_precision=self._target_precision,
            algorithm='autotuned', log_level='info')
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self._flann.build_index(X)

    def query(self, v, n):
        if self._metric == 'angular':
            v = sklearn.preprocessing.normalize([v], axis=1, norm='l2')[0]
        if v.dtype != numpy.float32:
            v = v.astype(numpy.float32)
        return self._flann.nn_index(v, n)[0][0]
