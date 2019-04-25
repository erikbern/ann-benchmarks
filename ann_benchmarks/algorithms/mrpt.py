from __future__ import absolute_import
import numpy
import sklearn.preprocessing
import mrpt
from ann_benchmarks.algorithms.base import BaseANN


class MRPT(BaseANN):
    def __init__(self, metric, count):
        self._metric = metric
        self._k = count

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        self._index_autotuned = mrpt.MRPTIndex(X)
        self._index_autotuned.build_autotune_sample(
            target_recall=None, k=self._k, n_test=1000)

    def set_query_arguments(self, target_recall):
        self._target_recall = target_recall
        self._index = self._index_autotuned.subset(target_recall)
        self._par = self._index.parameters()

    def query(self, v, n):
        if v.dtype != numpy.float32:
            v = v.astype(numpy.float32)
        if self._metric == 'angular':
            v = sklearn.preprocessing.normalize(
                v.reshape(1, -1), axis=1, norm='l2').flatten()
        return self._index.ann(v)

    def __str__(self):
        str_template = ('MRPT(target recall=%.3f, trees=%d, depth=%d, vote '
                        'threshold=%d, estimated recall=%.3f)')
        return str_template % (self._target_recall, self._par['n_trees'],
                               self._par['depth'], self._par['votes'],
                               self._par['estimated_recall'])
