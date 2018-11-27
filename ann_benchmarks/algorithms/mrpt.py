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
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        self._index_autotuned = mrpt.MRPTIndex(X)
        self._index_autotuned.build_autotune_sample(target_recall = None, k = self._k, n_test = 1000)

    def set_query_arguments(self, target_recall):
        self._target_recall = target_recall
        self._index = self._index_autotuned.subset(target_recall)

    def query(self, v, n):
        if self._metric == 'angular':
            v /= numpy.linalg.norm(v)

        return self._index.ann(v)

    def __str__(self):
        return 'MRPT(target recall = %.3f)' % (self._target_recall)
