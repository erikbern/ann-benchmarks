from __future__ import absolute_import
import sys
sys.path.append("install/lib-faiss")
import numpy
import ctypes
import faiss
from ann_benchmarks.algorithms.base import BaseANN

class FaissLSH(BaseANN):
    def __init__(self, n_bits):
        self.name = 'FaissLSH(n_bits={})'.format(n_bits)
        self._n_bits = n_bits
        self._index = None

    def fit(self, X):
        X = X.astype(numpy.float32)
        self._index = faiss.IndexLSH(len(X[0]), self._n_bits)
        self._index.train(X)
        self._index.add(X)

    def query(self, v, n):
        return [label for label, _ in self.query_with_distances(v, n)]

    def query_with_distances(self, v, n):
        v = v.astype(numpy.float32).reshape(1, -1)
        distances, labels = self._index.search(v, n)
        r = []
        for l, d in zip(labels[0], distances[0]):
            if l != -1:
                r.append((l, d))
        return r

    def use_threads(self):
        return False

import sklearn

class FaissIVF(BaseANN):
    def __init__(self, metric, n_list, n_probe):
        self.name = 'FaissIVF(n_list=%d, n_probe=%d)' % (n_list, n_probe)
        self._n_list = n_list
        self._n_probe = n_probe
        self._metric = metric

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        self.quantizer = faiss.IndexFlatL2(X.shape[1])
        index = faiss.IndexIVFFlat(self.quantizer, X.shape[1], self._n_list, faiss.METRIC_L2)
        index.train(X)
        index.add(X)
        index.nprobe = self._n_probe
        self._index = index

    def query(self, v, n):
        if self._metric == 'angular':
            v /= numpy.linalg.norm(v)
        (dist,), (ids,) = self._index.search(v.reshape(1, -1).astype('float32'), n)
        return ids
