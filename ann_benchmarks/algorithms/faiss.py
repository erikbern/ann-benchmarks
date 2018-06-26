from __future__ import absolute_import
import sys
sys.path.append("install/lib-faiss")
import numpy
import sklearn.preprocessing
import ctypes
import faiss
from ann_benchmarks.algorithms.base import BaseANN

class FaissLSH(BaseANN):
    def __init__(self, n_bits):
        self._n_bits = n_bits
        self._index = None
        self.name = 'FaissLSH(n_bits={})'.format(self._n_bits)

    def fit(self, X):
        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
        f = X.shape[1]
        self._index = faiss.IndexLSH(f, self._n_bits)
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


class FaissIVF(BaseANN):
    def __init__(self, metric, n_list):
        self._n_list = n_list
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
        self.index = index

    def set_query_arguments(self, n_probe):
        self._n_probe = n_probe
        self.index.nprobe = self._n_probe

    def query(self, v, n):
        if self._metric == 'angular':
            v /= numpy.linalg.norm(v)
        (dist,), (ids,) = self.index.search(v.reshape(1, -1).astype('float32'), n)
        return ids

    def batch_query(self, X, n):
        self.res = self.index.search(X.astype(numpy.float32), n)

    def get_batch_results(self):
        D, L = self.res
        res = []
        for i in range(len(D)):
            r = []
            for l, d in zip(L[i], D[i]):
                if l != -1:
                    r.append(l)
            res.append(r)
        return res

    def __str__(self):
        return 'FaissIVF(n_list=%d, n_probe=%d)' % (self._n_list, self._n_probe)
