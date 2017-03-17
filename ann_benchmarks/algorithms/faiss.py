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
