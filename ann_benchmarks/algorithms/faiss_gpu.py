from __future__ import absolute_import
import sys
# Assumes local installation of FAISS
sys.path.append("faiss")  # noqa
import numpy
import ctypes
import faiss
from ann_benchmarks.algorithms.base import BaseANN

# Implementation based on
# https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_sift1m.py  # noqa


class FaissGPU(BaseANN):
    def __init__(self, n_bits, n_probes):
        self.name = 'FaissGPU(n_bits={}, n_probes={})'.format(
            n_bits, n_probes)
        self._n_bits = n_bits
        self._n_probes = n_probes
        self._res = faiss.StandardGpuResources()
        self._index = None

    def fit(self, X):
        X = X.astype(numpy.float32)
        self._index = faiss.GpuIndexIVFFlat(self._res, len(X[0]), self._n_bits,
                                            faiss.METRIC_L2)
        # self._index = faiss.index_factory(len(X[0]),
        #                                   "IVF%d,Flat" % self._n_bits)
        # co = faiss.GpuClonerOptions()
        # co.useFloat16 = True
        # self._index = faiss.index_cpu_to_gpu(self._res, 0,
        #                                      self._index, co)
        self._index.train(X)
        self._index.add(X)
        self._index.setNumProbes(self._n_probes)

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

    def batch_query(self, X, n):
        self.res = self._index.search(X.astype(numpy.float32), n)

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
