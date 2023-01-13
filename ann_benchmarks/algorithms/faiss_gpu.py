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
    def __init__(self, n_lists):
        self.name = 'FaissGPU(n_lists={})'.format(n_lists)
        self._n_lists = n_lists
        self._n_probes = None
        self._k_reorder = None
        self._res = faiss.StandardGpuResources()
        self._index = None

    def fit(self, X):
        X = X.astype(numpy.float32)
        d = X.shape[1]
        index = faiss.index_factory(d, "IVF%s,PQ%s" % (self._n_lists, d//2))

        # faster, uses more memory
        # index = faiss.index_factory(d, "IVF16384,Flat")

        co = faiss.GpuClonerOptions()

        # here we are using a 64-byte PQ, so we must set the lookup tables to
        # 16 bit float (this is due to the limited temporary memory).
        co.useFloat16 = True

        index = faiss.index_cpu_to_gpu(self._res, 0, index, co)
        index.train(X)
        index.add(X)

        index_refine = faiss.IndexRefineFlat(index, faiss.swig_ptr(X))
        self.base_index = index
        self.refine_index = index_refine

    def query(self, v, n):
        return [label for label, _ in self.query_with_distances(v, n)]

    def query_with_distances(self, v, n):
        v = v.astype(numpy.float32).reshape(1, -1)
        distances, labels = self.index.search(v, n)
        r = []
        for l, d in zip(labels[0], distances[0]):
            if l != -1:
                r.append((l, d))
        return r

    def batch_query(self, X, n):
        self.res = self.index.search(X.astype(numpy.float32), n)

    def set_query_arguments(self, n_probe, k_reorder):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self._k_reorder = k_reorder
        self.base_index.nprobe = self._n_probe
        self.refine_index.k_factor = self._k_reorder
        if self._k_reorder == 0:
            self.index = self.base_index
        else:
            self.index = self.refine_index

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
