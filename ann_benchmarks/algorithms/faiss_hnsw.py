from __future__ import absolute_import
import os
import faiss
import numpy as np
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.algorithms.faiss import Faiss


class FaissHNSW(Faiss):
    def __init__(self, metric, method_param):
        self._metric = metric
        self.method_param = method_param
        self.name = 'faiss (%s)' % (self.method_param)

    def fit(self, X):
        self.index = faiss.IndexHNSWFlat(len(X[0]), self.method_param["M"])
        self.index.hnsw.efConstruction = self.method_param["efConstruction"]
        self.index.verbose = True

        if self._metric == 'angular':
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.index.add(X)
        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, ef):
        faiss.cvar.hnsw_stats.reset()
        self.index.hnsw.efSearch = ef

    def get_additional(self):
        return {"dist_comps": faiss.cvar.hnsw_stats.ndis}

    def freeIndex(self):
        del self.p
