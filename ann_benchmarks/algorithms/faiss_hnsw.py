from __future__ import absolute_import
import os
import faiss
import numpy as np
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.base import BaseANN


class FaissHNSW(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = metric
        self.method_param = method_param
        self.name = 'faiss (%s)' % (self.method_param)

    def fit(self, X):
        self.index = faiss.IndexHNSWFlat(len(X[0]),self.method_param["M"])
        self.index.hnsw.efConstruction = self.method_param["efConstruction"]
        self.index.verbose = True
        if(self.metric == 'angular'):
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        self.index.add(X)
        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, ef):
        self.index.hnsw.efSearch = ef

    def query(self, v, n):
        D, I = self.index.search(np.expand_dims(v,axis=0), n)
        return I[0]

    def freeIndex(self):
        del self.p
