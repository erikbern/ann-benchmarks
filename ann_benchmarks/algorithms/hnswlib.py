from __future__ import absolute_import
import os
import hnswlib
import scipy.sparse
import numpy as np

from sklearn.utils.extmath import randomized_svd
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.base import BaseANN


class HnswLib(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {'angular': 'cosine', 'euclidean': 'l2', 'jaccard': 'cosine'}[metric]
        if metric == 'jaccard':
            self._sparse = True
        else:
            self._sparse = False

        self.method_param = method_param
        # print(self.method_param,save_index,query_param)
        # self.ef=query_param['ef']
        self.name = 'hnswlib (%s)' % (self.method_param)

    def _sparse_convert_for_fit(self, X):
        lil_data = []
        self._n_cols = 1
        self._n_rows = len(X)
        for i in range(self._n_rows):
            lil_data.append([1] * len(X[i]))
            if max(X[i]) + 1 > self._n_cols:
                self._n_cols = max(X[i]) + 1

        result = scipy.sparse.lil_matrix(
            (self._n_rows, self._n_cols), dtype=np.int
        )
        result.rows[:] = list(X)
        result.data[:] = lil_data
        result = result.tocsr()

        n_components = self.method_param.get('n_components', 256)
        U, Sigma, VT = randomized_svd(result, n_components)
        self._sparse_components = VT.T
        result = U * Sigma

        return result

    def _sparse_convert_for_query(self, v):
        result = scipy.sparse.csr_matrix((1, self._n_cols), dtype=np.float32)
        result.indptr = np.array([0, len(v)])
        result.indices = np.array(v).astype(np.int32)
        result.data = np.ones(len(v), dtype=np.float32)
        result = result @ self._sparse_components

        return result

    def fit(self, X):
        if self._sparse:
            # Convert to SVD reduced from sparse matrix format
            X = self._sparse_convert_for_fit(X)

        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        self.p.init_index(max_elements=len(X),
                          ef_construction=self.method_param["efConstruction"],
                          M=self.method_param["M"])
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)

    def query(self, v, n):
        if self._sparse:
            v = self._sparse_convert_for_query(v)
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]

    def freeIndex(self):
        del self.p
