import os

import nmslib
import numpy as np
import scipy.sparse

from ...constants import INDEX_DIR
from ..base.module import BaseANN


def sparse_matrix_to_str(matrix):
    result = []
    matrix = matrix.tocsr()
    matrix.sort_indices()
    for row in range(matrix.shape[0]):
        arr = [k for k in matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]]
        result.append(" ".join([str(k) for k in arr]))
    return result


def dense_vector_to_str(vector):
    if vector.dtype == np.bool_:
        indices = np.flatnonzero(vector)
    else:
        indices = vector
    result = " ".join([str(k) for k in indices])
    return result


class NmslibReuseIndex(BaseANN):
    @staticmethod
    def encode(d):
        return ["%s=%s" % (a, b) for (a, b) in d.items()]

    def __init__(self, metric, method_name, index_param, query_param):
        self._nmslib_metric = {"angular": "cosinesimil", "euclidean": "l2", "jaccard": "jaccard_sparse"}[metric]
        self._method_name = method_name
        self._save_index = False
        self._index_param = NmslibReuseIndex.encode(index_param)
        if query_param is not False:
            self._query_param = NmslibReuseIndex.encode(query_param)
            self.name = "Nmslib(method_name={}, index_param={}, " "query_param={})".format(
                self._method_name, self._index_param, self._query_param
            )
        else:
            self._query_param = None
            self.name = "Nmslib(method_name=%s, index_param=%s)" % (self._method_name, self._index_param)

        self._index_name = os.path.join(
            INDEX_DIR, "nmslib_%s_%s_%s" % (self._method_name, metric, "_".join(self._index_param))
        )

        d = os.path.dirname(self._index_name)
        if not os.path.exists(d):
            os.makedirs(d)

    def fit(self, X):
        if self._method_name == "vptree":
            # To avoid this issue: terminate called after throwing an instance
            # of 'std::runtime_error'
            # what():  The data size is too small or the bucket size is too
            # big. Select the parameters so that <total # of records> is NOT
            # less than <bucket size> * 1000
            # Aborted (core dumped)
            self._index_param.append("bucketSize=%d" % min(int(len(X) * 0.0005), 1000))

        if self._nmslib_metric == "jaccard_sparse":
            self._index = nmslib.init(
                space=self._nmslib_metric,
                method=self._method_name,
                data_type=nmslib.DataType.OBJECT_AS_STRING,
            )
            if type(X) == list:
                sizes = [len(x) for x in X]
                n_cols = max([max(x) for x in X]) + 1
                sparse_matrix = scipy.sparse.csr_matrix((len(X), n_cols), dtype=np.float32)
                sparse_matrix.indices = np.hstack(X).astype(np.int32)
                sparse_matrix.indptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
                sparse_matrix.data = np.ones(sparse_matrix.indices.shape[0], dtype=np.float32)
                sparse_matrix.sort_indices()
            else:
                sparse_matrix = scipy.sparse.csr_matrix(X)
            string_data = sparse_matrix_to_str(sparse_matrix)
            self._index.addDataPointBatch(string_data)
        else:
            self._index = nmslib.init(space=self._nmslib_metric, method=self._method_name)
            self._index.addDataPointBatch(X)

        if os.path.exists(self._index_name):
            print("Loading index from file")
            self._index.loadIndex(self._index_name)
        else:
            self._index.createIndex(self._index_param)
            if self._save_index:
                self._index.saveIndex(self._index_name)
        if self._query_param is not None:
            self._index.setQueryTimeParams(self._query_param)

    def set_query_arguments(self, ef):
        if self._method_name == "hnsw" or self._method_name == "sw-graph":
            self._index.setQueryTimeParams(["efSearch=%s" % (ef)])

    def query(self, v, n):
        if self._nmslib_metric == "jaccard_sparse":
            v_string = dense_vector_to_str(v)
            ids, distances = self._index.knnQuery(v_string, n)
        else:
            ids, distances = self._index.knnQuery(v, n)
        return ids

    def batch_query(self, X, n):
        if self._nmslib_metric == "jaccard_sparse":
            sparse_matrix = scipy.sparse.csr_matrix(X)
            string_data = sparse_matrix_to_str(sparse_matrix)
            self.res = self._index.knnQueryBatch(string_data, n)
        else:
            self.res = self._index.knnQueryBatch(X, n)

    def get_batch_results(self):
        return [x for x, _ in self.res]
