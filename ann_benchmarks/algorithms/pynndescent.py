from __future__ import absolute_import
import pynndescent
from ann_benchmarks.algorithms.base import BaseANN
import numpy as np
import scipy.sparse


class PyNNDescent(BaseANN):
    def __init__(self, metric, index_param_dict, n_search_trees=1):
        if "n_neighbors" in index_param_dict:
            self._n_neighbors = int(index_param_dict["n_neighbors"])
        else:
            self._n_neighbors = 30

        if "pruning_degree_multiplier" in index_param_dict:
            self._pruning_degree_multiplier = float(
                index_param_dict["pruning_degree_multiplier"]
            )
        else:
            self._pruning_degree_multiplier = 1.5

        if "diversify_prob" in index_param_dict:
            self._diversify_prob = float(index_param_dict["diversify_prob"])
        else:
            self._diversify_prob = 1.0

        if "leaf_size" in index_param_dict:
            self._leaf_size = int(index_param_dict["leaf_size"])
        else:
            leaf_size = 32

        self._n_search_trees = int(n_search_trees)

        self._pynnd_metric = {
            "angular": "dot",
            # 'angular': 'cosine',
            "euclidean": "euclidean",
            "hamming": "hamming",
            "jaccard": "jaccard",
        }[metric]

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
        return result.tocsr()

    def _sparse_convert_for_query(self, v):
        result = scipy.sparse.csr_matrix((1, self._n_cols), dtype=np.int)
        result.indptr = np.array([0, len(v)])
        result.indices = np.array(v).astype(np.int32)
        result.data = np.ones(len(v), dtype=np.int)
        return result

    def fit(self, X):
        if self._pynnd_metric == "jaccard":
            # Convert to sparse matrix format
            X = self._sparse_convert_for_fit(X)

        self._index = pynndescent.NNDescent(
            X,
            n_neighbors=self._n_neighbors,
            metric=self._pynnd_metric,
            low_memory=True,
            leaf_size=self._leaf_size,
            pruning_degree_multiplier=self._pruning_degree_multiplier,
            diversify_prob=self._diversify_prob,
            n_search_trees=self._n_search_trees,
            compressed=True,
            verbose=True,
        )
        if hasattr(self._index, "prepare"):
            self._index.prepare()
        else:
            self._index._init_search_graph()
            if self._index._is_sparse:
                if hasattr(self._index, "_init_sparse_search_function"):
                    self._index._init_sparse_search_function()
            else:
                if hasattr(self._index, "_init_search_function"):
                    self._index._init_search_function()

    def set_query_arguments(self, epsilon=0.1):
        self._epsilon = float(epsilon)

    def query(self, v, n):
        if self._pynnd_metric == "jaccard":
            # convert index array to sparse matrix format and query
            v = self._sparse_convert_for_query(v)
            ind, dist = self._index.query(v, k=n, epsilon=self._epsilon)
        else:
            ind, dist = self._index.query(
                v.reshape(1, -1).astype("float32"), k=n, epsilon=self._epsilon
            )
        return ind[0]

    def __str__(self):
        str_template = "PyNNDescent(n_neighbors=%d, pruning_mult=%.2f, diversify_prob=%.3f, epsilon=%.3f, leaf_size=%02d)"
        return str_template % (
            self._n_neighbors,
            self._pruning_degree_multiplier,
            self._diversify_prob,
            self._epsilon,
            self._leaf_size,
        )
