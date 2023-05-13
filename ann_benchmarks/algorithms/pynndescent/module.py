import numpy as np
import pynndescent
import scipy.sparse

from ..base.module import BaseANN


class PyNNDescent(BaseANN):
    def __init__(self, metric, index_param_dict, n_search_trees=1):
        if "n_neighbors" in index_param_dict:
            self._n_neighbors = int(index_param_dict["n_neighbors"])
        else:
            self._n_neighbors = 30

        if "pruning_degree_multiplier" in index_param_dict:
            self._pruning_degree_multiplier = float(index_param_dict["pruning_degree_multiplier"])
        else:
            self._pruning_degree_multiplier = 1.5

        if "diversify_prob" in index_param_dict:
            self._diversify_prob = float(index_param_dict["diversify_prob"])
        else:
            self._diversify_prob = 1.0

        if "leaf_size" in index_param_dict:
            self._leaf_size = int(index_param_dict["leaf_size"])
        else:
            pass

        self._n_search_trees = int(n_search_trees)

        self._pynnd_metric = {
            "angular": "dot",
            # 'angular': 'cosine',
            "euclidean": "euclidean",
            "hamming": "hamming",
            "jaccard": "jaccard",
        }[metric]

    def fit(self, X):
        if self._pynnd_metric == "jaccard":
            # Convert to sparse matrix format
            if type(X) == list:
                sizes = [len(x) for x in X]
                n_cols = max([max(x) for x in X]) + 1
                matrix = scipy.sparse.csr_matrix((len(X), n_cols), dtype=np.float32)
                matrix.indices = np.hstack(X).astype(np.int32)
                matrix.indptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
                matrix.data = np.ones(matrix.indices.shape[0], dtype=np.float32)
                matrix.sort_indices()
                X = matrix
            else:
                X = scipy.sparse.csr_matrix(X)

            self._query_matrix = scipy.sparse.csr_matrix((1, X.shape[1]), dtype=np.float32)

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
        if self._index._is_sparse:
            # convert index array to sparse matrix format and query;
            # the overhead of direct conversion is high for single
            # queries (converting the entire test dataset and sending
            # single rows is better), so we just populate the required
            # structures.
            if v.dtype == np.bool_:
                self._query_matrix.indices = np.flatnonzero(v).astype(np.int32)
            else:
                self._query_matrix.indices = v.astype(np.int32)
            size = self._query_matrix.indices.shape[0]
            self._query_matrix.indptr = np.array([0, size], dtype=np.int32)
            self._query_matrix.data = np.ones(size, dtype=np.float32)
            ind, dist = self._index.query(self._query_matrix, k=n, epsilon=self._epsilon)
        else:
            ind, dist = self._index.query(v.reshape(1, -1).astype("float32"), k=n, epsilon=self._epsilon)
        return ind[0]

    def __str__(self):
        str_template = (
            "PyNNDescent(n_neighbors=%d, pruning_mult=%.2f, diversify_prob=%.3f, epsilon=%.3f, leaf_size=%02d)"
        )
        return str_template % (
            self._n_neighbors,
            self._pruning_degree_multiplier,
            self._diversify_prob,
            self._epsilon,
            self._leaf_size,
        )
