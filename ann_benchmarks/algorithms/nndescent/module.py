import numpy as np
import nndescent
import scipy.sparse

from ..base.module import BaseANN


class NNDescent(BaseANN):
    def __init__(self, metric, index_param_dict):
        if "n_neighbors" in index_param_dict:
            self.n_neighbors = int(index_param_dict["n_neighbors"])
        else:
            self.n_neighbors = 30

        if "pruning_degree_multiplier" in index_param_dict:
            self.pruning_degree_multiplier = float(
                index_param_dict["pruning_degree_multiplier"]
            )
        else:
            self.pruning_degree_multiplier = 1.5

        if "pruning_prob" in index_param_dict:
            self.pruning_prob = float(index_param_dict["pruning_prob"])
        else:
            self.pruning_prob = 1.0

        if "leaf_size" in index_param_dict:
            self.leaf_size = int(index_param_dict["leaf_size"])

        self.is_sparse = metric in ["jaccard"]

        self.nnd_metric = {
            "angular": "dot",
            "euclidean": "euclidean",
            "hamming": "hamming",
            "jaccard": "jaccard",
        }[metric]

    def fit(self, X):
        if self.is_sparse:
            # Convert to sparse matrix format
            if type(X) == list:
                sizes = [len(x) for x in X]
                n_cols = max([max(x) for x in X]) + 1
                matrix = scipy.sparse.csr_matrix(
                    (len(X), n_cols), dtype=np.float32
                )
                matrix.indices = np.hstack(X).astype(np.int32)
                matrix.indptr = np.concatenate([[0], np.cumsum(sizes)]).astype(
                    np.int32
                )
                matrix.data = np.ones(
                    matrix.indices.shape[0], dtype=np.float32
                )
                matrix.sort_indices()
                X = matrix
            else:
                X = scipy.sparse.csr_matrix(X)

            self.query_matrix = scipy.sparse.csr_matrix(
                (1, X.shape[1]), dtype=np.float32
            )
        elif not isinstance(X, np.ndarray) or X.dtype != np.float32:
            print("Convert data to float32")
            X = np.asarray(X, dtype=np.float32)

        # nndescent uses pointers to the data. Make shure X does not change
        # outside of this scope.
        self.X = X
        self.index = nndescent.NNDescent(
            self.X,
            n_neighbors=self.n_neighbors,
            metric=self.nnd_metric,
            leaf_size=self.leaf_size,
            pruning_degree_multiplier=self.pruning_degree_multiplier,
            pruning_prob=self.pruning_prob,
            verbose=True,
        )
        # Make a dummy query to prepare the search graph.
        if self.is_sparse:
            empty_mtx = np.empty((0, X.shape[0]), dtype=np.float32)
            empty_csr = scipy.sparse.csr_matrix(empty_mtx)
            self.index.query(empty_csr, k=1, epsilon=0.1)
        else:
            empty_mtx = np.empty((0, X.shape[0]), dtype=np.float32)
            self.index.query(empty_mtx, k=1, epsilon=0.1)

    def set_query_arguments(self, epsilon=0.1):
        self.epsilon = float(epsilon)

    def query(self, v, n):
        if self.is_sparse:
            # Convert index array to sparse matrix format and query; the
            # overhead of direct conversion is high for single queries
            # (converting the entire test dataset and sending single rows is
            # better), so we just populate the required structures.
            if v.dtype == np.bool_:
                self.query_matrix.indices = np.flatnonzero(v).astype(np.int32)
            else:
                self.query_matrix.indices = v.astype(np.int32)
            size = self.query_matrix.indices.shape[0]
            self.query_matrix.indptr = np.array([0, size], dtype=np.int32)
            self.query_matrix.data = np.ones(size, dtype=np.float32)
            ind, dist = self.index.query(
                self.query_matrix, k=n, epsilon=self.epsilon
            )
        else:
            ind, dist = self.index.query(
                v.reshape(1, -1).astype("float32"), k=n, epsilon=self.epsilon
            )
        return ind[0]

    def __str__(self):
        return (
            f"NNDescent(n_neighbors={self.n_neighbors}, "
            f"pruning_mult={self.pruning_degree_multiplier:.2f}, "
            f"pruning_prob={self.pruning_prob:.3f}, "
            f"epsilon={self.epsilon:.3f}, "
            f"leaf_size={self.leaf_size:02d})"
        )
