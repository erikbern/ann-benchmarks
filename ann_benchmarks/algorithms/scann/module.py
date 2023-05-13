import numpy as np
import scann

from ..base.module import BaseANN


class Scann(BaseANN):
    def __init__(self, n_leaves, avq_threshold, dims_per_block, dist):
        self.name = "scann n_leaves={} avq_threshold={:.02f} dims_per_block={}".format(
            n_leaves, avq_threshold, dims_per_block
        )
        self.n_leaves = n_leaves
        self.avq_threshold = avq_threshold
        self.dims_per_block = dims_per_block
        self.dist = dist

    def fit(self, X):
        if self.dist == "dot_product":
            spherical = True
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
        else:
            spherical = False

        self.searcher = (
            scann.scann_ops_pybind.builder(X, 10, self.dist)
            .tree(self.n_leaves, 1, training_sample_size=len(X), spherical=spherical, quantize_centroids=True)
            .score_ah(self.dims_per_block, anisotropic_quantization_threshold=self.avq_threshold)
            .reorder(1)
            .build()
        )

    def set_query_arguments(self, leaves_reorder):
        self.leaves_to_search, self.reorder = leaves_reorder

    def query(self, v, n):
        return self.searcher.search(v, n, self.reorder, self.leaves_to_search)[0]
