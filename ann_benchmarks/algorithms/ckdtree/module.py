from scipy.spatial import cKDTree

from ..base.module import BaseANN


class CKDTree(BaseANN):
    def __init__(self, metric, leaf_size=20):
        self._leaf_size = leaf_size
        self._metric = metric
        self.name = "CKDTree(leaf_size=%d)" % self._leaf_size

    def fit(self, X):
        self._tree = cKDTree(X, leafsize=self._leaf_size)

    def query(self, v, n):
        dist, ind = self._tree.query([v], k=n)
        return ind[0]
