import os

import numpy as np

from ..base.module import BaseANN
from quantal import Index


class Quantal(BaseANN):
    def __init__(self, metric, dimension, method_param):
        if metric not in ("angular", "dot"):
            raise NotImplementedError(f"quantal does not support metric {metric}")
        self._metric = metric
        self._dim = dimension
        self._ef_construction = method_param["ef_construction"]
        self._m = 128
        self._index = None
        self.name = f"quantal(ef_construction={self._ef_construction})"

    def _prep(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if self._metric == "angular":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.maximum(norms, 1e-30)
        return X

    def fit(self, X):
        X = self._prep(X)
        self._index = Index(dim=self._dim, ef_construction=self._ef_construction)
        self._index.add(X)  # auto-assigns ids 0..n-1, builds with all cores

    def set_query_arguments(self, m):
        # m is the candidate-pool size, the recall/QPS dial at query time.
        self._m = int(m)
        self.name = f"quantal(ef_construction={self._ef_construction}, m={self._m})"

    def query(self, v, n):
        v = self._prep(v.reshape(1, -1))
        return [id_ for id_, _ in self._index.search(v, k=n, m=self._m)]

    def batch_query(self, X, n):
        X = self._prep(X)
        _, ids, counts = self._index.search_batch(X, k=n, m=self._m, threads=os.cpu_count())
        self._batch = (ids, counts)

    def get_batch_results(self):
        ids, counts = self._batch
        return [row[:c] for row, c in zip(ids, counts)]

    def done(self):
        if self._index is not None:
            self._index.close()
            self._index = None
