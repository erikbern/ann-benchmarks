import shutil
import tempfile

import numpy as np

from ..base.module import BaseANN


class SereneDB(BaseANN):
    """Ann-benchmarks wrapper for iresearch HNSW vector search via pyiresearch."""

    def __init__(self, metric, segment_max_size):
        self._metric = metric
        self._segment_max_size = int(segment_max_size)
        self._ef_search = 64
        self._index_dir = tempfile.mkdtemp(prefix="serenedb_")
        self._index = None
        self.name = f"SereneDB(seg={segment_max_size})"

    def fit(self, X: np.ndarray) -> None:
        import pyiresearch

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        n, dim = X.shape
        self._index = pyiresearch.VectorIndex(self._index_dir, dim, self._segment_max_size)

        for row_idx in range(n):
            # label = row index so that query() can return indices into X_train
            self._index.insert(row_idx, X[row_idx])

        self._index.commit()

    def set_query_arguments(self, ef_search: int) -> None:
        self._ef_search = int(ef_search)

    def query(self, v: np.ndarray, n: int):
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        results = self._index.search(v, n, self._ef_search)
        # results is list of (label, distance); label == row index into X_train
        return [int(label) for label, _dist in results]

    def done(self) -> None:
        shutil.rmtree(self._index_dir, ignore_errors=True)

    def __str__(self) -> str:
        return f"SereneDB(seg={self._segment_max_size}, ef={self._ef_search})"
