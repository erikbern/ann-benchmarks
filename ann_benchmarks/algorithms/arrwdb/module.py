import numpy as np

from ..base.module import BaseANN


class ArrwDB(BaseANN):
    """arrwDB — Rust HNSW with AVX-512/AVX2/FMA intrinsics."""

    def __init__(self, metric, method_param):
        self._metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self._m = method_param.get("M", 32)
        self._ef_construction = method_param.get("efConstruction", 400)
        self._ef_search = 50
        self._index = None

    def fit(self, X):
        from rust_hnsw import RustFastHNSWIndex

        n, dim = X.shape
        X = np.ascontiguousarray(X, dtype=np.float32)

        self._index = RustFastHNSWIndex(
            dimension=dim,
            m=self._m,
            ef_construction=self._ef_construction,
            ef_search=self._ef_search,
            metric=self._metric,
        )

        ids = [str(i) for i in range(n)]
        self._index.build_bulk(ids, X.ravel())

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        if self._index is not None:
            self._index.set_ef_search(ef_search)
        self.name = "arrwdb (M=%d, efC=%d, efS=%d)" % (self._m, self._ef_construction, ef_search)

    def query(self, v, n):
        results = self._index.search(
            np.ascontiguousarray(v, dtype=np.float32),
            k=n,
            ef_override=self._ef_search,
        )
        return [int(vid) for vid, dist in results]

    def freeIndex(self):
        del self._index
        self._index = None
