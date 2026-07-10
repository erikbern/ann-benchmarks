"""ann-benchmarks adapter for vecdb (pip package: vecdbc).

vecdb is a from-scratch C vector database: HNSW graph + hand-written SIMD
distance kernels (AVX-512 on x86, NEON on ARM), with exact, quantized
(TurboQuant), and hybrid indexes. This adapter exposes the HNSW index for
Euclidean / angular benchmarks.

ann-benchmarks runs single-threaded (one saturated CPU), so search uses one
thread; ef (search beam width) is the per-query tunable.
"""
import numpy as np
from ..base.module import BaseANN

from pyvecdb import VecDB, set_threads


class Vecdbc(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param["M"]
        self._ef_construction = method_param["efConstruction"]
        self._ef = None
        self._index = None
        set_threads(1)                       # ann-benchmarks: single CPU

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if self._metric == "angular":
            # cosine == L2 on L2-normalized vectors; vecdb uses squared L2
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
        n, dim = X.shape
        self._index = VecDB(dim=dim, M=self._m,
                            ef_construction=self._ef_construction, capacity=n)
        self._index.add_bulk(np.arange(n, dtype=np.uint64), X, threads=1)

    def set_query_arguments(self, ef):
        self._ef = ef

    def query(self, v, n):
        v = np.ascontiguousarray(v, dtype=np.float32).reshape(1, -1)
        if self._metric == "angular":
            nv = np.linalg.norm(v)
            if nv > 0:
                v = v / nv
        ids, _ = self._index.search(v, k=n, ef=self._ef)
        return ids[0]

    def __str__(self):
        return f"Vecdb(M={self._m}, efC={self._ef_construction}, ef={self._ef})"
