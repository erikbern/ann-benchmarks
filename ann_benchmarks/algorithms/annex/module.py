import numpy as np
from ann_benchmarks.algorithms.base.module import BaseANN
import annexdb


class Annex(BaseANN):
    def __init__(self, metric, index_param):
        self._metric = metric
        self._m = index_param["M"]
        self._ef_construct = index_param.get("efConstruction", 300)
        self._sq8 = index_param.get("sq8", False)
        self._ef = 64
        self._index = None

    def fit(self, X):
        # ann-benchmarks guarantees unit-norm vectors for angular datasets,
        # so DistanceMetric::Cosine (dot product on unit vectors) is correct.
        ann_metric = {
            "angular": "cosine",
            "euclidean": "euclidean",
            "jaccard": "euclidean",
            "hamming": "euclidean",
            "dot": "dot",
        }.get(self._metric, "cosine")

        self._index = annexdb.Index.build(
            X.astype(np.float32, copy=False),
            metric=ann_metric,
            m=self._m,
            ef_construct=self._ef_construct,
            quantize=self._sq8,
        )

    def set_query_arguments(self, ef):
        self._ef = ef

    def query(self, v, n):
        ids, _ = self._index.search(
            v.astype(np.float32, copy=False),
            k=n,
            ef=self._ef,
            sq8_screen=self._sq8,
        )
        return ids.tolist()

    def batch_query(self, X, n, num_threads=1):
        ids, _ = self._index.search_batch(
            X.astype(np.float32, copy=False),
            k=n,
            ef=self._ef,
            sq8_screen=self._sq8,
            threads=num_threads,
        )
        self._res = ids

    def get_batch_results(self):
        return self._res

    def __str__(self):
        sq8_str = "+sq8" if self._sq8 else ""
        return f"Annex(M={self._m}, efC={self._ef_construct}{sq8_str}, ef={self._ef})"
