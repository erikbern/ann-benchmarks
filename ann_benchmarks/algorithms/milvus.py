from __future__ import absolute_import
import numpy
import pyknowhere
import sklearn.preprocessing
from ann_benchmarks.algorithms.base import BaseANN


def metric_mapping(_metric: str):
    _metric_type = {'angular': 'cosine', 'euclidean': 'l2'}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f"[Milvus] Not support metric type: {_metric}!!!")
    return _metric_type


class Milvus(BaseANN):
    """
    Needs `__AVX512F__` flag to run, otherwise the results are incorrect.
    Support HNSW index type
    """

    def __init__(self, metric, dim, index_param):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(self._metric)
        self._index_m = index_param.get("M", None)
        self._index_ef = index_param.get("efConstruction", None)
        self._search_ef = None

        self.client = None

    def fit(self, X):
        self.client = pyknowhere.Index(
            self._metric_type, self._dim, len(X), self._index_m, self._index_ef)
        self.client.add(X, numpy.arange(len(X)))

    def set_query_arguments(self, ef):
        self._search_ef = ef
        self.client.set_param(ef)

    def query(self, v, n):
        return self.client.search(v, k=n)

    def __str__(self):
        return f"Milvus(Knowhere)(index_M:{self._index_m},index_ef:{self._index_ef},search_ef={self._search_ef})"
