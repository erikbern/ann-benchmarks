from usearch.index import Index, MetricKind
import numpy as np

from .base import BaseANN

class USearch(BaseANN):

    def __init__(self, metric: str, accuracy: str, method_param: dict):
        assert accuracy in ['f64', 'f32', 'f16', 'f8']
        assert metric in ['angular', 'euclidean']
        assert 'M' in method_param
        assert 'efConstruction' in method_param

        self._metric = {'angular': MetricKind.Cos, 'euclidean': MetricKind.L2sq}[metric]
        self._method_param = method_param
        self._accuracy = accuracy

    def __str__(self):
        connectivity = self._method_param['M']
        expansion_add = self._method_param['efConstruction']
        return f'USearch(connecitivity={connectivity}, expansion_add={expansion_add})'

    def fit(self, X):
        connectivity = self._method_param['M']
        expansion_add = self._method_param['efConstruction']

        self._index = Index(
            ndim=len(X[0]),
            metric=self._metric,
            dtype=self._accuracy,
            connectivity=connectivity,
            expansion_add=expansion_add,
            jit=True
        )
        labels = np.arange(len(X), dtype=np.longlong)
        self._index.add(labels, np.asarray(X))

    def get_memory_usage(self) -> int:
        if not hasattr(self, '_index'):
            return 0

        return self._index.memory_usage / 1024

    def set_query_arguments(self, ef: int):
        self._index.expansion_search = ef

    def freeIndex(self):
        del self._index

    def query(self, v, n):
        return self._index.search(np.expand_dims(v, axis=0), k=n)[0][0]

    def batch_query(self, X, n):
        self._batch_results = self._index.search(np.asarray(X), n)

    def get_batch_results(self):
        return self._batch_results
