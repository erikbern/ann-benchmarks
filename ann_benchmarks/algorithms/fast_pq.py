from fast_pq import FastPQ as PQ, IVF
from .base import BaseANN


class FastPQ(BaseANN):
    def __init__(self, metric, build_probes, skew_factor):
        self._build_probes = build_probes
        self._skew_factor = skew_factor
        self._metric = metric
        self._query_probes = None
        self._ivf = None

    def fit(self, X):
        n = X.shape[0]
        self._ivf = IVF(self._metric, n_clusters=int(self._skew_factor * n**0.5 + 1))
        self._ivf.fit(X).build(X, n_probes=self._build_probes)

    def set_query_arguments(self, query_probes):
        self._query_probes = query_probes

    def query(self, v, n):
        return self._ivf.query(v, k=n, n_probes=self._query_probes)

    def __str__(self):
        return f"FastPQ(metric={self._metric}, build_probes={self._build_probes}, skew_factor={self._skew_factor}, query_probes={self._query_probes})"
