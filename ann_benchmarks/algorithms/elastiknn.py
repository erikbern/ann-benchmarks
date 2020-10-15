"""
ann-benchmarks interfaces for elastiknn: https://github.com/alexklibisz/elastiknn
Uses the elastiknn python client
To install a local copy of the client, run `pip install --upgrade -e /path/to/elastiknn/client-python/`
To monitor the Elasticsearch JVM using Visualvm, add `ports={ "8097": 8097 }` to the `containers.run` call in runner.py.
"""
from sys import stderr
from urllib.error import URLError

import numpy as np
from elastiknn.api import Vec
from elastiknn.models import ElastiknnModel
from elastiknn.utils import dealias_metric

from ann_benchmarks.algorithms.base import BaseANN

from urllib.request import Request, urlopen
from time import sleep, time


def es_wait():
    print("Waiting for elasticsearch health endpoint...")
    req = Request("http://localhost:9200/_cluster/health?wait_for_status=yellow&timeout=1s")
    for i in range(30):
        try:
            res = urlopen(req)
            if res.getcode() == 200:
                print("Elasticsearch is ready")
                return
        except URLError:
            pass
        sleep(1)
    raise RuntimeError("Failed to connec to local elasticsearch")


class Exact(BaseANN):

    def __init__(self, metric: str, dimension: int):
        self.name = f"eknn-exact-metric={metric}_dimension={dimension}"
        self.metric = metric
        self.dimension = dimension
        self.model = ElastiknnModel("exact", dealias_metric(metric))
        self.batch_res = None
        es_wait()

    def _handle_sparse(self, X):
        # convert list of lists of indices to sparse vectors.
        return [Vec.SparseBool(x, self.dimension) for x in X]

    def fit(self, X):
        if self.metric in {'jaccard', 'hamming'}:
            return self.model.fit(self._handle_sparse(X), shards=1)[0]
        else:
            return self.model.fit(X, shards=1)

    def query(self, q, n):
        if self.metric in {'jaccard', 'hamming'}:
            return self.model.kneighbors(self._handle_sparse([q]), n)[0]
        else:
            return self.model.kneighbors(np.expand_dims(q, 0), n)[0]

    def batch_query(self, X, n):
        if self.metric in {'jaccard', 'hamming'}:
            self.batch_res = self.model.kneighbors(self._handle_sparse(X), n)
        else:
            self.batch_res = self.model.kneighbors(X, n)

    def get_batch_results(self):
        return self.batch_res


class L2Lsh(BaseANN):

    def __init__(self, L: int, k: int, w: int):
        self.name_prefix = f"eknn-l2lsh-L={L}-k={k}-w={w}"
        self.name = None  # set based on query args.
        self.model = ElastiknnModel("lsh", "l2", mapping_params=dict(L=L, k=k, w=w))
        self.X_max = 1.0
        self.query_params = dict()
        self.batch_res = None
        self.num_queries = 0
        self.sum_query_dur = 0
        es_wait()

    def fit(self, X):
        self.X_max = X.max()
        return self.model.fit(X / self.X_max, shards=1)

    def set_query_arguments(self, candidates: int, probes: int):
        self.name = f"{self.name_prefix}_candidates={candidates}_probes={probes}"
        self.query_params = dict(candidates=candidates, probes=probes, limit=0.4)

    def query(self, q, n):
        # If mean latency is > 100ms after 100 queries, this means the parameter setting is bad for this dataset.
        # The results will be mediocre, so it's best to just exit instead of wasting time/money.
        if self.num_queries > 100 and self.sum_query_dur / self.num_queries > 40:
            stderr.write("Mean latency exceeds 40ms after 100 queries. Stopping to avoid wasteful computation.")
            exit(0)
        else:
            t0 = time()
            res = self.model.kneighbors(np.expand_dims(q, 0) / self.X_max, n, query_params=self.query_params)[0]
            self.sum_query_dur += (time() - t0) * 1000
            self.num_queries += 1
            return res

    def batch_query(self, X, n):
        self.batch_res = self.model.kneighbors(X, n)

    def get_batch_results(self):
        return self.batch_res
