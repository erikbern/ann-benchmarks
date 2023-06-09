"""
ann-benchmarks interfaces for elastiknn: https://github.com/alexklibisz/elastiknn
Uses the elastiknn python client
To install a local copy of the client, run `pip install --upgrade -e /path/to/elastiknn/client-python/`
To monitor the Elasticsearch JVM using Visualvm, add `ports={ "8097": 8097 }` to the `containers.run` call in runner.py.
"""
import logging
from time import perf_counter, sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from elastiknn.api import Vec
from elastiknn.models import ElastiknnModel

from ..base.module import BaseANN

# Mute the elasticsearch logger.
# By default, it writes an INFO statement for every request.
logging.getLogger("elasticsearch").setLevel(logging.WARN)


def es_wait():
    print("Waiting for elasticsearch health endpoint...")
    req = Request("http://localhost:9200/_cluster/health?wait_for_status=yellow&timeout=1s")
    for i in range(60):
        try:
            res = urlopen(req)
            if res.getcode() == 200:
                print(f"Elasticsearch is ready: {res.read().decode()}")
                return
        except URLError:
            pass
        sleep(1)
    raise RuntimeError("Failed to connect to local elasticsearch")


def dealias_metric(metric: str) -> str:
    mlower = metric.lower()
    if mlower == "euclidean":
        return "l2"
    elif mlower == "angular":
        return "cosine"
    else:
        return mlower


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
        if self.metric in {"jaccard", "hamming"}:
            return self.model.fit(self._handle_sparse(X), shards=1)[0]
        else:
            return self.model.fit(X, shards=1)

    def query(self, q, n):
        if self.metric in {"jaccard", "hamming"}:
            return self.model.kneighbors(self._handle_sparse([q]), n)[0]
        else:
            return self.model.kneighbors(np.expand_dims(q, 0), n)[0]

    def batch_query(self, X, n):
        if self.metric in {"jaccard", "hamming"}:
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
        self.query_params = dict()
        self.batch_res = None
        # Parameters that help us determine whether and when to give up on a particular set of parameters.
        # https://github.com/erikbern/ann-benchmarks/issues/178
        self.sum_query_dur = 0
        self.num_queries = 0
        es_wait()

    def fit(self, X):
        print(f"{self.name_prefix}: indexing {len(X)} vectors")
        res = self.model.fit(X, shards=1)
        print(f"{self.name_prefix}: finished indexing {len(X)} vectors")
        return res

    def set_query_arguments(self, candidates: int, probes: int):
        # This gets called when starting a new batch of queries.
        # Update the name and model's query parameters based on the given params.
        self.name = f"{self.name_prefix}_candidates={candidates}_probes={probes}"
        self.model.set_query_params(dict(candidates=candidates, probes=probes))
        # Reset the counters.
        self.num_queries = 0
        self.sum_query_dur = 0

    def query(self, q, n):

        t0 = perf_counter()
        res = self.model.kneighbors(np.expand_dims(q, 0), n, return_similarity=False)[0]
        dur = perf_counter() - t0

        # Maintain state and figure out if we should give up.
        self.sum_query_dur += dur
        self.num_queries += 1
        if self.num_queries > 500 and self.num_queries / self.sum_query_dur < 50:
            raise Exception(
                "Throughput after 500 queries is less than 50 q/s. Giving up to avoid wasteful computation."
            )
        elif res[-2:].sum() < 0:
            raise Exception(f"Model returned fewer than {n} neighbors. Giving up to avoid wasteful computation.")
        else:
            return res

    def batch_query(self, X, n):
        self.batch_res = self.model.kneighbors(X, n)

    def get_batch_results(self):
        return self.batch_res
