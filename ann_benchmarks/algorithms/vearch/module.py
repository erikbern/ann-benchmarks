import time

import numpy as np
import vearch

from ..base.module import BaseANN


class Vearch(BaseANN):
    def query(self, v, n):
        dists, ids = self.engine.search2(v, n)
        return ids[0]

    def batch_query(self, X, n):
        self.res = self.engine.search2(X, n)

    def get_batch_results(self):
        dists, ids = self.res
        res = []
        for single_ids in ids:
            res.append(single_ids.tolist())
        return res


class VearchIndex(Vearch):
    def __init__(self, metric, nlist, ns_threshold, n_dims_block):
        self.nlist = nlist
        if metric == "euclidean":
            self.metric = "L2"
        else:
            self.metric = "InnerProduct"
        self.ns_threshold = ns_threshold
        self.n_dims_block = n_dims_block

    def __str__(self):
        return "VearchIndex(nlist=%d, n_dims_block=%d, nprobe=%d, rerank=%d)" % (
            self.nlist,
            self.n_dims_block,
            self.nprobe,
            self.rerank,
        )

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self.metric == "InnerProduct":
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        d = X.shape[1]
        self.nsubvector = int(d / self.n_dims_block)
        self.engine = vearch.Engine("files", "logs")
        table = {
            "name": "test_table",
            "engine": {
                "index_size": X.shape[0],
                "retrieval_type": "VEARCH",
                "retrieval_param": {
                    "metric_type": self.metric,
                    "ncentroids": self.nlist,
                    "nsubvector": self.nsubvector,
                    "reordering": True,
                    "ns_threshold": self.ns_threshold,
                },
            },
            "properties": {"feature": {"type": "vector", "index": True, "dimension": d, "store_type": "Mmap"}},
        }
        self.engine.create_table(table)
        self.engine.add2(X)
        indexed_num = 0
        while indexed_num != X.shape[0]:
            indexed_num = self.engine.get_status()["min_indexed_num"]
            time.sleep(0.5)

    def set_query_arguments(self, n_probe, k_rerank):
        self.nprobe, self.rerank = n_probe, k_rerank * n_probe
        self.engine.set_nprobe(self.nprobe)
        self.engine.set_rerank(self.rerank)
