import glassppy as glass
import numpy as np
from sklearn import preprocessing
import os
from ..base.module import BaseANN

def metric_mapping(_metric: str):
    _metric_type = {"angular": "IP", "euclidean": "L2"}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f"[glass] Not support metric type: {_metric}!!!")
    return _metric_type

class Glass(BaseANN):
    def __init__(self, metric, dim, method_param):
        self.metric = metric_mapping(metric)
        self.R = method_param['R']
        self.L = method_param['L']
        self.level = method_param['level']
        self.name = 'glass_(%s)' % (method_param)
        self.dir = 'glass_indices'
        self.path = f'dim_{dim}_R_{self.R}_L_{self.L}.glass'

    def fit(self, X):
        if self.metric == "IP":
            X = preprocessing.normalize(X, "l2", axis=1)
        self.d = X.shape[1]
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        if self.path not in os.listdir(self.dir):
            p = glass.Index("HNSW", dim=self.d,
                            metric=self.metric, R=self.R, L=self.L)
            g = p.build(X)
            g.save(os.path.join(self.dir, self.path))
        g = glass.Graph(os.path.join(self.dir, self.path))
        self.searcher = glass.Searcher(g, X, self.metric, self.level)
        self.searcher.optimize(1)

    def set_query_arguments(self, ef):
        self.searcher.set_ef(ef)

    def prepare_query(self, q, n):
        if self.metric == 'IP':
            q = q / np.linalg.norm(q)
        self.q = q
        self.n = n

    def run_prepared_query(self):
        self.res = self.searcher.search(
            self.q, self.n)

    def get_prepared_query_results(self):
        return self.res

    def freeIndex(self):
        del self.searcher
