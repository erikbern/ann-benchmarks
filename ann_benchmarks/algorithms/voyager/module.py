import numpy as np
import voyager

from ..base.module import BaseANN


class Voyager(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": 2, "euclidean": 0}[metric]
        self.method_param = method_param
        self.name = "voyager (%s)" % (self.method_param)

    def fit(self, X):
        self.p = voyager.Index(
            space=voyager.Space(self.metric),
            num_dimensions=len(X[0]),
            max_elements=len(X),
            ef_construction=self.method_param["efConstruction"],
            M=self.method_param["M"],
        )
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, n):
        return self.p.query(np.expand_dims(v, axis=0), k=n, query_ef=self.ef)[0][0]

    def freeIndex(self):
        del self.p
