import numpy
import numpy as np
import puffinn

from ..base.module import BaseANN


class Puffinn(BaseANN):
    def __init__(self, metric, space=10**6, hash_function="fht_crosspolytope", hash_source="pool", hash_args=None):
        if metric not in ["jaccard", "angular"]:
            raise NotImplementedError("Puffinn doesn't support metric %s" % metric)
        self.metric = metric
        self.space = space
        self.hash_function = hash_function
        self.hash_source = hash_source
        self.hash_args = hash_args

    def fit(self, X):
        if self.metric == "angular":
            dimensions = len(X[0])
        else:
            dimensions = 0
            for x in X:
                dimensions = max(dimensions, max(x) + 1)

        if self.hash_args:
            self.index = puffinn.Index(
                self.metric,
                dimensions,
                self.space,
                hash_function=self.hash_function,
                hash_source=self.hash_source,
                hash_args=self.hash_args,
            )
        else:
            self.index = puffinn.Index(
                self.metric, dimensions, self.space, hash_function=self.hash_function, hash_source=self.hash_source
            )
        for i, x in enumerate(X):
            if self.metric == "jaccard" and x.dtype == np.bool_:
                x = np.flatnonzero(x)
            x = x.tolist()
            self.index.insert(x)
        self.index.rebuild()

    def set_query_arguments(self, recall):
        self.recall = recall

    def query(self, v, n):
        if self.metric == "jaccard" and v.dtype == np.bool_:
            v = np.flatnonzero(v)
        v = v.tolist()
        return self.index.search(v, n, self.recall)

    def __str__(self):
        return "PUFFINN(space=%d, recall=%f, hf=%s, hashsource=%s)" % (
            self.space,
            self.recall,
            self.hash_function,
            self.hash_source,
        )
