import pyvsag
import numpy as np
import json
import struct
from ..base.module import BaseANN

class Vsag(BaseANN):
    def __init__(self, metric, dim, method_param):
        self._metric = {"euclidean": "l2", "angular": "ip"}[metric]

        self._params = dict()
        self._params["M"] = method_param["M"]
        self._params["efc"] = method_param["ef_construction"]

        self._params["sq"] = -1
        if method_param["use_int8"] != 0:
            self._params["sq"] = method_param["use_int8"]

        if "alpha" in method_param:
            self._params["a"] = method_param["alpha"]
        else:
            self._params["a"] = 1.0

        if "rs" in method_param:
            self._params["rs"] = method_param["rs"]
        else:
            self._params["rs"] = 1

        self.name = "vsag (%s)" % (self._params)
        self._ef = 0
        print(self._params)

    def fit(self, X):
        index_params = {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": len(X[0]),
            "hnsw": {
                "max_degree": self._params["M"],
                "ef_construction": self._params["efc"],
                "ef_search": self._params["efc"],
                "max_elements": len(X),
                "use_static": False,
                "sq_num_bits": self._params["sq"],
                "alpha": self._params["a"],
                "redundant_rate": self._params["rs"]
            }
        }
        print(index_params)
        self._index = pyvsag.Index("hnsw", json.dumps(index_params))
        if self._metric == "ip":
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
        self._index.build(vectors=X,
                          ids=range(len(X)),
                          num_elements=len(X),
                          dim=len(X[0]))


    def set_query_arguments(self, ef):
        self._ef = ef
        self.name = "efs_%s_%s" % (self._ef, self._params)

    def query(self, v, n):
        search_params = {
            "hnsw": {
                "ef_search": self._ef
            }
        }
        length = 1
        if self._metric == "ip":
            length = np.linalg.norm(v)
        if length == 0:
            length = 1
        ids, dists = self._index.knn_search(vector=v / length, k=n, parameters=json.dumps(search_params))
        return ids

