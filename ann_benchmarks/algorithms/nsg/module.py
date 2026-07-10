import numpy as np
from pynsg import NSG, Metric, create_graph_file
import faiss
import os
import tempfile

from ..base.module import BaseANN


class NSGLib(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": Metric.L2, "euclidean": Metric.L2}[metric]
        self.L = method_param['L']
        self.R = method_param['R']
        self.C = method_param['C']
        self.knn = method_param['knn']
        self.normalize = metric == "angular"
        self.tempdir = tempfile.TemporaryDirectory()
        self.knn_file = os.path.join(self.tempdir.name, f"knn{self.knn}.graph")


    def fit(self, X: np.array):
        if self.normalize:
            faiss.normalize_L2(X)
        # will just build knn graph with metric L2 always since the vectors will be normalized for angular metrics
        create_graph_file(filename=self.knn_file, x=X, k=200)
        print(f"created graph file at {self.knn_file}")
        assert os.path.isfile(self.knn_file), f"KNN file missing: {self.knn_file}"
        self.index = NSG(dimension=len(X[0]), num_points=len(X), metric=self.metric)
        self.index.build_index(data=X, knng_path=self.knn_file, L=self.L, R=self.R, C=self.C)
        self.index.optimize_graph(data=X)

    def set_query_arguments(self, search_L):
        self.search_L = search_L
        self.name = f"nsg (L={self.L}, R={self.R}, C={self.C}, knn={self.knn}, search_L={search_L})" 

    def query(self, q: np.array, n: int) -> np.array:
        return self.index.search_opt(queries=np.expand_dims(q, axis=0), k=n, search_L=self.search_L)[0]

    def done(self):
        self.tempdir.cleanup()

