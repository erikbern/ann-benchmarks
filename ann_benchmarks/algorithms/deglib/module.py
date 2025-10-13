import sys
import time

import deglib
import numpy as np

from ..base.module import BaseANN


def build_from_data(
        data, edges_per_vertex = 30,
        metric = deglib.Metric.L2, optimization_target = deglib.builder.OptimizationTarget.StreamingData,
        extend_k = 60, extend_eps = 0.1,
        improve_k = 30, improve_eps = 0.001, max_path_length = 5,
        swap_tries = 0, additional_swap_tries = 0, remove_edges = True
):
    print('edges_per_vertex', edges_per_vertex)
    print('metric', metric)
    print('optimization_target', optimization_target)
    print('extend_k', extend_k)
    print('extend_eps', extend_eps)
    print('improve_k', improve_k)
    print('improve_eps', improve_eps)
    print('max_path_length', max_path_length)
    print('swap_tries', swap_tries)
    print('additional_swap_tries', additional_swap_tries)

    graph = deglib.graph.SizeBoundedGraph.create_empty(data.shape[0], data.shape[1], edges_per_vertex, metric)
    builder = deglib.builder.EvenRegularGraphBuilder(
        graph, optimization_target=optimization_target, extend_k=extend_k, extend_eps=extend_eps, improve_k=improve_k,
        improve_eps=improve_eps, max_path_length=max_path_length, swap_tries=swap_tries,
        additional_swap_tries=additional_swap_tries
    )
    builder.set_thread_count(1)
    labels = np.arange(data.shape[0], dtype=np.uint32)
    builder.add_entry(labels, data)
    builder.build()

    if remove_edges:
        graph.remove_non_mrng_edges()

    return graph


class DegLib(BaseANN):
    def __init__(self, metric, method_params):
        self.metric = {"angular": deglib.Metric.InnerProduct, "euclidean": deglib.Metric.L2}[metric]
        self.method_params = method_params
        self.query_params = {}
        self.name = None
        self.index = None

    def fit(self, X):
        # fix optimization target datatype
        build_params = {}
        for key, value in self.method_params.items():
            if key == 'optimization_target':
                value = deglib.builder.OptimizationTarget(value)
            build_params[key] = value

        if self.metric == deglib.Metric.InnerProduct:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.index = build_from_data(X, **build_params, metric=self.metric)

    def set_query_arguments(self, params):
        self.query_params = params
        self.name = "deglib (buildParams: {}, queryParams: {})".format(self.method_params, self.query_params)

    def query(self, v, n):
        v = np.expand_dims(v, axis=0)

        if self.metric == deglib.Metric.InnerProduct:
            v /= np.linalg.norm(v)

        if v.dtype != np.float32:
            v = v.astype(np.float32)

        return self.index.search(v, k=n, **self.query_params)[0][0]

    def freeIndex(self):
        del self.index
