from __future__ import absolute_import
import sys
# Assumes local installation of FAISS
sys.path.append("faiss")  # noqa
import numpy
import ctypes
import cupy
import pylibraft
from pylibraft.neighbors import ivf_pq
from ann_benchmarks.algorithms.base import BaseANN

pylibraft.config.set_output_as(lambda device_ndarray: device_ndarray.copy_to_host())


# Implementation based on
# https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_sift1m.py  # noqa


"""
    >>> import cupy as cp

    >>> from pylibraft.common import Handle
    >>> from pylibraft.neighbors import ivf_pq

    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000

    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> handle = Handle()
    >>> index_params = ivf_pq.IndexParams(
    ...     n_lists=1024,
    ...     metric="l2_expanded",
    ...     pq_dim=10)
    >>> index = ivf_pq.build(index_params, dataset, handle=handle)

    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> distances, neighbors = ivf_pq.search(ivf_pq.SearchParams(), index,
    ...                                      queries, k, handle=handle)

    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)

    >>> # pylibraft functions are often asynchronous so the
    >>> # handle needs to be explicitly synchronized
    >>> handle.sync()"""

class RAFTIVFPQ(BaseANN):
    def __init__(self, n_list):
        self.name = 'RAFTIVFPQ(n_list={})'.format(
        n_list)
        self._n_list = n_list
        self._index = None

    def fit(self, X):
        X = X.astype(numpy.float32)
        X = cupy.asarray(X)

        print("Called fit on %s with n_probes=%s" % (X, self._n_list))

        index_params = ivf_pq.IndexParams(n_lists=self._n_list,
                                          add_data_on_build=True,
                                          metric="l2_expanded")

        try:
            print("Running algo...")
            self._index = ivf_pq.build(index_params, X)
            printf("Done running: " + str(self._index))
        except Exception as e:
            print(str(e))

        print("Done.")


    def query(self, v, n):
        return [label for label, _ in self.query_with_distances(v, n)]

    def query_with_distances(self, v, n):
        print("Called query_with_distances on %s, %s" % (v, n))
        v = v.astype(numpy.float32).reshape(1, -1)

        search_params = ivf_pq.SearchParams(n_probes=self._n_probes)
        distances, labels = ivf_pq.search(search_params, self._index, v, n)
        r = []
        for l, d in zip(labels, distances):
            if l != -1:
                r.append((l, d))
        return r

    def batch_query(self, X, n):
        X = cupy.asarray(X.astype(numpy.float32))

        print("INDEX: " + str(self._index))
        print("Called batch_query on %s, %s" % (X, n))
        search_params = ivf_pq.SearchParams(n_probes=self._n_probes)
        self.res = ivf_pq.search(search_params, self._index, X, n)
        print("Done.")

    def get_batch_results(self):
        print("Called get_batch_results")
        D, L = self.res

        print("D %s, L %s" % (D, L))
        res = []
        for l, d in zip(L, D):
            res.append(l)
        return res

    def set_query_arguments(self, n_probe):
        self._n_probes = n_probe

