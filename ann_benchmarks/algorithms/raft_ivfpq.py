from __future__ import absolute_import
import sys
# Assumes local installation of FAISS
sys.path.append("faiss")  # noqa
import numpy
import ctypes
import cupy
import pylibraft
from pylibraft.neighbors import ivf_pq, refine
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
    def __init__(self, n_list, pq_bits, pq_dim, dtype):
        self.name = 'RAFTIVFPQ(n_list={})'.format(
        n_list)
        self._n_list = n_list
        self._index = None
        self._dataset = None
        self._k_refine = None
        self._pq_bits = pq_bits
        self._pq_dim = pq_dim
        self._dt = numpy.dtype(dtype)

    def fit(self, X):
        X = cupy.asarray(X).astype(self._dt)

        index_params = ivf_pq.IndexParams(n_lists=self._n_list,
                                          pq_bits=self._pq_bits,
                                          pq_dim=self._pq_dim,
                                          add_data_on_build=True,
                                          metric="l2_expanded")

        self._index = ivf_pq.build(index_params, X)
        self._dataset = X

    def query(self, v, k):
        v = cupy.asarray(v.reshape(1, -1).astype(self._dt))
        search_params = ivf_pq.SearchParams(n_probes=self._n_probes, lut_dtype=self._lut_dtype)

        k_refine = self._k_refine if self._k_refine is not None else k
        D, L = ivf_pq.search(search_params, self._index, v, k_refine)

        if self._k_refine is not None:
            D, L = refine(self._dataset, v, cupy.asarray(L), k=k)

        return cupy.asarray(L).flatten().get()

    def query_with_distances(self, v, n):

        v = cupy.asarray(v.astype(self._dt).reshape(1, -1))

        search_params = ivf_pq.SearchParams(n_probes=self._n_probes)
        distances, labels = ivf_pq.search(search_params, self._index, v, n)
        r = []
        for l, d in zip(labels[0], distances[0]):
            r.append((l, d))
        return r

    def batch_query(self, X, n):
        X = cupy.asarray(X.astype(numpy.byte))
        search_params = ivf_pq.SearchParams(n_probes=self._n_probes, lut_dtype=self._lut_dtype)

        k_refine = self._k_refine if self._k_refine is not None else n
        D, L = ivf_pq.search(search_params, self._index, X, k_refine)

        self.res = (D, L)

        if self._k_refine is not None:
            self.res = refine(self._dataset, X, cupy.asarray(L), k=n)

    def get_batch_results(self):
        _, L = self.res
        return L

    def set_query_arguments(self, n_probe, k_refine, lut_dtype):
        print("Setting refine: %s" % k_refine)
        self._n_probes = min(n_probe, self._n_list)
        self._lut_dtype = numpy.dtype(lut_dtype) if lut_dtype != "ubyte" else numpy.uint8
        if k_refine > 0:
            self._k_refine = k_refine

