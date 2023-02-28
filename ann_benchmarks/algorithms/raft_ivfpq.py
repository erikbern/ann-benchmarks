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

pylibraft.config.set_output_as(
    lambda device_ndarray: device_ndarray.copy_to_host())


def get_dtype(dt_str):
    if dt_str == "float":
        return numpy.float32
    elif dt_str == "ubyte":
        return numpy.uint8
    elif dt_str == "byte":
        return numpy.byte


class RAFTIVFPQ(BaseANN):
    def __init__(self, metric, n_list, pq_bits, pq_dim, dtype):
        self.name = 'RAFTIVFPQ(n_list={}, pq_bits={}, pq_dim={}, ' \
                    'dtype={})'.format(
        n_list, pq_bits, pq_dim, dtype)
        print(metric)
        self._n_list = n_list
        self._index = None
        self._dataset = None
        self._k_refine = None
        self._pq_bits = pq_bits
        self._pq_dim = pq_dim
        self._dt = get_dtype(dtype)
        self._metric = "sqeuclidean"

    def fit(self, X):
        X = cupy.asarray(X).astype(self._dt)

        index_params = ivf_pq.IndexParams(n_lists=self._n_list,
                                          pq_bits=self._pq_bits,
                                          pq_dim=self._pq_dim,
                                          add_data_on_build=True,
                                          metric=self._metric)

        self._index = ivf_pq.build(index_params, X)
        self._dataset = X

    def query(self, v, k):
        v = cupy.asarray(v.reshape(1, -1)).astype(self._dt)
        search_params = ivf_pq.SearchParams(n_probes=self._n_probes,
                                            lut_dtype=self._lut_dtype)

        k_refine = self._k_refine if self._k_refine is not None else k
        D, L = ivf_pq.search(search_params, self._index, v, k_refine)

        if self._k_refine is not None:
            D, L = refine(self._dataset, v, cupy.asarray(L), k=k,
                          metric=self._metric)
        return cupy.asarray(L).flatten().get()

    def batch_query(self, X, k):
        X = cupy.asarray(X).astype(self._dt)
        search_params = ivf_pq.SearchParams(n_probes=self._n_probes,
                                            lut_dtype=self._lut_dtype)

        k_refine = self._k_refine+k if self._k_refine is not None else k
        D, L = ivf_pq.search(search_params, self._index, X, k_refine)

        self.res = (D, L)

        if self._k_refine is not None:
            self.res = refine(self._dataset, X, cupy.asarray(L), k=k,
                              metric=self._metric)

    def get_batch_results(self):
        _, L = self.res
        return L

    def set_query_arguments(self, n_probe, k_refine, lut_dtype):
        self._n_probes = min(n_probe, self._n_list)
        self._lut_dtype = get_dtype(lut_dtype)
        if k_refine > 0:
            self._k_refine = k_refine

