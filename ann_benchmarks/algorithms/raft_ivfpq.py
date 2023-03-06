from __future__ import absolute_import
import sys
# Assumes local installation of FAISS
sys.path.append("faiss")  # noqa
import numpy
import ctypes
import cupy
import rmm
import pylibraft
from pylibraft.neighbors import ivf_pq, refine
from ann_benchmarks.algorithms.base import BaseANN

#pylibraft.config.set_output_as(
#    lambda device_ndarray: device_ndarray.copy_to_host())


def get_dtype(dt_str):
    if dt_str == "float32":
        return numpy.float32
    elif dt_str == "uint8":
        return numpy.uint8
    elif dt_str == "byte":
        return numpy.byte
    elif dt_str == "float16":
        return numpy.float16


class RAFTIVFPQ(BaseANN):
    def __init__(self, metric, n_list, pq_bits, pq_dim):
        self.name = 'RAFTIVFPQ(n_list={}, pq_bits={}, pq_dim={})'.format(
        n_list, pq_bits, pq_dim)

        # Will use 8GB of memory by default. Raise this if more is needed.
        mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(),
                                       initial_pool_size=2**30,
                                       maximum_pool_size=(2**32)*2)

        self._n_list = n_list
        self._index = None
        self._dataset = None
        self._refine_ratio = 1.0
        self._pq_bits = pq_bits
        self._pq_dim = pq_dim
        self._dist_dtype = None
        self._metric = "euclidean"
        self._mr = mr

        rmm.mr.set_current_device_resource(mr)
        cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

    def fit(self, X):
        X = cupy.asarray(X)#.astype(self._dt)
        index_params = ivf_pq.IndexParams(n_lists=self._n_list,
                                          pq_bits=self._pq_bits,
                                          pq_dim=self._pq_dim,
                                          add_data_on_build=True,
                                          metric=self._metric)

        self._index = ivf_pq.build(index_params, X)
        self._dataset = X

    def query(self, v, k):
        v = cupy.asarray(v.reshape(1, -1))
        search_params = ivf_pq.SearchParams(n_probes=self._n_probes,
                                            lut_dtype=self._lut_dtype,
                                            internal_distance_dtype=self._dist_dtype)

        refine_ratio = int(self._refine_ratio*k) if self._refine_ratio > 1.0 else k
        D, L = ivf_pq.search(search_params, self._index, v, refine_ratio,
                             memory_resource=self._mr)

        if self._refine_ratio > 1.0:
            D, L = refine(self._dataset, v, cupy.asarray(L), k=k,
                          metric=self._metric)
        return cupy.asarray(L).flatten().get()

    def batch_query(self, X, k):
        X = cupy.asarray(X)#.astype(self._dt)
        search_params = ivf_pq.SearchParams(n_probes=self._n_probes,
                                            lut_dtype=self._lut_dtype,
                                            internal_distance_dtype=self._dist_dtype)

        refine_ratio = int(self._refine_ratio*k) if self._refine_ratio > 1.0 else k
        D, L = ivf_pq.search(search_params, self._index, X, refine_ratio,
                             memory_resource=self._mr)

        self.res = (D, L)

        if self._refine_ratio > 1.0:
            self.res = refine(self._dataset, X, L, k=k,
                              metric=self._metric)

    def get_batch_results(self):
        _, L = self.res
        return L.copy_to_host()

    def set_query_arguments(self, n_probe, refine_ratio, lut_dtype, dist_dtype):
        self._n_probes = min(n_probe, self._n_list)
        self._lut_dtype = get_dtype(lut_dtype)
        self._dist_dtype=get_dtype(dist_dtype)
        if refine_ratio > 0:
            self._refine_ratio = refine_ratio

