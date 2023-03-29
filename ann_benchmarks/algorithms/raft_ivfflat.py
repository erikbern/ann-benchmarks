from __future__ import absolute_import
import sys
# Assumes local installation of FAISS
sys.path.append("faiss")  # noqa
import numpy
import ctypes
import cupy
import rmm
import pylibraft
from pylibraft.common import Handle
from pylibraft.neighbors import ivf_flat
from ann_benchmarks.algorithms.base import BaseANN


def get_dtype(dt_str):
    if dt_str == "float32":
        return numpy.float32
    elif dt_str == "uint8":
        return numpy.uint8
    elif dt_str == "byte":
        return numpy.byte
    elif dt_str == "float16":
        return numpy.float16


class RAFTIVFFlat(BaseANN):
    def __init__(self, metric, n_list):
        self.name = 'RAFTIVFFlat(n_list={})'.format(
        n_list)

        # Will use 8GB of memory by default. Raise this if more is needed.
        mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(),
                                       initial_pool_size=2**30,
                                       maximum_pool_size=(2**32)*2)

        self._n_list = n_list
        self._index = None
        self._dataset = None
        self._dist_dtype = None
        self._metric = "euclidean"
        self._mr = mr
        self._handle = Handle()

        rmm.mr.set_current_device_resource(mr)
        cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

    def fit(self, X):
        X = cupy.asarray(X)
        index_params = ivf_flat.IndexParams(n_lists=self._n_list,
                                          add_data_on_build=True,
                                          metric=self._metric)

        self._index = ivf_flat.build(index_params, X, handle=self._handle)
        self._dataset = X

    def query(self, v, k):
        v = cupy.asarray(v.reshape(1, -1))
        search_params = ivf_flat.SearchParams(n_probes=self._n_probes,
                                            internal_distance_dtype=self._dist_dtype)

        D, L = ivf_flat.search(search_params, self._index, v, k,
                               memory_resource=self._mr, handle=self._handle)

        return cupy.asarray(L).flatten().get()

    def batch_query(self, X, k):
        X = cupy.asarray(X)
        search_params = ivf_flat.SearchParams(n_probes=self._n_probes,
                                            internal_distance_dtype=self._dist_dtype)

        self.res = ivf_flat.search(search_params, self._index, X, k,
                             memory_resource=self._mr, handle=self._handle)

    def get_batch_results(self):
        _, L = self.res
        return L.copy_to_host()

    def set_query_arguments(self, n_probe, dist_dtype):
        self._n_probes = min(n_probe, self._n_list)
        self._dist_dtype=get_dtype(dist_dtype)

