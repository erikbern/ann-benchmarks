import ctypes
import os
import numpy as np

from ..base.module import BaseANN


class CortexHNSW(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = metric
        self.method_param = method_param
        self.engine_ptr = None
        self.dim = 0
        self.count = 0
        self.ef = 64
        self.name = f"cortex-hnsw ({method_param})"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.join(script_dir, "bin", "libcortex_ann_silicon.so")
        if not os.path.exists(so_path):
            so_path = os.path.join(script_dir, "libcortex_ann_silicon.so")

        self.lib = ctypes.CDLL(so_path)
        self.lib.cortex_ann_init.argtypes = [ctypes.c_size_t]
        self.lib.cortex_ann_init.restype = ctypes.c_void_p

        self.lib.cortex_ann_add.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        self.lib.cortex_ann_add.restype = None

        self.lib.cortex_ann_query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.cortex_ann_query.restype = None

        self.lib.cortex_ann_query_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.cortex_ann_query_batch.restype = None

        self.lib.cortex_ann_free.argtypes = [ctypes.c_void_p]
        self.lib.cortex_ann_free.restype = None

    def fit(self, X):
        data = np.ascontiguousarray(X, dtype=np.float32)
        self.count, self.dim = data.shape
        self.engine_ptr = self.lib.cortex_ann_init(self.dim)
        data_p = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.cortex_ann_add(self.engine_ptr, data_p, self.count)

    def set_query_arguments(self, ef):
        self.ef = ef
        self.name = f"cortex-hnsw ({self.method_param}, 'ef': {ef})"

    def query(self, v, n):
        query_vec = np.ascontiguousarray(v, dtype=np.float32)
        out_indices = np.empty(n, dtype=np.int32)
        self.lib.cortex_ann_query(
            self.engine_ptr,
            query_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
            out_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        return out_indices

    def batch_query(self, X, n):
        queries = np.ascontiguousarray(X, dtype=np.float32)
        num_q = queries.shape[0]
        out = np.empty((num_q, n), dtype=np.int32)
        self.lib.cortex_ann_query_batch(
            self.engine_ptr,
            queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_q,
            n,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        return out

    def freeIndex(self):
        if self.engine_ptr is not None:
            self.lib.cortex_ann_free(self.engine_ptr)
            self.engine_ptr = None

    def __del__(self):
        self.freeIndex()
