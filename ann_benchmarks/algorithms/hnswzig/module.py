"""ann-benchmarks wrapper for hnsw-zig via ctypes."""

import ctypes
import os
import numpy as np
import tempfile
from ..base.module import BaseANN

# Locate shared library relative to this file
_LIB_DIR = os.path.join(os.path.dirname(__file__), "lib")
_LIB_NAME = "libhnsw_zig.so"


def _load_lib():
    path = os.path.join(_LIB_DIR, _LIB_NAME)
    lib = ctypes.CDLL(path)

    # hnsw_init(max_nodes: u32, ef_construction: u32, dim: u32) -> *anyopaque
    lib.hnsw_init.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.hnsw_init.restype = ctypes.c_void_p

    # hnsw_free(handle: *anyopaque)
    lib.hnsw_free.argtypes = [ctypes.c_void_p]
    lib.hnsw_free.restype = None

    # hnsw_build(handle, vectors, labels, n, num_threads) -> bool
    lib.hnsw_build.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.hnsw_build.restype = ctypes.c_bool

    # hnsw_reorder(handle) -> bool
    lib.hnsw_reorder.argtypes = [ctypes.c_void_p]
    lib.hnsw_reorder.restype = ctypes.c_bool

    # hnsw_search(handle, query, k, ef_search, result_ids) -> u32
    lib.hnsw_search.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.hnsw_search.restype = ctypes.c_uint32

    # hnsw_save / hnsw_load
    lib.hnsw_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hnsw_save.restype = ctypes.c_bool
    lib.hnsw_load.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    lib.hnsw_load.restype = ctypes.c_void_p

    lib.hnsw_num_nodes.argtypes = [ctypes.c_void_p]
    lib.hnsw_num_nodes.restype = ctypes.c_uint32

    return lib


class HnswZig(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._M = method_param.get("M", 16)
        self._ef_construction = method_param.get("efConstruction", 200)
        self._ef_search = 100
        self._handle = None
        self._lib = _load_lib()
        self._result_buf = None

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        self._dim = d

        self._handle = self._lib.hnsw_init(n, self._ef_construction, d)
        if not self._handle:
            raise RuntimeError("hnsw_init failed")

        labels = np.arange(n, dtype=np.uint64)
        ok = self._lib.hnsw_build(
            self._handle,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            labels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            n,
            1,  # single-threaded build for fair comparison
        )
        if not ok:
            raise RuntimeError("hnsw_build failed")

        ok = self._lib.hnsw_reorder(self._handle)
        if not ok:
            raise RuntimeError("hnsw_reorder failed")

        # Pre-allocate result buffer
        self._result_buf = (ctypes.c_uint32 * 1000)()

    def set_query_arguments(self, ef):
        self._ef_search = ef

    def query(self, v, n):
        v = np.ascontiguousarray(v, dtype=np.float32)
        count = self._lib.hnsw_search(
            self._handle,
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
            self._ef_search,
            self._result_buf,
        )
        return np.array(self._result_buf[:count], dtype=np.uint32)

    def done(self):
        if self._handle:
            self._lib.hnsw_free(self._handle)
            self._handle = None

    def __str__(self):
        return f"hnsw-zig(M={self._M}, ef_c={self._ef_construction}, ef={self._ef_search})"
