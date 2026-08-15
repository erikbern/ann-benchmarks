import ctypes
import os

import numpy as np

from ..base.module import BaseANN

DEFAULT_LIB_PATH = "/home/app/libhann_bridge.so"


def _load_bridge():
    lib_path = os.environ.get("HANN_BRIDGE_LIB", DEFAULT_LIB_PATH)
    lib = ctypes.CDLL(lib_path)

    lib.hann_hnsw_new.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p]
    lib.hann_hnsw_new.restype = ctypes.c_int64

    lib.hann_hnsw_add_batch.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.hann_hnsw_add_batch.restype = ctypes.c_int64

    lib.hann_hnsw_set_ef.argtypes = [ctypes.c_int64, ctypes.c_int64]
    lib.hann_hnsw_set_ef.restype = ctypes.c_int64

    lib.hann_hnsw_search.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.hann_hnsw_search.restype = ctypes.c_int64

    lib.hann_hnsw_free.argtypes = [ctypes.c_int64]
    lib.hann_hnsw_free.restype = None

    return lib


class Hann(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "euclidean"}[metric]
        self.method_param = method_param
        self.lib = _load_bridge()
        self.handle = None
        self.dim = None

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, dim = X.shape
        self.dim = dim
        self.handle = self.lib.hann_hnsw_new(
            dim,
            self.method_param["M"],
            self.method_param["efConstruction"],
            self.metric.encode("ascii"),
        )
        if self.handle < 0:
            raise RuntimeError("hann: failed to create the index")
        added = self.lib.hann_hnsw_add_batch(
            self.handle, X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, dim
        )
        if added != n:
            raise RuntimeError("hann: added %d of %d vectors" % (added, n))

    def set_query_arguments(self, ef):
        if self.lib.hann_hnsw_set_ef(self.handle, ef) != 0:
            raise RuntimeError("hann: failed to set ef to %d" % ef)
        self.name = "hann (%s, 'efQuery': %s)" % (self.method_param, ef)

    def query(self, v, n):
        q = np.ascontiguousarray(v, dtype=np.float32)
        out = np.empty(n, dtype=np.int32)
        count = self.lib.hann_hnsw_search(
            self.handle,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.dim,
            n,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if count < 0:
            raise RuntimeError("hann: search failed")
        return out[:count]

    def freeIndex(self):
        if self.handle is not None:
            self.lib.hann_hnsw_free(self.handle)
            self.handle = None

    def done(self):
        self.freeIndex()
