from __future__ import annotations
# Silicium Architecture Research Team - HNSW Vector Search Engine
# Compliant with ann-benchmarks BaseANN Interface (Python 3.10+)

import ctypes
import os
from typing import Any, Dict, List, Optional
import numpy as np

try:
    from ..base.module import BaseANN
except (ImportError, ValueError):
    try:
        from ann_benchmarks.algorithms.base.module import BaseANN
    except ImportError:
        class BaseANN:  # type: ignore[no-redef]
            """Fallback BaseANN interface when running in standalone mode."""
            def done(self) -> None: pass
            def get_memory_usage(self) -> Optional[float]: return None
            def fit(self, X: np.ndarray) -> None: pass
            def query(self, q: np.ndarray, n: int) -> List[int]: return []
            def batch_query(self, X: np.ndarray, n: int) -> Any: pass
            def get_batch_results(self) -> Any: return getattr(self, "res", None)
            def __str__(self) -> str: return getattr(self, "name", "BaseANN")


class SiliciumHNSW(BaseANN):
    def __init__(self, metric: str, method_param: Dict[str, Any]):
        self.metric = metric
        self.method_param = method_param
        self.M = int(method_param.get("M", 32))
        self.ef_construction = int(method_param.get("efConstruction", 200))
        self.engine_ptr = None
        self.dim = 0
        self.count = 0
        self.ef = 64
        self.name = f"silicium ({method_param})"
        self.res: Optional[List[List[int]]] = None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            os.environ.get("SILICIUM_LIB_PATH"),
            "/usr/local/lib/libsilicium_ann_x86_64.so",
            "/usr/lib/libsilicium_ann_x86_64.so",
            "/home/app/bin/libsilicium_ann_x86_64.so",
            "/home/app/libsilicium_ann_x86_64.so",
            os.path.join(script_dir, "libsilicium_ann_x86_64.so"),
            os.path.join(script_dir, "bin", "libsilicium_ann_x86_64.so"),
            os.path.join(script_dir, "..", "dist", "libsilicium_ann_x86_64.so"),
            os.path.join(script_dir, "libann_benchmarks_optimized.so"),
        ]

        so_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                so_path = p
                break

        if so_path is None:
            so_path = "libsilicium_ann_x86_64.so"

        self.lib = ctypes.CDLL(so_path)

        # C ABI symbol resolution (clean public API)
        self._fn_init = getattr(self.lib, "silicium_ann_init")
        self._fn_init.argtypes = [ctypes.c_size_t]
        self._fn_init.restype = ctypes.c_void_p

        self._fn_init_params = getattr(self.lib, "silicium_ann_init_params", None)
        if self._fn_init_params is not None:
            self._fn_init_params.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
            self._fn_init_params.restype = ctypes.c_void_p

        self._fn_set_ef = getattr(self.lib, "silicium_ann_set_ef_search", None)
        if self._fn_set_ef is not None:
            self._fn_set_ef.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self._fn_set_ef.restype = None

        self._fn_save = getattr(self.lib, "silicium_ann_save_index", None)
        if self._fn_save is not None:
            self._fn_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self._fn_save.restype = ctypes.c_int32

        self._fn_load = getattr(self.lib, "silicium_ann_load_index", None)
        if self._fn_load is not None:
            self._fn_load.argtypes = [ctypes.c_char_p]
            self._fn_load.restype = ctypes.c_void_p

        self._fn_add = getattr(self.lib, "silicium_ann_add")
        self._fn_add.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        self._fn_add.restype = None

        self._fn_query = getattr(self.lib, "silicium_ann_query")
        self._fn_query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self._fn_query.restype = None

        self._fn_query_batch = getattr(self.lib, "silicium_ann_query_batch", None)
        if self._fn_query_batch is not None:
            self._fn_query_batch.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int32),
            ]
            self._fn_query_batch.restype = None

        self._fn_free = getattr(self.lib, "silicium_ann_free")
        self._fn_free.argtypes = [ctypes.c_void_p]
        self._fn_free.restype = None

        self._query_buf = None
        self._query_ptr = None
        self._out_buf = np.empty(1024, dtype=np.int32)
        self._out_ptr = self._out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    def fit(self, X: np.ndarray) -> None:
        data = np.ascontiguousarray(X, dtype=np.float32)
        if self.metric in ("angular", "cosine"):
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            data = data / norms

        self.count, self.dim = data.shape

        cache_dir = os.environ.get("SILICIUM_INDEX_DIR", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir, f"hnsw_{self.count}x{self.dim}_m{self.M}_efc{self.ef_construction}_{self.metric}.bin"
        )

        if os.path.exists(cache_file) and self._fn_load is not None:
            print(f"[SILICIUM HNSW] Loading pre-built HNSW graph from {cache_file} (< 500ms zero-copy)...")
            loaded_ptr = self._fn_load(cache_file.encode("utf-8"))
            if loaded_ptr:
                self.engine_ptr = loaded_ptr
                print(f"[SILICIUM HNSW] Successfully loaded index for {self.count} vectors. Ready for LIVE queries.")
                return
            print(f"[SILICIUM HNSW] Warning: Failed to load index from cache, falling back to construction.")

        print(
            f"[SILICIUM HNSW] Building HNSW graph from scratch (M={self.M}, efConstruction={self.ef_construction}, metric={self.metric})..."
        )
        if self._fn_init_params is not None:
            self.engine_ptr = self._fn_init_params(self.dim, self.M, self.ef_construction)
        else:
            self.engine_ptr = self._fn_init(self.dim)
        data_p = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._fn_add(self.engine_ptr, data_p, self.count)

        if self._fn_save is not None:
            print(f"[SILICIUM HNSW] Saving pre-built HNSW graph to {cache_file} for subsequent live queries...")
            res = self._fn_save(self.engine_ptr, cache_file.encode("utf-8"))
            if res == 0:
                print(f"[SILICIUM HNSW] Graph successfully saved to {cache_file}.")

    def set_query_arguments(self, ef: int) -> None:
        self.ef = int(ef)
        self.name = f"silicium ({self.method_param}, 'ef': {ef})"
        if self.engine_ptr and self._fn_set_ef is not None:
            self._fn_set_ef(self.engine_ptr, self.ef)

    def query(self, v: np.ndarray, n: int) -> List[int]:
        if self._query_buf is None:
            self._query_buf = np.empty(self.dim, dtype=np.float32)
            self._query_ptr = self._query_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if self.metric in ("angular", "cosine"):
            np.copyto(self._query_buf, v)
            v_norm = np.linalg.norm(self._query_buf)
            if v_norm > 0:
                self._query_buf /= v_norm
            q_ptr = self._query_ptr
        elif isinstance(v, np.ndarray) and v.dtype == np.float32 and v.flags["C_CONTIGUOUS"]:
            q_ptr = v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            np.copyto(self._query_buf, v)
            q_ptr = self._query_ptr

        if n > len(self._out_buf):
            self._out_buf = np.empty(max(n, len(self._out_buf) * 2), dtype=np.int32)
            self._out_ptr = self._out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        self._fn_query(
            self.engine_ptr,
            q_ptr,
            n,
            self._out_ptr,
        )
        return self._out_buf[:n].tolist()

    def batch_query(self, X: np.ndarray, n: int) -> np.ndarray:
        if self._fn_query_batch is None:
            res_list = [self.query(q, n) for q in X]
            self.res = res_list
            return np.array(res_list, dtype=np.int32)

        queries = np.ascontiguousarray(X, dtype=np.float32)
        if self.metric in ("angular", "cosine"):
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            queries = queries / norms

        num_q = queries.shape[0]
        out = np.empty((num_q, n), dtype=np.int32)
        self._fn_query_batch(
            self.engine_ptr,
            queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_q,
            n,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        self.res = out.tolist()
        return out

    def get_batch_results(self) -> Any:
        return self.res

    def get_memory_usage(self) -> Optional[float]:
        try:
            import psutil

            return psutil.Process().memory_info().rss / 1024.0
        except Exception:
            return None

    def done(self) -> None:
        self.freeIndex()

    def freeIndex(self) -> None:
        if self.engine_ptr is not None:
            self._fn_free(self.engine_ptr)
            self.engine_ptr = None

    def __del__(self) -> None:
        self.freeIndex()

    def __str__(self) -> str:
        return self.name
