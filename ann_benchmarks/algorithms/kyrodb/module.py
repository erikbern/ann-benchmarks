from __future__ import annotations

import ctypes
import os
from typing import Any, Optional

import numpy as np

try:
    from ..base.module import BaseANN
except ImportError:
    class BaseANN:
        pass


class _AnnFfi:
    _instance: Optional["_AnnFfi"] = None

    def __init__(self) -> None:
        self.lib = self._load_library()

        if not hasattr(self.lib, "kyrodb_ann_create_with_flags"):
            raise RuntimeError(
                "libkyrodb_engine is missing kyrodb_ann_create_with_flags; "
                "rebuild the benchmark library from this KyroDB tree"
            )
        self.lib.kyrodb_ann_create_with_flags.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint8,
            ctypes.c_uint32,
        ]
        self.lib.kyrodb_ann_create_with_flags.restype = ctypes.c_void_p

        self.lib.kyrodb_ann_free.argtypes = [ctypes.c_void_p]
        self.lib.kyrodb_ann_free.restype = None

        self.lib.kyrodb_ann_build_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self.lib.kyrodb_ann_build_f32.restype = ctypes.c_int

        self.lib.kyrodb_ann_query_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.kyrodb_ann_query_f32.restype = ctypes.c_int

        self.lib.kyrodb_ann_memory_bytes.argtypes = [ctypes.c_void_p]
        self.lib.kyrodb_ann_memory_bytes.restype = ctypes.c_uint64

        self.lib.kyrodb_ann_last_error_len.argtypes = []
        self.lib.kyrodb_ann_last_error_len.restype = ctypes.c_size_t

        self.lib.kyrodb_ann_last_error_copy.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.kyrodb_ann_last_error_copy.restype = ctypes.c_size_t

    @classmethod
    def instance(cls) -> "_AnnFfi":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _load_library() -> ctypes.CDLL:
        explicit = os.environ.get("KYRODB_ANN_LIB")
        candidates = []
        if explicit:
            candidates.append(explicit)

        candidates.extend(
            [
                "/usr/local/lib/libkyrodb_engine.so",
                "/usr/local/lib/libkyrodb_engine.dylib",
                os.path.join(os.path.dirname(__file__), "libkyrodb_engine.so"),
                os.path.join(os.path.dirname(__file__), "libkyrodb_engine.dylib"),
            ]
        )

        last_error: Optional[Exception] = None
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                return ctypes.CDLL(path)
            except OSError as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(f"failed to load kyrodb engine ffi library: {last_error}")
        raise RuntimeError(
            "kyrodb engine ffi library not found; set KYRODB_ANN_LIB or install libkyrodb_engine.so"
        )

    def last_error(self) -> str:
        msg_len = int(self.lib.kyrodb_ann_last_error_len())
        buf = ctypes.create_string_buffer(max(1, msg_len + 1))
        self.lib.kyrodb_ann_last_error_copy(ctypes.cast(buf, ctypes.c_void_p), len(buf))
        message = buf.value.decode("utf-8", errors="replace").strip()
        return message or "unknown kyrodb ffi error"


class KyroDB(BaseANN):
    def __init__(self, metric: str, params: dict[str, Any]):
        self.M = int(params.get("M", 16))
        self.ef_construction = int(params.get("ef_construction", 200))
        self.ef_search = int(params.get("ef_search", 50))

        self._distance_code = self._metric_to_distance_code(metric)
        self._normalize = metric.lower() in ("angular", "cosine")
        self._trusted_input = bool(params.get("trusted_input", True))

        self._ffi = _AnnFfi.instance()
        self._handle = ctypes.c_void_p(None)
        self._dimension = 0
        self._last_memory_kb = 0.0
        self._query_ids_buf = np.empty(0, dtype=np.uint32)
        self._batch_results: list[list[int]] = []

        self.name = "KyroDB"

    @staticmethod
    def _metric_to_distance_code(metric: str) -> int:
        lowered = metric.lower()
        if lowered in ("angular", "cosine"):
            return 0
        if lowered in ("euclidean", "l2"):
            return 1
        if lowered in ("inner_product", "inner-product"):
            raise ValueError(
                "KyroDB ANN-Benchmarks adapter does not support inner_product; "
                "use angular/cosine or euclidean/l2"
            )
        raise ValueError(
            f"Unsupported metric '{metric}'. Supported metrics: angular, cosine, euclidean, l2"
        )

    @staticmethod
    def _normalize_rows(arr: np.ndarray) -> np.ndarray:
        arr = np.array(arr, dtype=np.float32, order="C", copy=True)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        zero_mask = (norms < 1e-30).reshape(-1)
        if np.any(zero_mask):
            arr[zero_mask, 0] = 1e-12
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-30)
        np.divide(arr, norms, out=arr)
        return arr

    @staticmethod
    def _normalize_vector(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-30:
            out = np.zeros_like(vec)
            out[0] = 1.0
            return out
        return vec / norm

    def _destroy_handle(self) -> None:
        if self._handle.value:
            self._ffi.lib.kyrodb_ann_free(self._handle)
            self._handle = ctypes.c_void_p(None)

    def _create_handle(self, dimension: int, max_elements: int) -> None:
        self._destroy_handle()
        disable_norm_check = 1 if self._normalize else 0
        flags = 1 if self._trusted_input else 0
        handle = self._ffi.lib.kyrodb_ann_create_with_flags(
            ctypes.c_uint32(dimension),
            ctypes.c_uint64(max_elements),
            ctypes.c_uint32(self._distance_code),
            ctypes.c_uint32(self.M),
            ctypes.c_uint32(self.ef_construction),
            ctypes.c_uint8(disable_norm_check),
            ctypes.c_uint32(flags),
        )
        if not handle:
            raise RuntimeError(f"kyrodb_ann_create failed: {self._ffi.last_error()}")
        self._handle = ctypes.c_void_p(handle)
        self._dimension = dimension

    def fit(self, X: np.ndarray) -> None:
        vectors = np.asarray(X, dtype=np.float32, order="C")
        if vectors.ndim != 2:
            raise ValueError(f"expected 2D array, got shape={vectors.shape}")
        if vectors.shape[0] == 0:
            raise ValueError("fit requires at least one vector")

        if self._normalize:
            vectors = self._normalize_rows(vectors)

        rows = int(vectors.shape[0])
        cols = int(vectors.shape[1])

        self._create_handle(cols, rows)
        rc = self._ffi.lib.kyrodb_ann_build_f32(
            self._handle,
            vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(rows),
            ctypes.c_size_t(cols),
        )
        if rc != 0:
            err = self._ffi.last_error()
            self._destroy_handle()
            raise RuntimeError(f"kyrodb_ann_build_f32 failed: {err}")

        memory_bytes = int(self._ffi.lib.kyrodb_ann_memory_bytes(self._handle))
        if memory_bytes > 0:
            self._last_memory_kb = max(self._last_memory_kb, float(memory_bytes) / 1024.0)

    def set_query_arguments(self, *args: Any, **kwargs: Any) -> None:
        candidate: Any = None
        if kwargs.get("ef_search") is not None:
            candidate = kwargs.get("ef_search")
        elif args:
            candidate = args[0]
        elif kwargs.get("args"):
            candidate = kwargs["args"][0]

        if isinstance(candidate, (list, tuple)):
            if not candidate:
                raise ValueError("ef_search query argument cannot be empty")
            candidate = candidate[0]

        if candidate is not None:
            self.ef_search = int(candidate)
        self.name = f"KyroDB(M={self.M}, efConstruction={self.ef_construction}, efSearch={self.ef_search})"

    def query(self, q: np.ndarray, k: int) -> list[int]:
        if not self._handle.value:
            raise RuntimeError("query called before fit")

        kk = int(k)
        if kk <= 0:
            return []

        vec = np.asarray(q, dtype=np.float32, order="C").reshape(-1)
        if vec.size != self._dimension:
            raise ValueError(
                f"query dimension mismatch: expected {self._dimension}, got {vec.size}"
            )
        if self._normalize and not self._trusted_input:
            vec = self._normalize_vector(vec)

        if self._query_ids_buf.size < kk:
            self._query_ids_buf = np.empty(kk, dtype=np.uint32)
        out_ids = self._query_ids_buf
        out_len = ctypes.c_size_t(0)

        rc = self._ffi.lib.kyrodb_ann_query_f32(
            self._handle,
            vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(vec.size),
            ctypes.c_size_t(kk),
            ctypes.c_size_t(max(self.ef_search, kk)),
            out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.byref(out_len),
        )
        if rc != 0:
            raise RuntimeError(f"kyrodb_ann_query_f32 failed: {self._ffi.last_error()}")

        count = int(out_len.value)
        if count <= 0:
            return []
        return out_ids[:count].tolist()

    def batch_query(self, X: np.ndarray, n: int) -> None:
        queries = np.asarray(X, dtype=np.float32, order="C")
        if queries.ndim != 2:
            raise ValueError(f"expected 2D array for batch_query, got shape={queries.shape}")
        self._batch_results = [self.query(q, n) for q in queries]

    def get_batch_results(self) -> list[list[int]]:
        return self._batch_results

    def get_memory_usage(self) -> float:
        if self._handle.value:
            memory_bytes = int(self._ffi.lib.kyrodb_ann_memory_bytes(self._handle))
            if memory_bytes > 0:
                self._last_memory_kb = max(self._last_memory_kb, float(memory_bytes) / 1024.0)
                return float(memory_bytes) / 1024.0
        return self._last_memory_kb

    def done(self) -> None:
        self._destroy_handle()

    def __del__(self) -> None:
        try:
            self.done()
        except Exception:
            pass

    def __str__(self) -> str:
        return self.name
