from typing import Any, Dict, Union
import numpy as np
import deglib

from ..base.module import BaseANN


_METRIC_MAP = {
    "euclidean": (deglib.Metric.FP32_L2, deglib.Metric.Int8_L2, deglib.Metric.FP16_L2),
    "angular": (deglib.Metric.FP32_InnerProduct, deglib.Metric.Int8_InnerProduct, deglib.Metric.FP16_InnerProduct),
    "cosine": (deglib.Metric.FP32_InnerProduct, deglib.Metric.Int8_InnerProduct, deglib.Metric.FP16_InnerProduct),
    "ip": (deglib.Metric.FP32_InnerProduct, deglib.Metric.Int8_InnerProduct, deglib.Metric.FP16_InnerProduct),
    "normalized": (deglib.Metric.FP32_InnerProduct, deglib.Metric.Int8_InnerProduct, deglib.Metric.FP16_InnerProduct),
}


class DEG(BaseANN):
    """
    Dynamic Exploration Graph (DEG) supporting FP32 search and INT8 quantized search (QG) with FP16 reranking.
    """

    def __init__(
        self,
        metric: str,
        method_param: Any = None,
        k: int = 30,
        opt_target: str = "LowLID",
        prune_non_rng: bool = False,
        dtype: str = "float32",
    ):
        self.metric = metric.lower().strip()
        if self.metric not in _METRIC_MAP:
            raise ValueError(f"Unsupported metric '{self.metric}'. Choose from: {list(_METRIC_MAP.keys())}")

        if isinstance(method_param, dict):
            self.k = int(method_param.get("k", method_param.get("edges_per_vertex", k)))
            self.opt_target = method_param.get("opt_target", opt_target)
            self.prune_non_rng = bool(method_param.get("prune_non_rng", method_param.get("remove_edges", prune_non_rng)))
            self.dtype = str(method_param.get("dtype", method_param.get("search_dtype", dtype))).lower().strip()
        elif method_param is not None:
            self.k = int(method_param)
            self.opt_target = opt_target
            self.prune_non_rng = bool(prune_non_rng)
            self.dtype = str(dtype).lower().strip()
        else:
            self.k = int(k)
            self.opt_target = opt_target
            self.prune_non_rng = bool(prune_non_rng)
            self.dtype = str(dtype).lower().strip()

        self.is_quantized = self.dtype in ("int8", "qg", "quantized")
        self.search_eps = 0.1
        self.rerank_size_factor = 1.0

        self.base_metric, self.int8_metric, self.fp16_metric = _METRIC_MAP[self.metric]

        if isinstance(self.opt_target, deglib.builder.OptimizationTarget):
            self.opt_enum = self.opt_target
        elif isinstance(self.opt_target, int):
            self.opt_enum = deglib.builder.OptimizationTarget(self.opt_target)
        else:
            self.opt_enum = deglib.builder.OptimizationTarget[str(self.opt_target)]

        self.graph = None
        self.searcher = None
        self.quantizer = None
        self.original_features_fp16 = None
        self.rerank_space_fp16 = None
        self._update_name()

    def _update_name(self):
        if self.is_quantized:
            self.name = (
                f"deg(dtype=int8, k={self.k}, opt={self.opt_target}, prune_rng={self.prune_non_rng}, "
                f"rerank={self.rerank_size_factor}, eps={self.search_eps})"
            )
        else:
            self.name = f"deg(dtype=fp32, k={self.k}, opt={self.opt_target}, prune_rng={self.prune_non_rng}, eps={self.search_eps})"

    def fit(self, X: np.ndarray):
        """Builds DEG graph, optionally quantizes to INT8 with FP16 reranking, and prepares C++ searcher."""
        if self.metric in ("angular", "cosine", "ip", "normalized"):
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
        X = np.ascontiguousarray(X, dtype=np.float32)
        dims = X.shape[1]

        # 1. FLAS 1D Pre-sorting
        sorted_indices = deglib.optimization.presort(
            X,
            metric=self.base_metric,
            threads=1,
        )

        # 2. Build graph in FP32
        graph = deglib.builder.build_from_data(
            data=X[sorted_indices],
            labels=sorted_indices,
            edges_per_vertex=self.k,
            metric=self.base_metric,
            seed=7,
            optimization_target=self.opt_enum,
            thread_count=1,
        )

        # 3. Optional MRNG edge pruning
        if self.prune_non_rng:
            deglib.optimization.prune_non_rng_edges(graph, num_threads=1)

        # 4. Searcher setup according to dtype (FP32 vs INT8-QG)
        if self.is_quantized:
            self.original_features_fp16 = deglib.distances.floats_to_fp16(X)
            self.rerank_space_fp16 = deglib.FloatSpace.create(dim=dims, metric=self.fp16_metric)

            self.quantizer = deglib.optimization.make_scalar_quantizer_int8(X)
            int8_features = self.quantizer.quantize(X, num_threads=1)
            target_space = deglib.FloatSpace.create(dim=dims, metric=self.int8_metric)
            self.graph = graph.to_readonly(target_space, int8_features)

            self.searcher = deglib.search.create_searcher(
                graph=self.graph,
                quantizer=self.quantizer,
                refine_space=self.rerank_space_fp16,
                refine_data=self.original_features_fp16,
            )
        else:
            self.graph = graph.to_readonly()
            self.searcher = deglib.search.create_searcher(graph=self.graph)

    def set_query_arguments(self, *args, **kwargs):
        """Sets query-time parameters: (search_eps) for FP32 or (rerank_size_factor, search_eps) for INT8."""
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, dict):
                self.rerank_size_factor = float(arg.get("rerank_size_factor", arg.get("rerank_factor", self.rerank_size_factor)))
                self.search_eps = float(arg.get("eps", arg.get("search_eps", self.search_eps)))
            else:
                self.search_eps = float(arg)
        elif len(args) >= 2:
            self.rerank_size_factor = float(args[0])
            self.search_eps = float(args[1])

        if "search_eps" in kwargs:
            self.search_eps = float(kwargs["search_eps"])
        elif "eps" in kwargs:
            self.search_eps = float(kwargs["eps"])
        if "rerank_size_factor" in kwargs:
            self.rerank_size_factor = float(kwargs["rerank_size_factor"])
        elif "rerank_factor" in kwargs:
            self.rerank_size_factor = float(kwargs["rerank_factor"])

        self._update_name()

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        """Single query search on 1 thread via C++ searcher."""
        if self.metric in ("angular", "cosine", "ip", "normalized"):
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm

        v_arr = np.ascontiguousarray(v, dtype=np.float32)

        if self.is_quantized:
            return self.searcher.search(
                v_arr,
                k=n,
                eps=self.search_eps,
                rerank_factor=self.rerank_size_factor,
                threads=1,
                return_distances=False,
                unsorted=True,
            )
        else:
            return self.searcher.search(
                v_arr,
                k=n,
                eps=self.search_eps,
                threads=1,
                return_distances=False,
                unsorted=True,
            )

    def done(self) -> None:
        self.graph = None
        self.searcher = None
        self.quantizer = None
        self.original_features_fp16 = None
        self.rerank_space_fp16 = None

    def freeIndex(self) -> None:
        self.done()

    def __str__(self) -> str:
        return self.name


# Backwards compatibility aliases
DegLib = DEG
QG = DEG
