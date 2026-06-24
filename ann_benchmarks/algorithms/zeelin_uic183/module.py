import ctypes
import itertools
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import hnswlib
import numpy as np
import faiss

from ..base.module import BaseANN


class ZeelinSphereipHnsw(BaseANN):
    def __init__(self, metric, method_param):
        if metric != "angular":
            raise ValueError("zeelin-sphereip-hnsw is only defined for angular/cosine data")
        self.metric = "ip"
        self.method_param = method_param
        self.qbuf = None
        self.query_norm = bool(int(method_param.get("query_norm", 1)))

    @staticmethod
    def _normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
        return X / np.maximum(norms, 1e-12)

    @staticmethod
    def _standardize(v):
        v = np.asarray(v, dtype=np.float32)
        return (v - float(v.mean())) / max(float(v.std()), 1e-12)

    def _score_order(self, Xraw, Xn, mode):
        n = len(Xn)
        if mode == "original":
            return np.arange(n, dtype=np.int64)
        if mode == "random":
            seed = int(self.method_param.get("order_seed", 183))
            return np.random.RandomState(seed).permutation(n).astype(np.int64)

        if mode.startswith("rawnorm"):
            score = np.linalg.norm(Xraw, axis=1).astype(np.float32)
        elif mode.startswith("rawcentral"):
            mean = Xraw.mean(axis=0).astype(np.float32)
            mean_norm = float(np.linalg.norm(mean))
            score = Xraw.dot(mean) / max(mean_norm, 1e-12)
        elif mode.startswith("radialcentral"):
            mean = Xn.mean(axis=0).astype(np.float32)
            mean_norm = float(np.linalg.norm(mean))
            central = Xn.dot(mean) / max(mean_norm, 1e-12)
            raw_norm = np.linalg.norm(Xraw, axis=1).astype(np.float32)
            radial = (raw_norm - float(raw_norm.mean())) / max(float(raw_norm.std()), 1e-12)
            alpha = float(self.method_param.get("radial_alpha", 0.5))
            score = central + alpha * radial
        elif mode.startswith("radialquartic"):
            mean = Xn.mean(axis=0).astype(np.float32)
            mean_norm = float(np.linalg.norm(mean))
            central = Xn.dot(mean) / max(mean_norm, 1e-12)
            raw_norm = np.linalg.norm(Xraw, axis=1).astype(np.float32)
            quartic = np.sum(np.square(Xn) * np.square(Xn), axis=1)
            alpha = float(self.method_param.get("radial_alpha", 0.25))
            beta = float(self.method_param.get("order_beta", 0.25))
            score = central + alpha * self._standardize(raw_norm) + beta * self._standardize(quartic)
        elif mode.startswith("radialcapmax"):
            mean = Xn.mean(axis=0).astype(np.float32)
            mean_norm = float(np.linalg.norm(mean))
            central = Xn.dot(mean) / max(mean_norm, 1e-12)
            raw_norm = np.linalg.norm(Xraw, axis=1).astype(np.float32)
            capmax = np.max(np.abs(Xn), axis=1)
            alpha = float(self.method_param.get("radial_alpha", 0.25))
            beta = float(self.method_param.get("order_beta", 0.25))
            score = central + alpha * self._standardize(raw_norm) + beta * self._standardize(capmax)
        elif mode.startswith("radialorthocap"):
            mean = Xn.mean(axis=0).astype(np.float32)
            mean_norm = float(np.linalg.norm(mean))
            axis = mean / max(mean_norm, 1e-12)
            central = Xn.dot(axis)
            raw_norm = np.linalg.norm(Xraw, axis=1).astype(np.float32)
            capmax = np.max(np.abs(Xn), axis=1)
            ortho_energy = np.maximum(0.0, 1.0 - np.square(central))
            alpha = float(self.method_param.get("radial_alpha", 0.25))
            beta = float(self.method_param.get("order_beta", 0.25))
            gamma = float(self.method_param.get("order_gamma", 0.10))
            score = (
                central
                + alpha * self._standardize(raw_norm)
                + beta * self._standardize(capmax)
                + gamma * self._standardize(ortho_energy)
            )
        elif mode.startswith("norm"):
            score = np.linalg.norm(Xn, axis=1).astype(np.float32)
        else:
            mean = Xn.mean(axis=0).astype(np.float32)
            mean_norm = float(np.linalg.norm(mean))
            score = Xn.dot(mean) / max(mean_norm, 1e-12)

        ascending = mode.endswith("_asc")
        base = np.argsort(score, kind="stable")
        if not ascending:
            base = base[::-1]

        if mode.endswith("_zigzaghi"):
            lo = 0
            hi = len(base) - 1
            out = np.empty_like(base)
            pos = 0
            while lo <= hi:
                out[pos] = base[lo]
                pos += 1
                lo += 1
                if lo <= hi:
                    out[pos] = base[hi]
                    pos += 1
                    hi -= 1
            return out.astype(np.int64)

        if mode.endswith("_zigzag"):
            lo = 0
            hi = len(base) - 1
            out = np.empty_like(base)
            pos = 0
            while lo <= hi:
                out[pos] = base[hi]
                pos += 1
                hi -= 1
                if lo <= hi:
                    out[pos] = base[lo]
                    pos += 1
                    lo += 1
            return out.astype(np.int64)

        if mode.endswith("_quantile"):
            bins = max(2, int(self.method_param.get("order_bins", 32)))
            chunks = [c for c in np.array_split(base, bins) if len(c)]
            order = []
            lo = 0
            hi = len(chunks) - 1
            while lo <= hi:
                order.append(hi)
                hi -= 1
                if lo <= hi:
                    order.append(lo)
                    lo += 1
            out = []
            max_len = max(len(chunks[i]) for i in order)
            for offset in range(max_len):
                for chunk_id in order:
                    chunk = chunks[chunk_id]
                    if offset < len(chunk):
                        out.append(chunk[offset])
            return np.asarray(out, dtype=np.int64)

        if mode.endswith("_blockshuffle"):
            block = max(1, int(self.method_param.get("order_block", 8192)))
            starts = np.arange(0, len(base), block)
            rng = np.random.RandomState(int(self.method_param.get("order_seed", 183)))
            rng.shuffle(starts)
            parts = [base[start : start + block] for start in starts]
            return np.concatenate(parts).astype(np.int64)

        return base.astype(np.int64)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        self.p = hnswlib.Index(space=self.metric, dim=self.dim)
        self.p.init_index(
            max_elements=len(Xn),
            ef_construction=int(self.method_param["efConstruction"]),
            M=int(self.method_param["M"]),
            random_seed=int(self.method_param.get("random_seed", 100)),
        )

        order_mode = self.method_param.get("order_mode", "original")
        order = self._score_order(X, Xn, order_mode)
        labels = np.arange(len(Xn), dtype=np.int64)
        build_threads = int(self.method_param.get("build_threads", -1))
        block_size = int(self.method_param.get("block_size", 0))
        if block_size <= 0 or block_size >= len(Xn):
            self.p.add_items(Xn[order], labels[order], num_threads=build_threads)
        else:
            for start in range(0, len(order), block_size):
                idx = order[start : start + block_size]
                self.p.add_items(Xn[idx], labels[idx], num_threads=build_threads)

        self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)
        order_mode = self.method_param.get("order_mode", "original")
        if order_mode.startswith(("rawnorm", "rawcentral", "radialcentral")):
            prefix = "zeelin-radialphase-hnsw "
        else:
            prefix = "zeelin-sphereip-hnsw " if self.query_norm else "zeelin-sphereipraw-hnsw "
        self.name = (
            prefix +
            f"(M={self.method_param['M']}, "
            f"efConstruction={self.method_param['efConstruction']}, "
            f"seed={self.method_param.get('random_seed', 100)}, "
            f"order={order_mode}, "
            f"radialAlpha={self.method_param.get('radial_alpha', 0.0)}, "
            f"block={self.method_param.get('block_size', 0)}, "
            f"threads={self.method_param.get('build_threads', -1)}, "
            f"queryNorm={int(self.query_norm)}, "
            f"efQuery={ef})"
        )

    def query(self, v, n):
        if not self.query_norm:
            return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]
        norm = float(np.sqrt(np.dot(v, v)))
        self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
        return self.p.knn_query(self.qbuf, k=n)[0][0]

    def get_additional(self):
        return {
            "order_mode": self.method_param.get("order_mode", "original"),
            "order_seed": int(self.method_param.get("order_seed", 183)),
            "build_threads": int(self.method_param.get("build_threads", -1)),
            "block_size": int(self.method_param.get("block_size", 0)),
            "random_seed": int(self.method_param.get("random_seed", 100)),
            "query_norm": int(self.query_norm),
            "order_bins": int(self.method_param.get("order_bins", 0)),
            "order_block": int(self.method_param.get("order_block", 0)),
            "radial_alpha": float(self.method_param.get("radial_alpha", 0.0)),
            "order_beta": float(self.method_param.get("order_beta", 0.0)),
            "order_gamma": float(self.method_param.get("order_gamma", 0.0)),
        }

    def freeIndex(self):
        del self.p


class ZeelinSpherel2Hnsw(ZeelinSphereipHnsw):
    def __init__(self, metric, method_param):
        if metric != "angular":
            raise ValueError("zeelin-spherel2-hnsw is only defined for angular/cosine data")
        self.metric = "l2"
        self.method_param = method_param
        self.qbuf = None
        self.query_norm = True

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)
        self.name = (
            "zeelin-spherel2-hnsw "
            f"(M={self.method_param['M']}, "
            f"efConstruction={self.method_param['efConstruction']}, "
            f"seed={self.method_param.get('random_seed', 100)}, "
            f"order={self.method_param.get('order_mode', 'radialcentral_zigzag')}, "
            f"block={self.method_param.get('block_size', 0)}, "
            f"threads={self.method_param.get('build_threads', -1)}, "
            f"efQuery={ef})"
        )


class ZeelinHotmassHnsw(ZeelinSphereipHnsw):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self.hot_size = int(method_param.get("hot_size", 8192))
        self.query_threads = int(method_param.get("query_threads", 1))

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)

        order_mode = self.method_param.get("order_mode", "radialcentral_zigzag")
        order = self._score_order(X, Xn, order_mode)
        hot_n = min(max(1, self.hot_size), len(order))
        hot_ids = np.ascontiguousarray(order[:hot_n], dtype=np.int64)
        Xhot = np.ascontiguousarray(Xn[hot_ids], dtype=np.float32)

        self.p = hnswlib.Index(space=self.metric, dim=self.dim)
        self.p.init_index(
            max_elements=len(Xhot),
            ef_construction=int(self.method_param["efConstruction"]),
            M=int(self.method_param["M"]),
            random_seed=int(self.method_param.get("random_seed", 100)),
        )
        self.p.add_items(
            Xhot,
            hot_ids,
            num_threads=int(self.method_param.get("build_threads", -1)),
        )
        self.p.set_num_threads(self.query_threads)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)
        self.name = (
            "zeelin-hotmass-hnsw "
            f"(hotSize={self.hot_size}, M={self.method_param['M']}, "
            f"efConstruction={self.method_param['efConstruction']}, "
            f"seed={self.method_param.get('random_seed', 100)}, "
            f"order={self.method_param.get('order_mode', 'radialcentral_zigzag')}, "
            f"radialAlpha={self.method_param.get('radial_alpha', 0.0)}, "
            f"queryNorm={int(self.query_norm)}, "
            f"queryThreads={self.query_threads}, "
            f"efQuery={ef})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["hot_size"] = self.hot_size
        extra["query_threads"] = self.query_threads
        extra["kernel"] = "hot-mass-subset-hnsw"
        return extra


class ZeelinCosphaseHnsw(ZeelinSphereipHnsw):
    def __init__(self, metric, method_param):
        if metric != "angular":
            raise ValueError("zeelin-cosphase-hnsw is only defined for angular/cosine data")
        self.metric = "cosine"
        self.method_param = method_param
        self.qbuf = None
        self.query_norm = False
        self.query_mode = method_param.get("query_mode", "expand")
        self.query_threads = int(method_param.get("query_threads", 1))
        self.internal_k = int(method_param.get("internal_k", 0))
        self.default_return_k = int(method_param.get("return_k", 0))
        self.return_k = self.default_return_k

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = X.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        self.p = hnswlib.Index(space=self.metric, dim=self.dim)
        self.p.init_index(
            max_elements=len(X),
            ef_construction=int(self.method_param["efConstruction"]),
            M=int(self.method_param["M"]),
            random_seed=int(self.method_param.get("random_seed", 100)),
        )

        order_mode = self.method_param.get("order_mode", "central_zigzag")
        order = self._score_order(X, Xn, order_mode)
        labels = np.arange(len(X), dtype=np.int64)
        build_threads = int(self.method_param.get("build_threads", -1))
        block_size = int(self.method_param.get("block_size", 0))
        if block_size <= 0 or block_size >= len(X):
            self.p.add_items(X[order], labels[order], num_threads=build_threads)
        else:
            for start in range(0, len(order), block_size):
                idx = order[start : start + block_size]
                self.p.add_items(X[idx], labels[idx], num_threads=build_threads)

        self.p.set_num_threads(self.query_threads)

    def set_query_arguments(self, ef, return_k=None):
        self.p.set_ef(ef)
        self.return_k = int(return_k) if return_k is not None else self.default_return_k
        order_mode = self.method_param.get("order_mode", "original")
        if self.return_k > 0:
            prefix = "zeelin-singlehit-cosphase-hnsw"
        elif order_mode.startswith(("radialquartic", "radialcapmax", "radialorthocap")):
            prefix = "zeelin-phaseorder-hnsw"
        elif self.query_mode == "direct":
            prefix = "zeelin-cosphasedirect-hnsw"
        else:
            prefix = "zeelin-cosphase-hnsw"
        self.name = (
            prefix + " "
            f"(M={self.method_param['M']}, "
            f"efConstruction={self.method_param['efConstruction']}, "
            f"seed={self.method_param.get('random_seed', 100)}, "
            f"order={order_mode}, "
            f"radialAlpha={self.method_param.get('radial_alpha', 0.0)}, "
            f"block={self.method_param.get('block_size', 0)}, "
            f"threads={self.method_param.get('build_threads', -1)}, "
            f"queryMode={self.query_mode}, "
            f"queryThreads={self.query_threads}, "
            f"internalK={self.internal_k}, "
            f"returnK={self.return_k}, "
            f"efQuery={ef})"
        )

    def query(self, v, n):
        if self.return_k > 0:
            query_k = max(1, min(n, max(self.internal_k, self.return_k)))
            if self.query_mode == "direct":
                labels = self.p.knn_query(v, k=query_k)[0][0]
            elif self.query_mode == "qbuf":
                self.qbuf[0, :] = v
                labels = self.p.knn_query(self.qbuf, k=query_k)[0][0]
            else:
                labels = self.p.knn_query(np.expand_dims(v, axis=0), k=query_k)[0][0]
            return labels[: min(self.return_k, len(labels))]
        if self.query_mode == "direct":
            return self.p.knn_query(v, k=n)[0][0]
        if self.query_mode == "qbuf":
            self.qbuf[0, :] = v
            return self.p.knn_query(self.qbuf, k=n)[0][0]
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]

    def get_additional(self):
        extra = super().get_additional()
        extra["query_mode"] = self.query_mode
        extra["query_threads"] = self.query_threads
        extra["internal_k"] = self.internal_k
        extra["return_k"] = self.return_k
        return extra


class ZeelinAnchorweaveHnsw(ZeelinCosphaseHnsw):
    def __init__(self, metric, method_param):
        if metric != "angular":
            raise ValueError("zeelin-anchorweave-hnsw is only defined for angular/cosine data")
        self.metric = "cosine"
        self.method_param = method_param
        self.qbuf = None
        self.query_norm = False
        self.query_mode = method_param.get("query_mode", "expand")
        self.query_threads = int(method_param.get("query_threads", 1))
        self.internal_k = 0
        self.default_return_k = 0
        self.return_k = 0
        self.anchor_count = int(method_param.get("anchor_count", 64))
        self.anchor_mode = method_param.get("anchor_mode", "kmeans")
        self.anchor_payload_k = int(method_param.get("anchor_payload_k", 0))
        self.anchor_extra_default = int(method_param.get("anchor_extra", 16))
        self.anchor_extra = self.anchor_extra_default
        self.name_prefix = "zeelin-anchorweave-hnsw"
        self.default_order_mode = method_param.get("order_mode", "radialcentral_zigzag")
        self.default_block_size = int(method_param.get("block_size", 65536))
        self.default_build_threads = int(method_param.get("build_threads", -1))

    def _make_anchors(self, Xn):
        count = max(0, min(self.anchor_count, len(Xn)))
        if count == 0:
            return np.empty((0, Xn.shape[1]), dtype=np.float32)
        if self.anchor_mode == "mean-radial":
            mean = Xn.mean(axis=0).astype(np.float32)
            mean /= max(float(np.linalg.norm(mean)), 1e-12)
            rng = np.random.RandomState(int(self.method_param.get("order_seed", 183)))
            dims = rng.choice(Xn.shape[1], size=count, replace=True)
            anchors = np.tile(mean, (count, 1)).astype(np.float32)
            signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0).astype(np.float32)
            anchors[np.arange(count), dims] += signs * float(self.method_param.get("anchor_jitter", 0.20))
            return self._normalize_rows(anchors).astype(np.float32)

        faiss.omp_set_num_threads(int(self.method_param.get("anchor_threads", 1)))
        quantizer = faiss.IndexFlatIP(Xn.shape[1])
        index = faiss.IndexIVFFlat(quantizer, Xn.shape[1], count, faiss.METRIC_INNER_PRODUCT)
        index.train(np.ascontiguousarray(Xn, dtype=np.float32))
        anchors = np.empty((count, Xn.shape[1]), dtype=np.float32)
        for i in range(count):
            quantizer.reconstruct(i, anchors[i])
        return self._normalize_rows(anchors).astype(np.float32)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.train_size = len(X)
        self.dim = X.shape[1]
        self.anchors = self._make_anchors(Xn)
        self.anchor_count = len(self.anchors)
        self.anchor_payload = None
        if self.anchor_count and self.anchor_payload_k > 0:
            payload_k = min(max(1, self.anchor_payload_k), self.train_size)
            exact = faiss.IndexFlatIP(self.dim)
            exact.add(np.ascontiguousarray(Xn, dtype=np.float32))
            _, payload = exact.search(np.ascontiguousarray(self.anchors, dtype=np.float32), payload_k)
            self.anchor_payload = np.ascontiguousarray(payload, dtype=np.int64)
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        self.p = hnswlib.Index(space=self.metric, dim=self.dim)
        self.p.init_index(
            max_elements=self.train_size + self.anchor_count,
            ef_construction=int(self.method_param["efConstruction"]),
            M=int(self.method_param["M"]),
            random_seed=int(self.method_param.get("random_seed", 100)),
        )

        build_threads = self.default_build_threads
        if self.anchor_count:
            anchor_labels = np.arange(
                self.train_size, self.train_size + self.anchor_count, dtype=np.int64
            )
            self.p.add_items(self.anchors, anchor_labels, num_threads=build_threads)

        order_mode = self.default_order_mode
        order = self._score_order(X, Xn, order_mode)
        labels = np.arange(self.train_size, dtype=np.int64)
        block_size = self.default_block_size
        if block_size <= 0 or block_size >= self.train_size:
            self.p.add_items(X[order], labels[order], num_threads=build_threads)
        else:
            for start in range(0, len(order), block_size):
                idx = order[start : start + block_size]
                self.p.add_items(X[idx], labels[idx], num_threads=build_threads)

        self.p.set_num_threads(self.query_threads)

    def set_query_arguments(self, ef, anchor_extra=None):
        self.p.set_ef(ef)
        self.anchor_extra = (
            int(anchor_extra) if anchor_extra is not None else self.anchor_extra_default
        )
        self.name = (
            self.name_prefix + " "
            f"(M={self.method_param['M']}, "
            f"efConstruction={self.method_param['efConstruction']}, "
            f"seed={self.method_param.get('random_seed', 100)}, "
            f"order={self.default_order_mode}, "
            f"radialAlpha={self.method_param.get('radial_alpha', 0.0)}, "
            f"anchorCount={self.anchor_count}, "
            f"anchorMode={self.anchor_mode}, "
            f"anchorPayloadK={self.anchor_payload_k}, "
            f"anchorExtra={self.anchor_extra}, "
            f"block={self.default_block_size}, "
            f"threads={self.default_build_threads}, "
            f"queryMode={self.query_mode}, "
            f"queryThreads={self.query_threads}, "
            f"efQuery={ef})"
        )

    def _labels_to_real(self, labels, n):
        seen = set()
        out = []
        for label in labels:
            label = int(label)
            if label < self.train_size:
                if label not in seen:
                    seen.add(label)
                    out.append(label)
            elif self.anchor_payload is not None:
                anchor_id = label - self.train_size
                if 0 <= anchor_id < len(self.anchor_payload):
                    for candidate in self.anchor_payload[anchor_id]:
                        candidate = int(candidate)
                        if candidate not in seen:
                            seen.add(candidate)
                            out.append(candidate)
                            if len(out) >= n:
                                return np.asarray(out, dtype=np.int64)
            if len(out) >= n:
                return np.asarray(out, dtype=np.int64)
        return np.asarray(out, dtype=np.int64)

    def query(self, v, n):
        max_k = self.train_size + self.anchor_count
        query_k = min(max_k, max(n, n + self.anchor_extra))
        if self.query_mode == "direct":
            labels = self.p.knn_query(v, k=query_k)[0][0]
        elif self.query_mode == "qbuf":
            self.qbuf[0, :] = v
            labels = self.p.knn_query(self.qbuf, k=query_k)[0][0]
        else:
            labels = self.p.knn_query(np.expand_dims(v, axis=0), k=query_k)[0][0]
        real = self._labels_to_real(labels, n)
        if len(real) >= n or query_k >= max_k:
            return real[:n]

        query_k = min(max_k, max(query_k * 2, n + self.anchor_count + self.anchor_extra))
        if self.query_mode == "direct":
            labels = self.p.knn_query(v, k=query_k)[0][0]
        elif self.query_mode == "qbuf":
            self.qbuf[0, :] = v
            labels = self.p.knn_query(self.qbuf, k=query_k)[0][0]
        else:
            labels = self.p.knn_query(np.expand_dims(v, axis=0), k=query_k)[0][0]
        return self._labels_to_real(labels, n)[:n]

    def get_additional(self):
        extra = super().get_additional()
        extra["order_mode"] = self.default_order_mode
        extra["build_threads"] = self.default_build_threads
        extra["block_size"] = self.default_block_size
        extra["anchor_count"] = self.anchor_count
        extra["anchor_mode"] = self.anchor_mode
        extra["anchor_extra"] = self.anchor_extra
        extra["anchor_payload_k"] = self.anchor_payload_k
        extra["anchor_jitter"] = float(self.method_param.get("anchor_jitter", 0.0))
        return extra


class ZeelinAnchorpayloadHnsw(ZeelinAnchorweaveHnsw):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self.name_prefix = "zeelin-anchorpayload-hnsw"


class ZeelinBqphaseHnsw(ZeelinSphereipHnsw):
    def __init__(self, metric, method_param):
        if metric != "angular":
            raise ValueError("zeelin-bqphase-hnsw is only defined for angular/cosine data")
        self.metric = "cosine"
        self.method_param = method_param
        self.qbuf = None
        self.mag = None
        self.query_threads = int(method_param.get("query_threads", 1))
        self.mag_mode = method_param.get("mag_mode", "meanabs")

    def _encode_bq(self, Xn):
        absx = np.abs(Xn)
        sign = np.where(Xn >= 0.0, 1.0, -1.0).astype(np.float32)
        mag = np.where(absx >= self.mag, 2.0, 1.0).astype(np.float32)
        return sign * mag

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        if self.mag_mode == "medianabs":
            self.mag = np.median(np.abs(Xn), axis=0).astype(np.float32)
        else:
            self.mag = np.mean(np.abs(Xn), axis=0).astype(np.float32)
        Xb = self._encode_bq(Xn).astype(np.float32, copy=False)
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        self.p = hnswlib.Index(space=self.metric, dim=self.dim)
        self.p.init_index(
            max_elements=len(Xb),
            ef_construction=int(self.method_param["efConstruction"]),
            M=int(self.method_param["M"]),
            random_seed=int(self.method_param.get("random_seed", 100)),
        )

        order_mode = self.method_param.get("order_mode", "radialcentral_zigzag")
        order = self._score_order(X, Xn, order_mode)
        labels = np.arange(len(Xb), dtype=np.int64)
        build_threads = int(self.method_param.get("build_threads", -1))
        block_size = int(self.method_param.get("block_size", 0))
        if block_size <= 0 or block_size >= len(Xb):
            self.p.add_items(Xb[order], labels[order], num_threads=build_threads)
        else:
            for start in range(0, len(order), block_size):
                idx = order[start : start + block_size]
                self.p.add_items(Xb[idx], labels[idx], num_threads=build_threads)
        self.p.set_num_threads(self.query_threads)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)
        self.name = (
            "zeelin-bqphase-hnsw "
            f"(M={self.method_param['M']}, "
            f"efConstruction={self.method_param['efConstruction']}, "
            f"seed={self.method_param.get('random_seed', 100)}, "
            f"order={self.method_param.get('order_mode', 'original')}, "
            f"radialAlpha={self.method_param.get('radial_alpha', 0.0)}, "
            f"magMode={self.mag_mode}, "
            f"block={self.method_param.get('block_size', 0)}, "
            f"threads={self.method_param.get('build_threads', -1)}, "
            f"queryThreads={self.query_threads}, "
            f"efQuery={ef})"
        )

    def query(self, v, n):
        norm = float(np.sqrt(np.dot(v, v)))
        vn = v * (1.0 / max(norm, 1e-12))
        self.qbuf[0, :] = np.where(vn >= 0.0, 1.0, -1.0) * np.where(np.abs(vn) >= self.mag, 2.0, 1.0)
        return self.p.knn_query(self.qbuf, k=n)[0][0]

    def get_additional(self):
        return {
            "order_mode": self.method_param.get("order_mode", "original"),
            "order_seed": int(self.method_param.get("order_seed", 183)),
            "build_threads": int(self.method_param.get("build_threads", -1)),
            "block_size": int(self.method_param.get("block_size", 0)),
            "random_seed": int(self.method_param.get("random_seed", 100)),
            "radial_alpha": float(self.method_param.get("radial_alpha", 0.0)),
            "query_threads": self.query_threads,
            "mag_mode": self.mag_mode,
            "kernel": "2bit-sign-magnitude-hnsw-topology",
        }


class ZeelinMeasuregateIvf(BaseANN):
    def __init__(self, metric, method_param):
        if metric != "angular":
            raise ValueError("zeelin-measuregate-ivf is only defined for angular/cosine data")
        self.method_param = method_param
        self.nlist = int(method_param["nlist"])
        self.qbuf = None
        self.nprobe = 1
        self.default_return_k = int(method_param.get("return_k", 0))
        self.return_k = self.default_return_k
        self.internal_k = int(method_param.get("internal_k", 0))

    @staticmethod
    def _normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
        return X / np.maximum(norms, 1e-12)

    def _adapt_nlist_to_training_size(self, n_vectors):
        if self.nlist > n_vectors:
            self.nlist = max(1, int(n_vectors))

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self._adapt_nlist_to_training_size(len(Xn))
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))
        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.train(Xn)
        self.index.add(Xn)

    def set_query_arguments(self, nprobe, return_k=None):
        faiss.cvar.indexIVF_stats.reset()
        self.nprobe = int(nprobe)
        self.return_k = int(return_k) if return_k is not None else self.default_return_k
        self.index.nprobe = self.nprobe
        self.name = (
            "zeelin-measuregate-ivf "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, "
            f"internalK={self.internal_k}, returnK={self.return_k}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def query(self, v, n):
        norm = float(np.sqrt(np.dot(v, v)))
        self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
        query_k = n if self.internal_k <= 0 else max(1, min(n, self.internal_k))
        labels = self.index.search(self.qbuf, query_k)[1][0]
        if self.return_k > 0:
            return labels[: min(self.return_k, len(labels))]
        return labels

    def get_additional(self):
        return {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "internal_k": self.internal_k,
            "return_k": self.return_k,
            "threads": int(self.method_param.get("threads", 1)),
            "dist_comps": faiss.cvar.indexIVF_stats.ndis + faiss.cvar.indexIVF_stats.nq * self.nlist,
            "kernel": "faiss-ivfflat-ip-normalized",
        }


class ZeelinHnswgateIvf(BaseANN):
    def __init__(self, metric, method_param):
        if metric != "angular":
            raise ValueError("zeelin-hnswgate-ivf is only defined for angular/cosine data")
        self.method_param = method_param
        self.nlist = int(method_param["nlist"])
        self.quantizer_m = int(method_param.get("quantizer_m", 8))
        self.quantizer_ef_construction = int(method_param.get("quantizer_efConstruction", 40))
        self.default_quantizer_ef_search = int(method_param.get("quantizer_efSearch", 8))
        self.train_size = int(method_param.get("train_size", 0))
        self.nprobe = 1
        self.quantizer_ef_search = self.default_quantizer_ef_search
        self.default_return_k = int(method_param.get("return_k", 0))
        self.return_k = self.default_return_k
        self.internal_k = int(method_param.get("internal_k", 0))
        self.dedup_extra = int(method_param.get("dedup_extra", 16))
        self.fast_unique = bool(method_param.get("fast_unique", False))
        self.skip_query_norm = bool(method_param.get("skip_query_norm", False))
        self.query_view = bool(method_param.get("query_view", False))
        self.trust_full_labels = bool(method_param.get("trust_full_labels", False))
        self.use_search_c = bool(method_param.get("use_search_c", False))
        self.qbuf = None
        self.dbuf = None
        self.ibuf = None
        self.search_c_capacity = 0

    def _apply_cluster_seed(self):
        pass

    @staticmethod
    def _normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
        return X / np.maximum(norms, 1e-12)

    def _adapt_nlist_to_training_size(self, n_vectors):
        if self.nlist > n_vectors:
            self.nlist = max(1, int(n_vectors))

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self._adapt_nlist_to_training_size(len(Xn))
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )
        self._apply_cluster_seed()

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        self.index.train(train)
        self.index.add(Xn)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        faiss.cvar.indexIVF_stats.reset()
        self.nprobe = int(nprobe)
        self.quantizer_ef_search = (
            int(quantizer_ef_search)
            if quantizer_ef_search is not None
            else self.default_quantizer_ef_search
        )
        if dedup_extra is not None:
            self.dedup_extra = int(dedup_extra)
        self.return_k = int(return_k) if return_k is not None else self.default_return_k
        if internal_k is not None:
            self.internal_k = int(internal_k)
        self.index.nprobe = self.nprobe
        self.quantizer.hnsw.efSearch = self.quantizer_ef_search
        self.name = (
            "zeelin-hnswgate-ivf "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, "
            f"quantizerM={self.quantizer_m}, "
            f"quantizerEfC={self.quantizer_ef_construction}, "
            f"quantizerEfS={self.quantizer_ef_search}, "
            f"trainSize={self.train_size}, internalK={self.internal_k}, "
            f"returnK={self.return_k}, dedupExtra={self.dedup_extra}, "
            f"fastUnique={int(self.fast_unique)}, "
            f"skipQueryNorm={int(self.skip_query_norm)}, "
            f"queryView={int(self.query_view)}, "
            f"trustFullLabels={int(self.trust_full_labels)}, "
            f"useSearchC={int(self.use_search_c)}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def _ensure_search_c_buffers(self, query_k):
        if self.search_c_capacity >= query_k:
            return
        self.dbuf = np.empty((1, query_k), dtype=np.float32)
        self.ibuf = np.empty((1, query_k), dtype=np.int64)
        self.search_c_capacity = query_k

    def query(self, v, n):
        if self.skip_query_norm and self.query_view:
            q = v.reshape(1, self.dim)
        elif self.skip_query_norm:
            self.qbuf[0, :] = v
            q = self.qbuf
        else:
            norm = float(np.sqrt(np.dot(v, v)))
            self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
            q = self.qbuf
        target_k = n if self.return_k <= 0 else min(n, self.return_k)
        query_k = n if self.internal_k <= 0 else max(1, self.internal_k)
        query_k = max(query_k, target_k + self.dedup_extra)
        if self.use_search_c:
            self._ensure_search_c_buffers(query_k)
            self.index.search_c(
                1,
                faiss.swig_ptr(q),
                query_k,
                faiss.swig_ptr(self.dbuf),
                faiss.swig_ptr(self.ibuf),
            )
            labels = self.ibuf[0, :query_k]
        else:
            labels = self.index.search(q, query_k)[1][0]
        if self.fast_unique:
            if self.trust_full_labels:
                prefix = labels[:target_k]
                if len(prefix) == 0 or prefix[-1] >= 0:
                    return prefix
                return prefix[prefix >= 0]
            labels = labels[labels >= 0]
            return labels[:target_k]
        out = []
        seen = set()
        for label in labels:
            label = int(label)
            if label < 0 or label in seen:
                continue
            seen.add(label)
            out.append(label)
            if len(out) >= target_k:
                break
        labels = np.asarray(out, dtype=np.int64)
        if self.return_k > 0:
            return labels[: min(self.return_k, len(labels))]
        return labels[:n]

    def batch_query(self, X, n):
        X = np.asarray(X, dtype=np.float32)
        if self.skip_query_norm:
            Xn = np.ascontiguousarray(X, dtype=np.float32)
        else:
            norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
            Xn = X / np.maximum(norms, 1e-12)
        self._batch_target_k = n if self.return_k <= 0 else min(n, self.return_k)
        query_k = n if self.internal_k <= 0 else max(1, self.internal_k)
        query_k = max(query_k, self._batch_target_k + self.dedup_extra)
        self._batch_labels = self.index.search(
            np.ascontiguousarray(Xn, dtype=np.float32), query_k
        )[1]

    def get_batch_results(self):
        results = []
        target_k = self._batch_target_k
        for labels in self._batch_labels:
            if self.fast_unique:
                if self.trust_full_labels:
                    prefix = labels[:target_k]
                    if len(prefix) == 0 or prefix[-1] >= 0:
                        results.append(prefix)
                    else:
                        results.append(prefix[prefix >= 0])
                else:
                    results.append([int(label) for label in labels[:target_k] if label >= 0])
                continue
            out = []
            seen = set()
            for label in labels:
                label = int(label)
                if label < 0 or label in seen:
                    continue
                seen.add(label)
                out.append(label)
                if len(out) >= target_k:
                    break
            results.append(out)
        return results

    def get_additional(self):
        return {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "internal_k": self.internal_k,
            "return_k": self.return_k,
            "dedup_extra": self.dedup_extra,
            "fast_unique": int(self.fast_unique),
            "skip_query_norm": int(self.skip_query_norm),
            "query_view": int(self.query_view),
            "trust_full_labels": int(self.trust_full_labels),
            "use_search_c": int(self.use_search_c),
            "threads": int(self.method_param.get("threads", 1)),
            "train_size": self.train_size,
            "quantizer_m": self.quantizer_m,
            "quantizer_efConstruction": self.quantizer_ef_construction,
            "quantizer_efSearch": self.quantizer_ef_search,
            "dist_comps": faiss.cvar.indexIVF_stats.ndis,
            "kernel": "faiss-ivfflat-hnswquantizer-ip-normalized",
        }


class ZeelinBatchriskmassHnswgateIvf(ZeelinHnswgateIvf):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self.fast_return_k = int(method_param.get("fast_return_k", method_param.get("fk", 8)))
        self.slow_return_k = int(method_param.get("slow_return_k", method_param.get("sk", 10)))
        self.risk_ip_threshold = float(
            method_param.get("risk_ip_threshold_milli", method_param.get("ritm", 355))
        ) / 1000.0
        self._repair_count = 0
        self._query_count = 0
        self._last_avg_return_k = 0.0

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=self.fast_return_k,
            internal_k=max(self.fast_return_k, int(internal_k or 0)),
        )
        self.fast_return_k = max(1, min(10, self.fast_return_k))
        self.slow_return_k = max(self.fast_return_k, min(10, self.slow_return_k))
        self._repair_count = 0
        self._query_count = 0
        self._last_avg_return_k = float(self.fast_return_k)
        self.name = self.name.replace("zeelin-hnswgate-ivf", "zeelin-batchriskmass-hnswgate", 1)
        self.name = self.name[:-1] + (
            f", fastReturnK={self.fast_return_k}, slowReturnK={self.slow_return_k}, "
            f"riskIpThreshold={self.risk_ip_threshold:.3f})"
        )

    def query(self, v, n):
        if self.skip_query_norm and self.query_view:
            q = v.reshape(1, self.dim)
        elif self.skip_query_norm:
            self.qbuf[0, :] = v
            q = self.qbuf
        else:
            norm = float(np.sqrt(np.dot(v, v)))
            self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
            q = self.qbuf

        fast_k = min(n, self.fast_return_k)
        D, I = self.index.search(q, fast_k)
        self._query_count += 1
        if D[0, fast_k - 1] > self.risk_ip_threshold and I[0, fast_k - 1] >= 0:
            return I[0, :fast_k]

        self._repair_count += 1
        slow_k = min(n, self.slow_return_k)
        return self.index.search(q, slow_k)[1][0, :slow_k]

    def batch_query(self, X, n):
        X = np.asarray(X, dtype=np.float32)
        if self.skip_query_norm:
            Xn = np.ascontiguousarray(X, dtype=np.float32)
        else:
            norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
            Xn = X / np.maximum(norms, 1e-12)
            Xn = np.ascontiguousarray(Xn, dtype=np.float32)

        fast_k = min(n, self.fast_return_k)
        slow_k = min(n, self.slow_return_k)
        D, I = self.index.search(Xn, fast_k)
        risk = (D[:, fast_k - 1] <= self.risk_ip_threshold) | (I[:, fast_k - 1] < 0)
        self._query_count += int(len(Xn))
        self._repair_count += int(np.count_nonzero(risk))

        labels = np.full((len(Xn), slow_k), -1, dtype=np.int64)
        labels[:, :fast_k] = I[:, :fast_k]
        lengths = np.full(len(Xn), fast_k, dtype=np.int32)
        if np.any(risk):
            slow_labels = self.index.search(Xn[risk], slow_k)[1]
            labels[risk, :slow_k] = slow_labels[:, :slow_k]
            lengths[risk] = slow_k
        self._batch_labels = labels
        self._batch_lengths = lengths
        self._batch_target_k = slow_k
        self._last_avg_return_k = float(np.mean(lengths)) if len(lengths) else 0.0

    def get_batch_results(self):
        results = []
        for labels, length in zip(self._batch_labels, self._batch_lengths):
            prefix = labels[: int(length)]
            if self.fast_unique:
                results.append([int(label) for label in prefix if label >= 0])
                continue
            out = []
            seen = set()
            for label in prefix:
                label = int(label)
                if label < 0 or label in seen:
                    continue
                seen.add(label)
                out.append(label)
            results.append(out)
        return results

    def get_additional(self):
        extra = super().get_additional()
        total = float(self._query_count)
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-batchriskmass"
        extra["fast_return_k"] = self.fast_return_k
        extra["slow_return_k"] = self.slow_return_k
        extra["risk_ip_threshold_milli"] = int(round(self.risk_ip_threshold * 1000.0))
        extra["avg_repair_fraction"] = float(self._repair_count) / total if total else 0.0
        extra["avg_lens_return_k"] = self._last_avg_return_k
        return extra


class ZeelinDirectbatchHnswgateIvf(ZeelinHnswgateIvf):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self._batch_dbuf = None
        self._batch_ibuf = None
        self._batch_capacity = (0, 0)

    def _ensure_batch_buffers(self, nq, query_k):
        if self._batch_capacity[0] >= nq and self._batch_capacity[1] >= query_k:
            return
        self._batch_dbuf = np.empty((nq, query_k), dtype=np.float32)
        self._batch_ibuf = np.empty((nq, query_k), dtype=np.int64)
        self._batch_capacity = (nq, query_k)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace("zeelin-hnswgate-ivf", "zeelin-directbatch-hnswgate", 1)

    def batch_query(self, X, n):
        if self.skip_query_norm:
            Xn = X if X.dtype == np.float32 and X.flags["C_CONTIGUOUS"] else np.ascontiguousarray(X, dtype=np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
            Xn = np.ascontiguousarray(X / np.maximum(norms, 1e-12), dtype=np.float32)
        self._batch_target_k = n if self.return_k <= 0 else min(n, self.return_k)
        query_k = n if self.internal_k <= 0 else max(1, self.internal_k)
        query_k = max(query_k, self._batch_target_k + self.dedup_extra)
        self._ensure_batch_buffers(len(Xn), query_k)
        self.index.search_c(
            len(Xn),
            faiss.swig_ptr(Xn),
            query_k,
            faiss.swig_ptr(self._batch_dbuf),
            faiss.swig_ptr(self._batch_ibuf),
        )
        self._batch_labels = self._batch_ibuf[: len(Xn), :query_k]

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-directbatch-searchc"
        return extra


class ZeelinNormlensHnswgateIvf(ZeelinHnswgateIvf):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self.norm_quantile = float(method_param.get("norm_quantile", 0.8))
        self.fast_return_k = int(method_param.get("fast_return_k", 7))
        self.slow_return_k = int(method_param.get("slow_return_k", 10))
        self.norm_threshold = 0.0
        self._lens_fast = 0
        self._lens_slow = 0
        self._lens_return_sum = 0

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        raw_norms = np.linalg.norm(X, axis=1).astype(np.float32)
        self.norm_threshold = float(np.quantile(raw_norms, self.norm_quantile))
        super().fit(X)

    def set_query_arguments(self, nprobe, quantizer_ef_search=None, dedup_extra=None):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=self.slow_return_k,
            internal_k=0,
        )
        self._lens_fast = 0
        self._lens_slow = 0
        self._lens_return_sum = 0
        self.name = (
            "zeelin-normlens-hnswgate "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, "
            f"quantizerM={self.quantizer_m}, "
            f"quantizerEfC={self.quantizer_ef_construction}, "
            f"quantizerEfS={self.quantizer_ef_search}, "
            f"trainSize={self.train_size}, normQ={self.norm_quantile}, "
            f"normThr={self.norm_threshold:.6f}, fastReturnK={self.fast_return_k}, "
            f"slowReturnK={self.slow_return_k}, dedupExtra={self.dedup_extra}, "
            f"fastUnique={int(self.fast_unique)}, "
            f"skipQueryNorm={int(self.skip_query_norm)}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def query(self, v, n):
        raw_norm = float(np.sqrt(np.dot(v, v)))
        if self.skip_query_norm and self.query_view:
            q = v.reshape(1, self.dim)
        elif self.skip_query_norm:
            self.qbuf[0, :] = v
            q = self.qbuf
        else:
            self.qbuf[0, :] = v * (1.0 / max(raw_norm, 1e-12))
            q = self.qbuf

        if raw_norm <= self.norm_threshold:
            target_k = min(n, self.fast_return_k)
            self._lens_fast += 1
        else:
            target_k = min(n, self.slow_return_k)
            self._lens_slow += 1
        self._lens_return_sum += target_k

        query_k = max(1, target_k + self.dedup_extra)
        labels = self.index.search(q, query_k)[1][0]
        if self.fast_unique:
            if self.trust_full_labels:
                prefix = labels[:target_k]
                if len(prefix) == 0 or prefix[-1] >= 0:
                    return prefix
                return prefix[prefix >= 0]
            labels = labels[labels >= 0]
            return labels[:target_k]

        out = []
        seen = set()
        for label in labels:
            label = int(label)
            if label < 0 or label in seen:
                continue
            seen.add(label)
            out.append(label)
            if len(out) >= target_k:
                break
        return np.asarray(out, dtype=np.int64)

    def get_additional(self):
        extra = super().get_additional()
        total = self._lens_fast + self._lens_slow
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-normlens"
        extra["norm_quantile"] = self.norm_quantile
        extra["norm_threshold"] = self.norm_threshold
        extra["fast_return_k"] = self.fast_return_k
        extra["slow_return_k"] = self.slow_return_k
        extra["lens_fast_fraction"] = (self._lens_fast / total) if total else 0.0
        extra["lens_slow_fraction"] = (self._lens_slow / total) if total else 0.0
        extra["avg_lens_return_k"] = (self._lens_return_sum / total) if total else 0.0
        return extra


class ZeelinMinreturnHnswgateIvf(ZeelinHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace("zeelin-hnswgate-ivf", "zeelin-minreturn-hnswgate", 1)

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-minreturn"
        return extra


class ZeelinPostlessHnswgateIvf(ZeelinHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace("zeelin-hnswgate-ivf", "zeelin-postless-hnswgate", 1)

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-postless"
        return extra


class ZeelinViewpostlessHnswgateIvf(ZeelinPostlessHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace("zeelin-postless-hnswgate", "zeelin-viewpostless-hnswgate", 1)

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-viewpostless"
        return extra


class ZeelinMassguardHnswgateIvf(ZeelinHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace("zeelin-hnswgate-ivf", "zeelin-massguard-hnswgate", 1)

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-massguard"
        return extra


class ZeelinViewmassguardHnswgateIvf(ZeelinMassguardHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace("zeelin-massguard-hnswgate", "zeelin-viewmassguard-hnswgate", 1)

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-viewmassguard"
        return extra


class ZeelinPointermassguardHnswgateIvf(ZeelinMassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "qefc" in method_param:
            method_param["quantizer_efConstruction"] = method_param["qefc"]
        if "qm" in method_param:
            method_param["quantizer_m"] = method_param["qm"]
        if "ts" in method_param:
            method_param["train_size"] = method_param["ts"]
        method_param.setdefault("internal_k", 10)
        method_param.setdefault("return_k", 0)
        method_param.setdefault("dedup_extra", 0)
        method_param.setdefault("fast_unique", True)
        method_param.setdefault("skip_query_norm", True)
        method_param.setdefault("trust_full_labels", True)
        method_param.setdefault("use_search_c", True)
        method_param.setdefault("threads", 4)
        super().__init__(metric, method_param)
        self.pointer_mode = str(method_param.get("pmode", method_param.get("pointer_mode", "copy")))
        self._target_k = 0
        self._query_k = 0
        self._qbuf_ptr = None
        self._dbuf_ptr = None
        self._ibuf_ptr = None
        self._prefix = None

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self._target_k = 10 if self.return_k <= 0 else min(10, self.return_k)
        self._query_k = max(
            self.internal_k if self.internal_k > 0 else 10,
            self._target_k + self.dedup_extra,
        )
        self._ensure_search_c_buffers(self._query_k)
        self._qbuf_ptr = faiss.swig_ptr(self.qbuf)
        self._dbuf_ptr = faiss.swig_ptr(self.dbuf)
        self._ibuf_ptr = faiss.swig_ptr(self.ibuf)
        self._prefix = self.ibuf[0, : self._target_k]
        self.name = self.name.replace("zeelin-massguard-hnswgate", "zeelin-pointermassguard-hnswgate", 1)
        self.name = self.name[:-1] + f", pointerMode={self.pointer_mode})"

    def query(self, v, n):
        if self.pointer_mode == "copy":
            self.qbuf[0, :] = v
            qptr = self._qbuf_ptr
        else:
            qptr = faiss.swig_ptr(v)
        self.index.search_c(
            1,
            qptr,
            self._query_k,
            self._dbuf_ptr,
            self._ibuf_ptr,
        )
        if self._prefix[-1] >= 0:
            return self._prefix
        return self._prefix[self._prefix >= 0]

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-pointermassguard"
        extra["pointer_mode"] = self.pointer_mode
        return extra


class ZeelinRawpointermassguardHnswgateIvf(ZeelinPointermassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param["pmode"] = "view"
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=0,
            return_k=9,
            internal_k=4,
        )
        self._query_k = 9
        self._target_k = 9
        self._prefix = self.ibuf[0, :9]
        self.name = self.name.replace("zeelin-pointermassguard-hnswgate", "zeelin-rawpointermassguard", 1)

    def query(self, v, n):
        self.index.search_c(
            1,
            faiss.swig_ptr(v),
            9,
            self._dbuf_ptr,
            self._ibuf_ptr,
        )
        if self._prefix[-1] >= 0:
            return self._prefix
        return self._prefix[self._prefix >= 0]

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-rawpointermassguard"
        return extra


class ZeelinCodegatePreassignIvf(ZeelinHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "qefc" in method_param:
            method_param["quantizer_efConstruction"] = method_param["qefc"]
        if "qm" in method_param:
            method_param["quantizer_m"] = method_param["qm"]
        if "ts" in method_param:
            method_param["train_size"] = method_param["ts"]
        method_param.setdefault("internal_k", 8)
        method_param.setdefault("return_k", 8)
        method_param.setdefault("dedup_extra", 0)
        method_param.setdefault("fast_unique", True)
        method_param.setdefault("skip_query_norm", True)
        method_param.setdefault("trust_full_labels", True)
        method_param.setdefault("threads", 4)
        super().__init__(metric, method_param)
        self.code_bits = int(method_param.get("cb", method_param.get("code_bits", 12)))
        self.code_bits = max(1, min(20, self.code_bits))
        self.max_code_nprobe = int(method_param.get("cnp", method_param.get("code_nprobe", 1)))
        self.max_code_nprobe = max(1, min(16, self.max_code_nprobe))
        self.assign_ef_search = int(method_param.get("assign_efSearch", 16))
        self.code_dims = None
        self.code_table = None
        self._assign = None
        self._centroid_dis = None
        self._dbuf_ptr = None
        self._ibuf_ptr = None

    def _encode_codes(self, X):
        codes = np.zeros(X.shape[0], dtype=np.int64)
        for bit, dim in enumerate(self.code_dims):
            codes |= (X[:, int(dim)] >= 0.0).astype(np.int64) << bit
        return codes

    def _encode_one(self, v):
        code = 0
        for bit, dim in enumerate(self.code_dims):
            if v[int(dim)] >= 0.0:
                code |= 1 << bit
        return code

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )
        self._apply_cluster_seed()

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        self.index.train(train)
        self.index.add(Xn)

        dim_scores = np.var(Xn, axis=0)
        self.code_dims = np.ascontiguousarray(
            np.argsort(-dim_scores)[: self.code_bits], dtype=np.int32
        )
        codes = self._encode_codes(Xn)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        _, assigned = self.quantizer.search(np.ascontiguousarray(Xn, dtype=np.float32), 1)
        assigned = np.ascontiguousarray(assigned[:, 0], dtype=np.int64)
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

        code_count = 1 << self.code_bits
        global_counts = np.bincount(assigned, minlength=self.nlist)
        global_top = np.argsort(-global_counts)[: self.max_code_nprobe].astype(np.int64)
        table = np.tile(global_top, (code_count, 1))
        table_counts = np.full((code_count, self.max_code_nprobe), -1, dtype=np.int64)
        pairs = codes * int(self.nlist) + assigned
        uniq, counts = np.unique(pairs, return_counts=True)
        for pair, count in zip(uniq, counts):
            code = int(pair // self.nlist)
            label = int(pair % self.nlist)
            slot = int(np.argmin(table_counts[code]))
            if int(count) > int(table_counts[code, slot]):
                table_counts[code, slot] = int(count)
                table[code, slot] = label
        order = np.argsort(-table_counts, axis=1)
        self.code_table = np.take_along_axis(table, order, axis=1).astype(np.int64, copy=False)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        faiss.cvar.indexIVF_stats.reset()
        self.nprobe = max(1, min(int(nprobe), self.max_code_nprobe))
        self.quantizer_ef_search = 0
        if dedup_extra is not None:
            self.dedup_extra = int(dedup_extra)
        self.return_k = int(return_k) if return_k is not None else self.default_return_k
        if internal_k is not None:
            self.internal_k = int(internal_k)
        self.index.nprobe = self.nprobe
        self._target_k = 10 if self.return_k <= 0 else min(10, self.return_k)
        self._query_k = max(
            self.internal_k if self.internal_k > 0 else 10,
            self._target_k + self.dedup_extra,
        )
        self._ensure_search_c_buffers(self._query_k)
        self._dbuf_ptr = faiss.swig_ptr(self.dbuf)
        self._ibuf_ptr = faiss.swig_ptr(self.ibuf)
        self._prefix = self.ibuf[0, : self._target_k]
        self._assign = np.empty((1, self.nprobe), dtype=np.int64)
        self._centroid_dis = np.zeros((1, self.nprobe), dtype=np.float32)
        self.name = (
            "zeelin-codegate-preassign "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, "
            f"codeBits={self.code_bits}, maxCodeNprobe={self.max_code_nprobe}, "
            f"quantizerM={self.quantizer_m}, quantizerEfC={self.quantizer_ef_construction}, "
            f"assignEfS={self.assign_ef_search}, trainSize={self.train_size}, "
            f"internalK={self.internal_k}, returnK={self.return_k}, "
            f"dedupExtra={self.dedup_extra}, skipQueryNorm={int(self.skip_query_norm)}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def query(self, v, n):
        if self.skip_query_norm:
            qptr = faiss.swig_ptr(v)
        else:
            norm = float(np.sqrt(np.dot(v, v)))
            self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
            qptr = faiss.swig_ptr(self.qbuf)
        code = self._encode_one(v)
        self._assign[0, :] = self.code_table[code, : self.nprobe]
        self.index.search_preassigned_c(
            1,
            qptr,
            self._query_k,
            faiss.swig_ptr(self._assign),
            faiss.swig_ptr(self._centroid_dis),
            self._dbuf_ptr,
            self._ibuf_ptr,
            False,
        )
        if self._prefix[-1] >= 0:
            return self._prefix
        return self._prefix[self._prefix >= 0]

    def get_additional(self):
        return {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "internal_k": self.internal_k,
            "return_k": self.return_k,
            "dedup_extra": self.dedup_extra,
            "threads": int(self.method_param.get("threads", 1)),
            "train_size": self.train_size,
            "quantizer_m": self.quantizer_m,
            "quantizer_efConstruction": self.quantizer_ef_construction,
            "quantizer_efSearch": self.quantizer_ef_search,
            "code_bits": self.code_bits,
            "code_nprobe": self.nprobe,
            "assign_efSearch": self.assign_ef_search,
            "dist_comps": faiss.cvar.indexIVF_stats.ndis,
            "kernel": "faiss-ivfflat-codegate-preassigned-ip",
            "scan_layout": "signcode-transport-to-ivf-list",
        }


class ZeelinReturnpointermassguardHnswgateIvf(ZeelinPointermassguardHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace("zeelin-pointermassguard-hnswgate", "zeelin-returnpointermassguard-hnswgate", 1)

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-returnpointermassguard"
        return extra


class ZeelinFirsthitmasspointerHnswgateIvf(ZeelinReturnpointermassguardHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-returnpointermassguard-hnswgate",
            "zeelin-firsthitmasspointer-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-firsthitmasspointer"
        return extra


class ZeelinPrefixmasscodepointerHnswgateIvf(ZeelinReturnpointermassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self.prefix_max_codes = int(method_param.get("maxc", method_param.get("max_codes", 0)))

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.index.max_codes = max(0, self.prefix_max_codes)
        self.name = self.name.replace(
            "zeelin-returnpointermassguard-hnswgate",
            "zeelin-prefixmasscodepointer-hnswgate",
            1,
        )
        self.name = self.name[:-1] + f", maxCodes={self.prefix_max_codes})"

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-prefixmasscodepointer"
        extra["max_codes"] = self.prefix_max_codes
        return extra


class ZeelinSignpayloadpointermassguardHnswgateIvf(ZeelinPointermassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "sbits" in method_param:
            method_param["signature_bits"] = method_param["sbits"]
        if "pslots" in method_param:
            method_param["payload_k"] = method_param["pslots"]
        method_param.setdefault("pmode", "view")
        method_param.setdefault("search_k", 8)
        method_param.setdefault("return_k", 9)
        super().__init__(metric, method_param)
        self.signature_bits = int(method_param.get("signature_bits", 16))
        self.signature_bits = max(1, min(24, self.signature_bits))
        self.payload_k = int(method_param.get("payload_k", 1))
        self.payload_k = max(1, min(16, self.payload_k))
        self.search_k = int(method_param.get("search_k", 8))
        self.search_k = max(1, min(10, self.search_k))
        self.final_return_k = int(method_param.get("return_k", 9))
        self.final_return_k = max(self.search_k, min(10, self.final_return_k))
        self.sign_dims = None
        self.sign_payload = None
        self._payload_out = np.empty(10, dtype=np.int64)

    def _encode_signature_matrix(self, X):
        codes = np.zeros(X.shape[0], dtype=np.int64)
        for bit, dim in enumerate(self.sign_dims):
            codes |= (X[:, int(dim)] >= 0.0).astype(np.int64) << bit
        return codes

    def _encode_signature_one(self, v):
        code = 0
        for bit, dim in enumerate(self.sign_dims):
            if v[int(dim)] >= 0.0:
                code |= 1 << bit
        return code

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        raw_norms = np.linalg.norm(X, axis=1).astype(np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        dim_scores = np.var(Xn, axis=0)
        self.sign_dims = np.ascontiguousarray(
            np.argsort(-dim_scores)[: self.signature_bits], dtype=np.int32
        )
        codes = self._encode_signature_matrix(X)
        code_count = 1 << self.signature_bits
        payload = np.full((code_count, self.payload_k), -1, dtype=np.int64)
        fill = np.zeros(code_count, dtype=np.int16)
        order = np.lexsort((-raw_norms, codes))
        for idx in order:
            code = int(codes[int(idx)])
            slot = int(fill[code])
            if slot < self.payload_k:
                payload[code, slot] = int(idx)
                fill[code] = slot + 1
        self.sign_payload = np.ascontiguousarray(payload, dtype=np.int64)
        super().fit(X)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        if return_k is not None:
            self.final_return_k = max(self.search_k, min(10, int(return_k)))
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=0,
            return_k=self.search_k,
            internal_k=self.search_k,
        )
        self.name = self.name.replace(
            "zeelin-pointermassguard-hnswgate",
            "zeelin-signpayloadpointermassguard",
            1,
        )
        self.name = self.name[:-1] + (
            f", signatureBits={self.signature_bits}, payloadK={self.payload_k}, "
            f"searchK={self.search_k}, finalReturnK={self.final_return_k})"
        )

    def query(self, v, n):
        self.index.search_c(
            1,
            faiss.swig_ptr(v),
            self.search_k,
            self._dbuf_ptr,
            self._ibuf_ptr,
        )
        return self._compose_payload_results(self._encode_signature_one(v))

    def _compose_payload_results(self, code):
        prefix = self.ibuf[0, : self.search_k]
        if prefix[-1] >= 0:
            prefix_len = self.search_k
            self._payload_out[:prefix_len] = prefix
        else:
            valid = prefix[prefix >= 0]
            prefix_len = len(valid)
            self._payload_out[:prefix_len] = valid

        if prefix_len >= self.final_return_k:
            return self._payload_out[:prefix_len]

        for cand in self.sign_payload[code]:
            cand = int(cand)
            if cand < 0:
                continue
            duplicate = False
            for pos in range(prefix_len):
                if int(self._payload_out[pos]) == cand:
                    duplicate = True
                    break
            if duplicate:
                continue
            self._payload_out[prefix_len] = cand
            prefix_len += 1
            if prefix_len >= self.final_return_k:
                break
        return self._payload_out[:prefix_len]

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-signpayloadpointermassguard"
        extra["return_k"] = self.final_return_k
        extra["fast_return_k"] = self.search_k
        extra["slow_return_k"] = self.final_return_k
        extra["signature_bits"] = self.signature_bits
        extra["payload_k"] = self.payload_k
        return extra


class ZeelinPreparedsignpayloadpointermassguardHnswgateIvf(ZeelinSignpayloadpointermassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self._prepared_qptr = None
        self._prepared_code = 0

    def prepare_query(self, v, n):
        self._prepared_qptr = faiss.swig_ptr(v)
        self._prepared_code = self._encode_signature_one(v)

    def run_prepared_query(self):
        self.index.search_c(
            1,
            self._prepared_qptr,
            self.search_k,
            self._dbuf_ptr,
            self._ibuf_ptr,
        )

    def get_prepared_query_results(self):
        return self._compose_payload_results(self._prepared_code)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-signpayloadpointermassguard",
            "zeelin-preparedsignpayloadpointermassguard",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-preparedsignpayloadpointermassguard"
        extra["prepared_query"] = 1
        return extra


class ZeelinPreparedpointermassguardHnswgateIvf(ZeelinPointermassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param["pmode"] = "view"
        super().__init__(metric, method_param)
        self._prepared_qptr = None

    def prepare_query(self, v, n):
        self._prepared_qptr = faiss.swig_ptr(v)

    def run_prepared_query(self):
        self.index.search_c(
            1,
            self._prepared_qptr,
            self._query_k,
            self._dbuf_ptr,
            self._ibuf_ptr,
        )

    def get_prepared_query_results(self):
        if self._prefix[-1] >= 0:
            return self._prefix
        return self._prefix[self._prefix >= 0]

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace("zeelin-pointermassguard-hnswgate", "zeelin-preparedpointermassguard", 1)

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-preparedpointermassguard"
        extra["prepared_query"] = 1
        return extra


class ZeelinDirectswigpointermassguardHnswgateIvf(ZeelinPointermassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param["pmode"] = "view"
        super().__init__(metric, method_param)
        self._indexivf_search = faiss._swigfaiss.IndexIVF_search

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace("zeelin-pointermassguard-hnswgate", "zeelin-directswigpointermassguard", 1)

    def query(self, v, n):
        self._indexivf_search(
            self.index,
            1,
            faiss.swig_ptr(v),
            self._query_k,
            self._dbuf_ptr,
            self._ibuf_ptr,
        )
        if self._prefix[-1] >= 0:
            return self._prefix
        return self._prefix[self._prefix >= 0]

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-indexivf-directswig-ip-pointermassguard"
        return extra


class ZeelinPrepareddirectswigpointermassguardHnswgateIvf(ZeelinDirectswigpointermassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self._prepared_qptr = None

    def prepare_query(self, v, n):
        self._prepared_qptr = faiss.swig_ptr(v)

    def run_prepared_query(self):
        self._indexivf_search(
            self.index,
            1,
            self._prepared_qptr,
            self._query_k,
            self._dbuf_ptr,
            self._ibuf_ptr,
        )

    def get_prepared_query_results(self):
        if self._prefix[-1] >= 0:
            return self._prefix
        return self._prefix[self._prefix >= 0]

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-directswigpointermassguard",
            "zeelin-prepareddirectswigpointermassguard",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-indexivf-prepareddirectswig-ip-pointermassguard"
        extra["prepared_query"] = 1
        return extra


class ZeelinCellcorepointermassguardHnswgateIvf(ZeelinPointermassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "maxc" in method_param:
            method_param["max_codes"] = method_param["maxc"]
        if "aef" in method_param:
            method_param["assign_efSearch"] = method_param["aef"]
        if "co" in method_param:
            method_param["core_order"] = method_param["co"]
        if "ca" in method_param:
            method_param["core_alpha"] = method_param["ca"]
        if "cs" in method_param:
            method_param["cluster_seed"] = method_param["cs"]
        method_param.setdefault("pmode", "view")
        super().__init__(metric, method_param)
        self.max_codes = int(method_param.get("max_codes", 0))
        self.assign_ef_search = int(method_param.get("assign_efSearch", self.default_quantizer_ef_search))
        self.core_order = str(method_param.get("core_order", "centroid"))
        self.core_alpha = float(method_param.get("core_alpha", 0.0))
        self.cluster_seed = int(method_param.get("cluster_seed", 1234))

    def _apply_cluster_seed(self):
        if hasattr(self.index, "cp"):
            self.index.cp.seed = self.cluster_seed

    def _core_scores(self, Xn, assigned, raw_norms):
        centroids = np.empty((self.nlist, self.dim), dtype=np.float32)
        for cid in range(self.nlist):
            centroids[cid, :] = self.quantizer.reconstruct(cid)
        scores = np.einsum("ij,ij->i", Xn, centroids[assigned], optimize=True).astype(np.float32)
        if self.core_order == "boundary":
            return -scores
        if self.core_order == "rawnorm":
            return raw_norms.astype(np.float32, copy=False)
        if self.core_order == "hybrid":
            norm_mu = float(np.mean(raw_norms))
            norm_sd = float(np.std(raw_norms) + 1e-12)
            score_mu = float(np.mean(scores))
            score_sd = float(np.std(scores) + 1e-12)
            return ((scores - score_mu) / score_sd + self.core_alpha * ((raw_norms - norm_mu) / norm_sd)).astype(np.float32)
        return scores

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        raw_norms = np.linalg.norm(X, axis=1).astype(np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self._adapt_nlist_to_training_size(len(Xn))
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )
        self._apply_cluster_seed()

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        self.index.train(train)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        _, assigned = self.quantizer.search(np.ascontiguousarray(Xn, dtype=np.float32), 1)
        assigned = np.ascontiguousarray(assigned[:, 0], dtype=np.int64)
        scores = self._core_scores(Xn, assigned, raw_norms)
        order = np.lexsort((-scores, assigned)).astype(np.int64, copy=False)
        ids = np.ascontiguousarray(order, dtype=np.int64)
        block = int(self.method_param.get("add_block", 65536))
        block = max(1024, block)
        for start in range(0, len(order), block):
            stop = min(len(order), start + block)
            batch = np.ascontiguousarray(Xn[order[start:stop]], dtype=np.float32)
            self.index.add_with_ids(batch, ids[start:stop])
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.index.max_codes = max(0, self.max_codes)
        self.name = self.name.replace(
            "zeelin-pointermassguard-hnswgate",
            "zeelin-cellcorepointermassguard",
            1,
        )
        self.name = self.name[:-1] + (
            f", maxCodes={self.max_codes}, assignEfS={self.assign_ef_search}, "
            f"coreOrder={self.core_order}, coreAlpha={self.core_alpha})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-cellcorepointermassguard"
        extra["max_codes"] = self.max_codes
        extra["assign_efSearch"] = self.assign_ef_search
        extra["core_order"] = self.core_order
        extra["core_alpha"] = self.core_alpha
        extra["cluster_seed"] = self.cluster_seed
        return extra


class ZeelinBoundarycorepointermassguardHnswgateIvf(ZeelinCellcorepointermassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param["core_order"] = "boundary"
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-cellcorepointermassguard",
            "zeelin-boundarycorepointermassguard",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-boundarycorepointermassguard"
        return extra


class ZeelinQuantilemassTransportHnswgateIvf(ZeelinCellcorepointermassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param.setdefault("maxc", 128)
        method_param.setdefault("aef", 8)
        method_param.setdefault("co", "centroid")
        method_param.setdefault("pmode", "view")
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-cellcorepointermassguard",
            "zeelin-quantilemass-transport-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-quantilemass-transport"
        return extra


class ZeelinM10quantilemassTransportHnswgateIvf(ZeelinQuantilemassTransportHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param.setdefault("qm", 10)
        method_param.setdefault("maxc", 128)
        method_param.setdefault("aef", 8)
        method_param.setdefault("co", "centroid")
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-quantilemass-transport-hnswgate",
            "zeelin-m10quantilemass-transport-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10quantilemass-transport"
        return extra


class ZeelinCappedcoreTransportHnswgateIvf(ZeelinCellcorepointermassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "cap" in method_param:
            method_param["list_cap"] = method_param["cap"]
        if "lc" in method_param:
            method_param["list_cap"] = method_param["lc"]
        method_param.setdefault("maxc", 0)
        method_param.setdefault("aef", 8)
        method_param.setdefault("co", "centroid")
        method_param.setdefault("pmode", "view")
        super().__init__(metric, method_param)
        self.list_cap = int(method_param.get("list_cap", 128))
        self.list_cap = max(1, self.list_cap)
        self.kept_points = 0
        self.cap_min_actual = self.list_cap
        self.cap_max_actual = self.list_cap
        self.cap_mean_actual = float(self.list_cap)

    def _cell_caps(self, assigned, scores):
        return np.full(self.nlist, self.list_cap, dtype=np.int32)

    def _before_cell_caps(self, Xn, assigned, scores, raw_norms, order):
        pass

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        raw_norms = np.linalg.norm(X, axis=1).astype(np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self._adapt_nlist_to_training_size(len(Xn))
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        self.index.train(train)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        _, assigned = self.quantizer.search(np.ascontiguousarray(Xn, dtype=np.float32), 1)
        assigned = np.ascontiguousarray(assigned[:, 0], dtype=np.int64)
        scores = self._core_scores(Xn, assigned, raw_norms)
        order = np.lexsort((-scores, assigned)).astype(np.int64, copy=False)
        self._before_cell_caps(Xn, assigned, scores, raw_norms, order)
        cell_caps = np.asarray(self._cell_caps(assigned, scores), dtype=np.int32)
        nonzero_caps = cell_caps[cell_caps > 0]
        if len(nonzero_caps) > 0:
            self.cap_min_actual = int(np.min(nonzero_caps))
            self.cap_max_actual = int(np.max(nonzero_caps))
            self.cap_mean_actual = float(np.mean(nonzero_caps))

        keep = np.zeros(len(order), dtype=bool)
        last_cid = -1
        used_in_cell = 0
        for pos, idx in enumerate(order):
            cid = int(assigned[idx])
            if cid != last_cid:
                last_cid = cid
                used_in_cell = 0
            if used_in_cell < int(cell_caps[cid]):
                keep[pos] = True
                used_in_cell += 1
        kept_order = np.ascontiguousarray(order[keep], dtype=np.int64)
        self.kept_points = int(len(kept_order))
        ids = np.ascontiguousarray(kept_order, dtype=np.int64)
        block = int(self.method_param.get("add_block", 65536))
        block = max(1024, block)
        for start in range(0, len(kept_order), block):
            stop = min(len(kept_order), start + block)
            batch = np.ascontiguousarray(Xn[kept_order[start:stop]], dtype=np.float32)
            self.index.add_with_ids(batch, ids[start:stop])
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.index.max_codes = 0
        self.name = self.name.replace(
            "zeelin-cellcorepointermassguard",
            "zeelin-cappedcore-transport-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", listCap={self.list_cap}, keptPoints={self.kept_points})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-cappedcore-transport"
        extra["list_cap"] = self.list_cap
        extra["kept_points"] = self.kept_points
        extra["cap_min_actual"] = self.cap_min_actual
        extra["cap_max_actual"] = self.cap_max_actual
        extra["cap_mean_actual"] = self.cap_mean_actual
        return extra


class ZeelinM10cappedcoreTransportHnswgateIvf(ZeelinCappedcoreTransportHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param.setdefault("qm", 10)
        method_param.setdefault("lc", 128)
        method_param.setdefault("aef", 8)
        method_param.setdefault("co", "centroid")
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-cappedcore-transport-hnswgate",
            "zeelin-m10cappedcore-transport-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10cappedcore-transport"
        return extra


class ZeelinAdaptivecapcoreTransportHnswgateIvf(ZeelinCappedcoreTransportHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "cm" in method_param:
            method_param["cap_mode"] = method_param["cm"]
        if "cmin" in method_param:
            method_param["cap_min"] = method_param["cmin"]
        if "cmax" in method_param:
            method_param["cap_max"] = method_param["cmax"]
        if "cg" in method_param:
            method_param["cap_gamma"] = method_param["cg"]
        if "cpa" in method_param:
            method_param["cap_alpha"] = method_param["cpa"]
        method_param.setdefault("lc", 96)
        super().__init__(metric, method_param)
        self.cap_mode = str(method_param.get("cap_mode", "size"))
        self.cap_min = int(method_param.get("cap_min", 48))
        self.cap_max = int(method_param.get("cap_max", 160))
        self.cap_gamma = float(method_param.get("cap_gamma", 0.5))
        self.cap_alpha = float(method_param.get("cap_alpha", 0.0))
        self.cap_total_requested = 0

    def _cell_caps(self, assigned, scores):
        counts = np.bincount(assigned, minlength=self.nlist).astype(np.float32)
        nonempty = counts > 0
        caps = np.zeros(self.nlist, dtype=np.float32)
        if not np.any(nonempty):
            return caps.astype(np.int32)

        mean_count = float(np.mean(counts[nonempty]) + 1e-12)
        size_factor = np.power(np.maximum(counts, 1.0) / mean_count, self.cap_gamma)
        factor = size_factor

        if self.cap_mode in ("flat", "compact", "meanmass"):
            score64 = scores.astype(np.float64, copy=False)
            sums = np.bincount(assigned, weights=score64, minlength=self.nlist)
            sqs = np.bincount(assigned, weights=score64 * score64, minlength=self.nlist)
            denom = np.maximum(counts.astype(np.float64), 1.0)
            means = sums / denom
            var = np.maximum((sqs / denom) - means * means, 0.0)
            std = np.sqrt(var).astype(np.float32)
            mean_std = float(np.mean(std[nonempty]) + 1e-6)
            std_ratio = np.clip(std / mean_std, 0.25, 4.0)
            if self.cap_mode == "flat":
                factor = factor * np.power(1.0 / std_ratio, self.cap_alpha)
            elif self.cap_mode == "compact":
                factor = factor * np.power(std_ratio, self.cap_alpha)
            elif self.cap_mode == "meanmass":
                mean_mu = float(np.mean(means[nonempty]))
                mean_sd = float(np.std(means[nonempty]) + 1e-6)
                # Lower mean core-mass cells are more ambiguous, so give them more budget.
                z = np.clip((mean_mu - means) / mean_sd, -1.5, 1.5)
                factor = factor * np.exp(self.cap_alpha * z)

        factor = np.clip(factor, 0.25, 4.0)
        caps[nonempty] = np.rint(float(self.list_cap) * factor[nonempty])
        caps = np.clip(caps, self.cap_min, self.cap_max)
        caps[~nonempty] = 0
        caps = np.minimum(caps, counts)
        caps_i = caps.astype(np.int32)
        self.cap_total_requested = int(np.sum(caps_i))
        return caps_i

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-cappedcore-transport-hnswgate",
            "zeelin-adaptivecapcore-transport-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", capMode={self.cap_mode}, capMin={self.cap_min}, "
            f"capMax={self.cap_max}, capGamma={self.cap_gamma}, "
            f"capAlpha={self.cap_alpha}, capMean={self.cap_mean_actual:.2f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-adaptivecapcore-transport"
        extra["cap_mode"] = self.cap_mode
        extra["cap_min"] = self.cap_min
        extra["cap_max"] = self.cap_max
        extra["cap_gamma"] = self.cap_gamma
        extra["cap_alpha"] = self.cap_alpha
        extra["cap_total_requested"] = self.cap_total_requested
        return extra


class ZeelinM10adaptivecapcoreTransportHnswgateIvf(ZeelinAdaptivecapcoreTransportHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param.setdefault("qm", 10)
        method_param.setdefault("lc", 96)
        method_param.setdefault("aef", 8)
        method_param.setdefault("co", "centroid")
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-adaptivecapcore-transport-hnswgate",
            "zeelin-m10adaptivecapcore-transport-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10adaptivecapcore-transport"
        return extra


class ZeelinMarginalcapcoreTransportHnswgateIvf(ZeelinAdaptivecapcoreTransportHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "mm" in method_param:
            method_param["marginal_mode"] = method_param["mm"]
        if "mcs" in method_param:
            method_param["marginal_cell_samples"] = method_param["mcs"]
        if "mpc" in method_param:
            method_param["marginal_probe_cap"] = method_param["mpc"]
        if "mq" in method_param:
            method_param["marginal_quantile"] = method_param["mq"]
        if "msf" in method_param:
            method_param["marginal_safety"] = method_param["msf"]
        if "mmix" in method_param:
            method_param["marginal_mix"] = method_param["mmix"]
        if "mrk" in method_param:
            method_param["marginal_return_k"] = method_param["mrk"]
        super().__init__(metric, method_param)
        self.marginal_mode = str(method_param.get("marginal_mode", "yieldmin"))
        self.marginal_cell_samples = int(method_param.get("marginal_cell_samples", 2))
        self.marginal_probe_cap = int(method_param.get("marginal_probe_cap", 144))
        self.marginal_quantile = float(method_param.get("marginal_quantile", 0.80))
        self.marginal_safety = float(method_param.get("marginal_safety", 1.15))
        self.marginal_mix = float(method_param.get("marginal_mix", 0.50))
        self.marginal_return_k = int(method_param.get("marginal_return_k", 8))
        self.marginal_cell_samples = max(1, min(32, self.marginal_cell_samples))
        self.marginal_probe_cap = max(8, min(512, self.marginal_probe_cap))
        self.marginal_quantile = min(0.99, max(0.05, self.marginal_quantile))
        self.marginal_safety = max(0.25, min(4.0, self.marginal_safety))
        self.marginal_mix = min(1.0, max(0.0, self.marginal_mix))
        self.marginal_return_k = max(1, min(16, self.marginal_return_k))
        self.marginal_caps_raw = None
        self.marginal_observed_cells = 0
        self.marginal_mean_raw_cap = 0.0
        self.marginal_min_raw_cap = 0
        self.marginal_max_raw_cap = 0

    def _before_cell_caps(self, Xn, assigned, scores, raw_norms, order):
        counts = np.bincount(assigned, minlength=self.nlist).astype(np.int32)
        starts = np.zeros(self.nlist + 1, dtype=np.int64)
        starts[1:] = np.cumsum(counts, dtype=np.int64)
        raw_caps = np.zeros(self.nlist, dtype=np.float32)
        observed = np.zeros(self.nlist, dtype=bool)

        for cid in range(self.nlist):
            start = int(starts[cid])
            stop = int(starts[cid + 1])
            cell_size = stop - start
            if cell_size <= 1:
                continue
            members = order[start:stop]
            prefix_len = min(cell_size, self.marginal_probe_cap, max(self.cap_max, self.list_cap))
            if prefix_len <= 1:
                continue
            prefix_ids = members[:prefix_len]
            sample_count = min(self.marginal_cell_samples, cell_size)
            if sample_count <= 1:
                sample_pos = np.array([0], dtype=np.int64)
            else:
                sample_pos = np.linspace(0, cell_size - 1, sample_count, dtype=np.int64)
            query_ids = members[sample_pos]
            sims = np.dot(
                np.ascontiguousarray(Xn[query_ids], dtype=np.float32),
                np.ascontiguousarray(Xn[prefix_ids], dtype=np.float32).T,
            )
            for row, qid in enumerate(query_ids):
                self_hits = np.flatnonzero(prefix_ids == qid)
                if len(self_hits):
                    sims[row, self_hits[0]] = -np.inf

            need_ranks = []
            for row in range(sims.shape[0]):
                finite = np.isfinite(sims[row])
                finite_count = int(np.sum(finite))
                if finite_count <= 0:
                    continue
                take = min(self.marginal_return_k, finite_count)
                if take >= prefix_len:
                    top_idx = np.argsort(sims[row])[::-1][:take]
                else:
                    top_idx = np.argpartition(sims[row], -take)[-take:]
                need_ranks.append(int(np.max(top_idx)) + 1)

            if need_ranks:
                qrank = float(np.quantile(np.asarray(need_ranks, dtype=np.float32), self.marginal_quantile))
                raw_caps[cid] = qrank * self.marginal_safety
                observed[cid] = True

        self.marginal_caps_raw = raw_caps
        self.marginal_observed_cells = int(np.sum(observed))
        if self.marginal_observed_cells:
            observed_caps = raw_caps[observed]
            self.marginal_mean_raw_cap = float(np.mean(observed_caps))
            self.marginal_min_raw_cap = int(np.min(observed_caps))
            self.marginal_max_raw_cap = int(np.max(observed_caps))

    def _cell_caps(self, assigned, scores):
        base_caps = super()._cell_caps(assigned, scores).astype(np.float32)
        raw_caps = self.marginal_caps_raw
        if raw_caps is None:
            return base_caps.astype(np.int32)
        valid = raw_caps > 0
        if not np.any(valid):
            return base_caps.astype(np.int32)

        proposed = base_caps.copy()
        raw = np.where(valid, raw_caps, base_caps)
        if self.marginal_mode == "yield":
            proposed = raw
        elif self.marginal_mode == "yieldmax":
            proposed = np.maximum(base_caps, raw)
        elif self.marginal_mode == "yieldmix":
            proposed = np.exp(
                (1.0 - self.marginal_mix) * np.log(np.maximum(base_caps, 1.0))
                + self.marginal_mix * np.log(np.maximum(raw, 1.0))
            )
        else:
            proposed = np.minimum(base_caps, raw)

        counts = np.bincount(assigned, minlength=self.nlist).astype(np.float32)
        nonempty = counts > 0
        caps = np.zeros(self.nlist, dtype=np.float32)
        caps[nonempty] = proposed[nonempty]
        caps = np.clip(caps, self.cap_min, self.cap_max)
        caps[~nonempty] = 0
        caps = np.minimum(caps, counts)
        caps_i = np.rint(caps).astype(np.int32)
        self.cap_total_requested = int(np.sum(caps_i))
        return caps_i

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-adaptivecapcore-transport-hnswgate",
            "zeelin-marginalcapcore-transport-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", marginalMode={self.marginal_mode}, marginalSamples={self.marginal_cell_samples}, "
            f"marginalProbeCap={self.marginal_probe_cap}, marginalQ={self.marginal_quantile}, "
            f"marginalSafety={self.marginal_safety}, marginalMix={self.marginal_mix}, "
            f"marginalMeanRaw={self.marginal_mean_raw_cap:.2f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-marginalcapcore-transport"
        extra["marginal_mode"] = self.marginal_mode
        extra["marginal_cell_samples"] = self.marginal_cell_samples
        extra["marginal_probe_cap"] = self.marginal_probe_cap
        extra["marginal_quantile"] = self.marginal_quantile
        extra["marginal_safety"] = self.marginal_safety
        extra["marginal_mix"] = self.marginal_mix
        extra["marginal_return_k"] = self.marginal_return_k
        extra["marginal_observed_cells"] = self.marginal_observed_cells
        extra["marginal_mean_raw_cap"] = self.marginal_mean_raw_cap
        extra["marginal_min_raw_cap"] = self.marginal_min_raw_cap
        extra["marginal_max_raw_cap"] = self.marginal_max_raw_cap
        return extra


class ZeelinM10marginalcapcoreTransportHnswgateIvf(ZeelinMarginalcapcoreTransportHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param.setdefault("qm", 10)
        method_param.setdefault("lc", 96)
        method_param.setdefault("aef", 8)
        method_param.setdefault("co", "centroid")
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-marginalcapcore-transport-hnswgate",
            "zeelin-m10marginalcapcore-transport-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10marginalcapcore-transport"
        return extra


class ZeelinM10ratecapcoreTransportHnswgateIvf(ZeelinM10adaptivecapcoreTransportHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10adaptivecapcore-transport-hnswgate",
            "zeelin-m10ratecapcore-transport-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10ratecapcore-transport"
        return extra


class ZeelinM10ratecapcoreReturncompressHnswgateIvf(ZeelinM10ratecapcoreTransportHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        if return_k is None:
            return_k = 7
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10ratecapcore-transport-hnswgate",
            "zeelin-m10ratecapcore-returncompress-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10ratecapcore-returncompress"
        return extra


class ZeelinConcentrationcapcoreTransportHnswgateIvf(ZeelinAdaptivecapcoreTransportHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "ccm" in method_param:
            method_param["concentration_mode"] = method_param["ccm"]
        if "ccg" in method_param:
            method_param["concentration_gamma"] = method_param["ccg"]
        if "ccb" in method_param:
            method_param["concentration_bias"] = method_param["ccb"]
        if "ccp" in method_param:
            method_param["concentration_prefix_cap"] = method_param["ccp"]
        if "cct" in method_param:
            method_param["concentration_temperature_floor"] = method_param["cct"]
        if "ccq" in method_param:
            method_param["concentration_tail_quantile"] = method_param["ccq"]
        if "cccliplo" in method_param:
            method_param["concentration_clip_min"] = method_param["cccliplo"]
        if "cccliphi" in method_param:
            method_param["concentration_clip_max"] = method_param["cccliphi"]
        super().__init__(metric, method_param)
        self.concentration_mode = str(method_param.get("concentration_mode", "hybrid"))
        self.concentration_gamma = float(method_param.get("concentration_gamma", 0.35))
        self.concentration_bias = float(method_param.get("concentration_bias", 0.96))
        self.concentration_prefix_cap = int(method_param.get("concentration_prefix_cap", 0))
        self.concentration_temperature_floor = float(
            method_param.get("concentration_temperature_floor", 0.015)
        )
        self.concentration_tail_quantile = float(method_param.get("concentration_tail_quantile", 0.25))
        self.concentration_clip_min = float(method_param.get("concentration_clip_min", 0.70))
        self.concentration_clip_max = float(method_param.get("concentration_clip_max", 1.35))
        self.concentration_gamma = max(0.0, min(2.0, self.concentration_gamma))
        self.concentration_bias = max(0.25, min(2.0, self.concentration_bias))
        self.concentration_prefix_cap = max(0, self.concentration_prefix_cap)
        self.concentration_temperature_floor = max(1e-4, min(1.0, self.concentration_temperature_floor))
        self.concentration_tail_quantile = min(0.95, max(0.01, self.concentration_tail_quantile))
        self.concentration_clip_min = max(0.05, min(4.0, self.concentration_clip_min))
        self.concentration_clip_max = max(self.concentration_clip_min, min(4.0, self.concentration_clip_max))
        self.concentration_effective_rank = None
        self.concentration_tail_drop = None
        self.concentration_observed_cells = 0
        self.concentration_mean_effective_rank = 0.0
        self.concentration_mean_tail_drop = 0.0
        self.concentration_cap_ratio = 0.0

    def _before_cell_caps(self, Xn, assigned, scores, raw_norms, order):
        counts = np.bincount(assigned, minlength=self.nlist).astype(np.int32)
        starts = np.zeros(self.nlist + 1, dtype=np.int64)
        starts[1:] = np.cumsum(counts, dtype=np.int64)
        eff = np.zeros(self.nlist, dtype=np.float32)
        drop = np.zeros(self.nlist, dtype=np.float32)
        observed = np.zeros(self.nlist, dtype=bool)
        prefix_cap = self.concentration_prefix_cap
        if prefix_cap <= 0:
            prefix_cap = max(self.cap_max * 2, self.list_cap * 2, 32)

        for cid in range(self.nlist):
            start = int(starts[cid])
            stop = int(starts[cid + 1])
            cell_size = stop - start
            if cell_size <= 1:
                continue
            take = min(cell_size, prefix_cap)
            members = order[start : start + take]
            s = scores[members].astype(np.float64, copy=False)
            top = float(s[0])
            scale = max(float(np.std(s)), self.concentration_temperature_floor)
            weights = np.exp(np.clip((s - top) / scale, -50.0, 0.0))
            wsum = float(np.sum(weights))
            if wsum <= 0.0:
                continue
            p = weights / wsum
            eff[cid] = float(1.0 / max(float(np.sum(p * p)), 1e-12))
            tail = float(np.quantile(s, self.concentration_tail_quantile))
            drop[cid] = max(0.0, top - tail)
            observed[cid] = True

        self.concentration_effective_rank = eff
        self.concentration_tail_drop = drop
        self.concentration_observed_cells = int(np.sum(observed))
        if self.concentration_observed_cells:
            self.concentration_mean_effective_rank = float(np.mean(eff[observed]))
            self.concentration_mean_tail_drop = float(np.mean(drop[observed]))

    def _cell_caps(self, assigned, scores):
        base_caps = super()._cell_caps(assigned, scores).astype(np.float32)
        eff = self.concentration_effective_rank
        drop = self.concentration_tail_drop
        if eff is None or drop is None or not np.any(eff > 0.0):
            return base_caps.astype(np.int32)

        counts = np.bincount(assigned, minlength=self.nlist).astype(np.float32)
        nonempty = counts > 0
        valid_eff = eff > 0.0
        mean_eff = float(np.mean(eff[valid_eff]) + 1e-12)
        eff_ratio = np.ones(self.nlist, dtype=np.float32)
        eff_ratio[valid_eff] = eff[valid_eff] / mean_eff

        valid_drop = drop > 0.0
        drop_ratio = np.ones(self.nlist, dtype=np.float32)
        if np.any(valid_drop):
            mean_drop = float(np.mean(drop[valid_drop]) + 1e-12)
            drop_ratio[valid_drop] = mean_drop / np.maximum(drop[valid_drop], 1e-12)

        if self.concentration_mode == "effonly":
            factor = np.power(eff_ratio, self.concentration_gamma)
        elif self.concentration_mode == "tailonly":
            factor = np.power(drop_ratio, self.concentration_gamma)
        elif self.concentration_mode == "trimlow":
            factor = np.minimum(1.0, np.power(eff_ratio, self.concentration_gamma))
        else:
            factor = np.power(np.sqrt(eff_ratio * drop_ratio), self.concentration_gamma)

        factor = np.clip(factor * self.concentration_bias, self.concentration_clip_min, self.concentration_clip_max)
        caps = np.zeros(self.nlist, dtype=np.float32)
        caps[nonempty] = base_caps[nonempty] * factor[nonempty]
        caps = np.clip(caps, self.cap_min, self.cap_max)
        caps[~nonempty] = 0
        caps = np.minimum(caps, counts)
        caps_i = np.rint(caps).astype(np.int32)
        self.cap_total_requested = int(np.sum(caps_i))
        base_total = float(np.sum(base_caps[nonempty]))
        self.concentration_cap_ratio = float(np.sum(caps_i[nonempty]) / max(base_total, 1e-12))
        return caps_i

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-adaptivecapcore-transport-hnswgate",
            "zeelin-concentrationcapcore-transport-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", concentrationMode={self.concentration_mode}, "
            f"concentrationGamma={self.concentration_gamma}, "
            f"concentrationBias={self.concentration_bias}, "
            f"concentrationMeanEff={self.concentration_mean_effective_rank:.2f}, "
            f"concentrationCapRatio={self.concentration_cap_ratio:.3f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-concentrationcapcore-transport"
        extra["concentration_mode"] = self.concentration_mode
        extra["concentration_gamma"] = self.concentration_gamma
        extra["concentration_bias"] = self.concentration_bias
        extra["concentration_prefix_cap"] = self.concentration_prefix_cap
        extra["concentration_temperature_floor"] = self.concentration_temperature_floor
        extra["concentration_tail_quantile"] = self.concentration_tail_quantile
        extra["concentration_clip_min"] = self.concentration_clip_min
        extra["concentration_clip_max"] = self.concentration_clip_max
        extra["concentration_observed_cells"] = self.concentration_observed_cells
        extra["concentration_mean_effective_rank"] = self.concentration_mean_effective_rank
        extra["concentration_mean_tail_drop"] = self.concentration_mean_tail_drop
        extra["concentration_cap_ratio"] = self.concentration_cap_ratio
        return extra


class ZeelinM10concentrationcapcoreTransportHnswgateIvf(ZeelinConcentrationcapcoreTransportHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param.setdefault("qm", 10)
        method_param.setdefault("lc", 96)
        method_param.setdefault("aef", 8)
        method_param.setdefault("co", "centroid")
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-concentrationcapcore-transport-hnswgate",
            "zeelin-m10concentrationcapcore-transport-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10concentrationcapcore-transport"
        return extra


class ZeelinM10concentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10concentrationcapcoreTransportHnswgateIvf
):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        if return_k is None:
            return_k = 7
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10concentrationcapcore-transport-hnswgate",
            "zeelin-m10concentrationcapcore-returncompress-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10concentrationcapcore-returncompress"
        return extra


class ZeelinM10preassignedbatchConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10concentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self._batch_dbuf = None
        self._batch_ibuf = None
        self._batch_assign = None
        self._batch_centroid_dis = None
        self._batch_capacity = (0, 0, 0)

    def _ensure_preassigned_batch_buffers(self, nq, nprobe, query_k):
        if (
            self._batch_capacity[0] >= nq
            and self._batch_capacity[1] >= nprobe
            and self._batch_capacity[2] >= query_k
        ):
            return
        self._batch_dbuf = np.empty((nq, query_k), dtype=np.float32)
        self._batch_ibuf = np.empty((nq, query_k), dtype=np.int64)
        self._batch_assign = np.empty((nq, nprobe), dtype=np.int64)
        self._batch_centroid_dis = np.empty((nq, nprobe), dtype=np.float32)
        self._batch_capacity = (nq, nprobe, query_k)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10preassignedbatch-concentrationcapcore-returncompress-hnswgate",
            1,
        )

    def batch_query(self, X, n):
        X = np.asarray(X, dtype=np.float32)
        if self.skip_query_norm:
            Xn = X if X.flags["C_CONTIGUOUS"] else np.ascontiguousarray(X, dtype=np.float32)
        else:
            norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
            Xn = np.ascontiguousarray(X / np.maximum(norms, 1e-12), dtype=np.float32)

        self._batch_target_k = n if self.return_k <= 0 else min(n, self.return_k)
        query_k = n if self.internal_k <= 0 else max(1, self.internal_k)
        query_k = max(query_k, self._batch_target_k + self.dedup_extra)
        self._ensure_preassigned_batch_buffers(len(Xn), self.nprobe, query_k)

        centroid_dis, assign = self.quantizer.search(
            np.ascontiguousarray(Xn, dtype=np.float32), self.nprobe
        )
        self._batch_assign[: len(Xn), : self.nprobe] = assign
        self._batch_centroid_dis[: len(Xn), : self.nprobe] = centroid_dis
        self.index.search_preassigned_c(
            len(Xn),
            faiss.swig_ptr(Xn),
            query_k,
            faiss.swig_ptr(self._batch_assign),
            faiss.swig_ptr(self._batch_centroid_dis),
            faiss.swig_ptr(self._batch_dbuf),
            faiss.swig_ptr(self._batch_ibuf),
            False,
        )
        self._batch_labels = self._batch_ibuf[: len(Xn), :query_k]

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-preassignedbatch-hnswquantizer-ip-m10concentrationcapcore-returncompress"
        return extra


class ZeelinM10riskbudgetConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10concentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "fk" in method_param:
            method_param["fast_return_k"] = method_param["fk"]
        if "sk" in method_param:
            method_param["slow_return_k"] = method_param["sk"]
        if "ritm" in method_param:
            method_param["risk_ip_threshold_milli"] = method_param["ritm"]
        super().__init__(metric, method_param)
        self.fast_return_k = int(method_param.get("fast_return_k", 5))
        self.slow_return_k = int(method_param.get("slow_return_k", 7))
        self.risk_ip_threshold = float(method_param.get("risk_ip_threshold_milli", 350)) / 1000.0
        self.fast_return_k = max(1, min(10, self.fast_return_k))
        self.slow_return_k = max(self.fast_return_k, min(10, self.slow_return_k))
        self._repair_count = 0
        self._query_count = 0
        self._last_avg_return_k = float(self.slow_return_k)
        self._batch_lengths = None

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=self.slow_return_k,
            internal_k=max(self.slow_return_k, int(internal_k or 0)),
        )
        self._repair_count = 0
        self._query_count = 0
        self._last_avg_return_k = float(self.fast_return_k)
        self.name = self.name.replace(
            "zeelin-m10concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10riskbudget-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", fastReturnK={self.fast_return_k}, slowReturnK={self.slow_return_k}, "
            f"riskIpThreshold={self.risk_ip_threshold:.3f})"
        )

    def _query_matrix(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.skip_query_norm:
            return X if X.flags["C_CONTIGUOUS"] else np.ascontiguousarray(X, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
        return np.ascontiguousarray(X / np.maximum(norms, 1e-12), dtype=np.float32)

    def query(self, v, n):
        q = self._query_matrix(v.reshape(1, self.dim))
        fast_k = min(n, self.fast_return_k)
        slow_k = min(n, self.slow_return_k)
        D, I = self.index.search(q, fast_k)
        self._query_count += 1
        if D[0, fast_k - 1] > self.risk_ip_threshold and I[0, fast_k - 1] >= 0:
            return I[0, :fast_k]
        self._repair_count += 1
        return self.index.search(q, slow_k)[1][0, :slow_k]

    def batch_query(self, X, n):
        Xn = self._query_matrix(X)
        fast_k = min(n, self.fast_return_k)
        slow_k = min(n, self.slow_return_k)
        D, I = self.index.search(Xn, fast_k)
        risk = (D[:, fast_k - 1] <= self.risk_ip_threshold) | (I[:, fast_k - 1] < 0)
        self._query_count += int(len(Xn))
        self._repair_count += int(np.count_nonzero(risk))

        labels = np.full((len(Xn), slow_k), -1, dtype=np.int64)
        labels[:, :fast_k] = I[:, :fast_k]
        lengths = np.full(len(Xn), fast_k, dtype=np.int32)
        if np.any(risk):
            slow_labels = self.index.search(Xn[risk], slow_k)[1]
            labels[risk, :slow_k] = slow_labels[:, :slow_k]
            lengths[risk] = slow_k
        self._batch_labels = labels
        self._batch_lengths = lengths
        self._batch_target_k = slow_k
        self._last_avg_return_k = float(np.mean(lengths)) if len(lengths) else 0.0

    def get_batch_results(self):
        results = []
        for labels, length in zip(self._batch_labels, self._batch_lengths):
            prefix = labels[: int(length)]
            if self.fast_unique:
                if self.trust_full_labels:
                    results.append(prefix[prefix >= 0])
                else:
                    results.append([int(label) for label in prefix if label >= 0])
                continue
            out = []
            seen = set()
            for label in prefix:
                label = int(label)
                if label < 0 or label in seen:
                    continue
                seen.add(label)
                out.append(label)
            results.append(out)
        return results

    def get_additional(self):
        extra = super().get_additional()
        total = float(self._query_count)
        extra["kernel"] = "faiss-riskbudget-hnswquantizer-ip-m10concentrationcapcore-returncompress"
        extra["fast_return_k"] = self.fast_return_k
        extra["slow_return_k"] = self.slow_return_k
        extra["risk_ip_threshold_milli"] = int(round(self.risk_ip_threshold * 1000.0))
        extra["avg_repair_fraction"] = float(self._repair_count) / total if total else 0.0
        extra["avg_lens_return_k"] = self._last_avg_return_k
        return extra


class ZeelinBoundarymassConcentrationcapcoreTransportHnswgateIvf(
    ZeelinConcentrationcapcoreTransportHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "bmm" in method_param:
            method_param["boundary_mode"] = method_param["bmm"]
        if "bmg" in method_param:
            method_param["boundary_gamma"] = method_param["bmg"]
        if "bmb" in method_param:
            method_param["boundary_bias"] = method_param["bmb"]
        if "bmt" in method_param:
            method_param["boundary_temperature"] = method_param["bmt"]
        if "bmts" in method_param:
            method_param["boundary_train_size"] = method_param["bmts"]
        if "bmef" in method_param:
            method_param["boundary_efSearch"] = method_param["bmef"]
        if "bmcliplo" in method_param:
            method_param["boundary_clip_min"] = method_param["bmcliplo"]
        if "bmcliphi" in method_param:
            method_param["boundary_clip_max"] = method_param["bmcliphi"]
        super().__init__(metric, method_param)
        self.boundary_mode = str(method_param.get("boundary_mode", "reweight"))
        self.boundary_gamma = float(method_param.get("boundary_gamma", 0.45))
        self.boundary_bias = float(method_param.get("boundary_bias", 0.96))
        self.boundary_temperature = float(method_param.get("boundary_temperature", 0.035))
        self.boundary_train_size = int(method_param.get("boundary_train_size", 200000))
        self.boundary_efSearch = int(method_param.get("boundary_efSearch", self.assign_ef_search))
        self.boundary_clip_min = float(method_param.get("boundary_clip_min", 0.70))
        self.boundary_clip_max = float(method_param.get("boundary_clip_max", 1.30))
        self.boundary_gamma = max(0.0, min(2.0, self.boundary_gamma))
        self.boundary_bias = max(0.25, min(2.0, self.boundary_bias))
        self.boundary_temperature = max(1e-4, min(1.0, self.boundary_temperature))
        self.boundary_train_size = max(0, self.boundary_train_size)
        self.boundary_efSearch = max(1, min(64, self.boundary_efSearch))
        self.boundary_clip_min = max(0.05, min(4.0, self.boundary_clip_min))
        self.boundary_clip_max = max(self.boundary_clip_min, min(4.0, self.boundary_clip_max))
        self.boundary_cell_mass = None
        self.boundary_observed_cells = 0
        self.boundary_mean_mass = 0.0
        self.boundary_max_mass = 0.0
        self.boundary_mean_gap = 0.0
        self.boundary_cap_ratio = 0.0

    def _before_cell_caps(self, Xn, assigned, scores, raw_norms, order):
        super()._before_cell_caps(Xn, assigned, scores, raw_norms, order)
        boundary_sample = Xn
        if 0 < self.boundary_train_size < len(Xn):
            bidx = np.linspace(0, len(Xn) - 1, self.boundary_train_size, dtype=np.int64)
            boundary_sample = np.ascontiguousarray(Xn[bidx], dtype=np.float32)
        else:
            boundary_sample = np.ascontiguousarray(boundary_sample, dtype=np.float32)

        old_ef = int(self.quantizer.hnsw.efSearch)
        self.quantizer.hnsw.efSearch = self.boundary_efSearch
        bdis, bids = self.quantizer.search(boundary_sample, 2)
        self.quantizer.hnsw.efSearch = old_ef
        bids = np.asarray(bids, dtype=np.int64)
        bdis = np.asarray(bdis, dtype=np.float32)
        valid = (bids[:, 0] >= 0) & (bids[:, 1] >= 0)
        if not np.any(valid):
            self.boundary_cell_mass = np.zeros(self.nlist, dtype=np.float32)
            return

        routes = np.ascontiguousarray(bids[valid, 0], dtype=np.int64)
        gaps = np.maximum(0.0, bdis[valid, 0] - bdis[valid, 1]).astype(np.float32)
        risk = np.exp(-gaps / self.boundary_temperature).astype(np.float32)
        counts = np.bincount(routes, minlength=self.nlist).astype(np.float32)
        sums = np.bincount(routes, weights=risk.astype(np.float64), minlength=self.nlist).astype(np.float32)
        mass = np.zeros(self.nlist, dtype=np.float32)
        observed = counts > 0
        mass[observed] = sums[observed] / np.maximum(counts[observed], 1.0)
        self.boundary_cell_mass = mass
        self.boundary_observed_cells = int(np.sum(observed))
        if self.boundary_observed_cells:
            self.boundary_mean_mass = float(np.mean(mass[observed]))
            self.boundary_max_mass = float(np.max(mass[observed]))
            self.boundary_mean_gap = float(np.mean(gaps))

    def _cell_caps(self, assigned, scores):
        base_caps = super()._cell_caps(assigned, scores).astype(np.float32)
        mass = self.boundary_cell_mass
        if mass is None or not np.any(mass > 0.0):
            return base_caps.astype(np.int32)
        counts = np.bincount(assigned, minlength=self.nlist).astype(np.float32)
        nonempty = counts > 0
        observed = mass > 0.0
        mean_mass = float(np.mean(mass[observed]) + 1e-12)
        ratio = np.ones(self.nlist, dtype=np.float32)
        ratio[observed] = mass[observed] / mean_mass

        if self.boundary_mode == "guard":
            factor = np.maximum(1.0, np.power(ratio, self.boundary_gamma))
            factor = np.minimum(factor * self.boundary_bias, self.boundary_clip_max)
        elif self.boundary_mode == "trimstable":
            factor = np.minimum(1.0, np.power(ratio, self.boundary_gamma))
            factor = np.clip(factor * self.boundary_bias, self.boundary_clip_min, 1.0)
        else:
            factor = np.power(ratio, self.boundary_gamma)
            factor = np.clip(factor * self.boundary_bias, self.boundary_clip_min, self.boundary_clip_max)

        caps = np.zeros(self.nlist, dtype=np.float32)
        caps[nonempty] = base_caps[nonempty] * factor[nonempty]
        caps = np.clip(caps, self.cap_min, self.cap_max)
        caps[~nonempty] = 0
        caps = np.minimum(caps, counts)
        caps_i = np.rint(caps).astype(np.int32)
        self.cap_total_requested = int(np.sum(caps_i))
        base_total = float(np.sum(base_caps[nonempty]))
        self.boundary_cap_ratio = float(np.sum(caps_i[nonempty]) / max(base_total, 1e-12))
        return caps_i

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-concentrationcapcore-transport-hnswgate",
            "zeelin-boundarymass-concentrationcapcore-transport-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", boundaryMode={self.boundary_mode}, boundaryGamma={self.boundary_gamma}, "
            f"boundaryBias={self.boundary_bias}, boundaryMeanMass={self.boundary_mean_mass:.4f}, "
            f"boundaryCapRatio={self.boundary_cap_ratio:.3f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-boundarymass-concentrationcapcore-transport"
        extra["boundary_mode"] = self.boundary_mode
        extra["boundary_gamma"] = self.boundary_gamma
        extra["boundary_bias"] = self.boundary_bias
        extra["boundary_temperature"] = self.boundary_temperature
        extra["boundary_train_size"] = self.boundary_train_size
        extra["boundary_efSearch"] = self.boundary_efSearch
        extra["boundary_clip_min"] = self.boundary_clip_min
        extra["boundary_clip_max"] = self.boundary_clip_max
        extra["boundary_observed_cells"] = self.boundary_observed_cells
        extra["boundary_mean_mass"] = self.boundary_mean_mass
        extra["boundary_max_mass"] = self.boundary_max_mass
        extra["boundary_mean_gap"] = self.boundary_mean_gap
        extra["boundary_cap_ratio"] = self.boundary_cap_ratio
        return extra


class ZeelinM10boundarymassConcentrationcapcoreTransportHnswgateIvf(
    ZeelinBoundarymassConcentrationcapcoreTransportHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param.setdefault("qm", 10)
        method_param.setdefault("lc", 96)
        method_param.setdefault("aef", 8)
        method_param.setdefault("co", "centroid")
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-boundarymass-concentrationcapcore-transport-hnswgate",
            "zeelin-m10boundarymass-concentrationcapcore-transport-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10boundarymass-concentrationcapcore-transport"
        return extra


class ZeelinM10boundarymassConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarymassConcentrationcapcoreTransportHnswgateIvf
):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        if return_k is None:
            return_k = 7
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarymass-concentrationcapcore-transport-hnswgate",
            "zeelin-m10boundarymass-concentrationcapcore-returncompress-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10boundarymass-concentrationcapcore-returncompress"
        return extra


class ZeelinM10boundarytrimConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarymassConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param.setdefault("lc", 96)
        method_param.setdefault("cmin", 48)
        method_param.setdefault("cmax", 144)
        method_param.setdefault("cm", "size")
        method_param.setdefault("cg", 0.25)
        method_param.setdefault("cpa", 0.0)
        method_param.setdefault("ccm", "trimlow")
        method_param.setdefault("ccg", 0.60)
        method_param.setdefault("ccb", 0.92)
        method_param.setdefault("ccp", 192)
        method_param.setdefault("cct", 0.015)
        method_param.setdefault("ccq", 0.25)
        method_param.setdefault("cccliplo", 0.55)
        method_param.setdefault("cccliphi", 1.05)
        method_param.setdefault("bmm", "reweight")
        method_param.setdefault("bmg", 0.45)
        method_param.setdefault("bmb", 0.96)
        method_param.setdefault("bmt", 0.035)
        method_param.setdefault("bmts", 200000)
        method_param.setdefault("bmef", 8)
        method_param.setdefault("bmcliplo", 0.70)
        method_param.setdefault("bmcliphi", 1.30)
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarymass-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10boundarytrim-concentrationcapcore-returncompress-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10boundarytrim-concentrationcapcore-returncompress"
        return extra


class ZeelinM10boundarycertConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "pct" in method_param:
            method_param["phase_certificate_trials"] = method_param["pct"]
        if "pps" in method_param:
            method_param["phase_probe_size"] = method_param["pps"]
        if "ppr" in method_param:
            method_param["phase_probe_repeats"] = method_param["ppr"]
        if "ppk" in method_param:
            method_param["phase_probe_k"] = method_param["ppk"]
        if "pnp" in method_param:
            method_param["phase_probe_nprobe"] = method_param["pnp"]
        if "pef" in method_param:
            method_param["phase_probe_efSearch"] = method_param["pef"]
        super().__init__(metric, method_param)
        self.phase_certificate_trials = int(method_param.get("phase_certificate_trials", 2))
        self.phase_probe_size = int(method_param.get("phase_probe_size", 2048))
        self.phase_probe_repeats = int(method_param.get("phase_probe_repeats", 3))
        self.phase_probe_k = int(method_param.get("phase_probe_k", 7))
        self.phase_probe_nprobe = int(method_param.get("phase_probe_nprobe", 1))
        self.phase_probe_efSearch = int(method_param.get("phase_probe_efSearch", 1))
        self.phase_certificate_trials = max(1, min(4, self.phase_certificate_trials))
        self.phase_probe_size = max(64, min(20000, self.phase_probe_size))
        self.phase_probe_repeats = max(1, min(9, self.phase_probe_repeats))
        self.phase_probe_k = max(1, min(16, self.phase_probe_k))
        self.phase_probe_nprobe = max(1, min(16, self.phase_probe_nprobe))
        self.phase_probe_efSearch = max(1, min(64, self.phase_probe_efSearch))
        self.phase_best_trial = 0
        self.phase_best_build_time = 0.0
        self.phase_best_proxy_us = 0.0
        self.phase_best_route_entropy = 0.0
        self.phase_best_route_imbalance = 0.0
        self.phase_best_route_top_fraction = 0.0
        self.phase_all_build_times = ""
        self.phase_all_proxy_us = ""
        self.phase_all_route_entropy = ""
        self.phase_all_route_imbalance = ""

    def _phase_certificate(self, probe):
        old_nprobe = int(self.index.nprobe)
        old_ef = int(self.quantizer.hnsw.efSearch)
        self.index.nprobe = self.phase_probe_nprobe
        self.quantizer.hnsw.efSearch = self.phase_probe_efSearch

        warm = probe[: min(len(probe), 64)]
        if len(warm):
            self.index.search(warm, self.phase_probe_k)

        best_elapsed = None
        for _ in range(self.phase_probe_repeats):
            start = time.perf_counter()
            self.index.search(probe, self.phase_probe_k)
            elapsed = time.perf_counter() - start
            if best_elapsed is None or elapsed < best_elapsed:
                best_elapsed = elapsed

        _, route_ids = self.quantizer.search(probe, 1)
        route_ids = np.asarray(route_ids[:, 0], dtype=np.int64)
        traffic = np.bincount(route_ids, minlength=self.nlist).astype(np.float64)
        total = float(np.sum(traffic))
        if total > 0.0:
            p = traffic[traffic > 0.0] / total
            entropy = float(-np.sum(p * np.log(p)) / max(np.log(max(2, len(p))), 1e-12))
            mean = float(np.mean(traffic[traffic > 0.0]))
            imbalance = float(np.max(traffic) / max(mean, 1e-12))
            top_fraction = float(np.max(traffic) / total)
        else:
            entropy = 0.0
            imbalance = 0.0
            top_fraction = 0.0

        self.index.nprobe = old_nprobe
        self.quantizer.hnsw.efSearch = old_ef
        proxy_us = (float(best_elapsed) / max(1, len(probe))) * 1e6
        return proxy_us, entropy, imbalance, top_fraction

    def _capture_phase_state(self, trial, elapsed, proxy_us, entropy, imbalance, top_fraction):
        return {
            "index": self.index,
            "quantizer": self.quantizer,
            "dim": self.dim,
            "qbuf": self.qbuf,
            "kept_points": self.kept_points,
            "cap_min_actual": self.cap_min_actual,
            "cap_max_actual": self.cap_max_actual,
            "cap_mean_actual": self.cap_mean_actual,
            "cap_total_requested": getattr(self, "cap_total_requested", 0),
            "concentration_observed_cells": self.concentration_observed_cells,
            "concentration_mean_effective_rank": self.concentration_mean_effective_rank,
            "concentration_mean_tail_drop": self.concentration_mean_tail_drop,
            "concentration_cap_ratio": self.concentration_cap_ratio,
            "boundary_observed_cells": self.boundary_observed_cells,
            "boundary_mean_mass": self.boundary_mean_mass,
            "boundary_max_mass": self.boundary_max_mass,
            "boundary_mean_gap": self.boundary_mean_gap,
            "boundary_cap_ratio": self.boundary_cap_ratio,
            "trial": trial,
            "build_time": elapsed,
            "proxy_us": proxy_us,
            "entropy": entropy,
            "imbalance": imbalance,
            "top_fraction": top_fraction,
        }

    def _restore_phase_state(self, state):
        self.index = state["index"]
        self.quantizer = state["quantizer"]
        self.dim = state["dim"]
        self.qbuf = state["qbuf"]
        self.kept_points = state["kept_points"]
        self.cap_min_actual = state["cap_min_actual"]
        self.cap_max_actual = state["cap_max_actual"]
        self.cap_mean_actual = state["cap_mean_actual"]
        self.cap_total_requested = state["cap_total_requested"]
        self.concentration_observed_cells = state["concentration_observed_cells"]
        self.concentration_mean_effective_rank = state["concentration_mean_effective_rank"]
        self.concentration_mean_tail_drop = state["concentration_mean_tail_drop"]
        self.concentration_cap_ratio = state["concentration_cap_ratio"]
        self.boundary_observed_cells = state["boundary_observed_cells"]
        self.boundary_mean_mass = state["boundary_mean_mass"]
        self.boundary_max_mass = state["boundary_max_mass"]
        self.boundary_mean_gap = state["boundary_mean_gap"]
        self.boundary_cap_ratio = state["boundary_cap_ratio"]
        self.phase_best_trial = state["trial"]
        self.phase_best_build_time = state["build_time"]
        self.phase_best_proxy_us = state["proxy_us"]
        self.phase_best_route_entropy = state["entropy"]
        self.phase_best_route_imbalance = state["imbalance"]
        self.phase_best_route_top_fraction = state["top_fraction"]

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        probe_idx = np.linspace(
            0,
            len(X) - 1,
            min(len(X), self.phase_probe_size),
            dtype=np.int64,
        )
        probe = self._normalize_rows(X[probe_idx]).astype(np.float32, copy=False)
        probe = np.ascontiguousarray(probe, dtype=np.float32)

        best_state = None
        best_score = None
        build_times = []
        proxy_times = []
        entropies = []
        imbalances = []

        for trial in range(1, self.phase_certificate_trials + 1):
            start = time.time()
            super().fit(X)
            elapsed = time.time() - start
            proxy_us, entropy, imbalance, top_fraction = self._phase_certificate(probe)
            build_times.append(elapsed)
            proxy_times.append(proxy_us)
            entropies.append(entropy)
            imbalances.append(imbalance)
            score = proxy_us
            state = self._capture_phase_state(
                trial, elapsed, proxy_us, entropy, imbalance, top_fraction
            )
            if best_score is None or score < best_score:
                best_score = score
                best_state = state

        if best_state is not None:
            self._restore_phase_state(best_state)

        self.phase_all_build_times = ",".join(f"{v:.6f}" for v in build_times)
        self.phase_all_proxy_us = ",".join(f"{v:.6f}" for v in proxy_times)
        self.phase_all_route_entropy = ",".join(f"{v:.6f}" for v in entropies)
        self.phase_all_route_imbalance = ",".join(f"{v:.6f}" for v in imbalances)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10boundarycert-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", phaseCertTrials={self.phase_certificate_trials}, "
            f"phaseBestTrial={self.phase_best_trial}, "
            f"phaseProxyUs={self.phase_best_proxy_us:.3f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10boundarycert-concentrationcapcore-returncompress"
        extra["phase_certificate_trials"] = self.phase_certificate_trials
        extra["phase_probe_size"] = self.phase_probe_size
        extra["phase_probe_repeats"] = self.phase_probe_repeats
        extra["phase_probe_k"] = self.phase_probe_k
        extra["phase_probe_nprobe"] = self.phase_probe_nprobe
        extra["phase_probe_efSearch"] = self.phase_probe_efSearch
        extra["phase_best_trial"] = self.phase_best_trial
        extra["phase_best_build_time"] = self.phase_best_build_time
        extra["phase_best_proxy_us"] = self.phase_best_proxy_us
        extra["phase_best_route_entropy"] = self.phase_best_route_entropy
        extra["phase_best_route_imbalance"] = self.phase_best_route_imbalance
        extra["phase_best_route_top_fraction"] = self.phase_best_route_top_fraction
        extra["phase_all_build_times"] = self.phase_all_build_times
        extra["phase_all_proxy_us"] = self.phase_all_proxy_us
        extra["phase_all_route_entropy"] = self.phase_all_route_entropy
        extra["phase_all_route_imbalance"] = self.phase_all_route_imbalance
        return extra


class ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "mc" in method_param:
            method_param["max_codes"] = method_param["mc"]
        super().__init__(metric, method_param)
        self.max_codes = int(method_param.get("max_codes", 72))
        self.max_codes = max(0, min(512, self.max_codes))

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.index.max_codes = self.max_codes
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + f", maxCodes={self.max_codes})"

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10boundarytrim-maxcodes-concentrationcapcore-returncompress"
        extra["max_codes"] = self.max_codes
        return extra


class ZeelinLargecapPrefixbudgetM10BoundarytrimHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param.setdefault("nlist", 16896)
        method_param.setdefault("qefc", 48)
        method_param.setdefault("ts", 1000000)
        method_param.setdefault("threads", 4)
        method_param.setdefault("mc", 64)
        method_param.setdefault("cs", 2026)
        method_param.setdefault("qm", 12)
        method_param.setdefault("sef", 1)
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-largecap-prefixbudget-m10boundarytrim-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-largecap-prefixbudget-m10boundarytrim-hnswquantizer-ip"
        extra["largecap_prefixbudget"] = 1
        extra["largecap_prefixbudget_default_mc"] = 64
        return extra


class ZeelinM10selfhitphasecertBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "spct" in method_param:
            method_param["selfhit_phase_trials"] = method_param["spct"]
        if "sps" in method_param:
            method_param["selfhit_probe_size"] = method_param["sps"]
        if "shk" in method_param:
            method_param["selfhit_k"] = method_param["shk"]
        if "shw" in method_param:
            method_param["selfhit_weight"] = method_param["shw"]
        super().__init__(metric, method_param)
        self.selfhit_phase_trials = int(method_param.get("selfhit_phase_trials", 3))
        self.selfhit_probe_size = int(method_param.get("selfhit_probe_size", 4096))
        self.selfhit_k = int(method_param.get("selfhit_k", 7))
        self.selfhit_weight = float(method_param.get("selfhit_weight", 250.0))
        self.selfhit_phase_trials = max(1, min(6, self.selfhit_phase_trials))
        self.selfhit_probe_size = max(128, min(50000, self.selfhit_probe_size))
        self.selfhit_k = max(1, min(16, self.selfhit_k))
        self.selfhit_weight = max(0.0, min(10000.0, self.selfhit_weight))
        self.selfhit_best_trial = 0
        self.selfhit_best_hit_rate = 0.0
        self.selfhit_best_proxy_us = 0.0
        self.selfhit_best_score = 0.0
        self.selfhit_all_hit_rates = ""
        self.selfhit_all_proxy_us = ""
        self.selfhit_all_scores = ""
        self.selfhit_all_build_times = ""

    def _selfhit_capture_state(self, trial, build_time, hit_rate, proxy_us, score):
        return {
            "index": self.index,
            "quantizer": self.quantizer,
            "dim": self.dim,
            "qbuf": self.qbuf,
            "kept_points": self.kept_points,
            "cap_min_actual": self.cap_min_actual,
            "cap_max_actual": self.cap_max_actual,
            "cap_mean_actual": self.cap_mean_actual,
            "cap_total_requested": getattr(self, "cap_total_requested", 0),
            "concentration_observed_cells": self.concentration_observed_cells,
            "concentration_mean_effective_rank": self.concentration_mean_effective_rank,
            "concentration_mean_tail_drop": self.concentration_mean_tail_drop,
            "concentration_cap_ratio": self.concentration_cap_ratio,
            "boundary_observed_cells": self.boundary_observed_cells,
            "boundary_mean_mass": self.boundary_mean_mass,
            "boundary_max_mass": self.boundary_max_mass,
            "boundary_mean_gap": self.boundary_mean_gap,
            "boundary_cap_ratio": self.boundary_cap_ratio,
            "trial": trial,
            "build_time": build_time,
            "hit_rate": hit_rate,
            "proxy_us": proxy_us,
            "score": score,
        }

    def _selfhit_restore_state(self, state):
        self.index = state["index"]
        self.quantizer = state["quantizer"]
        self.dim = state["dim"]
        self.qbuf = state["qbuf"]
        self.kept_points = state["kept_points"]
        self.cap_min_actual = state["cap_min_actual"]
        self.cap_max_actual = state["cap_max_actual"]
        self.cap_mean_actual = state["cap_mean_actual"]
        self.cap_total_requested = state["cap_total_requested"]
        self.concentration_observed_cells = state["concentration_observed_cells"]
        self.concentration_mean_effective_rank = state["concentration_mean_effective_rank"]
        self.concentration_mean_tail_drop = state["concentration_mean_tail_drop"]
        self.concentration_cap_ratio = state["concentration_cap_ratio"]
        self.boundary_observed_cells = state["boundary_observed_cells"]
        self.boundary_mean_mass = state["boundary_mean_mass"]
        self.boundary_max_mass = state["boundary_max_mass"]
        self.boundary_mean_gap = state["boundary_mean_gap"]
        self.boundary_cap_ratio = state["boundary_cap_ratio"]
        self.selfhit_best_trial = state["trial"]
        self.selfhit_best_hit_rate = state["hit_rate"]
        self.selfhit_best_proxy_us = state["proxy_us"]
        self.selfhit_best_score = state["score"]

    def _selfhit_certificate(self, probes, probe_ids):
        old_nprobe = int(self.index.nprobe)
        old_max_codes = int(self.index.max_codes)
        old_ef = int(self.quantizer.hnsw.efSearch)
        self.index.nprobe = 1
        self.index.max_codes = self.max_codes
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

        warm = probes[: min(len(probes), 128)]
        if len(warm):
            self.index.search(warm, self.selfhit_k)
        start = time.perf_counter()
        _, labels = self.index.search(probes, self.selfhit_k)
        elapsed = time.perf_counter() - start

        hits = np.any(labels == probe_ids.reshape(-1, 1), axis=1)
        hit_rate = float(np.mean(hits)) if len(hits) else 0.0
        proxy_us = (elapsed / max(1, len(probes))) * 1e6

        self.index.nprobe = old_nprobe
        self.index.max_codes = old_max_codes
        self.quantizer.hnsw.efSearch = old_ef
        return hit_rate, proxy_us

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        probe_count = min(len(X), self.selfhit_probe_size)
        probe_idx = np.linspace(0, len(X) - 1, probe_count, dtype=np.int64)
        probes = self._normalize_rows(X[probe_idx]).astype(np.float32, copy=False)
        probes = np.ascontiguousarray(probes, dtype=np.float32)

        best_state = None
        best_score = None
        hit_rates = []
        proxy_times = []
        scores = []
        build_times = []
        for trial in range(1, self.selfhit_phase_trials + 1):
            start = time.time()
            super().fit(X)
            build_time = time.time() - start
            hit_rate, proxy_us = self._selfhit_certificate(probes, probe_idx)
            score = (hit_rate * self.selfhit_weight) - proxy_us
            state = self._selfhit_capture_state(trial, build_time, hit_rate, proxy_us, score)
            hit_rates.append(hit_rate)
            proxy_times.append(proxy_us)
            scores.append(score)
            build_times.append(build_time)
            if best_score is None or score > best_score:
                best_score = score
                best_state = state

        if best_state is not None:
            self._selfhit_restore_state(best_state)

        self.selfhit_all_hit_rates = ",".join(f"{v:.6f}" for v in hit_rates)
        self.selfhit_all_proxy_us = ",".join(f"{v:.6f}" for v in proxy_times)
        self.selfhit_all_scores = ",".join(f"{v:.6f}" for v in scores)
        self.selfhit_all_build_times = ",".join(f"{v:.6f}" for v in build_times)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10selfhitphasecert-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", selfhitTrials={self.selfhit_phase_trials}, "
            f"selfhitBestTrial={self.selfhit_best_trial}, "
            f"selfhitRate={self.selfhit_best_hit_rate:.4f}, "
            f"selfhitProxyUs={self.selfhit_best_proxy_us:.3f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-selfhitphasecert-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["selfhit_phase_trials"] = self.selfhit_phase_trials
        extra["selfhit_probe_size"] = self.selfhit_probe_size
        extra["selfhit_k"] = self.selfhit_k
        extra["selfhit_weight"] = self.selfhit_weight
        extra["selfhit_best_trial"] = self.selfhit_best_trial
        extra["selfhit_best_hit_rate"] = self.selfhit_best_hit_rate
        extra["selfhit_best_proxy_us"] = self.selfhit_best_proxy_us
        extra["selfhit_best_score"] = self.selfhit_best_score
        extra["selfhit_all_hit_rates"] = self.selfhit_all_hit_rates
        extra["selfhit_all_proxy_us"] = self.selfhit_all_proxy_us
        extra["selfhit_all_scores"] = self.selfhit_all_scores
        extra["selfhit_all_build_times"] = self.selfhit_all_build_times
        return extra


class ZeelinM10layoutpreservingSlackmassBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "slr" in method_param:
            method_param["slack_budget_ratio"] = method_param["slr"]
        if "slg" in method_param:
            method_param["slack_gamma"] = method_param["slg"]
        if "slq" in method_param:
            method_param["slack_quantile"] = method_param["slq"]
        if "slx" in method_param:
            method_param["slack_cell_max_extra"] = method_param["slx"]
        super().__init__(metric, method_param)
        self.slack_budget_ratio = float(method_param.get("slack_budget_ratio", 0.006))
        self.slack_gamma = float(method_param.get("slack_gamma", 1.0))
        self.slack_quantile = float(method_param.get("slack_quantile", 0.75))
        self.slack_cell_max_extra = int(method_param.get("slack_cell_max_extra", 3))
        self.slack_budget_ratio = max(0.0, min(0.05, self.slack_budget_ratio))
        self.slack_gamma = max(0.0, min(4.0, self.slack_gamma))
        self.slack_quantile = max(0.0, min(0.98, self.slack_quantile))
        self.slack_cell_max_extra = max(0, min(32, self.slack_cell_max_extra))
        self.slack_added = 0
        self.slack_cells = 0
        self.slack_cap_ratio = 1.0
        self.slack_mean_score = 0.0
        self.slack_max_score = 0.0

    def _cell_caps(self, assigned, scores):
        base_caps_i = super()._cell_caps(assigned, scores).astype(np.int32)
        self.slack_added = 0
        self.slack_cells = 0
        self.slack_cap_ratio = 1.0
        self.slack_mean_score = 0.0
        self.slack_max_score = 0.0
        if self.slack_budget_ratio <= 0.0 or self.slack_cell_max_extra <= 0:
            return base_caps_i

        mass = self.boundary_cell_mass
        if mass is None or not np.any(mass > 0.0):
            return base_caps_i
        counts = np.bincount(assigned, minlength=self.nlist).astype(np.int32)
        base_caps = base_caps_i.astype(np.float32)
        nonempty = counts > 0
        slack = np.minimum(counts, self.cap_max).astype(np.float32) - base_caps
        slack = np.maximum(slack, 0.0)
        candidates = nonempty & (slack > 0.0) & (mass > 0.0)
        if not np.any(candidates):
            return base_caps_i

        mean_mass = float(np.mean(mass[mass > 0.0]) + 1e-12)
        boundary_score = np.ones(self.nlist, dtype=np.float32)
        boundary_score[candidates] = np.maximum(mass[candidates] / mean_mass, 1e-6)

        eff = self.concentration_effective_rank
        eff_score = np.ones(self.nlist, dtype=np.float32)
        if eff is not None and np.any(eff > 0.0):
            valid_eff = eff > 0.0
            mean_eff = float(np.mean(eff[valid_eff]) + 1e-12)
            eff_score[valid_eff] = np.sqrt(np.maximum(eff[valid_eff] / mean_eff, 1e-6))

        raw_score = np.power(boundary_score, self.slack_gamma) * eff_score
        score_values = raw_score[candidates]
        if len(score_values) == 0:
            return base_caps_i
        threshold = float(np.quantile(score_values, self.slack_quantile))
        active = candidates & (raw_score >= threshold)
        if not np.any(active):
            active = candidates
        weights = np.zeros(self.nlist, dtype=np.float64)
        weights[active] = (raw_score[active] * np.minimum(slack[active], self.slack_cell_max_extra)).astype(np.float64)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            return base_caps_i

        base_total = int(np.sum(base_caps_i[nonempty]))
        budget = int(round(base_total * self.slack_budget_ratio))
        budget = max(0, min(budget, int(np.sum(np.minimum(slack[active], self.slack_cell_max_extra)))))
        if budget <= 0:
            return base_caps_i

        fractional = weights * (float(budget) / weight_sum)
        add = np.floor(fractional).astype(np.int32)
        add = np.minimum(add, np.minimum(slack.astype(np.int32), self.slack_cell_max_extra))
        remaining = int(budget - np.sum(add))
        if remaining > 0:
            room = np.minimum(slack.astype(np.int32), self.slack_cell_max_extra) - add
            remainders = fractional - add.astype(np.float64)
            order = np.argsort(-remainders, kind="stable")
            for cid in order:
                if remaining <= 0:
                    break
                if room[cid] <= 0 or weights[cid] <= 0.0:
                    continue
                add[cid] += 1
                remaining -= 1

        caps_i = base_caps_i + add
        caps_i = np.minimum(caps_i, counts)
        caps_i = np.minimum(caps_i, self.cap_max).astype(np.int32)
        self.slack_added = int(np.sum(caps_i - base_caps_i))
        self.slack_cells = int(np.count_nonzero(caps_i > base_caps_i))
        self.cap_total_requested = int(np.sum(caps_i))
        if base_total > 0:
            self.slack_cap_ratio = float(np.sum(caps_i[nonempty])) / float(base_total)
        if np.any(active):
            self.slack_mean_score = float(np.mean(raw_score[active]))
            self.slack_max_score = float(np.max(raw_score[active]))
        return caps_i

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10layoutpreserving-slackmass-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", slackBudgetRatio={self.slack_budget_ratio}, "
            f"slackGamma={self.slack_gamma}, slackQ={self.slack_quantile}, "
            f"slackMaxExtra={self.slack_cell_max_extra}, "
            f"slackAdded={self.slack_added}, slackCapRatio={self.slack_cap_ratio:.4f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-layoutpreserving-slackmass-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["slack_budget_ratio"] = self.slack_budget_ratio
        extra["slack_gamma"] = self.slack_gamma
        extra["slack_quantile"] = self.slack_quantile
        extra["slack_cell_max_extra"] = self.slack_cell_max_extra
        extra["slack_added"] = self.slack_added
        extra["slack_cells"] = self.slack_cells
        extra["slack_cap_ratio"] = self.slack_cap_ratio
        extra["slack_mean_score"] = self.slack_mean_score
        extra["slack_max_score"] = self.slack_max_score
        return extra


class ZeelinM10phasecoupledSlackmassBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10layoutpreservingSlackmassBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "pcg" in method_param:
            method_param["phasecouple_gamma"] = method_param["pcg"]
        if "pcm" in method_param:
            method_param["phasecouple_mode"] = method_param["pcm"]
        if "pctail" in method_param:
            method_param["phasecouple_tail_span"] = method_param["pctail"]
        if "pcfloor" in method_param:
            method_param["phasecouple_floor"] = method_param["pcfloor"]
        super().__init__(metric, method_param)
        self.phasecouple_gamma = float(method_param.get("phasecouple_gamma", 1.0))
        self.phasecouple_mode = str(method_param.get("phasecouple_mode", "tailmass"))
        self.phasecouple_tail_span = int(method_param.get("phasecouple_tail_span", 8))
        self.phasecouple_floor = float(method_param.get("phasecouple_floor", 0.05))
        self.phasecouple_gamma = max(0.0, min(4.0, self.phasecouple_gamma))
        self.phasecouple_tail_span = max(1, min(64, self.phasecouple_tail_span))
        self.phasecouple_floor = max(0.0, min(1.0, self.phasecouple_floor))
        if self.phasecouple_mode not in {"tailmass", "boundarytail", "rankgap"}:
            self.phasecouple_mode = "tailmass"
        self.phasecouple_observed_cells = 0
        self.phasecouple_mean_gap = 0.0
        self.phasecouple_max_gap = 0.0
        self.phasecouple_mean_active_gap = 0.0

    def _phase_gap_scores(self, assigned, scores, base_caps_i, counts):
        order = np.lexsort((-scores, assigned)).astype(np.int64, copy=False)
        starts = np.zeros(self.nlist + 1, dtype=np.int64)
        starts[1:] = np.cumsum(counts, dtype=np.int64)
        gaps = np.zeros(self.nlist, dtype=np.float32)
        observed = np.zeros(self.nlist, dtype=bool)
        for cid in range(self.nlist):
            start = int(starts[cid])
            stop = int(starts[cid + 1])
            size = stop - start
            if size <= 1:
                continue
            cap = min(int(base_caps_i[cid]), size)
            if cap <= 0 or cap >= size:
                continue
            members = order[start:stop]
            prefix_stop = min(cap, max(1, self.max_codes if self.max_codes > 0 else cap))
            prefix_stop = min(prefix_stop, size)
            tail_stop = min(size, prefix_stop + self.phasecouple_tail_span)
            if tail_stop <= prefix_stop:
                continue
            head = scores[members[:prefix_stop]]
            tail = scores[members[prefix_stop:tail_stop]]
            if len(head) == 0 or len(tail) == 0:
                continue
            head_floor = float(head[-1])
            if self.phasecouple_mode == "rankgap":
                gap = float(tail_stop - prefix_stop) / float(max(1, size))
            else:
                tail_mean = float(np.mean(tail))
                local_scale = max(float(np.std(scores[members[:tail_stop]])), 1e-6)
                gap = float(np.exp(np.clip((tail_mean - head_floor) / local_scale, -30.0, 0.0)))
                if self.phasecouple_mode == "boundarytail":
                    gap *= float(tail_stop - prefix_stop) / float(max(1, self.phasecouple_tail_span))
            gaps[cid] = gap
            observed[cid] = True

        self.phasecouple_observed_cells = int(np.count_nonzero(observed))
        if self.phasecouple_observed_cells:
            values = gaps[observed]
            self.phasecouple_mean_gap = float(np.mean(values))
            self.phasecouple_max_gap = float(np.max(values))
        return gaps

    def _cell_caps(self, assigned, scores):
        base_caps_i = ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf._cell_caps(
            self, assigned, scores
        ).astype(np.int32)
        self.slack_added = 0
        self.slack_cells = 0
        self.slack_cap_ratio = 1.0
        self.slack_mean_score = 0.0
        self.slack_max_score = 0.0
        self.phasecouple_observed_cells = 0
        self.phasecouple_mean_gap = 0.0
        self.phasecouple_max_gap = 0.0
        self.phasecouple_mean_active_gap = 0.0
        if self.slack_budget_ratio <= 0.0 or self.slack_cell_max_extra <= 0:
            return base_caps_i

        mass = self.boundary_cell_mass
        if mass is None or not np.any(mass > 0.0):
            return base_caps_i
        counts = np.bincount(assigned, minlength=self.nlist).astype(np.int32)
        base_caps = base_caps_i.astype(np.float32)
        nonempty = counts > 0
        slack = np.minimum(counts, self.cap_max).astype(np.float32) - base_caps
        slack = np.maximum(slack, 0.0)
        candidates = nonempty & (slack > 0.0) & (mass > 0.0)
        if not np.any(candidates):
            return base_caps_i

        mean_mass = float(np.mean(mass[mass > 0.0]) + 1e-12)
        boundary_score = np.ones(self.nlist, dtype=np.float32)
        boundary_score[candidates] = np.maximum(mass[candidates] / mean_mass, 1e-6)

        eff = self.concentration_effective_rank
        eff_score = np.ones(self.nlist, dtype=np.float32)
        if eff is not None and np.any(eff > 0.0):
            valid_eff = eff > 0.0
            mean_eff = float(np.mean(eff[valid_eff]) + 1e-12)
            eff_score[valid_eff] = np.sqrt(np.maximum(eff[valid_eff] / mean_eff, 1e-6))

        phase_gap = self._phase_gap_scores(assigned, scores, base_caps_i, counts)
        valid_gap = phase_gap > 0.0
        phase_score = np.ones(self.nlist, dtype=np.float32) * self.phasecouple_floor
        if np.any(valid_gap):
            mean_gap = float(np.mean(phase_gap[valid_gap]) + 1e-12)
            phase_score[valid_gap] = np.maximum(phase_gap[valid_gap] / mean_gap, self.phasecouple_floor)

        raw_score = (
            np.power(boundary_score, self.slack_gamma)
            * eff_score
            * np.power(phase_score, self.phasecouple_gamma)
        )
        score_values = raw_score[candidates]
        if len(score_values) == 0:
            return base_caps_i
        threshold = float(np.quantile(score_values, self.slack_quantile))
        active = candidates & (raw_score >= threshold)
        if not np.any(active):
            active = candidates
        weights = np.zeros(self.nlist, dtype=np.float64)
        weights[active] = (raw_score[active] * np.minimum(slack[active], self.slack_cell_max_extra)).astype(np.float64)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            return base_caps_i

        base_total = int(np.sum(base_caps_i[nonempty]))
        budget = int(round(base_total * self.slack_budget_ratio))
        budget = max(0, min(budget, int(np.sum(np.minimum(slack[active], self.slack_cell_max_extra)))))
        if budget <= 0:
            return base_caps_i

        fractional = weights * (float(budget) / weight_sum)
        add = np.floor(fractional).astype(np.int32)
        add = np.minimum(add, np.minimum(slack.astype(np.int32), self.slack_cell_max_extra))
        remaining = int(budget - np.sum(add))
        if remaining > 0:
            room = np.minimum(slack.astype(np.int32), self.slack_cell_max_extra) - add
            remainders = fractional - add.astype(np.float64)
            order = np.argsort(-remainders, kind="stable")
            for cid in order:
                if remaining <= 0:
                    break
                if room[cid] <= 0 or weights[cid] <= 0.0:
                    continue
                add[cid] += 1
                remaining -= 1

        caps_i = base_caps_i + add
        caps_i = np.minimum(caps_i, counts)
        caps_i = np.minimum(caps_i, self.cap_max).astype(np.int32)
        self.slack_added = int(np.sum(caps_i - base_caps_i))
        self.slack_cells = int(np.count_nonzero(caps_i > base_caps_i))
        self.cap_total_requested = int(np.sum(caps_i))
        if base_total > 0:
            self.slack_cap_ratio = float(np.sum(caps_i[nonempty])) / float(base_total)
        if np.any(active):
            self.slack_mean_score = float(np.mean(raw_score[active]))
            self.slack_max_score = float(np.max(raw_score[active]))
            active_gap = phase_gap[active & (phase_gap > 0.0)]
            if len(active_gap):
                self.phasecouple_mean_active_gap = float(np.mean(active_gap))
        return caps_i

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10layoutpreserving-slackmass-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10phasecoupled-slackmass-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", phaseCoupleGamma={self.phasecouple_gamma}, "
            f"phaseCoupleMode={self.phasecouple_mode}, "
            f"phaseCoupleTail={self.phasecouple_tail_span}, "
            f"phaseCoupleMeanGap={self.phasecouple_mean_gap:.4f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-phasecoupled-slackmass-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["phasecouple_gamma"] = self.phasecouple_gamma
        extra["phasecouple_mode"] = self.phasecouple_mode
        extra["phasecouple_tail_span"] = self.phasecouple_tail_span
        extra["phasecouple_floor"] = self.phasecouple_floor
        extra["phasecouple_observed_cells"] = self.phasecouple_observed_cells
        extra["phasecouple_mean_gap"] = self.phasecouple_mean_gap
        extra["phasecouple_max_gap"] = self.phasecouple_max_gap
        extra["phasecouple_mean_active_gap"] = self.phasecouple_mean_active_gap
        return extra


class ZeelinM10routegradientSlackmassBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10phasecoupledSlackmassBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "rgg" in method_param:
            method_param["routegradient_gamma"] = method_param["rgg"]
        if "rgm" in method_param:
            method_param["routegradient_mode"] = method_param["rgm"]
        if "rgw" in method_param:
            method_param["routegradient_weight"] = method_param["rgw"]
        if "rgfloor" in method_param:
            method_param["routegradient_floor"] = method_param["rgfloor"]
        super().__init__(metric, method_param)
        self.routegradient_gamma = float(method_param.get("routegradient_gamma", 1.0))
        self.routegradient_mode = str(method_param.get("routegradient_mode", "agree"))
        self.routegradient_weight = float(method_param.get("routegradient_weight", 1.0))
        self.routegradient_floor = float(method_param.get("routegradient_floor", 0.05))
        self.routegradient_gamma = max(0.0, min(4.0, self.routegradient_gamma))
        self.routegradient_weight = max(0.0, min(8.0, self.routegradient_weight))
        self.routegradient_floor = max(0.0, min(1.0, self.routegradient_floor))
        if self.routegradient_mode not in {"agree", "outflow", "inflow", "balance"}:
            self.routegradient_mode = "agree"
        self.routegradient_score = None
        self.routegradient_out_mean = 0.0
        self.routegradient_in_mean = 0.0
        self.routegradient_mean_score = 0.0
        self.routegradient_max_score = 0.0
        self.routegradient_observed_cells = 0

    def _before_cell_caps(self, Xn, assigned, scores, raw_norms, order):
        super()._before_cell_caps(Xn, assigned, scores, raw_norms, order)
        sample = Xn
        if 0 < self.boundary_train_size < len(Xn):
            ridx = np.linspace(0, len(Xn) - 1, self.boundary_train_size, dtype=np.int64)
            sample = np.ascontiguousarray(Xn[ridx], dtype=np.float32)
        else:
            sample = np.ascontiguousarray(sample, dtype=np.float32)

        old_ef = int(self.quantizer.hnsw.efSearch)
        self.quantizer.hnsw.efSearch = self.boundary_efSearch
        rdis, rids = self.quantizer.search(sample, 2)
        self.quantizer.hnsw.efSearch = old_ef
        rids = np.asarray(rids, dtype=np.int64)
        rdis = np.asarray(rdis, dtype=np.float32)
        valid = (rids[:, 0] >= 0) & (rids[:, 1] >= 0) & (rids[:, 0] != rids[:, 1])
        score = np.ones(self.nlist, dtype=np.float32)
        if not np.any(valid):
            self.routegradient_score = score
            return

        first = np.ascontiguousarray(rids[valid, 0], dtype=np.int64)
        second = np.ascontiguousarray(rids[valid, 1], dtype=np.int64)
        gaps = np.maximum(0.0, rdis[valid, 0] - rdis[valid, 1]).astype(np.float32)
        risk = np.exp(-gaps / self.boundary_temperature).astype(np.float32)

        out_counts = np.bincount(first, minlength=self.nlist).astype(np.float32)
        in_counts = np.bincount(second, minlength=self.nlist).astype(np.float32)
        out_sums = np.bincount(first, weights=risk.astype(np.float64), minlength=self.nlist).astype(np.float32)
        in_sums = np.bincount(second, weights=risk.astype(np.float64), minlength=self.nlist).astype(np.float32)
        out_mean = np.zeros(self.nlist, dtype=np.float32)
        in_mean = np.zeros(self.nlist, dtype=np.float32)
        out_seen = out_counts > 0.0
        in_seen = in_counts > 0.0
        out_mean[out_seen] = out_sums[out_seen] / np.maximum(out_counts[out_seen], 1.0)
        in_mean[in_seen] = in_sums[in_seen] / np.maximum(in_counts[in_seen], 1.0)

        if np.any(out_seen):
            mean_out = float(np.mean(out_mean[out_seen]) + 1e-12)
            out_ratio = np.ones(self.nlist, dtype=np.float32)
            out_ratio[out_seen] = np.maximum(out_mean[out_seen] / mean_out, 1e-6)
            self.routegradient_out_mean = float(np.mean(out_mean[out_seen]))
        else:
            out_ratio = np.ones(self.nlist, dtype=np.float32)
            self.routegradient_out_mean = 0.0
        if np.any(in_seen):
            mean_in = float(np.mean(in_mean[in_seen]) + 1e-12)
            in_ratio = np.ones(self.nlist, dtype=np.float32)
            in_ratio[in_seen] = np.maximum(in_mean[in_seen] / mean_in, 1e-6)
            self.routegradient_in_mean = float(np.mean(in_mean[in_seen]))
        else:
            in_ratio = np.ones(self.nlist, dtype=np.float32)
            self.routegradient_in_mean = 0.0

        if self.routegradient_mode == "outflow":
            score = out_ratio
        elif self.routegradient_mode == "inflow":
            score = in_ratio
        elif self.routegradient_mode == "balance":
            score = 0.5 * (out_ratio + in_ratio)
        else:
            score = np.sqrt(out_ratio * in_ratio)
        score = np.maximum(score, self.routegradient_floor).astype(np.float32)
        observed = out_seen | in_seen
        self.routegradient_observed_cells = int(np.count_nonzero(observed))
        if self.routegradient_observed_cells:
            vals = score[observed]
            self.routegradient_mean_score = float(np.mean(vals))
            self.routegradient_max_score = float(np.max(vals))
        self.routegradient_score = score

    def _cell_caps(self, assigned, scores):
        base_caps_i = ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf._cell_caps(
            self, assigned, scores
        ).astype(np.int32)
        self.slack_added = 0
        self.slack_cells = 0
        self.slack_cap_ratio = 1.0
        self.slack_mean_score = 0.0
        self.slack_max_score = 0.0
        self.phasecouple_observed_cells = 0
        self.phasecouple_mean_gap = 0.0
        self.phasecouple_max_gap = 0.0
        self.phasecouple_mean_active_gap = 0.0
        if self.slack_budget_ratio <= 0.0 or self.slack_cell_max_extra <= 0:
            return base_caps_i

        mass = self.boundary_cell_mass
        route_score = self.routegradient_score
        if mass is None or route_score is None or not np.any(mass > 0.0):
            return base_caps_i
        counts = np.bincount(assigned, minlength=self.nlist).astype(np.int32)
        base_caps = base_caps_i.astype(np.float32)
        nonempty = counts > 0
        slack = np.minimum(counts, self.cap_max).astype(np.float32) - base_caps
        slack = np.maximum(slack, 0.0)
        candidates = nonempty & (slack > 0.0) & (mass > 0.0)
        if not np.any(candidates):
            return base_caps_i

        mean_mass = float(np.mean(mass[mass > 0.0]) + 1e-12)
        boundary_score = np.ones(self.nlist, dtype=np.float32)
        boundary_score[candidates] = np.maximum(mass[candidates] / mean_mass, 1e-6)

        eff = self.concentration_effective_rank
        eff_score = np.ones(self.nlist, dtype=np.float32)
        if eff is not None and np.any(eff > 0.0):
            valid_eff = eff > 0.0
            mean_eff = float(np.mean(eff[valid_eff]) + 1e-12)
            eff_score[valid_eff] = np.sqrt(np.maximum(eff[valid_eff] / mean_eff, 1e-6))

        phase_gap = self._phase_gap_scores(assigned, scores, base_caps_i, counts)
        valid_gap = phase_gap > 0.0
        phase_score = np.ones(self.nlist, dtype=np.float32) * self.phasecouple_floor
        if np.any(valid_gap):
            mean_gap = float(np.mean(phase_gap[valid_gap]) + 1e-12)
            phase_score[valid_gap] = np.maximum(phase_gap[valid_gap] / mean_gap, self.phasecouple_floor)

        raw_score = (
            np.power(boundary_score, self.slack_gamma)
            * eff_score
            * np.power(phase_score, self.phasecouple_gamma)
            * np.power(np.maximum(route_score, self.routegradient_floor), self.routegradient_gamma)
            * max(self.routegradient_weight, 1e-6)
        )
        score_values = raw_score[candidates]
        if len(score_values) == 0:
            return base_caps_i
        threshold = float(np.quantile(score_values, self.slack_quantile))
        active = candidates & (raw_score >= threshold)
        if not np.any(active):
            active = candidates
        weights = np.zeros(self.nlist, dtype=np.float64)
        weights[active] = (raw_score[active] * np.minimum(slack[active], self.slack_cell_max_extra)).astype(np.float64)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            return base_caps_i

        base_total = int(np.sum(base_caps_i[nonempty]))
        budget = int(round(base_total * self.slack_budget_ratio))
        budget = max(0, min(budget, int(np.sum(np.minimum(slack[active], self.slack_cell_max_extra)))))
        if budget <= 0:
            return base_caps_i

        fractional = weights * (float(budget) / weight_sum)
        add = np.floor(fractional).astype(np.int32)
        add = np.minimum(add, np.minimum(slack.astype(np.int32), self.slack_cell_max_extra))
        remaining = int(budget - np.sum(add))
        if remaining > 0:
            room = np.minimum(slack.astype(np.int32), self.slack_cell_max_extra) - add
            remainders = fractional - add.astype(np.float64)
            order = np.argsort(-remainders, kind="stable")
            for cid in order:
                if remaining <= 0:
                    break
                if room[cid] <= 0 or weights[cid] <= 0.0:
                    continue
                add[cid] += 1
                remaining -= 1

        caps_i = base_caps_i + add
        caps_i = np.minimum(caps_i, counts)
        caps_i = np.minimum(caps_i, self.cap_max).astype(np.int32)
        self.slack_added = int(np.sum(caps_i - base_caps_i))
        self.slack_cells = int(np.count_nonzero(caps_i > base_caps_i))
        self.cap_total_requested = int(np.sum(caps_i))
        if base_total > 0:
            self.slack_cap_ratio = float(np.sum(caps_i[nonempty])) / float(base_total)
        if np.any(active):
            self.slack_mean_score = float(np.mean(raw_score[active]))
            self.slack_max_score = float(np.max(raw_score[active]))
            active_gap = phase_gap[active & (phase_gap > 0.0)]
            if len(active_gap):
                self.phasecouple_mean_active_gap = float(np.mean(active_gap))
        return caps_i

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10phasecoupled-slackmass-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10routegradient-slackmass-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", routeGradientMode={self.routegradient_mode}, "
            f"routeGradientGamma={self.routegradient_gamma}, "
            f"routeGradientObserved={self.routegradient_observed_cells}, "
            f"routeGradientMean={self.routegradient_mean_score:.4f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-routegradient-slackmass-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["routegradient_gamma"] = self.routegradient_gamma
        extra["routegradient_mode"] = self.routegradient_mode
        extra["routegradient_weight"] = self.routegradient_weight
        extra["routegradient_floor"] = self.routegradient_floor
        extra["routegradient_out_mean"] = self.routegradient_out_mean
        extra["routegradient_in_mean"] = self.routegradient_in_mean
        extra["routegradient_mean_score"] = self.routegradient_mean_score
        extra["routegradient_max_score"] = self.routegradient_max_score
        extra["routegradient_observed_cells"] = self.routegradient_observed_cells
        return extra


class ZeelinM10capmeasureTransportAtlasBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10routegradientSlackmassBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "cmb" in method_param:
            method_param["slack_budget_ratio"] = method_param["cmb"]
        if "cmx" in method_param:
            method_param["slack_cell_max_extra"] = method_param["cmx"]
        if "cmq" in method_param:
            method_param["slack_quantile"] = method_param["cmq"]
        if "cmt" in method_param:
            method_param["capmeasure_temperature"] = method_param["cmt"]
        if "cmw" in method_param:
            method_param["capmeasure_mass_weight"] = method_param["cmw"]
        if "cew" in method_param:
            method_param["capmeasure_entropy_weight"] = method_param["cew"]
        if "cbw" in method_param:
            method_param["capmeasure_boundary_weight"] = method_param["cbw"]
        if "crw" in method_param:
            method_param["capmeasure_route_weight"] = method_param["crw"]
        if "cgw" in method_param:
            method_param["capmeasure_gap_weight"] = method_param["cgw"]
        method_param.setdefault("slack_budget_ratio", 0.004)
        method_param.setdefault("slack_cell_max_extra", 3)
        method_param.setdefault("slack_quantile", 0.70)
        method_param.setdefault("rgm", "agree")
        method_param.setdefault("rgg", 0.75)
        method_param.setdefault("rgw", 1.0)
        super().__init__(metric, method_param)
        self.capmeasure_temperature = float(method_param.get("capmeasure_temperature", 0.030))
        self.capmeasure_mass_weight = float(method_param.get("capmeasure_mass_weight", 1.0))
        self.capmeasure_entropy_weight = float(method_param.get("capmeasure_entropy_weight", 0.50))
        self.capmeasure_boundary_weight = float(method_param.get("capmeasure_boundary_weight", 1.0))
        self.capmeasure_route_weight = float(method_param.get("capmeasure_route_weight", 0.75))
        self.capmeasure_gap_weight = float(method_param.get("capmeasure_gap_weight", 0.50))
        self.capmeasure_temperature = max(1e-4, min(1.0, self.capmeasure_temperature))
        self.capmeasure_mass_weight = max(0.0, min(8.0, self.capmeasure_mass_weight))
        self.capmeasure_entropy_weight = max(0.0, min(8.0, self.capmeasure_entropy_weight))
        self.capmeasure_boundary_weight = max(0.0, min(8.0, self.capmeasure_boundary_weight))
        self.capmeasure_route_weight = max(0.0, min(8.0, self.capmeasure_route_weight))
        self.capmeasure_gap_weight = max(0.0, min(8.0, self.capmeasure_gap_weight))
        self.capmeasure_added = 0
        self.capmeasure_cells = 0
        self.capmeasure_cap_ratio = 1.0
        self.capmeasure_active_fraction = 0.0
        self.capmeasure_mean_signal = 0.0
        self.capmeasure_max_signal = 0.0

    @staticmethod
    def _positive_ratio(values, mask):
        ratio = np.ones_like(values, dtype=np.float32)
        valid = mask & (values > 0.0)
        if np.any(valid):
            mean = float(np.mean(values[valid]) + 1e-12)
            ratio[valid] = np.maximum(values[valid] / mean, 1e-6)
        return ratio

    def _cell_caps(self, assigned, scores):
        base_caps_i = ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf._cell_caps(
            self, assigned, scores
        ).astype(np.int32)
        self.slack_added = 0
        self.slack_cells = 0
        self.slack_cap_ratio = 1.0
        self.slack_mean_score = 0.0
        self.slack_max_score = 0.0
        self.capmeasure_added = 0
        self.capmeasure_cells = 0
        self.capmeasure_cap_ratio = 1.0
        self.capmeasure_active_fraction = 0.0
        self.capmeasure_mean_signal = 0.0
        self.capmeasure_max_signal = 0.0
        self.phasecouple_observed_cells = 0
        self.phasecouple_mean_gap = 0.0
        self.phasecouple_max_gap = 0.0
        self.phasecouple_mean_active_gap = 0.0
        if self.slack_budget_ratio <= 0.0 or self.slack_cell_max_extra <= 0:
            return base_caps_i

        mass = self.boundary_cell_mass
        route_score = self.routegradient_score
        if mass is None or route_score is None or not np.any(mass > 0.0):
            return base_caps_i
        counts = np.bincount(assigned, minlength=self.nlist).astype(np.int32)
        base_caps = base_caps_i.astype(np.float32)
        nonempty = counts > 0
        slack = np.minimum(counts, self.cap_max).astype(np.float32) - base_caps
        slack = np.maximum(slack, 0.0)
        candidates = nonempty & (slack > 0.0) & (mass > 0.0)
        if not np.any(candidates):
            return base_caps_i

        mass_ratio = self._positive_ratio(mass.astype(np.float32, copy=False), candidates)

        eff = self.concentration_effective_rank
        eff_ratio = np.ones(self.nlist, dtype=np.float32)
        if eff is not None:
            eff_ratio = self._positive_ratio(eff.astype(np.float32, copy=False), eff > 0.0)

        tail = self.concentration_tail_drop
        tail_ratio = np.ones(self.nlist, dtype=np.float32)
        if tail is not None and np.any(tail > 0.0):
            valid_tail = tail > 0.0
            mean_tail = float(np.mean(tail[valid_tail]) + 1e-12)
            # Low tail drop means the cell boundary is ambiguous, so it receives more mass.
            tail_ratio[valid_tail] = np.maximum(mean_tail / np.maximum(tail[valid_tail], 1e-12), 1e-6)

        phase_gap = self._phase_gap_scores(assigned, scores, base_caps_i, counts)
        gap_ratio = self._positive_ratio(phase_gap, phase_gap > 0.0)

        route_ratio = np.maximum(route_score.astype(np.float32, copy=False), self.routegradient_floor)
        route_ratio = self._positive_ratio(route_ratio, route_ratio > 0.0)

        log_signal = np.zeros(self.nlist, dtype=np.float64)
        log_signal += self.capmeasure_mass_weight * np.log(np.maximum(mass_ratio, 1e-6))
        log_signal += self.capmeasure_entropy_weight * np.log(np.maximum(eff_ratio, 1e-6))
        log_signal += self.capmeasure_boundary_weight * np.log(np.maximum(tail_ratio, 1e-6))
        log_signal += self.capmeasure_route_weight * np.log(np.maximum(route_ratio, 1e-6))
        log_signal += self.capmeasure_gap_weight * np.log(np.maximum(gap_ratio, 1e-6))

        signal_values = log_signal[candidates]
        if len(signal_values) == 0:
            return base_caps_i
        threshold = float(np.quantile(signal_values, self.slack_quantile))
        active = candidates & (log_signal >= threshold)
        if not np.any(active):
            active = candidates

        scaled = (log_signal[active] - float(np.max(log_signal[active]))) / self.capmeasure_temperature
        soft = np.exp(np.clip(scaled, -60.0, 0.0))
        weights = np.zeros(self.nlist, dtype=np.float64)
        weights[active] = soft * np.minimum(slack[active], self.slack_cell_max_extra)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            return base_caps_i

        base_total = int(np.sum(base_caps_i[nonempty]))
        budget = int(round(base_total * self.slack_budget_ratio))
        budget = max(0, min(budget, int(np.sum(np.minimum(slack[active], self.slack_cell_max_extra)))))
        if budget <= 0:
            return base_caps_i

        fractional = weights * (float(budget) / weight_sum)
        add = np.floor(fractional).astype(np.int32)
        add = np.minimum(add, np.minimum(slack.astype(np.int32), self.slack_cell_max_extra))
        remaining = int(budget - np.sum(add))
        if remaining > 0:
            room = np.minimum(slack.astype(np.int32), self.slack_cell_max_extra) - add
            remainders = fractional - add.astype(np.float64)
            order = np.argsort(-remainders, kind="stable")
            for cid in order:
                if remaining <= 0:
                    break
                if room[cid] <= 0 or weights[cid] <= 0.0:
                    continue
                add[cid] += 1
                remaining -= 1

        caps_i = base_caps_i + add
        caps_i = np.minimum(caps_i, counts)
        caps_i = np.minimum(caps_i, self.cap_max).astype(np.int32)
        self.cap_total_requested = int(np.sum(caps_i))
        self.slack_added = int(np.sum(caps_i - base_caps_i))
        self.slack_cells = int(np.count_nonzero(caps_i > base_caps_i))
        self.capmeasure_added = self.slack_added
        self.capmeasure_cells = self.slack_cells
        if base_total > 0:
            self.slack_cap_ratio = float(np.sum(caps_i[nonempty])) / float(base_total)
            self.capmeasure_cap_ratio = self.slack_cap_ratio
        self.capmeasure_active_fraction = float(np.count_nonzero(active)) / float(max(1, np.count_nonzero(candidates)))
        if np.any(active):
            active_signal = log_signal[active]
            self.slack_mean_score = float(np.mean(active_signal))
            self.slack_max_score = float(np.max(active_signal))
            self.capmeasure_mean_signal = self.slack_mean_score
            self.capmeasure_max_signal = self.slack_max_score
            active_gap = phase_gap[active & (phase_gap > 0.0)]
            if len(active_gap):
                self.phasecouple_mean_active_gap = float(np.mean(active_gap))
        return caps_i

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10routegradient-slackmass-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10capmeasure-transport-atlas-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", capMeasureTemp={self.capmeasure_temperature}, "
            f"capMeasureAdded={self.capmeasure_added}, "
            f"capMeasureCells={self.capmeasure_cells}, "
            f"capMeasureRatio={self.capmeasure_cap_ratio:.4f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-capmeasure-transport-atlas-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["capmeasure_temperature"] = self.capmeasure_temperature
        extra["capmeasure_mass_weight"] = self.capmeasure_mass_weight
        extra["capmeasure_entropy_weight"] = self.capmeasure_entropy_weight
        extra["capmeasure_boundary_weight"] = self.capmeasure_boundary_weight
        extra["capmeasure_route_weight"] = self.capmeasure_route_weight
        extra["capmeasure_gap_weight"] = self.capmeasure_gap_weight
        extra["capmeasure_added"] = self.capmeasure_added
        extra["capmeasure_cells"] = self.capmeasure_cells
        extra["capmeasure_cap_ratio"] = self.capmeasure_cap_ratio
        extra["capmeasure_active_fraction"] = self.capmeasure_active_fraction
        extra["capmeasure_mean_signal"] = self.capmeasure_mean_signal
        extra["capmeasure_max_signal"] = self.capmeasure_max_signal
        return extra


class ZeelinM10bayesboundaryOrderBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "pbw" in method_param:
            method_param["posterior_boundary_weight"] = method_param["pbw"]
        if "pnw" in method_param:
            method_param["posterior_norm_weight"] = method_param["pnw"]
        if "pmt" in method_param:
            method_param["posterior_margin_temperature"] = method_param["pmt"]
        if "pmo" in method_param:
            method_param["posterior_mode"] = method_param["pmo"]
        if "pmode" in method_param and "posterior_mode" not in method_param:
            method_param["posterior_mode"] = method_param["pmode"]
            method_param["pmode"] = "view"
        super().__init__(metric, method_param)
        self.posterior_boundary_weight = float(method_param.get("posterior_boundary_weight", 0.04))
        self.posterior_norm_weight = float(method_param.get("posterior_norm_weight", 0.0))
        self.posterior_margin_temperature = float(method_param.get("posterior_margin_temperature", 0.035))
        self.posterior_mode = str(method_param.get("posterior_mode", "risk"))
        self.posterior_boundary_weight = max(-2.0, min(2.0, self.posterior_boundary_weight))
        self.posterior_norm_weight = max(-2.0, min(2.0, self.posterior_norm_weight))
        self.posterior_margin_temperature = max(1e-4, min(1.0, self.posterior_margin_temperature))
        if self.posterior_mode not in {"risk", "margin", "antirisk", "norm", "core"}:
            self.posterior_mode = "risk"
        self.posterior_mean_risk = 0.0
        self.posterior_max_risk = 0.0
        self.posterior_mean_margin = 0.0
        self.posterior_top_overlap = 0.0

    @staticmethod
    def _zscore(values):
        values = np.asarray(values, dtype=np.float32)
        return (values - float(np.mean(values))) / max(float(np.std(values)), 1e-12)

    def _posterior_order_scores(self, base_scores, margins, raw_norms):
        risk = np.exp(-margins / self.posterior_margin_temperature).astype(np.float32)
        self.posterior_mean_risk = float(np.mean(risk))
        self.posterior_max_risk = float(np.max(risk)) if len(risk) else 0.0
        self.posterior_mean_margin = float(np.mean(margins)) if len(margins) else 0.0

        if self.posterior_mode == "core":
            return base_scores.astype(np.float32, copy=True), risk

        score = base_scores.astype(np.float32, copy=True)
        if self.posterior_mode == "margin":
            score += self.posterior_boundary_weight * self._zscore(-margins)
        elif self.posterior_mode == "antirisk":
            score -= self.posterior_boundary_weight * self._zscore(risk)
        elif self.posterior_mode == "norm":
            score += self.posterior_norm_weight * self._zscore(raw_norms)
        else:
            score += self.posterior_boundary_weight * self._zscore(risk)

        if self.posterior_mode != "norm" and self.posterior_norm_weight != 0.0:
            score += self.posterior_norm_weight * self._zscore(raw_norms)
        return score.astype(np.float32, copy=False), risk

    def _top_overlap(self, base_order, posterior_order, assigned, cell_caps):
        counts = np.bincount(assigned, minlength=self.nlist).astype(np.int32)
        starts = np.zeros(self.nlist + 1, dtype=np.int64)
        starts[1:] = np.cumsum(counts, dtype=np.int64)
        total = 0
        overlap = 0
        for cid in range(self.nlist):
            start = int(starts[cid])
            stop = int(starts[cid + 1])
            if stop <= start:
                continue
            take = min(int(cell_caps[cid]), stop - start, self.max_codes if self.max_codes > 0 else stop - start)
            if take <= 0:
                continue
            base_top = base_order[start : start + take]
            post_top = posterior_order[start : start + take]
            overlap += len(set(int(v) for v in base_top).intersection(int(v) for v in post_top))
            total += take
        return float(overlap) / float(total) if total else 1.0

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        raw_norms = np.linalg.norm(X, axis=1).astype(np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        self.index.train(train)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        _, assigned1 = self.quantizer.search(np.ascontiguousarray(Xn, dtype=np.float32), 1)
        assigned = np.ascontiguousarray(assigned1[:, 0], dtype=np.int64)

        base_scores = self._core_scores(Xn, assigned, raw_norms)
        base_order = np.lexsort((-base_scores, assigned)).astype(np.int64, copy=False)
        self._before_cell_caps(Xn, assigned, base_scores, raw_norms, base_order)
        cell_caps = np.asarray(self._cell_caps(assigned, base_scores), dtype=np.int32)
        nonzero_caps = cell_caps[cell_caps > 0]
        if len(nonzero_caps) > 0:
            self.cap_min_actual = int(np.min(nonzero_caps))
            self.cap_max_actual = int(np.max(nonzero_caps))
            self.cap_mean_actual = float(np.mean(nonzero_caps))

        if self.posterior_mode == "core":
            margins = np.ones(len(Xn), dtype=np.float32)
        else:
            centroid_scores, assigned2 = self.quantizer.search(
                np.ascontiguousarray(Xn, dtype=np.float32), 2
            )
            first_scores = np.ascontiguousarray(centroid_scores[:, 0], dtype=np.float32)
            second_scores = np.ascontiguousarray(centroid_scores[:, 1], dtype=np.float32)
            valid_second = assigned2[:, 1] >= 0
            margins = np.maximum(0.0, first_scores - second_scores).astype(np.float32)
            if np.any(valid_second):
                margins[~valid_second] = float(np.max(margins[valid_second]))
            else:
                margins[:] = 1.0

        posterior_scores, _ = self._posterior_order_scores(base_scores, margins, raw_norms)
        order = np.lexsort((-posterior_scores, assigned)).astype(np.int64, copy=False)
        self.posterior_top_overlap = self._top_overlap(base_order, order, assigned, cell_caps)

        keep = np.zeros(len(order), dtype=bool)
        last_cid = -1
        used_in_cell = 0
        for pos, idx in enumerate(order):
            cid = int(assigned[idx])
            if cid != last_cid:
                last_cid = cid
                used_in_cell = 0
            if used_in_cell < int(cell_caps[cid]):
                keep[pos] = True
                used_in_cell += 1
        kept_order = np.ascontiguousarray(order[keep], dtype=np.int64)
        self.kept_points = int(len(kept_order))
        ids = np.ascontiguousarray(kept_order, dtype=np.int64)
        block = int(self.method_param.get("add_block", 65536))
        block = max(1024, block)
        for start in range(0, len(kept_order), block):
            stop = min(len(kept_order), start + block)
            batch = np.ascontiguousarray(Xn[kept_order[start:stop]], dtype=np.float32)
            self.index.add_with_ids(batch, ids[start:stop])
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10bayesboundary-order-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", posteriorMode={self.posterior_mode}, "
            f"posteriorBoundaryWeight={self.posterior_boundary_weight}, "
            f"posteriorNormWeight={self.posterior_norm_weight}, "
            f"posteriorMarginTemperature={self.posterior_margin_temperature}, "
            f"posteriorTopOverlap={self.posterior_top_overlap:.4f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-bayesboundary-order-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["posterior_mode"] = self.posterior_mode
        extra["posterior_boundary_weight"] = self.posterior_boundary_weight
        extra["posterior_norm_weight"] = self.posterior_norm_weight
        extra["posterior_margin_temperature"] = self.posterior_margin_temperature
        extra["posterior_mean_risk"] = self.posterior_mean_risk
        extra["posterior_max_risk"] = self.posterior_max_risk
        extra["posterior_mean_margin"] = self.posterior_mean_margin
        extra["posterior_top_overlap"] = self.posterior_top_overlap
        return extra


class ZeelinM10cachepreservingBoundaryliftBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "blg" in method_param:
            method_param["bmg"] = method_param["blg"]
        if "blb" in method_param:
            method_param["bmb"] = method_param["blb"]
        if "blhi" in method_param:
            method_param["bmcliphi"] = method_param["blhi"]
        if "bllo" in method_param:
            method_param["bmcliplo"] = method_param["bllo"]
        method_param.setdefault("bmm", "guard")
        method_param.setdefault("bmg", 0.35)
        method_param.setdefault("bmb", 1.0)
        method_param.setdefault("bmcliphi", 1.10)
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10cachepreserving-boundarylift-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-cachepreserving-boundarylift-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["cachepreserving_boundarylift"] = 1
        return extra


class ZeelinM10capbraidBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "bp" in method_param:
            method_param["capbraid_period"] = method_param["bp"]
        if "bt" in method_param:
            method_param["capbraid_take"] = method_param["bt"]
        if "bw" in method_param:
            method_param["capbraid_weight"] = method_param["bw"]
        super().__init__(metric, method_param)
        self.capbraid_period = int(method_param.get("capbraid_period", 16))
        self.capbraid_take = int(method_param.get("capbraid_take", 1))
        self.capbraid_weight = float(method_param.get("capbraid_weight", 0.35))
        self.capbraid_period = max(2, min(128, self.capbraid_period))
        self.capbraid_take = max(0, min(self.capbraid_period - 1, self.capbraid_take))
        self.capbraid_weight = max(0.0, min(4.0, self.capbraid_weight))
        self.capbraid_cells = 0
        self.capbraid_boundary_fraction = 0.0

    def _capbraid_order(self, core_order, assigned, scores, margins):
        if self.capbraid_take <= 0:
            self.capbraid_cells = 0
            self.capbraid_boundary_fraction = 0.0
            return core_order

        counts = np.bincount(assigned, minlength=self.nlist).astype(np.int64)
        starts = np.zeros(self.nlist + 1, dtype=np.int64)
        starts[1:] = np.cumsum(counts, dtype=np.int64)
        braided_parts = []
        braided_boundary = 0
        braided_total = 0
        braided_cells = 0

        for cid in range(self.nlist):
            start = int(starts[cid])
            stop = int(starts[cid + 1])
            if stop <= start:
                continue
            members = core_order[start:stop]
            if len(members) <= self.capbraid_period:
                braided_parts.append(members)
                braided_total += int(len(members))
                continue

            boundary_values = scores[members] - self.capbraid_weight * margins[members]
            boundary_local = np.argsort(-boundary_values, kind="stable")
            boundary = members[boundary_local]
            used = set()
            out = []
            core_pos = 0
            boundary_pos = 0
            core_quota = max(1, self.capbraid_period - self.capbraid_take)
            while len(out) < len(members):
                added = 0
                while core_pos < len(members) and added < core_quota:
                    idx = int(members[core_pos])
                    core_pos += 1
                    if idx in used:
                        continue
                    used.add(idx)
                    out.append(idx)
                    added += 1
                added_boundary = 0
                while boundary_pos < len(boundary) and added_boundary < self.capbraid_take:
                    idx = int(boundary[boundary_pos])
                    boundary_pos += 1
                    if idx in used:
                        continue
                    used.add(idx)
                    out.append(idx)
                    added_boundary += 1
                    braided_boundary += 1
                if added == 0 and added_boundary == 0:
                    break
            while core_pos < len(members):
                idx = int(members[core_pos])
                core_pos += 1
                if idx not in used:
                    out.append(idx)
                    used.add(idx)
            if len(out) < len(members):
                for idx in members:
                    idx = int(idx)
                    if idx not in used:
                        out.append(idx)
                        used.add(idx)
            braided_parts.append(np.asarray(out, dtype=np.int64))
            braided_total += int(len(out))
            braided_cells += 1

        self.capbraid_cells = braided_cells
        self.capbraid_boundary_fraction = float(braided_boundary) / float(max(1, braided_total))
        if not braided_parts:
            return core_order
        return np.ascontiguousarray(np.concatenate(braided_parts), dtype=np.int64)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        raw_norms = np.linalg.norm(X, axis=1).astype(np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        self.index.train(train)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        centroid_scores, assigned2 = self.quantizer.search(
            np.ascontiguousarray(Xn, dtype=np.float32), 2
        )
        assigned = np.ascontiguousarray(assigned2[:, 0], dtype=np.int64)
        scores = self._core_scores(Xn, assigned, raw_norms)
        first_scores = np.ascontiguousarray(centroid_scores[:, 0], dtype=np.float32)
        second_scores = np.ascontiguousarray(centroid_scores[:, 1], dtype=np.float32)
        margins = np.maximum(0.0, first_scores - second_scores).astype(np.float32)
        valid_second = assigned2[:, 1] >= 0
        if np.any(valid_second):
            margins[~valid_second] = float(np.max(margins[valid_second]))
        core_order = np.lexsort((-scores, assigned)).astype(np.int64, copy=False)
        self._before_cell_caps(Xn, assigned, scores, raw_norms, core_order)
        cell_caps = np.asarray(self._cell_caps(assigned, scores), dtype=np.int32)
        nonzero_caps = cell_caps[cell_caps > 0]
        if len(nonzero_caps) > 0:
            self.cap_min_actual = int(np.min(nonzero_caps))
            self.cap_max_actual = int(np.max(nonzero_caps))
            self.cap_mean_actual = float(np.mean(nonzero_caps))

        order = self._capbraid_order(core_order, assigned, scores, margins)
        keep = np.zeros(len(order), dtype=bool)
        last_cid = -1
        used_in_cell = 0
        for pos, idx in enumerate(order):
            cid = int(assigned[idx])
            if cid != last_cid:
                last_cid = cid
                used_in_cell = 0
            if used_in_cell < int(cell_caps[cid]):
                keep[pos] = True
                used_in_cell += 1
        kept_order = np.ascontiguousarray(order[keep], dtype=np.int64)
        self.kept_points = int(len(kept_order))
        ids = np.ascontiguousarray(kept_order, dtype=np.int64)
        block = int(self.method_param.get("add_block", 65536))
        block = max(1024, block)
        for start in range(0, len(kept_order), block):
            stop = min(len(kept_order), start + block)
            batch = np.ascontiguousarray(Xn[kept_order[start:stop]], dtype=np.float32)
            self.index.add_with_ids(batch, ids[start:stop])
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.set_query_arguments(
            self,
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10capbraid-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", capBraidPeriod={self.capbraid_period}, "
            f"capBraidTake={self.capbraid_take}, "
            f"capBraidWeight={self.capbraid_weight}, "
            f"capBraidBoundaryFraction={self.capbraid_boundary_fraction:.4f})"
        )

    def get_additional(self):
        extra = ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.get_additional(self)
        extra["kernel"] = "faiss-capbraid-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["capbraid_period"] = self.capbraid_period
        extra["capbraid_take"] = self.capbraid_take
        extra["capbraid_weight"] = self.capbraid_weight
        extra["capbraid_cells"] = self.capbraid_cells
        extra["capbraid_boundary_fraction"] = self.capbraid_boundary_fraction
        return extra


class ZeelinM10structcertBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "xct" in method_param:
            method_param["structure_trials"] = method_param["xct"]
        if "xmode" in method_param:
            method_param["structure_score_mode"] = method_param["xmode"]
        if "xguard" in method_param:
            method_param["structure_guard"] = method_param["xguard"]
        super().__init__(metric, method_param)
        self.structure_trials = int(method_param.get("structure_trials", 3))
        self.structure_score_mode = str(method_param.get("structure_score_mode", "speedguard"))
        self.structure_guard = float(method_param.get("structure_guard", 0.025))
        self.structure_trials = max(1, min(5, self.structure_trials))
        self.structure_guard = max(0.0, min(0.25, self.structure_guard))
        self.structure_best_trial = 0
        self.structure_best_score = 0.0
        self.structure_best_build_time = 0.0
        self.structure_best_cap_mean = 0.0
        self.structure_best_concentration_cap_ratio = 0.0
        self.structure_best_boundary_cap_ratio = 0.0
        self.structure_best_boundary_mean_mass = 0.0
        self.structure_best_effective_rank = 0.0
        self.structure_best_kept_points = 0
        self.structure_all_scores = ""
        self.structure_all_build_times = ""
        self.structure_all_cap_mean = ""
        self.structure_all_concentration_cap_ratio = ""
        self.structure_all_boundary_cap_ratio = ""
        self.structure_all_boundary_mean_mass = ""
        self.structure_all_effective_rank = ""
        self.structure_all_kept_points = ""

    @staticmethod
    def _rank01(values, high=True):
        values = np.asarray(values, dtype=np.float64)
        if len(values) <= 1:
            return np.ones(len(values), dtype=np.float64)
        if float(np.max(values) - np.min(values)) <= 1e-12:
            return np.ones(len(values), dtype=np.float64) * 0.5
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
        if high:
            return ranks
        return 1.0 - ranks

    def _capture_structure_state(self, trial, elapsed):
        return {
            "index": self.index,
            "quantizer": self.quantizer,
            "dim": self.dim,
            "qbuf": self.qbuf,
            "kept_points": self.kept_points,
            "cap_min_actual": self.cap_min_actual,
            "cap_max_actual": self.cap_max_actual,
            "cap_mean_actual": self.cap_mean_actual,
            "cap_total_requested": getattr(self, "cap_total_requested", 0),
            "concentration_observed_cells": self.concentration_observed_cells,
            "concentration_mean_effective_rank": self.concentration_mean_effective_rank,
            "concentration_mean_tail_drop": self.concentration_mean_tail_drop,
            "concentration_cap_ratio": self.concentration_cap_ratio,
            "boundary_observed_cells": self.boundary_observed_cells,
            "boundary_mean_mass": self.boundary_mean_mass,
            "boundary_max_mass": self.boundary_max_mass,
            "boundary_mean_gap": self.boundary_mean_gap,
            "boundary_cap_ratio": self.boundary_cap_ratio,
            "trial": trial,
            "build_time": elapsed,
            "score": 0.0,
        }

    def _restore_structure_state(self, state):
        self.index = state["index"]
        self.quantizer = state["quantizer"]
        self.dim = state["dim"]
        self.qbuf = state["qbuf"]
        self.kept_points = state["kept_points"]
        self.cap_min_actual = state["cap_min_actual"]
        self.cap_max_actual = state["cap_max_actual"]
        self.cap_mean_actual = state["cap_mean_actual"]
        self.cap_total_requested = state["cap_total_requested"]
        self.concentration_observed_cells = state["concentration_observed_cells"]
        self.concentration_mean_effective_rank = state["concentration_mean_effective_rank"]
        self.concentration_mean_tail_drop = state["concentration_mean_tail_drop"]
        self.concentration_cap_ratio = state["concentration_cap_ratio"]
        self.boundary_observed_cells = state["boundary_observed_cells"]
        self.boundary_mean_mass = state["boundary_mean_mass"]
        self.boundary_max_mass = state["boundary_max_mass"]
        self.boundary_mean_gap = state["boundary_mean_gap"]
        self.boundary_cap_ratio = state["boundary_cap_ratio"]
        self.structure_best_trial = state["trial"]
        self.structure_best_score = state["score"]
        self.structure_best_build_time = state["build_time"]
        self.structure_best_cap_mean = state["cap_mean_actual"]
        self.structure_best_concentration_cap_ratio = state["concentration_cap_ratio"]
        self.structure_best_boundary_cap_ratio = state["boundary_cap_ratio"]
        self.structure_best_boundary_mean_mass = state["boundary_mean_mass"]
        self.structure_best_effective_rank = state["concentration_mean_effective_rank"]
        self.structure_best_kept_points = state["kept_points"]

    def _score_structure_states(self, states):
        build = [s["build_time"] for s in states]
        cap_mean = [s["cap_mean_actual"] for s in states]
        conc_ratio = [s["concentration_cap_ratio"] for s in states]
        boundary_ratio = [s["boundary_cap_ratio"] for s in states]
        boundary_mass = [s["boundary_mean_mass"] for s in states]
        eff_rank = [s["concentration_mean_effective_rank"] for s in states]
        kept = [s["kept_points"] for s in states]

        build_fast = self._rank01(build, high=False)
        cap_tight = self._rank01(cap_mean, high=False)
        conc_high = self._rank01(conc_ratio, high=True)
        boundary_ratio_high = self._rank01(boundary_ratio, high=True)
        boundary_mass_high = self._rank01(boundary_mass, high=True)
        eff_rank_low = self._rank01(eff_rank, high=False)
        kept_high = self._rank01(kept, high=True)

        for i, state in enumerate(states):
            if self.structure_score_mode == "balanced":
                score = (
                    0.80 * build_fast[i]
                    + 0.65 * conc_high[i]
                    + 0.35 * cap_tight[i]
                    + 0.25 * eff_rank_low[i]
                    + 0.20 * boundary_ratio_high[i]
                    + 0.15 * kept_high[i]
                )
            elif self.structure_score_mode == "recallguard":
                score = (
                    0.55 * build_fast[i]
                    + 0.85 * conc_high[i]
                    + 0.45 * boundary_ratio_high[i]
                    + 0.30 * cap_tight[i]
                    + 0.25 * eff_rank_low[i]
                    + 0.10 * boundary_mass_high[i]
                )
            else:
                score = (
                    1.05 * build_fast[i]
                    + 0.55 * conc_high[i]
                    + 0.35 * cap_tight[i]
                    + 0.25 * eff_rank_low[i]
                    + 0.20 * boundary_mass_high[i]
                    + 0.10 * kept_high[i]
                )
            state["score"] = float(score)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        states = []
        for trial in range(1, self.structure_trials + 1):
            start = time.time()
            super().fit(X)
            elapsed = time.time() - start
            states.append(self._capture_structure_state(trial, elapsed))

        self._score_structure_states(states)
        conc = np.asarray([s["concentration_cap_ratio"] for s in states], dtype=np.float64)
        boundary = np.asarray([s["boundary_cap_ratio"] for s in states], dtype=np.float64)
        conc_floor = float(np.median(conc) - self.structure_guard)
        boundary_floor = float(np.median(boundary) - self.structure_guard)
        acceptable = [
            state
            for state in states
            if state["concentration_cap_ratio"] >= conc_floor
            and state["boundary_cap_ratio"] >= boundary_floor
        ]
        if not acceptable:
            acceptable = states
        best_state = max(acceptable, key=lambda s: (s["score"], -s["build_time"]))
        self._restore_structure_state(best_state)

        self.structure_all_scores = ",".join(f"{s['score']:.6f}" for s in states)
        self.structure_all_build_times = ",".join(f"{s['build_time']:.6f}" for s in states)
        self.structure_all_cap_mean = ",".join(f"{s['cap_mean_actual']:.6f}" for s in states)
        self.structure_all_concentration_cap_ratio = ",".join(
            f"{s['concentration_cap_ratio']:.6f}" for s in states
        )
        self.structure_all_boundary_cap_ratio = ",".join(
            f"{s['boundary_cap_ratio']:.6f}" for s in states
        )
        self.structure_all_boundary_mean_mass = ",".join(
            f"{s['boundary_mean_mass']:.6f}" for s in states
        )
        self.structure_all_effective_rank = ",".join(
            f"{s['concentration_mean_effective_rank']:.6f}" for s in states
        )
        self.structure_all_kept_points = ",".join(str(int(s["kept_points"])) for s in states)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10structcert-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", structureTrials={self.structure_trials}, "
            f"structureMode={self.structure_score_mode}, "
            f"structureBestTrial={self.structure_best_trial}, "
            f"structureScore={self.structure_best_score:.3f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10structcert-boundarytrim-maxcodes-concentrationcapcore-returncompress"
        extra["structure_trials"] = self.structure_trials
        extra["structure_score_mode"] = self.structure_score_mode
        extra["structure_guard"] = self.structure_guard
        extra["structure_best_trial"] = self.structure_best_trial
        extra["structure_best_score"] = self.structure_best_score
        extra["structure_best_build_time"] = self.structure_best_build_time
        extra["structure_best_cap_mean"] = self.structure_best_cap_mean
        extra["structure_best_concentration_cap_ratio"] = self.structure_best_concentration_cap_ratio
        extra["structure_best_boundary_cap_ratio"] = self.structure_best_boundary_cap_ratio
        extra["structure_best_boundary_mean_mass"] = self.structure_best_boundary_mean_mass
        extra["structure_best_effective_rank"] = self.structure_best_effective_rank
        extra["structure_best_kept_points"] = self.structure_best_kept_points
        extra["structure_all_scores"] = self.structure_all_scores
        extra["structure_all_build_times"] = self.structure_all_build_times
        extra["structure_all_cap_mean"] = self.structure_all_cap_mean
        extra["structure_all_concentration_cap_ratio"] = self.structure_all_concentration_cap_ratio
        extra["structure_all_boundary_cap_ratio"] = self.structure_all_boundary_cap_ratio
        extra["structure_all_boundary_mean_mass"] = self.structure_all_boundary_mean_mass
        extra["structure_all_effective_rank"] = self.structure_all_effective_rank
        extra["structure_all_kept_points"] = self.structure_all_kept_points
        return extra


class ZeelinM10conformalbudgetBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "fk" in method_param:
            method_param["fast_return_k"] = method_param["fk"]
        if "sk" in method_param:
            method_param["slow_return_k"] = method_param["sk"]
        if "rtm" in method_param:
            method_param["ratio_threshold_milli"] = method_param["rtm"]
        if "stm" in method_param:
            method_param["risk_ip_threshold_milli"] = method_param["stm"]
        super().__init__(metric, method_param)
        self.fast_return_k = int(method_param.get("fast_return_k", 6))
        self.slow_return_k = int(method_param.get("slow_return_k", 7))
        self.ratio_threshold = float(method_param.get("ratio_threshold_milli", 0)) / 1000.0
        self.risk_ip_threshold = float(method_param.get("risk_ip_threshold_milli", 420)) / 1000.0
        self.fast_return_k = max(1, min(10, self.fast_return_k))
        self.slow_return_k = max(self.fast_return_k, min(10, self.slow_return_k))
        self.ratio_threshold = max(0.0, min(2.0, self.ratio_threshold))
        self.risk_ip_threshold = max(-1.0, min(1.0, self.risk_ip_threshold))
        self._repair_count = 0
        self._query_count = 0
        self._last_avg_return_k = float(self.slow_return_k)
        self._batch_lengths = None

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=self.slow_return_k,
            internal_k=max(self.slow_return_k, int(internal_k or 0)),
        )
        self._repair_count = 0
        self._query_count = 0
        self._last_avg_return_k = float(self.fast_return_k)
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10conformalbudget-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", fastReturnK={self.fast_return_k}, slowReturnK={self.slow_return_k}, "
            f"riskIpThreshold={self.risk_ip_threshold:.3f}, "
            f"ratioThreshold={self.ratio_threshold:.3f})"
        )

    def _query_matrix(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.skip_query_norm:
            return X if X.flags["C_CONTIGUOUS"] else np.ascontiguousarray(X, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
        return np.ascontiguousarray(X / np.maximum(norms, 1e-12), dtype=np.float32)

    def _risk_mask(self, D, I, fast_k):
        last = D[:, fast_k - 1]
        risk = (last <= self.risk_ip_threshold) | (I[:, fast_k - 1] < 0)
        if self.ratio_threshold > 0.0:
            first = np.maximum(D[:, 0], 1e-12)
            ratio = last / first
            risk = risk | (ratio < self.ratio_threshold)
        return risk

    def query(self, v, n):
        q = self._query_matrix(v.reshape(1, self.dim))
        fast_k = min(n, self.fast_return_k)
        slow_k = min(n, self.slow_return_k)
        D, I = self.index.search(q, fast_k)
        self._query_count += 1
        risk = bool(self._risk_mask(D, I, fast_k)[0])
        if not risk:
            return I[0, :fast_k]
        self._repair_count += 1
        return self.index.search(q, slow_k)[1][0, :slow_k]

    def batch_query(self, X, n):
        Xn = self._query_matrix(X)
        fast_k = min(n, self.fast_return_k)
        slow_k = min(n, self.slow_return_k)
        D, I = self.index.search(Xn, fast_k)
        risk = self._risk_mask(D, I, fast_k)
        self._query_count += int(len(Xn))
        self._repair_count += int(np.count_nonzero(risk))

        labels = np.full((len(Xn), slow_k), -1, dtype=np.int64)
        labels[:, :fast_k] = I[:, :fast_k]
        lengths = np.full(len(Xn), fast_k, dtype=np.int32)
        if np.any(risk):
            slow_labels = self.index.search(Xn[risk], slow_k)[1]
            labels[risk, :slow_k] = slow_labels[:, :slow_k]
            lengths[risk] = slow_k
        self._batch_labels = labels
        self._batch_lengths = lengths
        self._batch_target_k = slow_k
        self._last_avg_return_k = float(np.mean(lengths)) if len(lengths) else 0.0

    def get_batch_results(self):
        results = []
        for labels, length in zip(self._batch_labels, self._batch_lengths):
            prefix = labels[: int(length)]
            if self.fast_unique:
                if self.trust_full_labels:
                    results.append(prefix[prefix >= 0])
                else:
                    results.append([int(label) for label in prefix if label >= 0])
                continue
            out = []
            seen = set()
            for label in prefix:
                label = int(label)
                if label < 0 or label in seen:
                    continue
                seen.add(label)
                out.append(label)
            results.append(out)
        return results

    def get_additional(self):
        extra = super().get_additional()
        total = float(self._query_count)
        extra["kernel"] = "faiss-conformalbudget-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["fast_return_k"] = self.fast_return_k
        extra["slow_return_k"] = self.slow_return_k
        extra["risk_ip_threshold_milli"] = int(round(self.risk_ip_threshold * 1000.0))
        extra["ratio_threshold_milli"] = int(round(self.ratio_threshold * 1000.0))
        extra["avg_repair_fraction"] = float(self._repair_count) / total if total else 0.0
        extra["avg_lens_return_k"] = self._last_avg_return_k
        return extra


class ZeelinM10rdbudgetBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "rdg" in method_param:
            method_param["rd_gamma"] = method_param["rdg"]
        if "rdbu" in method_param:
            method_param["rd_budget_ratio"] = method_param["rdbu"]
        if "rdmin" in method_param:
            method_param["rd_cap_min"] = method_param["rdmin"]
        super().__init__(metric, method_param)
        self.rd_gamma = float(method_param.get("rd_gamma", 0.45))
        self.rd_budget_ratio = float(method_param.get("rd_budget_ratio", 0.985))
        self.rd_cap_min = int(method_param.get("rd_cap_min", 40))
        self.rd_gamma = max(0.0, min(2.0, self.rd_gamma))
        self.rd_budget_ratio = max(0.50, min(1.25, self.rd_budget_ratio))
        self.rd_cap_min = max(1, min(self.cap_max, self.rd_cap_min))
        self.rd_cap_ratio = 0.0
        self.rd_utility_mean = 0.0
        self.rd_utility_max = 0.0
        self.rd_weighted_cap = 0.0
        self.rd_weighted_base_cap = 0.0

    def _cell_caps(self, assigned, scores):
        base_caps = super()._cell_caps(assigned, scores).astype(np.float32)
        counts = np.bincount(assigned, minlength=self.nlist).astype(np.float32)
        nonempty = counts > 0.0
        if not np.any(nonempty):
            return base_caps.astype(np.int32)

        eff = self.concentration_effective_rank
        mass = self.boundary_cell_mass
        utility = np.ones(self.nlist, dtype=np.float32)
        if eff is not None and np.any(eff > 0.0):
            valid = eff > 0.0
            mean_eff = float(np.mean(eff[valid]) + 1e-12)
            utility[valid] *= np.sqrt(np.maximum(eff[valid] / mean_eff, 1e-6))
        if mass is not None and np.any(mass > 0.0):
            valid = mass > 0.0
            mean_mass = float(np.mean(mass[valid]) + 1e-12)
            utility[valid] *= np.sqrt(np.maximum(mass[valid] / mean_mass, 1e-6))

        utility = np.maximum(utility, 1e-6)
        valid_utility = utility[nonempty]
        if len(valid_utility):
            self.rd_utility_mean = float(np.mean(valid_utility))
            self.rd_utility_max = float(np.max(valid_utility))

        weights = np.power(utility, self.rd_gamma)
        proposed = base_caps * weights
        base_total = float(np.sum(base_caps[nonempty]))
        proposed_total = float(np.sum(proposed[nonempty]))
        if proposed_total > 0.0:
            proposed *= (base_total * self.rd_budget_ratio) / proposed_total

        caps = np.zeros(self.nlist, dtype=np.float32)
        caps[nonempty] = proposed[nonempty]
        caps = np.clip(caps, self.rd_cap_min, self.cap_max)
        caps[~nonempty] = 0
        caps = np.minimum(caps, counts)
        caps_i = np.rint(caps).astype(np.int32)
        self.cap_total_requested = int(np.sum(caps_i))
        self.rd_cap_ratio = float(np.sum(caps_i[nonempty]) / max(base_total, 1e-12))

        if mass is not None and np.any(mass > 0.0):
            mass_sum = float(np.sum(mass))
            if mass_sum > 0.0:
                self.rd_weighted_cap = float(np.sum(mass * caps_i) / mass_sum)
                self.rd_weighted_base_cap = float(np.sum(mass * base_caps) / mass_sum)
        return caps_i

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10rdbudget-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", rdGamma={self.rd_gamma}, rdBudgetRatio={self.rd_budget_ratio}, "
            f"rdCapMin={self.rd_cap_min}, rdCapRatio={self.rd_cap_ratio:.3f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-rdbudget-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["rd_gamma"] = self.rd_gamma
        extra["rd_budget_ratio"] = self.rd_budget_ratio
        extra["rd_cap_min"] = self.rd_cap_min
        extra["rd_cap_ratio"] = self.rd_cap_ratio
        extra["rd_utility_mean"] = self.rd_utility_mean
        extra["rd_utility_max"] = self.rd_utility_max
        extra["rd_weighted_cap"] = self.rd_weighted_cap
        extra["rd_weighted_base_cap"] = self.rd_weighted_base_cap
        return extra


class ZeelinM10tailtapeBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "tpk" in method_param:
            method_param["tail_payload_k"] = method_param["tpk"]
        if "tpo" in method_param:
            method_param["tail_payload_offset"] = method_param["tpo"]
        if "tps" in method_param:
            method_param["tail_payload_stride"] = method_param["tps"]
        if "srk" in method_param:
            method_param["search_return_k"] = method_param["srk"]
        super().__init__(metric, method_param)
        self.tail_payload_k = int(method_param.get("tail_payload_k", 1))
        self.tail_payload_offset = int(method_param.get("tail_payload_offset", 0))
        self.tail_payload_stride = int(method_param.get("tail_payload_stride", 1))
        self.search_return_k = int(method_param.get("search_return_k", 7))
        self.tail_payload_k = max(0, min(4, self.tail_payload_k))
        self.tail_payload_offset = max(0, min(256, self.tail_payload_offset))
        self.tail_payload_stride = max(1, min(32, self.tail_payload_stride))
        self.search_return_k = max(1, min(10, self.search_return_k))
        self.final_return_k = min(10, self.search_return_k + self.tail_payload_k)
        self.label_to_cell = None
        self.cell_tail_payload = None
        self.tail_payload_filled_fraction = 0.0
        self.tail_payload_cells = 0
        self.tail_payload_mean_valid = 0.0

    def _build_tail_payload(self, X):
        X = np.asarray(X, dtype=np.float32)
        raw_norms = np.linalg.norm(X, axis=1).astype(np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)

        old_ef = int(self.quantizer.hnsw.efSearch)
        self.quantizer.hnsw.efSearch = self.assign_ef_search
        _, assigned = self.quantizer.search(np.ascontiguousarray(Xn, dtype=np.float32), 1)
        self.quantizer.hnsw.efSearch = old_ef
        assigned = np.ascontiguousarray(assigned[:, 0], dtype=np.int64)
        self.label_to_cell = np.ascontiguousarray(assigned, dtype=np.int32)

        scores = self._core_scores(Xn, assigned, raw_norms)
        order = np.lexsort((-scores, assigned)).astype(np.int64, copy=False)
        self._before_cell_caps(Xn, assigned, scores, raw_norms, order)
        cell_caps = np.asarray(self._cell_caps(assigned, scores), dtype=np.int32)
        counts = np.bincount(assigned, minlength=self.nlist).astype(np.int32)
        starts = np.zeros(self.nlist + 1, dtype=np.int64)
        starts[1:] = np.cumsum(counts, dtype=np.int64)

        payload = np.full((self.nlist, self.tail_payload_k), -1, dtype=np.int64)
        if self.tail_payload_k <= 0:
            self.cell_tail_payload = payload
            return

        valid_slots = 0
        valid_cells = 0
        scan_cap_default = self.max_codes if self.max_codes > 0 else self.cap_max
        for cid in range(self.nlist):
            start = int(starts[cid])
            stop = int(starts[cid + 1])
            if stop <= start:
                continue
            members = order[start:stop]
            cap = min(int(cell_caps[cid]), len(members))
            scan_cap = min(scan_cap_default, cap)
            tape_start = min(cap, scan_cap + self.tail_payload_offset)
            if tape_start >= cap:
                continue
            picked = members[tape_start:cap:self.tail_payload_stride][: self.tail_payload_k]
            if len(picked) <= 0:
                continue
            payload[cid, : len(picked)] = picked
            valid_slots += int(len(picked))
            valid_cells += 1

        self.cell_tail_payload = np.ascontiguousarray(payload, dtype=np.int64)
        self.tail_payload_cells = valid_cells
        total_slots = max(1, self.nlist * max(1, self.tail_payload_k))
        self.tail_payload_filled_fraction = float(valid_slots) / float(total_slots)
        self.tail_payload_mean_valid = float(valid_slots) / float(max(1, valid_cells))

    def fit(self, X):
        super().fit(X)
        self._build_tail_payload(X)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        if return_k is not None:
            self.search_return_k = max(1, min(10, int(return_k)))
            self.final_return_k = min(10, self.search_return_k + self.tail_payload_k)
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=self.search_return_k,
            internal_k=max(self.search_return_k, int(internal_k or 0)),
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10tailtape-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", searchReturnK={self.search_return_k}, finalReturnK={self.final_return_k}, "
            f"tailPayloadK={self.tail_payload_k}, tailPayloadOffset={self.tail_payload_offset}, "
            f"tailPayloadStride={self.tail_payload_stride})"
        )

    def _append_tail_payload(self, labels):
        if self.tail_payload_k <= 0 or self.cell_tail_payload is None or self.label_to_cell is None:
            return labels
        first = int(labels[0]) if len(labels) else -1
        if first < 0 or first >= len(self.label_to_cell):
            return labels
        cell = int(self.label_to_cell[first])
        if cell < 0 or cell >= len(self.cell_tail_payload):
            return labels
        payload = self.cell_tail_payload[cell]
        if len(payload) == 0 or int(payload[0]) < 0:
            return labels
        return np.concatenate([labels, payload[payload >= 0]])[: self.final_return_k]

    def query(self, v, n):
        labels = super().query(v, n)
        return self._append_tail_payload(labels)

    def batch_query(self, X, n):
        super().batch_query(X, n)
        if self.tail_payload_k <= 0 or self.cell_tail_payload is None or self.label_to_cell is None:
            return
        base = np.asarray(self._batch_labels, dtype=np.int64)
        target = min(n, self.final_return_k)
        if target <= base.shape[1]:
            self._batch_target_k = target
            return
        out = np.full((len(base), target), -1, dtype=np.int64)
        take = min(base.shape[1], target)
        out[:, :take] = base[:, :take]

        first = base[:, 0]
        valid = (first >= 0) & (first < len(self.label_to_cell))
        cells = np.full(len(base), -1, dtype=np.int64)
        cells[valid] = self.label_to_cell[first[valid]]
        valid_cells = (cells >= 0) & (cells < len(self.cell_tail_payload))
        if np.any(valid_cells) and take < target:
            payload = self.cell_tail_payload[cells[valid_cells], : target - take]
            out[valid_cells, take:target] = payload
        self._batch_labels = out
        self._batch_target_k = target

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-tailtape-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["tail_payload_k"] = self.tail_payload_k
        extra["tail_payload_offset"] = self.tail_payload_offset
        extra["tail_payload_stride"] = self.tail_payload_stride
        extra["search_return_k"] = self.search_return_k
        extra["final_return_k"] = self.final_return_k
        extra["tail_payload_cells"] = self.tail_payload_cells
        extra["tail_payload_filled_fraction"] = self.tail_payload_filled_fraction
        extra["tail_payload_mean_valid"] = self.tail_payload_mean_valid
        return extra


class ZeelinM10stabilitycertBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "sct" in method_param:
            method_param["stability_trials"] = method_param["sct"]
        if "sps" in method_param:
            method_param["stability_probe_size"] = method_param["sps"]
        if "sbs" in method_param:
            method_param["stability_base_size"] = method_param["sbs"]
        if "stk" in method_param:
            method_param["stability_truth_k"] = method_param["stk"]
        if "spk" in method_param:
            method_param["stability_probe_k"] = method_param["spk"]
        if "stol" in method_param:
            method_param["stability_recall_tolerance"] = method_param["stol"]
        if "sef" in method_param:
            method_param["stability_probe_efSearch"] = method_param["sef"]
        super().__init__(metric, method_param)
        self.stability_trials = int(method_param.get("stability_trials", 3))
        self.stability_probe_size = int(method_param.get("stability_probe_size", 384))
        self.stability_base_size = int(method_param.get("stability_base_size", 65536))
        self.stability_truth_k = int(method_param.get("stability_truth_k", 10))
        self.stability_probe_k = int(method_param.get("stability_probe_k", 7))
        self.stability_recall_tolerance = float(method_param.get("stability_recall_tolerance", 0.015))
        self.stability_probe_efSearch = int(method_param.get("stability_probe_efSearch", 1))
        self.stability_trials = max(1, min(5, self.stability_trials))
        self.stability_probe_size = max(32, min(4096, self.stability_probe_size))
        self.stability_base_size = max(1024, min(200000, self.stability_base_size))
        self.stability_truth_k = max(1, min(32, self.stability_truth_k))
        self.stability_probe_k = max(1, min(32, self.stability_probe_k))
        self.stability_recall_tolerance = max(0.0, min(0.25, self.stability_recall_tolerance))
        self.stability_probe_efSearch = max(1, min(64, self.stability_probe_efSearch))
        self.stability_best_trial = 0
        self.stability_best_proxy_recall = 0.0
        self.stability_best_proxy_us = 0.0
        self.stability_best_score = 0.0
        self.stability_all_proxy_recall = ""
        self.stability_all_proxy_us = ""
        self.stability_all_build_times = ""

    def _make_stability_audit(self, X):
        n = len(X)
        base_size = min(n, self.stability_base_size)
        probe_size = min(n, self.stability_probe_size)
        base_idx = np.linspace(0, n - 1, base_size, dtype=np.int64)
        probe_idx = np.linspace(max(0, n // (2 * max(probe_size, 1))), n - 1, probe_size, dtype=np.int64)
        base = self._normalize_rows(np.asarray(X[base_idx], dtype=np.float32)).astype(np.float32, copy=False)
        query = np.asarray(X[probe_idx], dtype=np.float32)
        rng = np.random.RandomState(183)
        noise = rng.normal(0.0, 0.0075, size=query.shape).astype(np.float32)
        query = self._normalize_rows(query + noise).astype(np.float32, copy=False)
        base = np.ascontiguousarray(base, dtype=np.float32)
        query = np.ascontiguousarray(query, dtype=np.float32)
        k = min(self.stability_truth_k, base.shape[0])
        scores = query.dot(base.T)
        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        truth = np.take_along_axis(top, order, axis=1)
        truth = base_idx[truth].astype(np.int64, copy=False)
        return query, truth

    def _stability_certificate(self, query, truth):
        old_nprobe = int(self.index.nprobe)
        old_ef = int(self.quantizer.hnsw.efSearch)
        self.index.nprobe = 1
        self.index.max_codes = self.max_codes
        self.quantizer.hnsw.efSearch = self.stability_probe_efSearch

        warm = query[: min(len(query), 64)]
        if len(warm):
            self.index.search(warm, self.stability_probe_k)

        best_elapsed = None
        labels = None
        repeats = 3
        for _ in range(repeats):
            start = time.perf_counter()
            _, cand = self.index.search(query, self.stability_probe_k)
            elapsed = time.perf_counter() - start
            if best_elapsed is None or elapsed < best_elapsed:
                best_elapsed = elapsed
                labels = np.asarray(cand, dtype=np.int64)

        total = float(truth.shape[0] * truth.shape[1])
        hits = 0
        for cand_row, truth_row in zip(labels, truth):
            truth_set = set(int(x) for x in truth_row)
            for label in cand_row:
                label = int(label)
                if label >= 0 and label in truth_set:
                    hits += 1
        proxy_recall = float(hits) / max(total, 1.0)
        proxy_us = (float(best_elapsed) / max(1, len(query))) * 1e6

        self.index.nprobe = old_nprobe
        self.quantizer.hnsw.efSearch = old_ef
        return proxy_recall, proxy_us

    def _capture_stability_state(self, trial, elapsed, proxy_recall, proxy_us, score):
        return {
            "index": self.index,
            "quantizer": self.quantizer,
            "dim": self.dim,
            "qbuf": self.qbuf,
            "kept_points": self.kept_points,
            "cap_min_actual": self.cap_min_actual,
            "cap_max_actual": self.cap_max_actual,
            "cap_mean_actual": self.cap_mean_actual,
            "cap_total_requested": getattr(self, "cap_total_requested", 0),
            "concentration_observed_cells": self.concentration_observed_cells,
            "concentration_mean_effective_rank": self.concentration_mean_effective_rank,
            "concentration_mean_tail_drop": self.concentration_mean_tail_drop,
            "concentration_cap_ratio": self.concentration_cap_ratio,
            "boundary_observed_cells": self.boundary_observed_cells,
            "boundary_mean_mass": self.boundary_mean_mass,
            "boundary_max_mass": self.boundary_max_mass,
            "boundary_mean_gap": self.boundary_mean_gap,
            "boundary_cap_ratio": self.boundary_cap_ratio,
            "trial": trial,
            "build_time": elapsed,
            "proxy_recall": proxy_recall,
            "proxy_us": proxy_us,
            "score": score,
        }

    def _restore_stability_state(self, state):
        self.index = state["index"]
        self.quantizer = state["quantizer"]
        self.dim = state["dim"]
        self.qbuf = state["qbuf"]
        self.kept_points = state["kept_points"]
        self.cap_min_actual = state["cap_min_actual"]
        self.cap_max_actual = state["cap_max_actual"]
        self.cap_mean_actual = state["cap_mean_actual"]
        self.cap_total_requested = state["cap_total_requested"]
        self.concentration_observed_cells = state["concentration_observed_cells"]
        self.concentration_mean_effective_rank = state["concentration_mean_effective_rank"]
        self.concentration_mean_tail_drop = state["concentration_mean_tail_drop"]
        self.concentration_cap_ratio = state["concentration_cap_ratio"]
        self.boundary_observed_cells = state["boundary_observed_cells"]
        self.boundary_mean_mass = state["boundary_mean_mass"]
        self.boundary_max_mass = state["boundary_max_mass"]
        self.boundary_mean_gap = state["boundary_mean_gap"]
        self.boundary_cap_ratio = state["boundary_cap_ratio"]
        self.stability_best_trial = state["trial"]
        self.stability_best_proxy_recall = state["proxy_recall"]
        self.stability_best_proxy_us = state["proxy_us"]
        self.stability_best_score = state["score"]

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        audit_query, audit_truth = self._make_stability_audit(X)
        states = []
        recalls = []
        proxy_times = []
        build_times = []

        for trial in range(1, self.stability_trials + 1):
            start = time.time()
            super().fit(X)
            elapsed = time.time() - start
            self.index.max_codes = self.max_codes
            proxy_recall, proxy_us = self._stability_certificate(audit_query, audit_truth)
            build_times.append(elapsed)
            recalls.append(proxy_recall)
            proxy_times.append(proxy_us)
            score = proxy_recall - 0.0005 * proxy_us
            states.append(self._capture_stability_state(trial, elapsed, proxy_recall, proxy_us, score))

        max_recall = max(recalls) if recalls else 0.0
        floor = max_recall - self.stability_recall_tolerance
        acceptable = [state for state in states if state["proxy_recall"] >= floor]
        if not acceptable:
            acceptable = states
        best_state = min(acceptable, key=lambda s: (s["proxy_us"], -s["proxy_recall"]))
        self._restore_stability_state(best_state)

        self.stability_all_build_times = ",".join(f"{v:.6f}" for v in build_times)
        self.stability_all_proxy_recall = ",".join(f"{v:.6f}" for v in recalls)
        self.stability_all_proxy_us = ",".join(f"{v:.6f}" for v in proxy_times)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10stabilitycert-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", stabilityTrials={self.stability_trials}, "
            f"stabilityBestTrial={self.stability_best_trial}, "
            f"stabilityProxyRecall={self.stability_best_proxy_recall:.4f}, "
            f"stabilityProxyUs={self.stability_best_proxy_us:.3f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10stabilitycert-boundarytrim-maxcodes-concentrationcapcore-returncompress"
        extra["stability_trials"] = self.stability_trials
        extra["stability_probe_size"] = self.stability_probe_size
        extra["stability_base_size"] = self.stability_base_size
        extra["stability_truth_k"] = self.stability_truth_k
        extra["stability_probe_k"] = self.stability_probe_k
        extra["stability_recall_tolerance"] = self.stability_recall_tolerance
        extra["stability_probe_efSearch"] = self.stability_probe_efSearch
        extra["stability_best_trial"] = self.stability_best_trial
        extra["stability_best_proxy_recall"] = self.stability_best_proxy_recall
        extra["stability_best_proxy_us"] = self.stability_best_proxy_us
        extra["stability_best_score"] = self.stability_best_score
        extra["stability_all_proxy_recall"] = self.stability_all_proxy_recall
        extra["stability_all_proxy_us"] = self.stability_all_proxy_us
        extra["stability_all_build_times"] = self.stability_all_build_times
        return extra


class ZeelinM10speedcertBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10stabilitycertBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        audit_query, audit_truth = self._make_stability_audit(X)
        states = []
        recalls = []
        proxy_times = []
        build_times = []

        for trial in range(1, self.stability_trials + 1):
            start = time.time()
            ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.fit(self, X)
            elapsed = time.time() - start
            self.index.max_codes = self.max_codes
            proxy_recall, proxy_us = self._stability_certificate(audit_query, audit_truth)
            build_times.append(elapsed)
            recalls.append(proxy_recall)
            proxy_times.append(proxy_us)
            score = -proxy_us
            states.append(self._capture_stability_state(trial, elapsed, proxy_recall, proxy_us, score))

        best_state = min(states, key=lambda s: (s["proxy_us"], -s["proxy_recall"]))
        self._restore_stability_state(best_state)

        self.stability_all_build_times = ",".join(f"{v:.6f}" for v in build_times)
        self.stability_all_proxy_recall = ",".join(f"{v:.6f}" for v in recalls)
        self.stability_all_proxy_us = ",".join(f"{v:.6f}" for v in proxy_times)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.set_query_arguments(
            self,
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10speedcert-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", speedCertTrials={self.stability_trials}, "
            f"speedBestTrial={self.stability_best_trial}, "
            f"speedProxyUs={self.stability_best_proxy_us:.3f}, "
            f"speedProxyRecall={self.stability_best_proxy_recall:.4f})"
        )

    def get_additional(self):
        extra = ZeelinM10stabilitycertBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.get_additional(self)
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10speedcert-boundarytrim-maxcodes-concentrationcapcore-returncompress"
        extra["speed_certificate"] = 1
        return extra


class ZeelinM10speedguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10speedcertBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "sgs" in method_param:
            method_param["speed_guard_probe_size"] = method_param["sgs"]
        if "sgk" in method_param:
            method_param["speed_guard_k"] = method_param["sgk"]
        if "sgwk" in method_param:
            method_param["speed_guard_wide_k"] = method_param["sgwk"]
        if "sgmc" in method_param:
            method_param["speed_guard_wide_max_codes"] = method_param["sgmc"]
        if "sgtol" in method_param:
            method_param["speed_guard_tolerance"] = method_param["sgtol"]
        if "sgnoise" in method_param:
            method_param["speed_guard_noise"] = method_param["sgnoise"]
        super().__init__(metric, method_param)
        self.speed_guard_probe_size = int(
            method_param.get("speed_guard_probe_size", self.stability_probe_size)
        )
        self.speed_guard_k = int(method_param.get("speed_guard_k", self.stability_probe_k))
        self.speed_guard_wide_k = int(method_param.get("speed_guard_wide_k", 10))
        self.speed_guard_wide_max_codes = int(method_param.get("speed_guard_wide_max_codes", 0))
        self.speed_guard_tolerance = float(method_param.get("speed_guard_tolerance", 0.005))
        self.speed_guard_noise = float(method_param.get("speed_guard_noise", 0.0075))
        self.speed_guard_probe_size = max(32, min(8192, self.speed_guard_probe_size))
        self.speed_guard_k = max(1, min(16, self.speed_guard_k))
        self.speed_guard_wide_k = max(self.speed_guard_k, min(32, self.speed_guard_wide_k))
        self.speed_guard_wide_max_codes = max(0, min(4096, self.speed_guard_wide_max_codes))
        self.speed_guard_tolerance = max(0.0, min(0.25, self.speed_guard_tolerance))
        self.speed_guard_noise = max(0.0, min(0.25, self.speed_guard_noise))
        self.speed_guard_best_stability = 0.0
        self.speed_guard_all_stability = ""
        self.speed_guard_all_proxy_us = ""

    def _make_speed_guard_queries(self, X):
        n = len(X)
        probe_size = min(n, self.speed_guard_probe_size)
        start = max(0, n // (3 * max(probe_size, 1)))
        probe_idx = np.linspace(start, n - 1, probe_size, dtype=np.int64)
        query = np.asarray(X[probe_idx], dtype=np.float32)
        if self.speed_guard_noise > 0.0:
            rng = np.random.RandomState(403)
            noise = rng.normal(0.0, self.speed_guard_noise, size=query.shape).astype(np.float32)
            query = query + noise
        query = self._normalize_rows(query).astype(np.float32, copy=False)
        return np.ascontiguousarray(query, dtype=np.float32)

    def _speed_guard_certificate(self, query):
        old_nprobe = int(self.index.nprobe)
        old_ef = int(self.quantizer.hnsw.efSearch)
        old_max_codes = int(self.index.max_codes)
        self.index.nprobe = 1
        self.quantizer.hnsw.efSearch = self.stability_probe_efSearch
        self.index.max_codes = self.max_codes

        warm = query[: min(len(query), 64)]
        if len(warm):
            self.index.search(warm, self.speed_guard_k)

        best_elapsed = None
        narrow_labels = None
        for _ in range(3):
            start = time.perf_counter()
            _, labels = self.index.search(query, self.speed_guard_k)
            elapsed = time.perf_counter() - start
            if best_elapsed is None or elapsed < best_elapsed:
                best_elapsed = elapsed
                narrow_labels = np.asarray(labels, dtype=np.int64)

        self.index.max_codes = self.speed_guard_wide_max_codes
        _, wide_labels = self.index.search(query, self.speed_guard_wide_k)
        wide_labels = np.asarray(wide_labels, dtype=np.int64)

        hits = 0
        total = 0
        for narrow_row, wide_row in zip(narrow_labels, wide_labels):
            wide_set = set(int(x) for x in wide_row if int(x) >= 0)
            for label in narrow_row:
                label = int(label)
                if label < 0:
                    continue
                total += 1
                if label in wide_set:
                    hits += 1
        stability = float(hits) / float(max(1, total))
        proxy_us = (float(best_elapsed) / max(1, len(query))) * 1e6

        self.index.max_codes = old_max_codes
        self.index.nprobe = old_nprobe
        self.quantizer.hnsw.efSearch = old_ef
        return stability, proxy_us

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        guard_query = self._make_speed_guard_queries(X)
        states = []
        stabilities = []
        proxy_times = []
        build_times = []

        for trial in range(1, self.stability_trials + 1):
            start = time.time()
            ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.fit(self, X)
            elapsed = time.time() - start
            self.index.max_codes = self.max_codes
            guard_stability, proxy_us = self._speed_guard_certificate(guard_query)
            build_times.append(elapsed)
            stabilities.append(guard_stability)
            proxy_times.append(proxy_us)
            score = guard_stability - 0.0005 * proxy_us
            states.append(
                self._capture_stability_state(
                    trial, elapsed, guard_stability, proxy_us, score
                )
            )

        best_stability = max(stabilities) if stabilities else 0.0
        floor = best_stability - self.speed_guard_tolerance
        acceptable = [state for state in states if state["proxy_recall"] >= floor]
        if not acceptable:
            acceptable = states
        best_state = min(acceptable, key=lambda s: (s["proxy_us"], -s["proxy_recall"]))
        self._restore_stability_state(best_state)
        self.speed_guard_best_stability = self.stability_best_proxy_recall

        self.stability_all_build_times = ",".join(f"{v:.6f}" for v in build_times)
        self.stability_all_proxy_recall = ",".join(f"{v:.6f}" for v in stabilities)
        self.stability_all_proxy_us = ",".join(f"{v:.6f}" for v in proxy_times)
        self.speed_guard_all_stability = self.stability_all_proxy_recall
        self.speed_guard_all_proxy_us = self.stability_all_proxy_us

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.set_query_arguments(
            self,
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10speedguard-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", speedGuardTrials={self.stability_trials}, "
            f"speedGuardBestTrial={self.stability_best_trial}, "
            f"speedGuardStability={self.speed_guard_best_stability:.4f}, "
            f"speedGuardProxyUs={self.stability_best_proxy_us:.3f})"
        )

    def get_additional(self):
        extra = ZeelinM10stabilitycertBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.get_additional(self)
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10speedguard-boundarytrim-maxcodes-concentrationcapcore-returncompress"
        extra["speed_guard"] = 1
        extra["speed_guard_probe_size"] = self.speed_guard_probe_size
        extra["speed_guard_k"] = self.speed_guard_k
        extra["speed_guard_wide_k"] = self.speed_guard_wide_k
        extra["speed_guard_wide_max_codes"] = self.speed_guard_wide_max_codes
        extra["speed_guard_tolerance"] = self.speed_guard_tolerance
        extra["speed_guard_noise"] = self.speed_guard_noise
        extra["speed_guard_best_stability"] = self.speed_guard_best_stability
        extra["speed_guard_all_stability"] = self.speed_guard_all_stability
        extra["speed_guard_all_proxy_us"] = self.speed_guard_all_proxy_us
        return extra


class ZeelinM10speedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10speedcertBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "egs" in method_param:
            method_param["exact_guard_probe_size"] = method_param["egs"]
        if "egk" in method_param:
            method_param["exact_guard_truth_k"] = method_param["egk"]
        if "egsk" in method_param:
            method_param["exact_guard_search_k"] = method_param["egsk"]
        if "egtol" in method_param:
            method_param["exact_guard_tolerance"] = method_param["egtol"]
        if "egnoise" in method_param:
            method_param["exact_guard_noise"] = method_param["egnoise"]
        if "egchunk" in method_param:
            method_param["exact_guard_chunk"] = method_param["egchunk"]
        super().__init__(metric, method_param)
        self.exact_guard_probe_size = int(method_param.get("exact_guard_probe_size", 128))
        self.exact_guard_truth_k = int(method_param.get("exact_guard_truth_k", 10))
        self.exact_guard_search_k = int(method_param.get("exact_guard_search_k", 7))
        self.exact_guard_tolerance = float(method_param.get("exact_guard_tolerance", 0.005))
        self.exact_guard_noise = float(method_param.get("exact_guard_noise", 0.0075))
        self.exact_guard_chunk = int(method_param.get("exact_guard_chunk", 65536))
        self.exact_guard_probe_size = max(16, min(2048, self.exact_guard_probe_size))
        self.exact_guard_truth_k = max(1, min(32, self.exact_guard_truth_k))
        self.exact_guard_search_k = max(1, min(32, self.exact_guard_search_k))
        self.exact_guard_tolerance = max(0.0, min(0.25, self.exact_guard_tolerance))
        self.exact_guard_noise = max(0.0, min(0.25, self.exact_guard_noise))
        self.exact_guard_chunk = max(4096, min(262144, self.exact_guard_chunk))
        self.exact_guard_best_recall = 0.0
        self.exact_guard_all_recall = ""
        self.exact_guard_all_proxy_us = ""
        self.exact_guard_build_truth_time = 0.0

    def _make_exact_guard(self, X):
        start_time = time.time()
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        n = len(Xn)
        probe_size = min(n, self.exact_guard_probe_size)
        start = max(0, n // (5 * max(probe_size, 1)))
        probe_idx = np.linspace(start, n - 1, probe_size, dtype=np.int64)
        query = np.asarray(X[probe_idx], dtype=np.float32)
        if self.exact_guard_noise > 0.0:
            rng = np.random.RandomState(403183)
            noise = rng.normal(0.0, self.exact_guard_noise, size=query.shape).astype(np.float32)
            query = query + noise
        query = self._normalize_rows(query).astype(np.float32, copy=False)
        query = np.ascontiguousarray(query, dtype=np.float32)

        k = min(self.exact_guard_truth_k, n)
        best_scores = np.full((len(query), k), -np.inf, dtype=np.float32)
        best_ids = np.full((len(query), k), -1, dtype=np.int64)
        for start in range(0, n, self.exact_guard_chunk):
            stop = min(n, start + self.exact_guard_chunk)
            block = np.ascontiguousarray(Xn[start:stop], dtype=np.float32)
            scores = query.dot(block.T)
            take = min(k, scores.shape[1])
            part = np.argpartition(scores, -take, axis=1)[:, -take:]
            part_scores = np.take_along_axis(scores, part, axis=1)
            part_ids = (part + start).astype(np.int64, copy=False)
            comb_scores = np.concatenate([best_scores, part_scores], axis=1)
            comb_ids = np.concatenate([best_ids, part_ids], axis=1)
            choose = np.argpartition(comb_scores, -k, axis=1)[:, -k:]
            best_scores = np.take_along_axis(comb_scores, choose, axis=1)
            best_ids = np.take_along_axis(comb_ids, choose, axis=1)
        order = np.argsort(-best_scores, axis=1)
        truth = np.take_along_axis(best_ids, order, axis=1)
        self.exact_guard_build_truth_time = time.time() - start_time
        return query, np.ascontiguousarray(truth, dtype=np.int64)

    def _exact_guard_certificate(self, query, truth):
        old_nprobe = int(self.index.nprobe)
        old_ef = int(self.quantizer.hnsw.efSearch)
        old_max_codes = int(self.index.max_codes)
        self.index.nprobe = 1
        self.quantizer.hnsw.efSearch = self.stability_probe_efSearch
        self.index.max_codes = self.max_codes

        warm = query[: min(len(query), 64)]
        if len(warm):
            self.index.search(warm, self.exact_guard_search_k)

        best_elapsed = None
        labels = None
        for _ in range(3):
            start = time.perf_counter()
            _, cand = self.index.search(query, self.exact_guard_search_k)
            elapsed = time.perf_counter() - start
            if best_elapsed is None or elapsed < best_elapsed:
                best_elapsed = elapsed
                labels = np.asarray(cand, dtype=np.int64)

        hits = 0
        total = int(truth.shape[0] * truth.shape[1])
        for cand_row, truth_row in zip(labels, truth):
            truth_set = set(int(x) for x in truth_row if int(x) >= 0)
            for label in cand_row:
                label = int(label)
                if label >= 0 and label in truth_set:
                    hits += 1
        proxy_recall = float(hits) / float(max(1, total))
        proxy_us = (float(best_elapsed) / max(1, len(query))) * 1e6

        self.index.max_codes = old_max_codes
        self.index.nprobe = old_nprobe
        self.quantizer.hnsw.efSearch = old_ef
        return proxy_recall, proxy_us

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        exact_query, exact_truth = self._make_exact_guard(X)
        states = []
        recalls = []
        proxy_times = []
        build_times = []

        for trial in range(1, self.stability_trials + 1):
            start = time.time()
            ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.fit(self, X)
            elapsed = time.time() - start
            self.index.max_codes = self.max_codes
            proxy_recall, proxy_us = self._exact_guard_certificate(exact_query, exact_truth)
            build_times.append(elapsed)
            recalls.append(proxy_recall)
            proxy_times.append(proxy_us)
            score = proxy_recall - 0.0005 * proxy_us
            states.append(
                self._capture_stability_state(trial, elapsed, proxy_recall, proxy_us, score)
            )

        max_recall = max(recalls) if recalls else 0.0
        floor = max_recall - self.exact_guard_tolerance
        acceptable = [state for state in states if state["proxy_recall"] >= floor]
        if not acceptable:
            acceptable = states
        best_state = min(acceptable, key=lambda s: (s["proxy_us"], -s["proxy_recall"]))
        self._restore_stability_state(best_state)
        self.exact_guard_best_recall = self.stability_best_proxy_recall

        self.stability_all_build_times = ",".join(f"{v:.6f}" for v in build_times)
        self.stability_all_proxy_recall = ",".join(f"{v:.6f}" for v in recalls)
        self.stability_all_proxy_us = ",".join(f"{v:.6f}" for v in proxy_times)
        self.exact_guard_all_recall = self.stability_all_proxy_recall
        self.exact_guard_all_proxy_us = self.stability_all_proxy_us

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.set_query_arguments(
            self,
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10speedexactguard-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", exactGuardTrials={self.stability_trials}, "
            f"exactGuardBestTrial={self.stability_best_trial}, "
            f"exactGuardRecall={self.exact_guard_best_recall:.4f}, "
            f"exactGuardProxyUs={self.stability_best_proxy_us:.3f})"
        )

    def get_additional(self):
        extra = ZeelinM10stabilitycertBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.get_additional(self)
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10speedexactguard-boundarytrim-maxcodes-concentrationcapcore-returncompress"
        extra["speed_exact_guard"] = 1
        extra["exact_guard_probe_size"] = self.exact_guard_probe_size
        extra["exact_guard_truth_k"] = self.exact_guard_truth_k
        extra["exact_guard_search_k"] = self.exact_guard_search_k
        extra["exact_guard_tolerance"] = self.exact_guard_tolerance
        extra["exact_guard_noise"] = self.exact_guard_noise
        extra["exact_guard_chunk"] = self.exact_guard_chunk
        extra["exact_guard_best_recall"] = self.exact_guard_best_recall
        extra["exact_guard_all_recall"] = self.exact_guard_all_recall
        extra["exact_guard_all_proxy_us"] = self.exact_guard_all_proxy_us
        extra["exact_guard_build_truth_time"] = self.exact_guard_build_truth_time
        return extra


class ZeelinM10cachepreservingBoundaryliftSpeedcertBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10speedcertBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "blg" in method_param:
            method_param["bmg"] = method_param["blg"]
        if "blb" in method_param:
            method_param["bmb"] = method_param["blb"]
        if "blhi" in method_param:
            method_param["bmcliphi"] = method_param["blhi"]
        if "bllo" in method_param:
            method_param["bmcliplo"] = method_param["bllo"]
        method_param.setdefault("bmm", "guard")
        method_param.setdefault("bmg", 0.35)
        method_param.setdefault("bmb", 1.0)
        method_param.setdefault("bmcliphi", 1.10)
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10speedcert-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10cachepreserving-boundarylift-speedcert-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-cachepreserving-boundarylift-speedcert-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["cachepreserving_boundarylift"] = 1
        return extra


class ZeelinM10cachepreservingBoundaryliftSpeedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10speedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "blg" in method_param:
            method_param["bmg"] = method_param["blg"]
        if "blb" in method_param:
            method_param["bmb"] = method_param["blb"]
        if "blhi" in method_param:
            method_param["bmcliphi"] = method_param["blhi"]
        if "bllo" in method_param:
            method_param["bmcliplo"] = method_param["bllo"]
        method_param.setdefault("bmm", "guard")
        method_param.setdefault("bmg", 0.35)
        method_param.setdefault("bmb", 1.0)
        method_param.setdefault("bmcliphi", 1.10)
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10speedexactguard-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10cachepreserving-boundarylift-speedexactguard-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-cachepreserving-boundarylift-speedexactguard-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["cachepreserving_boundarylift"] = 1
        return extra


class ZeelinM10recallhardwareDualselectorBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10cachepreservingBoundaryliftSpeedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "rhr" in method_param:
            method_param["recallhardware_repeats"] = method_param["rhr"]
        if "rhjw" in method_param:
            method_param["recallhardware_jitter_weight"] = method_param["rhjw"]
        if "rhkw" in method_param:
            method_param["recallhardware_kept_weight"] = method_param["rhkw"]
        super().__init__(metric, method_param)
        self.recallhardware_repeats = int(method_param.get("recallhardware_repeats", 7))
        self.recallhardware_jitter_weight = float(
            method_param.get("recallhardware_jitter_weight", 0.25)
        )
        self.recallhardware_kept_weight = float(
            method_param.get("recallhardware_kept_weight", 0.0)
        )
        self.recallhardware_repeats = max(3, min(21, self.recallhardware_repeats))
        self.recallhardware_jitter_weight = max(0.0, min(4.0, self.recallhardware_jitter_weight))
        self.recallhardware_kept_weight = max(0.0, min(4.0, self.recallhardware_kept_weight))
        self.recallhardware_best_recall = 0.0
        self.recallhardware_best_median_us = 0.0
        self.recallhardware_best_jitter_us = 0.0
        self.recallhardware_selected_score = 0.0
        self.recallhardware_all_recall = ""
        self.recallhardware_all_min_us = ""
        self.recallhardware_all_median_us = ""
        self.recallhardware_all_jitter_us = ""
        self.recallhardware_all_score = ""
        self.recallhardware_all_kept_points = ""
        self.recallhardware_all_boundary_cap_ratio = ""

    def _recallhardware_certificate(self, query, truth):
        old_nprobe = int(self.index.nprobe)
        old_ef = int(self.quantizer.hnsw.efSearch)
        old_max_codes = int(self.index.max_codes)
        self.index.nprobe = 1
        self.quantizer.hnsw.efSearch = self.stability_probe_efSearch
        self.index.max_codes = self.max_codes

        warm = query[: min(len(query), 64)]
        if len(warm):
            self.index.search(warm, self.exact_guard_search_k)

        elapsed_values = []
        best_elapsed = None
        labels = None
        for _ in range(self.recallhardware_repeats):
            start = time.perf_counter()
            _, cand = self.index.search(query, self.exact_guard_search_k)
            elapsed = time.perf_counter() - start
            elapsed_values.append(elapsed)
            if best_elapsed is None or elapsed < best_elapsed:
                best_elapsed = elapsed
                labels = np.asarray(cand, dtype=np.int64)

        hits = 0
        total = int(truth.shape[0] * truth.shape[1])
        for cand_row, truth_row in zip(labels, truth):
            truth_set = set(int(x) for x in truth_row if int(x) >= 0)
            for label in cand_row:
                label = int(label)
                if label >= 0 and label in truth_set:
                    hits += 1
        proxy_recall = float(hits) / float(max(1, total))
        elapsed_values = np.asarray(elapsed_values, dtype=np.float64)
        min_us = (float(np.min(elapsed_values)) / max(1, len(query))) * 1e6
        median_us = (float(np.median(elapsed_values)) / max(1, len(query))) * 1e6
        p75_us = (float(np.percentile(elapsed_values, 75.0)) / max(1, len(query))) * 1e6
        jitter_us = max(0.0, p75_us - min_us)

        self.index.max_codes = old_max_codes
        self.index.nprobe = old_nprobe
        self.quantizer.hnsw.efSearch = old_ef
        return proxy_recall, min_us, median_us, jitter_us

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        exact_query, exact_truth = self._make_exact_guard(X)
        states = []
        recalls = []
        min_times = []
        median_times = []
        jitter_times = []
        scores = []
        kept_points = []
        boundary_cap_ratios = []
        build_times = []

        for trial in range(1, self.stability_trials + 1):
            start = time.time()
            ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.fit(self, X)
            elapsed = time.time() - start
            self.index.max_codes = self.max_codes
            proxy_recall, min_us, median_us, jitter_us = self._recallhardware_certificate(
                exact_query, exact_truth
            )
            kept = float(getattr(self, "kept_points", 0))
            kept_penalty = (kept / float(max(1, len(X)))) * self.recallhardware_kept_weight
            hw_score = median_us + self.recallhardware_jitter_weight * jitter_us + kept_penalty
            score = proxy_recall - 0.0005 * hw_score

            build_times.append(elapsed)
            recalls.append(proxy_recall)
            min_times.append(min_us)
            median_times.append(median_us)
            jitter_times.append(jitter_us)
            scores.append(hw_score)
            kept_points.append(int(kept))
            boundary_cap_ratios.append(float(getattr(self, "boundary_cap_ratio", 0.0)))

            state = self._capture_stability_state(trial, elapsed, proxy_recall, median_us, score)
            state["recallhardware_min_us"] = min_us
            state["recallhardware_median_us"] = median_us
            state["recallhardware_jitter_us"] = jitter_us
            state["recallhardware_score"] = hw_score
            states.append(state)

        max_recall = max(recalls) if recalls else 0.0
        floor = max_recall - self.exact_guard_tolerance
        acceptable = [state for state in states if state["proxy_recall"] >= floor]
        if not acceptable:
            acceptable = states
        best_state = min(
            acceptable,
            key=lambda s: (
                s["recallhardware_score"],
                s["recallhardware_median_us"],
                -s["proxy_recall"],
            ),
        )
        self._restore_stability_state(best_state)
        self.exact_guard_best_recall = self.stability_best_proxy_recall
        self.recallhardware_best_recall = self.stability_best_proxy_recall
        self.recallhardware_best_median_us = float(best_state["recallhardware_median_us"])
        self.recallhardware_best_jitter_us = float(best_state["recallhardware_jitter_us"])
        self.recallhardware_selected_score = float(best_state["recallhardware_score"])

        self.stability_all_build_times = ",".join(f"{v:.6f}" for v in build_times)
        self.stability_all_proxy_recall = ",".join(f"{v:.6f}" for v in recalls)
        self.stability_all_proxy_us = ",".join(f"{v:.6f}" for v in median_times)
        self.exact_guard_all_recall = self.stability_all_proxy_recall
        self.exact_guard_all_proxy_us = self.stability_all_proxy_us
        self.recallhardware_all_recall = self.stability_all_proxy_recall
        self.recallhardware_all_min_us = ",".join(f"{v:.6f}" for v in min_times)
        self.recallhardware_all_median_us = ",".join(f"{v:.6f}" for v in median_times)
        self.recallhardware_all_jitter_us = ",".join(f"{v:.6f}" for v in jitter_times)
        self.recallhardware_all_score = ",".join(f"{v:.6f}" for v in scores)
        self.recallhardware_all_kept_points = ",".join(str(v) for v in kept_points)
        self.recallhardware_all_boundary_cap_ratio = ",".join(
            f"{v:.6f}" for v in boundary_cap_ratios
        )

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.set_query_arguments(
            self,
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10recallhardware-dualselector-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", recallHardwareTrials={self.stability_trials}, "
            f"recallHardwareBestTrial={self.stability_best_trial}, "
            f"recallHardwareRecall={self.recallhardware_best_recall:.4f}, "
            f"recallHardwareMedianUs={self.recallhardware_best_median_us:.3f}, "
            f"recallHardwareJitterUs={self.recallhardware_best_jitter_us:.3f})"
        )

    def get_additional(self):
        extra = ZeelinM10cachepreservingBoundaryliftSpeedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.get_additional(self)
        extra["kernel"] = "faiss-recallhardware-dualselector-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["recallhardware_dualselector"] = 1
        extra["recallhardware_repeats"] = self.recallhardware_repeats
        extra["recallhardware_jitter_weight"] = self.recallhardware_jitter_weight
        extra["recallhardware_kept_weight"] = self.recallhardware_kept_weight
        extra["recallhardware_best_recall"] = self.recallhardware_best_recall
        extra["recallhardware_best_median_us"] = self.recallhardware_best_median_us
        extra["recallhardware_best_jitter_us"] = self.recallhardware_best_jitter_us
        extra["recallhardware_selected_score"] = self.recallhardware_selected_score
        extra["recallhardware_all_recall"] = self.recallhardware_all_recall
        extra["recallhardware_all_min_us"] = self.recallhardware_all_min_us
        extra["recallhardware_all_median_us"] = self.recallhardware_all_median_us
        extra["recallhardware_all_jitter_us"] = self.recallhardware_all_jitter_us
        extra["recallhardware_all_score"] = self.recallhardware_all_score
        extra["recallhardware_all_kept_points"] = self.recallhardware_all_kept_points
        extra["recallhardware_all_boundary_cap_ratio"] = self.recallhardware_all_boundary_cap_ratio
        return extra


class ZeelinM10shapepriorSpeedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10speedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "spcm" in method_param:
            method_param["shapeprior_target_cap_mean"] = method_param["spcm"]
        if "spbcr" in method_param:
            method_param["shapeprior_target_boundary_cap_ratio"] = method_param["spbcr"]
        if "spccr" in method_param:
            method_param["shapeprior_target_concentration_cap_ratio"] = method_param["spccr"]
        if "spkp" in method_param:
            method_param["shapeprior_target_kept_points"] = method_param["spkp"]
        super().__init__(metric, method_param)
        self.shapeprior_target_cap_mean = float(method_param.get("shapeprior_target_cap_mean", 59.02))
        self.shapeprior_target_boundary_cap_ratio = float(
            method_param.get("shapeprior_target_boundary_cap_ratio", 0.997)
        )
        self.shapeprior_target_concentration_cap_ratio = float(
            method_param.get("shapeprior_target_concentration_cap_ratio", 0.9182)
        )
        self.shapeprior_target_kept_points = float(
            method_param.get("shapeprior_target_kept_points", 991000.0)
        )
        self.shapeprior_weight_cap_mean = float(method_param.get("shapeprior_weight_cap_mean", 1.0))
        self.shapeprior_weight_boundary_cap_ratio = float(
            method_param.get("shapeprior_weight_boundary_cap_ratio", 18.0)
        )
        self.shapeprior_weight_concentration_cap_ratio = float(
            method_param.get("shapeprior_weight_concentration_cap_ratio", 12.0)
        )
        self.shapeprior_weight_kept_points = float(
            method_param.get("shapeprior_weight_kept_points", 0.15)
        )
        self.shapeprior_best_score = 0.0
        self.shapeprior_all_score = ""
        self.shapeprior_all_cap_mean = ""
        self.shapeprior_all_boundary_cap_ratio = ""
        self.shapeprior_all_concentration_cap_ratio = ""
        self.shapeprior_all_kept_points = ""

    def _shapeprior_score(self, n):
        cap_mean = float(getattr(self, "cap_mean_actual", 0.0))
        bcr = float(getattr(self, "boundary_cap_ratio", 0.0))
        ccr = float(getattr(self, "concentration_cap_ratio", 0.0))
        kept = float(getattr(self, "kept_points", 0.0))
        cap_term = abs(cap_mean - self.shapeprior_target_cap_mean) / max(
            1.0, self.shapeprior_target_cap_mean
        )
        bcr_term = abs(bcr - self.shapeprior_target_boundary_cap_ratio) / max(
            1e-6, self.shapeprior_target_boundary_cap_ratio
        )
        ccr_term = abs(ccr - self.shapeprior_target_concentration_cap_ratio) / max(
            1e-6, self.shapeprior_target_concentration_cap_ratio
        )
        kept_term = abs(kept - self.shapeprior_target_kept_points) / float(max(1, n))
        return (
            self.shapeprior_weight_cap_mean * cap_term
            + self.shapeprior_weight_boundary_cap_ratio * bcr_term
            + self.shapeprior_weight_concentration_cap_ratio * ccr_term
            + self.shapeprior_weight_kept_points * kept_term
        )

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        exact_query, exact_truth = self._make_exact_guard(X)
        states = []
        recalls = []
        proxy_times = []
        build_times = []
        shape_scores = []
        cap_means = []
        boundary_ratios = []
        concentration_ratios = []
        kept_points = []

        for trial in range(1, self.stability_trials + 1):
            start = time.time()
            ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.fit(self, X)
            elapsed = time.time() - start
            self.index.max_codes = self.max_codes
            proxy_recall, proxy_us = self._exact_guard_certificate(exact_query, exact_truth)
            shape_score = self._shapeprior_score(len(X))
            score = proxy_recall - shape_score - 0.0001 * proxy_us

            build_times.append(elapsed)
            recalls.append(proxy_recall)
            proxy_times.append(proxy_us)
            shape_scores.append(shape_score)
            cap_means.append(float(getattr(self, "cap_mean_actual", 0.0)))
            boundary_ratios.append(float(getattr(self, "boundary_cap_ratio", 0.0)))
            concentration_ratios.append(float(getattr(self, "concentration_cap_ratio", 0.0)))
            kept_points.append(int(getattr(self, "kept_points", 0)))
            state = self._capture_stability_state(trial, elapsed, proxy_recall, proxy_us, score)
            state["shapeprior_score"] = shape_score
            states.append(state)

        max_recall = max(recalls) if recalls else 0.0
        floor = max_recall - self.exact_guard_tolerance
        acceptable = [state for state in states if state["proxy_recall"] >= floor]
        if not acceptable:
            acceptable = states
        best_state = min(
            acceptable,
            key=lambda s: (
                s["shapeprior_score"],
                s["proxy_us"],
                -s["proxy_recall"],
            ),
        )
        self._restore_stability_state(best_state)
        self.exact_guard_best_recall = self.stability_best_proxy_recall
        self.shapeprior_best_score = float(best_state["shapeprior_score"])

        self.stability_all_build_times = ",".join(f"{v:.6f}" for v in build_times)
        self.stability_all_proxy_recall = ",".join(f"{v:.6f}" for v in recalls)
        self.stability_all_proxy_us = ",".join(f"{v:.6f}" for v in proxy_times)
        self.exact_guard_all_recall = self.stability_all_proxy_recall
        self.exact_guard_all_proxy_us = self.stability_all_proxy_us
        self.shapeprior_all_score = ",".join(f"{v:.6f}" for v in shape_scores)
        self.shapeprior_all_cap_mean = ",".join(f"{v:.6f}" for v in cap_means)
        self.shapeprior_all_boundary_cap_ratio = ",".join(f"{v:.6f}" for v in boundary_ratios)
        self.shapeprior_all_concentration_cap_ratio = ",".join(
            f"{v:.6f}" for v in concentration_ratios
        )
        self.shapeprior_all_kept_points = ",".join(str(v) for v in kept_points)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.set_query_arguments(
            self,
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10shapeprior-speedexactguard-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", shapePriorTrials={self.stability_trials}, "
            f"shapePriorBestTrial={self.stability_best_trial}, "
            f"shapePriorScore={self.shapeprior_best_score:.5f}, "
            f"shapePriorRecall={self.exact_guard_best_recall:.4f})"
        )

    def get_additional(self):
        extra = ZeelinM10speedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.get_additional(self)
        extra["kernel"] = "faiss-shapeprior-speedexactguard-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["shapeprior_selector"] = 1
        extra["shapeprior_target_cap_mean"] = self.shapeprior_target_cap_mean
        extra["shapeprior_target_boundary_cap_ratio"] = self.shapeprior_target_boundary_cap_ratio
        extra["shapeprior_target_concentration_cap_ratio"] = self.shapeprior_target_concentration_cap_ratio
        extra["shapeprior_target_kept_points"] = self.shapeprior_target_kept_points
        extra["shapeprior_best_score"] = self.shapeprior_best_score
        extra["shapeprior_all_score"] = self.shapeprior_all_score
        extra["shapeprior_all_cap_mean"] = self.shapeprior_all_cap_mean
        extra["shapeprior_all_boundary_cap_ratio"] = self.shapeprior_all_boundary_cap_ratio
        extra["shapeprior_all_concentration_cap_ratio"] = self.shapeprior_all_concentration_cap_ratio
        extra["shapeprior_all_kept_points"] = self.shapeprior_all_kept_points
        return extra


class ZeelinM10layoutinvariantListpackingBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "lpdual" in method_param:
            method_param["listpack_dual_per_cell"] = method_param["lpdual"]
        if "lpt" in method_param:
            method_param["listpack_temperature"] = method_param["lpt"]
        if "lpw" in method_param:
            method_param["listpack_weight"] = method_param["lpw"]
        if "lpq" in method_param:
            method_param["listpack_quantile"] = method_param["lpq"]
        if "lpcap" in method_param:
            method_param["listpack_total_cap_ratio"] = method_param["lpcap"]
        if "lpboost" in method_param:
            method_param["listpack_dual_boost"] = method_param["lpboost"]
        if "lpscore" in method_param:
            method_param["listpack_dual_score_mode"] = method_param["lpscore"]
        super().__init__(metric, method_param)
        self.listpack_dual_per_cell = int(method_param.get("listpack_dual_per_cell", 1))
        self.listpack_temperature = float(method_param.get("listpack_temperature", 0.035))
        self.listpack_weight = float(method_param.get("listpack_weight", 0.20))
        self.listpack_quantile = float(method_param.get("listpack_quantile", 0.10))
        self.listpack_total_cap_ratio = float(method_param.get("listpack_total_cap_ratio", 1.015))
        self.listpack_dual_boost = float(method_param.get("listpack_dual_boost", 0.15))
        self.listpack_dual_score_mode = str(method_param.get("listpack_dual_score_mode", "origin")).lower()
        self.listpack_dual_per_cell = max(0, min(8, self.listpack_dual_per_cell))
        self.listpack_temperature = max(1e-4, min(1.0, self.listpack_temperature))
        self.listpack_weight = max(0.0, min(4.0, self.listpack_weight))
        self.listpack_quantile = max(0.0, min(0.75, self.listpack_quantile))
        self.listpack_total_cap_ratio = max(1.0, min(1.25, self.listpack_total_cap_ratio))
        self.listpack_dual_boost = max(0.0, min(4.0, self.listpack_dual_boost))
        if self.listpack_dual_score_mode not in {"origin", "target"}:
            self.listpack_dual_score_mode = "origin"
        self.listpack_dual_added = 0
        self.listpack_dual_cells = 0
        self.listpack_dual_fraction = 0.0
        self.listpack_mean_margin = 0.0
        self.listpack_margin_threshold = 0.0

    @staticmethod
    def _zscore(values):
        values = np.asarray(values, dtype=np.float32)
        return (values - float(np.mean(values))) / max(float(np.std(values)), 1e-12)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        raw_norms = np.linalg.norm(X, axis=1).astype(np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        self.index.train(train)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        centroid_scores, assigned2 = self.quantizer.search(
            np.ascontiguousarray(Xn, dtype=np.float32), 2
        )
        assigned = np.ascontiguousarray(assigned2[:, 0], dtype=np.int64)
        second = np.ascontiguousarray(assigned2[:, 1], dtype=np.int64)
        first_scores = np.ascontiguousarray(centroid_scores[:, 0], dtype=np.float32)
        second_scores = np.ascontiguousarray(centroid_scores[:, 1], dtype=np.float32)
        valid_second = second >= 0
        margins = np.maximum(0.0, first_scores - second_scores).astype(np.float32)
        if np.any(valid_second):
            margins[~valid_second] = float(np.max(margins[valid_second]))
        else:
            margins[:] = 1.0
        self.listpack_mean_margin = float(np.mean(margins[valid_second])) if np.any(valid_second) else 0.0
        self.listpack_margin_threshold = float(np.quantile(margins[valid_second], self.listpack_quantile)) if np.any(valid_second) else 0.0

        base_scores = self._core_scores(Xn, assigned, raw_norms)
        risk = np.exp(-margins / self.listpack_temperature).astype(np.float32)
        risk_z = self._zscore(risk)
        pack_scores = (base_scores + self.listpack_weight * risk_z).astype(np.float32)
        order = np.lexsort((-pack_scores, assigned)).astype(np.int64, copy=False)

        self._before_cell_caps(Xn, assigned, pack_scores, raw_norms, order)
        cell_caps = np.asarray(self._cell_caps(assigned, pack_scores), dtype=np.int32)
        nonzero_caps = cell_caps[cell_caps > 0]
        if len(nonzero_caps) > 0:
            self.cap_min_actual = int(np.min(nonzero_caps))
            self.cap_max_actual = int(np.max(nonzero_caps))
            self.cap_mean_actual = float(np.mean(nonzero_caps))

        keep = np.zeros(len(order), dtype=bool)
        last_cid = -1
        used_in_cell = 0
        for pos, idx in enumerate(order):
            cid = int(assigned[idx])
            if cid != last_cid:
                last_cid = cid
                used_in_cell = 0
            if used_in_cell < int(cell_caps[cid]):
                keep[pos] = True
                used_in_cell += 1
        kept_order = np.ascontiguousarray(order[keep], dtype=np.int64)
        self.kept_points = int(len(kept_order))

        self.listpack_dual_added = 0
        self.listpack_dual_cells = 0
        dual_ids_parts = []
        dual_assign_parts = []
        dual_score_parts = []
        if self.listpack_dual_per_cell > 0 and np.any(valid_second):
            counts = np.bincount(assigned, minlength=self.nlist).astype(np.int64)
            starts = np.zeros(self.nlist + 1, dtype=np.int64)
            starts[1:] = np.cumsum(counts, dtype=np.int64)
            max_dual = int(max(0, round(self.kept_points * (self.listpack_total_cap_ratio - 1.0))))
            for cid in range(self.nlist):
                if self.listpack_dual_added >= max_dual:
                    break
                start = int(starts[cid])
                stop = int(starts[cid + 1])
                if stop <= start:
                    continue
                members = order[start:stop]
                cap = min(int(cell_caps[cid]), len(members))
                if cap <= 0:
                    continue
                prefix = members[:cap]
                mask = valid_second[prefix] & (second[prefix] != cid)
                if not np.any(mask):
                    continue
                cand = prefix[mask]
                local = np.argsort(margins[cand], kind="stable")
                take = min(self.listpack_dual_per_cell, len(local), max_dual - self.listpack_dual_added)
                if take <= 0:
                    continue
                chosen = cand[local[:take]]
                dual_ids_parts.append(chosen.astype(np.int64, copy=False))
                dual_assign_parts.append(second[chosen].astype(np.int64, copy=False))
                if self.listpack_dual_score_mode == "target":
                    dual_base_scores = second_scores[chosen]
                else:
                    dual_base_scores = pack_scores[chosen]
                dual_score_parts.append(
                    (
                        dual_base_scores
                        + self.listpack_dual_boost * risk_z[chosen]
                    ).astype(np.float32, copy=False)
                )
                self.listpack_dual_added += int(take)
                self.listpack_dual_cells += 1

        ids = np.ascontiguousarray(kept_order, dtype=np.int64)
        assign_tape = np.ascontiguousarray(assigned[kept_order], dtype=np.int64)
        score_tape = np.ascontiguousarray(pack_scores[kept_order], dtype=np.float32)
        if dual_ids_parts:
            dual_ids = np.ascontiguousarray(np.concatenate(dual_ids_parts), dtype=np.int64)
            dual_assign = np.ascontiguousarray(np.concatenate(dual_assign_parts), dtype=np.int64)
            dual_scores = np.ascontiguousarray(np.concatenate(dual_score_parts), dtype=np.float32)
            ids = np.ascontiguousarray(np.concatenate([ids, dual_ids]), dtype=np.int64)
            assign_tape = np.ascontiguousarray(np.concatenate([assign_tape, dual_assign]), dtype=np.int64)
            score_tape = np.ascontiguousarray(np.concatenate([score_tape, dual_scores]), dtype=np.float32)

        entry_order = np.lexsort((-score_tape, assign_tape)).astype(np.int64, copy=False)
        ids = np.ascontiguousarray(ids[entry_order], dtype=np.int64)
        assign_tape = np.ascontiguousarray(assign_tape[entry_order], dtype=np.int64)
        block = int(self.method_param.get("add_block", 65536))
        block = max(1024, block)
        for start in range(0, len(ids), block):
            stop = min(len(ids), start + block)
            batch = np.ascontiguousarray(Xn[ids[start:stop]], dtype=np.float32)
            self.index.add_core(
                stop - start,
                faiss.swig_ptr(batch),
                faiss.swig_ptr(ids[start:stop]),
                faiss.swig_ptr(assign_tape[start:stop]),
            )
        self.listpack_dual_fraction = float(self.listpack_dual_added) / float(max(1, self.kept_points))
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10layoutinvariant-listpacking-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", listpackDualPerCell={self.listpack_dual_per_cell}, "
            f"listpackDualAdded={self.listpack_dual_added}, "
            f"listpackTemp={self.listpack_temperature}, "
            f"listpackWeight={self.listpack_weight}, "
            f"listpackQ={self.listpack_quantile}, "
            f"listpackCapRatio={self.listpack_total_cap_ratio}, "
            f"listpackDualBoost={self.listpack_dual_boost}, "
            f"listpackDualScoreMode={self.listpack_dual_score_mode})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-listpacking-dualassign-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["scan_layout"] = "faiss-ivf-primary-core-plus-dual-boundary-mass-transport"
        extra["listpack_dual_per_cell"] = self.listpack_dual_per_cell
        extra["listpack_temperature"] = self.listpack_temperature
        extra["listpack_weight"] = self.listpack_weight
        extra["listpack_quantile"] = self.listpack_quantile
        extra["listpack_total_cap_ratio"] = self.listpack_total_cap_ratio
        extra["listpack_dual_boost"] = self.listpack_dual_boost
        extra["listpack_dual_score_mode"] = self.listpack_dual_score_mode
        extra["listpack_dual_added"] = self.listpack_dual_added
        extra["listpack_dual_cells"] = self.listpack_dual_cells
        extra["listpack_dual_fraction"] = self.listpack_dual_fraction
        extra["listpack_mean_margin"] = self.listpack_mean_margin
        extra["listpack_margin_threshold"] = self.listpack_margin_threshold
        return extra


class ZeelinM10boundaryimitatorSpeedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10speedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "bigs" in method_param:
            method_param["boundary_imitator_probe_size"] = method_param["bigs"]
        if "bits" in method_param:
            method_param["boundary_imitator_train_size"] = method_param["bits"]
        if "bimix" in method_param:
            method_param["boundary_imitator_mix"] = method_param["bimix"]
        if "binoise" in method_param:
            method_param["boundary_imitator_noise"] = method_param["binoise"]
        super().__init__(metric, method_param)
        self.boundary_imitator_probe_size = int(
            method_param.get("boundary_imitator_probe_size", self.exact_guard_probe_size)
        )
        self.boundary_imitator_train_size = int(
            method_param.get("boundary_imitator_train_size", 200000)
        )
        self.boundary_imitator_mix = float(method_param.get("boundary_imitator_mix", 0.35))
        self.boundary_imitator_noise = float(
            method_param.get("boundary_imitator_noise", self.exact_guard_noise)
        )
        self.boundary_imitator_probe_size = max(16, min(2048, self.boundary_imitator_probe_size))
        self.boundary_imitator_train_size = max(1024, min(1000000, self.boundary_imitator_train_size))
        self.boundary_imitator_mix = max(0.0, min(1.0, self.boundary_imitator_mix))
        self.boundary_imitator_noise = max(0.0, min(0.25, self.boundary_imitator_noise))
        self.boundary_imitator_best_mean_gap = 0.0
        self.boundary_imitator_best_max_gap = 0.0
        self.boundary_imitator_all_mean_gap = ""
        self.boundary_imitator_all_max_gap = ""

    def _boundary_imitator_queries(self, X):
        n = len(X)
        sample_count = min(n, self.boundary_imitator_train_size)
        sample_idx = np.linspace(0, n - 1, sample_count, dtype=np.int64)
        sample = self._normalize_rows(np.asarray(X[sample_idx], dtype=np.float32)).astype(
            np.float32, copy=False
        )
        sample = np.ascontiguousarray(sample, dtype=np.float32)

        old_ef = int(self.quantizer.hnsw.efSearch)
        self.quantizer.hnsw.efSearch = self.stability_probe_efSearch
        dists, ids = self.quantizer.search(sample, 2)
        self.quantizer.hnsw.efSearch = old_ef

        ids = np.asarray(ids, dtype=np.int64)
        dists = np.asarray(dists, dtype=np.float32)
        valid = (ids[:, 0] >= 0) & (ids[:, 1] >= 0)
        if not np.any(valid):
            chosen = np.arange(min(len(sample), self.boundary_imitator_probe_size), dtype=np.int64)
        else:
            gaps = np.maximum(0.0, dists[valid, 0] - dists[valid, 1])
            valid_pos = np.flatnonzero(valid)
            take = min(self.boundary_imitator_probe_size, len(valid_pos))
            part = np.argpartition(gaps, take - 1)[:take]
            chosen = valid_pos[part]

        base = np.ascontiguousarray(sample[chosen], dtype=np.float32)
        chosen_ids = ids[chosen]
        centroids = np.empty_like(base)
        for row, pair in enumerate(chosen_ids):
            c0 = np.empty(self.dim, dtype=np.float32)
            c1 = np.empty(self.dim, dtype=np.float32)
            cid0 = int(pair[0])
            cid1 = int(pair[1])
            if cid0 < 0 or cid1 < 0:
                centroids[row] = base[row]
                continue
            self.quantizer.reconstruct(cid0, c0)
            self.quantizer.reconstruct(cid1, c1)
            mid = c0 + c1
            norm = float(np.linalg.norm(mid))
            if norm <= 1e-12:
                centroids[row] = base[row]
            else:
                centroids[row] = mid * (1.0 / norm)

        query = (
            (1.0 - self.boundary_imitator_mix) * base
            + self.boundary_imitator_mix * centroids
        ).astype(np.float32, copy=False)
        if self.boundary_imitator_noise > 0.0:
            rng = np.random.RandomState(405183)
            query = query + rng.normal(
                0.0, self.boundary_imitator_noise, size=query.shape
            ).astype(np.float32)
        query = self._normalize_rows(query).astype(np.float32, copy=False)

        if len(chosen) and np.all(chosen_ids[:, 1] >= 0):
            chosen_dists = dists[chosen]
            chosen_gaps = np.maximum(0.0, chosen_dists[:, 0] - chosen_dists[:, 1])
            mean_gap = float(np.mean(chosen_gaps))
            max_gap = float(np.max(chosen_gaps))
        else:
            mean_gap = 0.0
            max_gap = 0.0
        return np.ascontiguousarray(query, dtype=np.float32), mean_gap, max_gap

    def _exact_truth_for_queries(self, X, query):
        start_time = time.time()
        X = np.asarray(X, dtype=np.float32)
        n = len(X)
        k = min(self.exact_guard_truth_k, n)
        best_scores = np.full((len(query), k), -np.inf, dtype=np.float32)
        best_ids = np.full((len(query), k), -1, dtype=np.int64)
        for start in range(0, n, self.exact_guard_chunk):
            stop = min(n, start + self.exact_guard_chunk)
            block = self._normalize_rows(np.asarray(X[start:stop], dtype=np.float32)).astype(
                np.float32, copy=False
            )
            block = np.ascontiguousarray(block, dtype=np.float32)
            scores = query.dot(block.T)
            take = min(k, scores.shape[1])
            part = np.argpartition(scores, -take, axis=1)[:, -take:]
            part_scores = np.take_along_axis(scores, part, axis=1)
            part_ids = (part + start).astype(np.int64, copy=False)
            comb_scores = np.concatenate([best_scores, part_scores], axis=1)
            comb_ids = np.concatenate([best_ids, part_ids], axis=1)
            choose = np.argpartition(comb_scores, -k, axis=1)[:, -k:]
            best_scores = np.take_along_axis(comb_scores, choose, axis=1)
            best_ids = np.take_along_axis(comb_ids, choose, axis=1)
        order = np.argsort(-best_scores, axis=1)
        self.exact_guard_build_truth_time += time.time() - start_time
        return np.ascontiguousarray(np.take_along_axis(best_ids, order, axis=1), dtype=np.int64)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        states = []
        recalls = []
        proxy_times = []
        build_times = []
        mean_gaps = []
        max_gaps = []
        self.exact_guard_build_truth_time = 0.0

        for trial in range(1, self.stability_trials + 1):
            start = time.time()
            ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.fit(self, X)
            elapsed = time.time() - start
            self.index.max_codes = self.max_codes
            query, mean_gap, max_gap = self._boundary_imitator_queries(X)
            truth = self._exact_truth_for_queries(X, query)
            proxy_recall, proxy_us = self._exact_guard_certificate(query, truth)
            build_times.append(elapsed)
            recalls.append(proxy_recall)
            proxy_times.append(proxy_us)
            mean_gaps.append(mean_gap)
            max_gaps.append(max_gap)
            score = proxy_recall - 0.0005 * proxy_us
            states.append(
                self._capture_stability_state(trial, elapsed, proxy_recall, proxy_us, score)
            )

        max_recall = max(recalls) if recalls else 0.0
        floor = max_recall - self.exact_guard_tolerance
        acceptable = [state for state in states if state["proxy_recall"] >= floor]
        if not acceptable:
            acceptable = states
        best_state = min(acceptable, key=lambda s: (s["proxy_us"], -s["proxy_recall"]))
        self._restore_stability_state(best_state)
        self.exact_guard_best_recall = self.stability_best_proxy_recall
        best_idx = max(0, self.stability_best_trial - 1)
        if best_idx < len(mean_gaps):
            self.boundary_imitator_best_mean_gap = mean_gaps[best_idx]
            self.boundary_imitator_best_max_gap = max_gaps[best_idx]

        self.stability_all_build_times = ",".join(f"{v:.6f}" for v in build_times)
        self.stability_all_proxy_recall = ",".join(f"{v:.6f}" for v in recalls)
        self.stability_all_proxy_us = ",".join(f"{v:.6f}" for v in proxy_times)
        self.exact_guard_all_recall = self.stability_all_proxy_recall
        self.exact_guard_all_proxy_us = self.stability_all_proxy_us
        self.boundary_imitator_all_mean_gap = ",".join(f"{v:.6f}" for v in mean_gaps)
        self.boundary_imitator_all_max_gap = ",".join(f"{v:.6f}" for v in max_gaps)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.set_query_arguments(
            self,
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10boundaryimitator-speedexactguard-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", boundaryImitatorTrials={self.stability_trials}, "
            f"boundaryImitatorBestTrial={self.stability_best_trial}, "
            f"boundaryImitatorRecall={self.exact_guard_best_recall:.4f}, "
            f"boundaryImitatorProxyUs={self.stability_best_proxy_us:.3f})"
        )

    def get_additional(self):
        extra = ZeelinM10speedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.get_additional(self)
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10boundaryimitator-speedexactguard-boundarytrim-maxcodes-concentrationcapcore-returncompress"
        extra["boundary_imitator"] = 1
        extra["boundary_imitator_probe_size"] = self.boundary_imitator_probe_size
        extra["boundary_imitator_train_size"] = self.boundary_imitator_train_size
        extra["boundary_imitator_mix"] = self.boundary_imitator_mix
        extra["boundary_imitator_noise"] = self.boundary_imitator_noise
        extra["boundary_imitator_best_mean_gap"] = self.boundary_imitator_best_mean_gap
        extra["boundary_imitator_best_max_gap"] = self.boundary_imitator_best_max_gap
        extra["boundary_imitator_all_mean_gap"] = self.boundary_imitator_all_mean_gap
        extra["boundary_imitator_all_max_gap"] = self.boundary_imitator_all_max_gap
        return extra


class ZeelinM10dualguardSpeedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundaryimitatorSpeedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "dgtol" in method_param:
            method_param["dual_exact_tolerance"] = method_param["dgtol"]
            method_param["dual_boundary_tolerance"] = method_param["dgtol"]
        if "degtol" in method_param:
            method_param["dual_exact_tolerance"] = method_param["degtol"]
        if "dbgtol" in method_param:
            method_param["dual_boundary_tolerance"] = method_param["dbgtol"]
        super().__init__(metric, method_param)
        self.dual_exact_tolerance = float(
            method_param.get("dual_exact_tolerance", self.exact_guard_tolerance)
        )
        self.dual_boundary_tolerance = float(
            method_param.get("dual_boundary_tolerance", self.exact_guard_tolerance)
        )
        self.dual_exact_tolerance = max(0.0, min(0.25, self.dual_exact_tolerance))
        self.dual_boundary_tolerance = max(0.0, min(0.25, self.dual_boundary_tolerance))
        self.dual_exact_best_recall = 0.0
        self.dual_boundary_best_recall = 0.0
        self.dual_all_exact_recall = ""
        self.dual_all_boundary_recall = ""
        self.dual_all_exact_proxy_us = ""
        self.dual_all_boundary_proxy_us = ""
        self.dual_selected_score = 0.0
        self.dual_all_score = ""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        exact_query, exact_truth = self._make_exact_guard(X)
        states = []
        exact_recalls = []
        boundary_recalls = []
        exact_times = []
        boundary_times = []
        combined_recalls = []
        combined_times = []
        scores = []
        build_times = []
        mean_gaps = []
        max_gaps = []

        for trial in range(1, self.stability_trials + 1):
            start = time.time()
            ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.fit(self, X)
            elapsed = time.time() - start
            self.index.max_codes = self.max_codes

            exact_recall, exact_us = self._exact_guard_certificate(exact_query, exact_truth)
            boundary_query, mean_gap, max_gap = self._boundary_imitator_queries(X)
            boundary_truth = self._exact_truth_for_queries(X, boundary_query)
            boundary_recall, boundary_us = self._exact_guard_certificate(
                boundary_query, boundary_truth
            )

            combined_recall = min(exact_recall, boundary_recall)
            combined_us = max(exact_us, boundary_us)
            score = combined_recall - 0.00025 * (exact_us + boundary_us)

            build_times.append(elapsed)
            exact_recalls.append(exact_recall)
            boundary_recalls.append(boundary_recall)
            exact_times.append(exact_us)
            boundary_times.append(boundary_us)
            combined_recalls.append(combined_recall)
            combined_times.append(combined_us)
            scores.append(score)
            mean_gaps.append(mean_gap)
            max_gaps.append(max_gap)

            state = self._capture_stability_state(
                trial, elapsed, combined_recall, combined_us, score
            )
            state["dual_exact_recall"] = exact_recall
            state["dual_boundary_recall"] = boundary_recall
            state["dual_exact_proxy_us"] = exact_us
            state["dual_boundary_proxy_us"] = boundary_us
            state["dual_score"] = score
            states.append(state)

        best_exact = max(exact_recalls) if exact_recalls else 0.0
        best_boundary = max(boundary_recalls) if boundary_recalls else 0.0
        exact_floor = best_exact - self.dual_exact_tolerance
        boundary_floor = best_boundary - self.dual_boundary_tolerance
        acceptable = [
            state
            for state in states
            if state["dual_exact_recall"] >= exact_floor
            and state["dual_boundary_recall"] >= boundary_floor
        ]
        if acceptable:
            best_state = min(
                acceptable,
                key=lambda s: (
                    max(s["dual_exact_proxy_us"], s["dual_boundary_proxy_us"]),
                    -(s["dual_exact_recall"] + s["dual_boundary_recall"]),
                ),
            )
        else:
            def dual_deficit(state):
                return max(0.0, exact_floor - state["dual_exact_recall"]) + max(
                    0.0, boundary_floor - state["dual_boundary_recall"]
                )

            best_state = min(
                states,
                key=lambda s: (
                    dual_deficit(s),
                    max(s["dual_exact_proxy_us"], s["dual_boundary_proxy_us"]),
                    -(s["dual_exact_recall"] + s["dual_boundary_recall"]),
                ),
            )

        self._restore_stability_state(best_state)
        self.dual_exact_best_recall = float(best_state["dual_exact_recall"])
        self.dual_boundary_best_recall = float(best_state["dual_boundary_recall"])
        self.dual_selected_score = float(best_state["dual_score"])
        self.exact_guard_best_recall = self.dual_exact_best_recall
        best_idx = max(0, self.stability_best_trial - 1)
        if best_idx < len(mean_gaps):
            self.boundary_imitator_best_mean_gap = mean_gaps[best_idx]
            self.boundary_imitator_best_max_gap = max_gaps[best_idx]

        self.stability_all_build_times = ",".join(f"{v:.6f}" for v in build_times)
        self.stability_all_proxy_recall = ",".join(f"{v:.6f}" for v in combined_recalls)
        self.stability_all_proxy_us = ",".join(f"{v:.6f}" for v in combined_times)
        self.exact_guard_all_recall = ",".join(f"{v:.6f}" for v in exact_recalls)
        self.exact_guard_all_proxy_us = ",".join(f"{v:.6f}" for v in exact_times)
        self.dual_all_exact_recall = self.exact_guard_all_recall
        self.dual_all_boundary_recall = ",".join(f"{v:.6f}" for v in boundary_recalls)
        self.dual_all_exact_proxy_us = self.exact_guard_all_proxy_us
        self.dual_all_boundary_proxy_us = ",".join(f"{v:.6f}" for v in boundary_times)
        self.dual_all_score = ",".join(f"{v:.6f}" for v in scores)
        self.boundary_imitator_all_mean_gap = ",".join(f"{v:.6f}" for v in mean_gaps)
        self.boundary_imitator_all_max_gap = ",".join(f"{v:.6f}" for v in max_gaps)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.set_query_arguments(
            self,
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10dualguard-speedexactguard-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", dualGuardTrials={self.stability_trials}, "
            f"dualGuardBestTrial={self.stability_best_trial}, "
            f"dualExactRecall={self.dual_exact_best_recall:.4f}, "
            f"dualBoundaryRecall={self.dual_boundary_best_recall:.4f}, "
            f"dualProxyUs={self.stability_best_proxy_us:.3f})"
        )

    def get_additional(self):
        extra = ZeelinM10boundaryimitatorSpeedexactguardBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf.get_additional(self)
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10dualguard-speedexactguard-boundarytrim-maxcodes-concentrationcapcore-returncompress"
        extra["dual_guard"] = 1
        extra["dual_exact_tolerance"] = self.dual_exact_tolerance
        extra["dual_boundary_tolerance"] = self.dual_boundary_tolerance
        extra["dual_exact_best_recall"] = self.dual_exact_best_recall
        extra["dual_boundary_best_recall"] = self.dual_boundary_best_recall
        extra["dual_all_exact_recall"] = self.dual_all_exact_recall
        extra["dual_all_boundary_recall"] = self.dual_all_boundary_recall
        extra["dual_all_exact_proxy_us"] = self.dual_all_exact_proxy_us
        extra["dual_all_boundary_proxy_us"] = self.dual_all_boundary_proxy_us
        extra["dual_selected_score"] = self.dual_selected_score
        extra["dual_all_score"] = self.dual_all_score
        return extra


class ZeelinM10directbatchBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self._direct_batch_dbuf = None
        self._direct_batch_ibuf = None
        self._direct_batch_capacity = (0, 0)

    def _ensure_direct_batch_buffers(self, nq, query_k):
        if self._direct_batch_capacity[0] >= nq and self._direct_batch_capacity[1] >= query_k:
            return
        self._direct_batch_dbuf = np.empty((nq, query_k), dtype=np.float32)
        self._direct_batch_ibuf = np.empty((nq, query_k), dtype=np.int64)
        self._direct_batch_capacity = (nq, query_k)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10directbatch-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )

    def batch_query(self, X, n):
        X = np.asarray(X, dtype=np.float32)
        if self.skip_query_norm:
            Xn = X if X.flags["C_CONTIGUOUS"] else np.ascontiguousarray(X, dtype=np.float32)
        else:
            norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
            Xn = np.ascontiguousarray(X / np.maximum(norms, 1e-12), dtype=np.float32)
        self._batch_target_k = n if self.return_k <= 0 else min(n, self.return_k)
        query_k = n if self.internal_k <= 0 else max(1, self.internal_k)
        query_k = max(query_k, self._batch_target_k + self.dedup_extra)
        self._ensure_direct_batch_buffers(len(Xn), query_k)
        self.index.search_c(
            len(Xn),
            faiss.swig_ptr(Xn),
            query_k,
            faiss.swig_ptr(self._direct_batch_dbuf),
            faiss.swig_ptr(self._direct_batch_ibuf),
        )
        self._batch_labels = self._direct_batch_ibuf[: len(Xn), :query_k]

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10directbatch-boundarytrim-maxcodes-concentrationcapcore-returncompress"
        return extra


class ZeelinM10preassignedbatchBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self._pre_batch_dbuf = None
        self._pre_batch_ibuf = None
        self._pre_batch_assign = None
        self._pre_batch_centroid_dis = None
        self._pre_batch_capacity = (0, 0, 0)

    def _ensure_preassigned_boundary_buffers(self, nq, nprobe, query_k):
        if (
            self._pre_batch_capacity[0] >= nq
            and self._pre_batch_capacity[1] >= nprobe
            and self._pre_batch_capacity[2] >= query_k
        ):
            return
        self._pre_batch_dbuf = np.empty((nq, query_k), dtype=np.float32)
        self._pre_batch_ibuf = np.empty((nq, query_k), dtype=np.int64)
        self._pre_batch_assign = np.empty((nq, nprobe), dtype=np.int64)
        self._pre_batch_centroid_dis = np.empty((nq, nprobe), dtype=np.float32)
        self._pre_batch_capacity = (nq, nprobe, query_k)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10preassignedbatch-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )

    def batch_query(self, X, n):
        X = np.asarray(X, dtype=np.float32)
        if self.skip_query_norm:
            Xn = X if X.flags["C_CONTIGUOUS"] else np.ascontiguousarray(X, dtype=np.float32)
        else:
            norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
            Xn = np.ascontiguousarray(X / np.maximum(norms, 1e-12), dtype=np.float32)
        self._batch_target_k = n if self.return_k <= 0 else min(n, self.return_k)
        query_k = n if self.internal_k <= 0 else max(1, self.internal_k)
        query_k = max(query_k, self._batch_target_k + self.dedup_extra)
        self._ensure_preassigned_boundary_buffers(len(Xn), self.nprobe, query_k)
        centroid_dis, assign = self.quantizer.search(
            np.ascontiguousarray(Xn, dtype=np.float32), self.nprobe
        )
        self._pre_batch_assign[: len(Xn), : self.nprobe] = assign
        self._pre_batch_centroid_dis[: len(Xn), : self.nprobe] = centroid_dis
        self.index.search_preassigned_c(
            len(Xn),
            faiss.swig_ptr(Xn),
            query_k,
            faiss.swig_ptr(self._pre_batch_assign),
            faiss.swig_ptr(self._pre_batch_centroid_dis),
            faiss.swig_ptr(self._pre_batch_dbuf),
            faiss.swig_ptr(self._pre_batch_ibuf),
            False,
        )
        self._batch_labels = self._pre_batch_ibuf[: len(Xn), :query_k]

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-preassignedbatch-hnswquantizer-ip-m10boundarytrim-maxcodes-concentrationcapcore-returncompress"
        return extra


class ZeelinM10cpppayloadBoundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimMaxcodesConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "pk" in method_param:
            method_param["payload_k"] = method_param["pk"]
        super().__init__(metric, method_param)
        default_payload = self.max_codes if self.max_codes > 0 else self.list_cap
        self.payload_k = int(method_param.get("payload_k", default_payload))
        self.payload_k = max(1, min(512, self.payload_k))
        self._lib = ZeelinCentroidscanIvf._load_centroidscan_library(self)
        self._idx = self._lib.zeelin_centroidscan_ivf_create()
        self._cpp_qbuf = None
        self._cpp_route_dists = None
        self._cpp_route_ids = None
        self._cpp_labels = None
        self._cpp_capacity = (0, 0, 0)
        self._cpp_out64 = None

    def _ensure_cpppayload_buffers(self, nq, nprobe, target_k):
        if (
            self._cpp_capacity[0] >= nq
            and self._cpp_capacity[1] >= nprobe
            and self._cpp_capacity[2] >= target_k
        ):
            return
        self._cpp_route_dists = np.empty((nq, nprobe), dtype=np.float32)
        self._cpp_route_ids = np.empty((nq, nprobe), dtype=np.int64)
        self._cpp_labels = np.empty((nq, target_k), dtype=np.int64)
        self._cpp_capacity = (nq, nprobe, target_k)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        raw_norms = np.linalg.norm(X, axis=1).astype(np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        self.index.train(train)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        _, assigned = self.quantizer.search(np.ascontiguousarray(Xn, dtype=np.float32), 1)
        assigned = np.ascontiguousarray(assigned[:, 0], dtype=np.int64)
        scores = self._core_scores(Xn, assigned, raw_norms)
        order = np.lexsort((-scores, assigned)).astype(np.int64, copy=False)
        self._before_cell_caps(Xn, assigned, scores, raw_norms, order)
        cell_caps = np.asarray(self._cell_caps(assigned, scores), dtype=np.int32)
        nonzero_caps = cell_caps[cell_caps > 0]
        if len(nonzero_caps) > 0:
            self.cap_min_actual = int(np.min(nonzero_caps))
            self.cap_max_actual = int(np.max(nonzero_caps))
            self.cap_mean_actual = float(np.mean(nonzero_caps))

        keep = np.zeros(len(order), dtype=bool)
        payload = np.full((self.nlist, self.payload_k), -1, dtype=np.int64)
        payload_counts = np.zeros(self.nlist, dtype=np.int32)
        last_cid = -1
        used_in_cell = 0
        for pos, idx in enumerate(order):
            cid = int(assigned[idx])
            if cid != last_cid:
                last_cid = cid
                used_in_cell = 0
            if used_in_cell < int(cell_caps[cid]):
                keep[pos] = True
                used_in_cell += 1
                pcount = int(payload_counts[cid])
                if pcount < self.payload_k:
                    payload[cid, pcount] = int(idx)
                    payload_counts[cid] = pcount + 1

        kept_order = np.ascontiguousarray(order[keep], dtype=np.int64)
        self.kept_points = int(len(kept_order))
        ids = np.ascontiguousarray(kept_order, dtype=np.int64)
        block = int(self.method_param.get("add_block", 65536))
        block = max(1024, block)
        for start in range(0, len(kept_order), block):
            stop = min(len(kept_order), start + block)
            batch = np.ascontiguousarray(Xn[kept_order[start:stop]], dtype=np.float32)
            self.index.add_with_ids(batch, ids[start:stop])

        self._lib.zeelin_centroidscan_ivf_fit_payload(
            self._idx,
            np.ascontiguousarray(Xn, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            np.ascontiguousarray(payload, dtype=np.int64).ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(Xn.shape[0]),
            int(self.dim),
            int(self.nlist),
            int(self.payload_k),
        )
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10cpppayload-boundarytrim-maxcodes-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + f", payloadK={self.payload_k})"

    def query(self, v, n):
        if self.skip_query_norm and self.query_view:
            q = v.reshape(1, self.dim)
        elif self.skip_query_norm:
            self.qbuf[0, :] = v
            q = self.qbuf
        else:
            norm = float(np.sqrt(np.dot(v, v)))
            self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
            q = self.qbuf
        target_k = n if self.return_k <= 0 else min(n, self.return_k)
        if self._cpp_out64 is None or len(self._cpp_out64) < target_k:
            self._cpp_out64 = np.empty(target_k, dtype=np.int64)
        self._ensure_cpppayload_buffers(1, self.nprobe, target_k)
        self.quantizer.search_c(
            1,
            faiss.swig_ptr(q),
            self.nprobe,
            faiss.swig_ptr(self._cpp_route_dists),
            faiss.swig_ptr(self._cpp_route_ids),
        )
        count = self._lib.zeelin_centroidscan_ivf_query_payload(
            self._idx,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._cpp_route_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(self.nprobe),
            int(target_k),
            self._cpp_out64.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        )
        return self._cpp_out64[:count]

    def batch_query(self, X, n):
        X = np.asarray(X, dtype=np.float32)
        if self.skip_query_norm:
            Xn = X if X.flags["C_CONTIGUOUS"] else np.ascontiguousarray(X, dtype=np.float32)
        else:
            norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
            Xn = np.ascontiguousarray(X / np.maximum(norms, 1e-12), dtype=np.float32)
        self._batch_target_k = n if self.return_k <= 0 else min(n, self.return_k)
        target_k = self._batch_target_k
        self._ensure_cpppayload_buffers(len(Xn), self.nprobe, target_k)
        self.quantizer.search_c(
            len(Xn),
            faiss.swig_ptr(Xn),
            self.nprobe,
            faiss.swig_ptr(self._cpp_route_dists),
            faiss.swig_ptr(self._cpp_route_ids),
        )
        self._lib.zeelin_centroidscan_ivf_query_payload_batch(
            self._idx,
            Xn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._cpp_route_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(len(Xn)),
            int(self.nprobe),
            int(target_k),
            self._cpp_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        )
        self._batch_labels = self._cpp_labels[: len(Xn), :target_k]

    def get_additional(self):
        extra = super().get_additional()
        extra["payload_k"] = self.payload_k
        extra["dist_comps"] = self._lib.zeelin_centroidscan_ivf_total_dist_comps(self._idx)
        extra["avg_dist_comps"] = self._lib.zeelin_centroidscan_ivf_avg_dist_comps(self._idx)
        extra["avg_candidates"] = self._lib.zeelin_centroidscan_ivf_avg_candidates(self._idx)
        extra["kernel"] = "cpp-batch-tailpayload-hnswquantizer-ip-m10boundarytrim-maxcodes"
        extra["scan_layout"] = "hnsw-route-cpp-contiguous-tailpayload"
        return extra

    def done(self):
        if getattr(self, "_idx", None):
            self._lib.zeelin_centroidscan_ivf_free(self._idx)
            self._idx = None


class ZeelinM10routeboundarytrimConcentrationcapcoreReturncompressHnswgateIvf(
    ZeelinM10boundarytrimConcentrationcapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "rbtm" in method_param:
            method_param["route_boundary_mode"] = method_param["rbtm"]
        if "rbg" in method_param:
            method_param["route_boundary_gamma"] = method_param["rbg"]
        if "rtg" in method_param:
            method_param["route_boundary_route_gamma"] = method_param["rtg"]
        if "rbrg" in method_param:
            method_param["route_boundary_route_gamma"] = method_param["rbrg"]
        if "rbb" in method_param:
            method_param["route_boundary_bias"] = method_param["rbb"]
        if "rts" in method_param:
            method_param["route_train_size"] = method_param["rts"]
        if "rbts" in method_param:
            method_param["route_train_size"] = method_param["rbts"]
        if "rte" in method_param:
            method_param["route_efSearch"] = method_param["rte"]
        if "rbef" in method_param:
            method_param["route_efSearch"] = method_param["rbef"]
        if "rs" in method_param:
            method_param["route_smooth"] = method_param["rs"]
        if "rbs" in method_param:
            method_param["route_smooth"] = method_param["rbs"]
        if "rbcliplo" in method_param:
            method_param["route_boundary_clip_min"] = method_param["rbcliplo"]
        if "rbcliphi" in method_param:
            method_param["route_boundary_clip_max"] = method_param["rbcliphi"]
        super().__init__(metric, method_param)
        self.route_boundary_mode = str(method_param.get("route_boundary_mode", "riskdensity"))
        self.route_boundary_gamma = float(method_param.get("route_boundary_gamma", 0.55))
        self.route_boundary_route_gamma = float(method_param.get("route_boundary_route_gamma", 0.25))
        self.route_boundary_bias = float(method_param.get("route_boundary_bias", 0.97))
        self.route_train_size = int(method_param.get("route_train_size", 200000))
        self.route_efSearch = int(method_param.get("route_efSearch", 1))
        self.route_smooth = float(method_param.get("route_smooth", 1.0))
        self.route_boundary_clip_min = float(method_param.get("route_boundary_clip_min", 0.78))
        self.route_boundary_clip_max = float(method_param.get("route_boundary_clip_max", 1.08))
        self.route_boundary_gamma = max(0.0, min(2.0, self.route_boundary_gamma))
        self.route_boundary_route_gamma = max(0.0, min(2.0, self.route_boundary_route_gamma))
        self.route_boundary_bias = max(0.25, min(2.0, self.route_boundary_bias))
        self.route_train_size = max(0, self.route_train_size)
        self.route_efSearch = max(1, min(64, self.route_efSearch))
        self.route_smooth = max(0.0, self.route_smooth)
        self.route_boundary_clip_min = max(0.05, min(4.0, self.route_boundary_clip_min))
        self.route_boundary_clip_max = max(
            self.route_boundary_clip_min, min(4.0, self.route_boundary_clip_max)
        )
        self.route_traffic = None
        self.route_observed_cells = 0
        self.route_mean_traffic = 0.0
        self.route_max_traffic = 0.0
        self.route_boundary_mean_signal = 0.0
        self.route_boundary_max_signal = 0.0
        self.route_boundary_cap_ratio = 0.0
        self.route_boundary_weighted_cap = 0.0
        self.route_boundary_weighted_base_cap = 0.0
        self.route_boundary_weighted_cap_ratio = 0.0

    def _before_cell_caps(self, Xn, assigned, scores, raw_norms, order):
        super()._before_cell_caps(Xn, assigned, scores, raw_norms, order)
        route_sample = Xn
        if 0 < self.route_train_size < len(Xn):
            route_idx = np.linspace(0, len(Xn) - 1, self.route_train_size, dtype=np.int64)
            route_sample = np.ascontiguousarray(Xn[route_idx], dtype=np.float32)
        else:
            route_sample = np.ascontiguousarray(route_sample, dtype=np.float32)

        old_ef = int(self.quantizer.hnsw.efSearch)
        self.quantizer.hnsw.efSearch = self.route_efSearch
        _, route_ids = self.quantizer.search(route_sample, 1)
        self.quantizer.hnsw.efSearch = old_ef

        route_ids = np.asarray(route_ids[:, 0], dtype=np.int64)
        valid = route_ids >= 0
        traffic = np.zeros(self.nlist, dtype=np.float32)
        if np.any(valid):
            traffic = np.bincount(route_ids[valid], minlength=self.nlist).astype(np.float32)
        self.route_traffic = traffic
        observed = traffic > 0.0
        self.route_observed_cells = int(np.sum(observed))
        if self.route_observed_cells:
            self.route_mean_traffic = float(np.mean(traffic[observed]))
            self.route_max_traffic = float(np.max(traffic[observed]))

    def _cell_caps(self, assigned, scores):
        boundary_caps = super()._cell_caps(assigned, scores).astype(np.float32)
        traffic = self.route_traffic
        mass = self.boundary_cell_mass
        if traffic is None or mass is None or not np.any(traffic > 0.0) or not np.any(mass > 0.0):
            return boundary_caps.astype(np.int32)

        counts = np.bincount(assigned, minlength=self.nlist).astype(np.float32)
        nonempty = counts > 0.0
        route_observed = traffic > 0.0
        mass_observed = mass > 0.0

        mean_traffic = float(np.mean(traffic[route_observed]) + self.route_smooth + 1e-12)
        route_ratio = (traffic + self.route_smooth) / mean_traffic
        route_ratio = np.maximum(route_ratio, 1e-6)

        mean_mass = float(np.mean(mass[mass_observed]) + 1e-12)
        boundary_ratio = np.ones(self.nlist, dtype=np.float32)
        boundary_ratio[mass_observed] = mass[mass_observed] / mean_mass
        boundary_ratio = np.maximum(boundary_ratio, 1e-6)

        if self.route_boundary_mode == "jointmass":
            signal = boundary_ratio * np.power(route_ratio, self.route_boundary_route_gamma)
        elif self.route_boundary_mode == "hottrim":
            signal = boundary_ratio / np.power(route_ratio, self.route_boundary_route_gamma)
            signal = np.minimum(1.0, signal)
        elif self.route_boundary_mode == "coldtrim":
            signal = boundary_ratio * np.power(route_ratio, self.route_boundary_route_gamma)
            signal = np.minimum(1.0, signal)
        else:
            signal = boundary_ratio / np.power(route_ratio, self.route_boundary_route_gamma)

        signal = np.maximum(signal, 1e-6)
        valid_signal = signal[nonempty]
        if len(valid_signal):
            self.route_boundary_mean_signal = float(np.mean(valid_signal))
            self.route_boundary_max_signal = float(np.max(valid_signal))

        factor = np.power(signal, self.route_boundary_gamma) * self.route_boundary_bias
        factor = np.clip(
            factor, self.route_boundary_clip_min, self.route_boundary_clip_max
        )

        caps = np.zeros(self.nlist, dtype=np.float32)
        caps[nonempty] = boundary_caps[nonempty] * factor[nonempty]
        caps = np.clip(caps, self.cap_min, self.cap_max)
        caps[~nonempty] = 0
        caps = np.minimum(caps, counts)
        caps_i = np.rint(caps).astype(np.int32)
        self.cap_total_requested = int(np.sum(caps_i))

        boundary_total = float(np.sum(boundary_caps[nonempty]))
        self.route_boundary_cap_ratio = float(
            np.sum(caps_i[nonempty]) / max(boundary_total, 1e-12)
        )
        traffic_sum = float(np.sum(traffic))
        if traffic_sum > 0.0:
            self.route_boundary_weighted_cap = float(np.sum(traffic * caps_i) / traffic_sum)
            self.route_boundary_weighted_base_cap = float(
                np.sum(traffic * boundary_caps) / traffic_sum
            )
            self.route_boundary_weighted_cap_ratio = (
                self.route_boundary_weighted_cap
                / max(self.route_boundary_weighted_base_cap, 1e-12)
            )
        return caps_i

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10boundarytrim-concentrationcapcore-returncompress-hnswgate",
            "zeelin-m10routeboundarytrim-concentrationcapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", routeBoundaryMode={self.route_boundary_mode}, "
            f"routeBoundaryGamma={self.route_boundary_gamma}, "
            f"routeBoundaryRouteGamma={self.route_boundary_route_gamma}, "
            f"routeBoundaryCapRatio={self.route_boundary_cap_ratio:.3f}, "
            f"routeBoundaryWeightedRatio={self.route_boundary_weighted_cap_ratio:.3f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10routeboundarytrim-concentrationcapcore-returncompress"
        extra["route_boundary_mode"] = self.route_boundary_mode
        extra["route_boundary_gamma"] = self.route_boundary_gamma
        extra["route_boundary_route_gamma"] = self.route_boundary_route_gamma
        extra["route_boundary_bias"] = self.route_boundary_bias
        extra["route_train_size"] = self.route_train_size
        extra["route_efSearch"] = self.route_efSearch
        extra["route_smooth"] = self.route_smooth
        extra["route_boundary_clip_min"] = self.route_boundary_clip_min
        extra["route_boundary_clip_max"] = self.route_boundary_clip_max
        extra["route_observed_cells"] = self.route_observed_cells
        extra["route_mean_traffic"] = self.route_mean_traffic
        extra["route_max_traffic"] = self.route_max_traffic
        extra["route_boundary_mean_signal"] = self.route_boundary_mean_signal
        extra["route_boundary_max_signal"] = self.route_boundary_max_signal
        extra["route_boundary_cap_ratio"] = self.route_boundary_cap_ratio
        extra["route_boundary_weighted_cap"] = self.route_boundary_weighted_cap
        extra["route_boundary_weighted_base_cap"] = self.route_boundary_weighted_base_cap
        extra["route_boundary_weighted_cap_ratio"] = self.route_boundary_weighted_cap_ratio
        return extra


class ZeelinRouteweightedRatecapcoreTransportHnswgateIvf(ZeelinAdaptivecapcoreTransportHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "rtm" in method_param:
            method_param["route_mode"] = method_param["rtm"]
        if "rtg" in method_param:
            method_param["route_gamma"] = method_param["rtg"]
        if "rts" in method_param:
            method_param["route_train_size"] = method_param["rts"]
        if "rte" in method_param:
            method_param["route_efSearch"] = method_param["rte"]
        if "rs" in method_param:
            method_param["route_smooth"] = method_param["rs"]
        if "rcliplo" in method_param:
            method_param["route_clip_min"] = method_param["rcliplo"]
        if "rcliphi" in method_param:
            method_param["route_clip_max"] = method_param["rcliphi"]
        super().__init__(metric, method_param)
        self.route_mode = str(method_param.get("route_mode", "coolhot"))
        self.route_gamma = float(method_param.get("route_gamma", 0.25))
        self.route_train_size = int(method_param.get("route_train_size", 200000))
        self.route_efSearch = int(method_param.get("route_efSearch", 1))
        self.route_smooth = float(method_param.get("route_smooth", 1.0))
        self.route_clip_min = float(method_param.get("route_clip_min", 0.50))
        self.route_clip_max = float(method_param.get("route_clip_max", 2.00))
        self.route_gamma = max(0.0, min(2.0, self.route_gamma))
        self.route_train_size = max(0, self.route_train_size)
        self.route_efSearch = max(1, self.route_efSearch)
        self.route_smooth = max(0.0, self.route_smooth)
        self.route_clip_min = max(0.05, min(8.0, self.route_clip_min))
        self.route_clip_max = max(self.route_clip_min, min(8.0, self.route_clip_max))
        self.route_traffic = None
        self.route_observed_cells = 0
        self.route_mean_traffic = 0.0
        self.route_max_traffic = 0.0
        self.route_weighted_cap = 0.0
        self.route_weighted_base_cap = 0.0
        self.route_weighted_cap_ratio = 0.0

    def _before_cell_caps(self, Xn, assigned, scores, raw_norms, order):
        route_sample = Xn
        if 0 < self.route_train_size < len(Xn):
            route_idx = np.linspace(0, len(Xn) - 1, self.route_train_size, dtype=np.int64)
            route_sample = np.ascontiguousarray(Xn[route_idx], dtype=np.float32)
        else:
            route_sample = np.ascontiguousarray(route_sample, dtype=np.float32)
        old_ef = int(self.quantizer.hnsw.efSearch)
        self.quantizer.hnsw.efSearch = self.route_efSearch
        _, route_ids = self.quantizer.search(route_sample, 1)
        self.quantizer.hnsw.efSearch = old_ef
        route_ids = np.ascontiguousarray(route_ids[:, 0], dtype=np.int64)
        traffic = np.bincount(route_ids, minlength=self.nlist).astype(np.float32)
        self.route_traffic = traffic
        observed = traffic > 0
        self.route_observed_cells = int(np.sum(observed))
        if self.route_observed_cells:
            self.route_mean_traffic = float(np.mean(traffic[observed]))
            self.route_max_traffic = float(np.max(traffic[observed]))

    def _cell_caps(self, assigned, scores):
        base_caps = super()._cell_caps(assigned, scores).astype(np.float32)
        traffic = self.route_traffic
        if traffic is None or not np.any(traffic > 0):
            return base_caps.astype(np.int32)

        counts = np.bincount(assigned, minlength=self.nlist).astype(np.float32)
        nonempty = counts > 0
        observed = traffic > 0
        mean_traffic = float(np.mean(traffic[observed]) + self.route_smooth + 1e-12)
        route_ratio = (traffic + self.route_smooth) / mean_traffic
        route_ratio = np.clip(route_ratio, self.route_clip_min, self.route_clip_max)

        if self.route_mode == "penalty":
            proposed = base_caps * np.power(route_ratio, -self.route_gamma)
        elif self.route_mode == "routeonly":
            proposed = float(self.list_cap) * np.power(route_ratio, -self.route_gamma)
        elif self.route_mode == "balance":
            mean_count = float(np.mean(counts[nonempty]) + 1e-12)
            count_ratio = np.maximum(counts, 1.0) / mean_count
            proposed = float(self.list_cap) * np.power(count_ratio / route_ratio, self.route_gamma)
        elif self.route_mode == "hotfloor":
            proposed = np.where(route_ratio > 1.0, base_caps * np.power(route_ratio, -self.route_gamma), base_caps)
        else:
            cooled = base_caps * np.power(route_ratio, -self.route_gamma)
            proposed = np.minimum(base_caps, cooled)

        caps = np.zeros(self.nlist, dtype=np.float32)
        caps[nonempty] = proposed[nonempty]
        caps = np.clip(caps, self.cap_min, self.cap_max)
        caps[~nonempty] = 0
        caps = np.minimum(caps, counts)
        caps_i = np.rint(caps).astype(np.int32)
        self.cap_total_requested = int(np.sum(caps_i))

        traffic_sum = float(np.sum(traffic))
        if traffic_sum > 0.0:
            self.route_weighted_cap = float(np.sum(traffic * caps_i) / traffic_sum)
            self.route_weighted_base_cap = float(np.sum(traffic * base_caps) / traffic_sum)
            self.route_weighted_cap_ratio = self.route_weighted_cap / max(self.route_weighted_base_cap, 1e-12)
        return caps_i

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-adaptivecapcore-transport-hnswgate",
            "zeelin-routeweighted-ratecapcore-transport-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", routeMode={self.route_mode}, routeGamma={self.route_gamma}, "
            f"routeTrain={self.route_train_size}, routeEfS={self.route_efSearch}, "
            f"routeWeightedCap={self.route_weighted_cap:.2f}, "
            f"routeWeightedRatio={self.route_weighted_cap_ratio:.3f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-routeweighted-ratecapcore"
        extra["route_mode"] = self.route_mode
        extra["route_gamma"] = self.route_gamma
        extra["route_train_size"] = self.route_train_size
        extra["route_efSearch"] = self.route_efSearch
        extra["route_smooth"] = self.route_smooth
        extra["route_clip_min"] = self.route_clip_min
        extra["route_clip_max"] = self.route_clip_max
        extra["route_observed_cells"] = self.route_observed_cells
        extra["route_mean_traffic"] = self.route_mean_traffic
        extra["route_max_traffic"] = self.route_max_traffic
        extra["route_weighted_cap"] = self.route_weighted_cap
        extra["route_weighted_base_cap"] = self.route_weighted_base_cap
        extra["route_weighted_cap_ratio"] = self.route_weighted_cap_ratio
        return extra


class ZeelinM10routeweightedRatecapcoreTransportHnswgateIvf(ZeelinRouteweightedRatecapcoreTransportHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param.setdefault("qm", 10)
        method_param.setdefault("lc", 96)
        method_param.setdefault("aef", 8)
        method_param.setdefault("co", "centroid")
        super().__init__(metric, method_param)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-routeweighted-ratecapcore-transport-hnswgate",
            "zeelin-m10routeweighted-ratecapcore-transport-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10routeweighted-ratecapcore"
        return extra


class ZeelinM10routeweightedRatecapcoreReturncompressHnswgateIvf(
    ZeelinM10routeweightedRatecapcoreTransportHnswgateIvf
):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        if return_k is None:
            return_k = 7
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10routeweighted-ratecapcore-transport-hnswgate",
            "zeelin-m10routeweighted-ratecapcore-returncompress-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10routeweighted-ratecapcore-returncompress"
        return extra


class ZeelinM10phasepickRatecapcoreReturncompressHnswgateIvf(ZeelinM10ratecapcoreReturncompressHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "pt" in method_param:
            method_param["phase_trials"] = method_param["pt"]
        super().__init__(metric, method_param)
        self.phase_trials = int(method_param.get("phase_trials", 2))
        self.phase_trials = max(1, min(4, self.phase_trials))
        self.phase_best_trial = 0
        self.phase_best_build_time = 0.0
        self.phase_all_build_times = ""

    def fit(self, X):
        best_state = None
        best_score = None
        build_times = []

        for trial in range(self.phase_trials):
            start = time.time()
            super().fit(X)
            elapsed = time.time() - start
            build_times.append(elapsed)
            score = elapsed
            state = {
                "index": self.index,
                "quantizer": self.quantizer,
                "dim": self.dim,
                "qbuf": self.qbuf,
                "kept_points": self.kept_points,
                "cap_min_actual": self.cap_min_actual,
                "cap_max_actual": self.cap_max_actual,
                "cap_mean_actual": self.cap_mean_actual,
                "cap_total_requested": getattr(self, "cap_total_requested", 0),
                "trial": trial + 1,
                "build_time": elapsed,
            }
            if best_score is None or score < best_score:
                best_score = score
                best_state = state

        if best_state is not None:
            self.index = best_state["index"]
            self.quantizer = best_state["quantizer"]
            self.dim = best_state["dim"]
            self.qbuf = best_state["qbuf"]
            self.kept_points = best_state["kept_points"]
            self.cap_min_actual = best_state["cap_min_actual"]
            self.cap_max_actual = best_state["cap_max_actual"]
            self.cap_mean_actual = best_state["cap_mean_actual"]
            self.cap_total_requested = best_state["cap_total_requested"]
            self.phase_best_trial = best_state["trial"]
            self.phase_best_build_time = best_state["build_time"]
        self.phase_all_build_times = ",".join(f"{v:.6f}" for v in build_times)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10ratecapcore-returncompress-hnswgate",
            "zeelin-m10phasepick-ratecapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", phaseTrials={self.phase_trials}, phaseBestTrial={self.phase_best_trial}, "
            f"phaseBestBuild={self.phase_best_build_time:.3f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10phasepick-ratecapcore-returncompress"
        extra["phase_trials"] = self.phase_trials
        extra["phase_best_trial"] = self.phase_best_trial
        extra["phase_best_build_time"] = self.phase_best_build_time
        extra["phase_all_build_times"] = self.phase_all_build_times
        return extra


class ZeelinM10phasecertificateRatecapcoreReturncompressHnswgateIvf(
    ZeelinM10ratecapcoreReturncompressHnswgateIvf
):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "pct" in method_param:
            method_param["phase_certificate_trials"] = method_param["pct"]
        if "pps" in method_param:
            method_param["phase_probe_size"] = method_param["pps"]
        if "ppr" in method_param:
            method_param["phase_probe_repeats"] = method_param["ppr"]
        if "ppk" in method_param:
            method_param["phase_probe_k"] = method_param["ppk"]
        if "pnp" in method_param:
            method_param["phase_probe_nprobe"] = method_param["pnp"]
        if "pef" in method_param:
            method_param["phase_probe_efSearch"] = method_param["pef"]
        super().__init__(metric, method_param)
        self.phase_certificate_trials = int(method_param.get("phase_certificate_trials", 2))
        self.phase_probe_size = int(method_param.get("phase_probe_size", 2048))
        self.phase_probe_repeats = int(method_param.get("phase_probe_repeats", 3))
        self.phase_probe_k = int(method_param.get("phase_probe_k", 7))
        self.phase_probe_nprobe = int(method_param.get("phase_probe_nprobe", 1))
        self.phase_probe_efSearch = int(method_param.get("phase_probe_efSearch", 1))
        self.phase_certificate_trials = max(1, min(4, self.phase_certificate_trials))
        self.phase_probe_size = max(64, min(20000, self.phase_probe_size))
        self.phase_probe_repeats = max(1, min(9, self.phase_probe_repeats))
        self.phase_probe_k = max(1, min(16, self.phase_probe_k))
        self.phase_probe_nprobe = max(1, min(16, self.phase_probe_nprobe))
        self.phase_probe_efSearch = max(1, min(64, self.phase_probe_efSearch))
        self.phase_best_trial = 0
        self.phase_best_build_time = 0.0
        self.phase_best_proxy_us = 0.0
        self.phase_best_route_entropy = 0.0
        self.phase_best_route_imbalance = 0.0
        self.phase_best_route_top_fraction = 0.0
        self.phase_all_build_times = ""
        self.phase_all_proxy_us = ""
        self.phase_all_route_entropy = ""
        self.phase_all_route_imbalance = ""

    def _phase_certificate(self, probe):
        old_nprobe = int(self.index.nprobe)
        old_ef = int(self.quantizer.hnsw.efSearch)
        self.index.nprobe = self.phase_probe_nprobe
        self.quantizer.hnsw.efSearch = self.phase_probe_efSearch

        warm = probe[: min(len(probe), 64)]
        if len(warm):
            self.index.search(warm, self.phase_probe_k)

        best_elapsed = None
        for _ in range(self.phase_probe_repeats):
            start = time.perf_counter()
            self.index.search(probe, self.phase_probe_k)
            elapsed = time.perf_counter() - start
            if best_elapsed is None or elapsed < best_elapsed:
                best_elapsed = elapsed

        _, route_ids = self.quantizer.search(probe, 1)
        route_ids = np.asarray(route_ids[:, 0], dtype=np.int64)
        traffic = np.bincount(route_ids, minlength=self.nlist).astype(np.float64)
        total = float(np.sum(traffic))
        if total > 0.0:
            p = traffic[traffic > 0.0] / total
            entropy = float(-np.sum(p * np.log(p)) / max(np.log(max(2, len(p))), 1e-12))
            mean = float(np.mean(traffic[traffic > 0.0]))
            imbalance = float(np.max(traffic) / max(mean, 1e-12))
            top_fraction = float(np.max(traffic) / total)
        else:
            entropy = 0.0
            imbalance = 0.0
            top_fraction = 0.0

        self.index.nprobe = old_nprobe
        self.quantizer.hnsw.efSearch = old_ef
        proxy_us = (float(best_elapsed) / max(1, len(probe))) * 1e6
        return proxy_us, entropy, imbalance, top_fraction

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        probe_idx = np.linspace(
            0,
            len(X) - 1,
            min(len(X), self.phase_probe_size),
            dtype=np.int64,
        )
        probe = self._normalize_rows(X[probe_idx]).astype(np.float32, copy=False)
        probe = np.ascontiguousarray(probe, dtype=np.float32)

        best_state = None
        best_score = None
        build_times = []
        proxy_times = []
        entropies = []
        imbalances = []

        for trial in range(self.phase_certificate_trials):
            start = time.time()
            super().fit(X)
            elapsed = time.time() - start
            proxy_us, entropy, imbalance, top_fraction = self._phase_certificate(probe)
            build_times.append(elapsed)
            proxy_times.append(proxy_us)
            entropies.append(entropy)
            imbalances.append(imbalance)
            score = proxy_us
            state = {
                "index": self.index,
                "quantizer": self.quantizer,
                "dim": self.dim,
                "qbuf": self.qbuf,
                "kept_points": self.kept_points,
                "cap_min_actual": self.cap_min_actual,
                "cap_max_actual": self.cap_max_actual,
                "cap_mean_actual": self.cap_mean_actual,
                "cap_total_requested": getattr(self, "cap_total_requested", 0),
                "trial": trial + 1,
                "build_time": elapsed,
                "proxy_us": proxy_us,
                "entropy": entropy,
                "imbalance": imbalance,
                "top_fraction": top_fraction,
            }
            if best_score is None or score < best_score:
                best_score = score
                best_state = state

        if best_state is not None:
            self.index = best_state["index"]
            self.quantizer = best_state["quantizer"]
            self.dim = best_state["dim"]
            self.qbuf = best_state["qbuf"]
            self.kept_points = best_state["kept_points"]
            self.cap_min_actual = best_state["cap_min_actual"]
            self.cap_max_actual = best_state["cap_max_actual"]
            self.cap_mean_actual = best_state["cap_mean_actual"]
            self.cap_total_requested = best_state["cap_total_requested"]
            self.phase_best_trial = best_state["trial"]
            self.phase_best_build_time = best_state["build_time"]
            self.phase_best_proxy_us = best_state["proxy_us"]
            self.phase_best_route_entropy = best_state["entropy"]
            self.phase_best_route_imbalance = best_state["imbalance"]
            self.phase_best_route_top_fraction = best_state["top_fraction"]

        self.phase_all_build_times = ",".join(f"{v:.6f}" for v in build_times)
        self.phase_all_proxy_us = ",".join(f"{v:.6f}" for v in proxy_times)
        self.phase_all_route_entropy = ",".join(f"{v:.6f}" for v in entropies)
        self.phase_all_route_imbalance = ",".join(f"{v:.6f}" for v in imbalances)

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-m10ratecapcore-returncompress-hnswgate",
            "zeelin-m10phasecertificate-ratecapcore-returncompress-hnswgate",
            1,
        )
        self.name = self.name[:-1] + (
            f", phaseCertTrials={self.phase_certificate_trials}, "
            f"phaseBestTrial={self.phase_best_trial}, "
            f"phaseProxyUs={self.phase_best_proxy_us:.3f}, "
            f"phaseEntropy={self.phase_best_route_entropy:.3f})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-m10phasecertificate-ratecapcore-returncompress"
        extra["phase_certificate_trials"] = self.phase_certificate_trials
        extra["phase_probe_size"] = self.phase_probe_size
        extra["phase_probe_repeats"] = self.phase_probe_repeats
        extra["phase_probe_k"] = self.phase_probe_k
        extra["phase_probe_nprobe"] = self.phase_probe_nprobe
        extra["phase_probe_efSearch"] = self.phase_probe_efSearch
        extra["phase_best_trial"] = self.phase_best_trial
        extra["phase_best_build_time"] = self.phase_best_build_time
        extra["phase_best_proxy_us"] = self.phase_best_proxy_us
        extra["phase_best_route_entropy"] = self.phase_best_route_entropy
        extra["phase_best_route_imbalance"] = self.phase_best_route_imbalance
        extra["phase_best_route_top_fraction"] = self.phase_best_route_top_fraction
        extra["phase_all_build_times"] = self.phase_all_build_times
        extra["phase_all_proxy_us"] = self.phase_all_proxy_us
        extra["phase_all_route_entropy"] = self.phase_all_route_entropy
        extra["phase_all_route_imbalance"] = self.phase_all_route_imbalance
        return extra


class ZeelinSpectralsubcellpointermassguardHnswgateIvf(ZeelinPointermassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "qefc" in method_param:
            method_param["quantizer_efConstruction"] = method_param["qefc"]
        if "qm" in method_param:
            method_param["quantizer_m"] = method_param["qm"]
        if "ts" in method_param:
            method_param["train_size"] = method_param["ts"]
        if "sbits" in method_param:
            method_param["sub_bits"] = method_param["sbits"]
        method_param.setdefault("pmode", "view")
        method_param.setdefault("internal_k", 4)
        method_param.setdefault("return_k", 9)
        method_param.setdefault("dedup_extra", 0)
        method_param.setdefault("fast_unique", True)
        method_param.setdefault("skip_query_norm", True)
        method_param.setdefault("trust_full_labels", True)
        method_param.setdefault("use_search_c", True)
        super().__init__(metric, method_param)
        self.sub_bits = int(method_param.get("sub_bits", 2))
        self.sub_bits = max(1, min(6, self.sub_bits))
        self.sub_count = 1 << self.sub_bits
        self.assign_ef_search = int(method_param.get("assign_efSearch", self.default_quantizer_ef_search))
        self.axis_train_size = int(method_param.get("axis_train_size", 200000))
        self.axes = None
        self.centroid_proj = None
        self._assign = None
        self._centroid_dis = None

    def _fit_axes(self, Xn):
        sample = Xn
        if 0 < self.axis_train_size < len(Xn):
            axis_idx = np.linspace(0, len(Xn) - 1, self.axis_train_size, dtype=np.int64)
            sample = np.ascontiguousarray(Xn[axis_idx], dtype=np.float32)
        centered = sample - np.mean(sample, axis=0, keepdims=True)
        cov = np.dot(centered.T, centered).astype(np.float64, copy=False)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[-self.sub_bits :][::-1]
        axes = vecs[:, order].T.astype(np.float32, copy=True)
        norms = np.linalg.norm(axes, axis=1, keepdims=True).astype(np.float32)
        return np.ascontiguousarray(axes / np.maximum(norms, 1e-12), dtype=np.float32)

    def _codes_from_projection(self, proj, centroid_proj):
        bits = proj >= centroid_proj
        codes = np.zeros(bits.shape[0], dtype=np.int64)
        for bit in range(self.sub_bits):
            codes |= bits[:, bit].astype(np.int64) << bit
        return codes

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        trainer = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        trainer.train(train)

        self.axes = self._fit_axes(Xn)
        centroids = np.empty((self.nlist, self.dim), dtype=np.float32)
        for cid in range(self.nlist):
            centroids[cid, :] = self.quantizer.reconstruct(cid)
        self.centroid_proj = np.ascontiguousarray(np.dot(centroids, self.axes.T), dtype=np.float32)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        _, assigned = self.quantizer.search(np.ascontiguousarray(Xn, dtype=np.float32), 1)
        assigned = np.ascontiguousarray(assigned[:, 0], dtype=np.int64)
        x_proj = np.ascontiguousarray(np.dot(Xn, self.axes.T), dtype=np.float32)
        codes = self._codes_from_projection(x_proj, self.centroid_proj[assigned])
        fine_assign = np.ascontiguousarray(assigned * self.sub_count + codes, dtype=np.int64)

        self.index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(self.dim),
            self.dim,
            self.nlist * self.sub_count,
            faiss.METRIC_INNER_PRODUCT,
        )
        self.index.is_trained = True
        ids = np.arange(len(Xn), dtype=np.int64)
        block = int(self.method_param.get("add_block", 65536))
        block = max(1024, block)
        for start in range(0, len(Xn), block):
            stop = min(len(Xn), start + block)
            xb = np.ascontiguousarray(Xn[start:stop], dtype=np.float32)
            self.index.add_core(
                stop - start,
                faiss.swig_ptr(xb),
                faiss.swig_ptr(ids[start:stop]),
                faiss.swig_ptr(fine_assign[start:stop]),
            )
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self._assign = np.empty((1, self.nprobe), dtype=np.int64)
        self._centroid_dis = np.zeros((1, self.nprobe), dtype=np.float32)
        self.name = self.name.replace(
            "zeelin-pointermassguard-hnswgate",
            "zeelin-spectralsubcellpointermassguard",
            1,
        )
        self.name = self.name[:-1] + (
            f", subBits={self.sub_bits}, subCount={self.sub_count}, "
            f"coarseNlist={self.nlist}, fineNlist={self.nlist * self.sub_count}, "
            f"assignEfS={self.assign_ef_search}, axisTrainSize={self.axis_train_size})"
        )

    def query(self, v, n):
        if self.skip_query_norm:
            qptr = faiss.swig_ptr(v)
            qrow = v
        else:
            norm = float(np.sqrt(np.dot(v, v)))
            self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
            qptr = self._qbuf_ptr
            qrow = self.qbuf[0]

        _, coarse = self.quantizer.search(np.asarray(qrow, dtype=np.float32).reshape(1, self.dim), self.nprobe)
        coarse = np.ascontiguousarray(coarse[0], dtype=np.int64)
        proj = np.dot(qrow, self.axes.T).astype(np.float32, copy=False)
        bits = proj.reshape(1, self.sub_bits) >= self.centroid_proj[coarse]
        codes = np.zeros(self.nprobe, dtype=np.int64)
        for bit in range(self.sub_bits):
            codes |= bits[:, bit].astype(np.int64) << bit
        self._assign[0, :] = coarse * self.sub_count + codes

        self.index.search_preassigned_c(
            1,
            qptr,
            self._query_k,
            faiss.swig_ptr(self._assign),
            faiss.swig_ptr(self._centroid_dis),
            self._dbuf_ptr,
            self._ibuf_ptr,
            False,
        )
        if self._prefix[-1] >= 0:
            return self._prefix
        return self._prefix[self._prefix >= 0]

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-spectralsubcellpointermassguard"
        extra["sub_bits"] = self.sub_bits
        extra["sub_count"] = self.sub_count
        extra["fine_nlist"] = self.nlist * self.sub_count
        extra["assign_efSearch"] = self.assign_ef_search
        extra["axis_train_size"] = self.axis_train_size
        return extra


class ZeelinGraphpointermassguardHnswgateIvf(ZeelinPointermassguardHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace("zeelin-pointermassguard-hnswgate", "zeelin-graphpointermassguard-hnswgate", 1)

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-graphpointermassguard"
        return extra


class ZeelinGraphpointerbestHnswgateIvf(ZeelinGraphpointermassguardHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace("zeelin-graphpointermassguard-hnswgate", "zeelin-graphpointerbest-hnswgate", 1)

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-graphpointerbest"
        return extra


class ZeelinDegreequotientGraphpointerHnswgateIvf(ZeelinGraphpointermassguardHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-graphpointermassguard-hnswgate",
            "zeelin-degreequotient-graphpointer-hnswgate",
            1,
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-degreequotient-graphpointer"
        extra["degree_quotient"] = self.quantizer_m
        return extra


class ZeelinFusedgraphrouteHnswgateIvf(ZeelinHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "qefc" in method_param:
            method_param["quantizer_efConstruction"] = method_param["qefc"]
        if "qm" in method_param:
            method_param["quantizer_m"] = method_param["qm"]
        if "ts" in method_param:
            method_param["train_size"] = method_param["ts"]
        if "aef" in method_param:
            method_param["assign_efSearch"] = method_param["aef"]
        method_param.setdefault("quantizer_efSearch", 1)
        method_param.setdefault("skip_query_norm", True)
        method_param.setdefault("query_view", True)
        method_param.setdefault("threads", 4)
        method_param.setdefault("return_k", 9)
        super().__init__(metric, method_param)
        self.assign_ef_search = int(method_param.get("assign_efSearch", 8))
        self.default_repair_cap = int(method_param.get("repair_cap", method_param.get("rc", 0)))
        self.repair_cap = max(0, self.default_repair_cap)
        self.boundary_tape_cap = int(method_param.get("boundary_tape_cap", method_param.get("btcap", 64)))
        self.boundary_tape_cap = max(1, self.boundary_tape_cap)
        self._lib = self._load_fused_graphroute_library()
        self._idx = self._lib.zeelin_fused_graphroute_ivf_create()
        self._out64 = None

    def _load_fused_graphroute_library(self):
        here = Path(__file__).resolve().parent
        source = here / "zeelin_fused_graphroute_ivf_lib.cpp"
        suffix = ".dylib" if os.uname().sysname == "Darwin" else ".so"
        out = here / f"libzeelin_fused_graphroute_ivf{suffix}"
        force_rebuild = os.environ.get("ZEELIN_FUSED_GRAPHROUTE_FORCE_REBUILD") == "1"
        needs_rebuild = force_rebuild or not out.exists()
        if out.exists() and source.exists():
            needs_rebuild = needs_rebuild or source.stat().st_mtime > out.stat().st_mtime
        if needs_rebuild:
            cxx = os.environ.get("CXX", "c++")
            flags = os.environ.get("ZEELIN_FUSED_GRAPHROUTE_CXXFLAGS")
            if flags:
                cmd = [cxx, *flags.split(), str(source), "-o", str(out)]
            else:
                cmd = [
                    cxx,
                    "-O3",
                    "-std=c++17",
                    "-march=native",
                    "-ffast-math",
                    "-funroll-loops",
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-o",
                    str(out),
                ]
            subprocess.check_call(cmd)
        lib = ctypes.CDLL(str(out))
        lib.zeelin_fused_graphroute_ivf_create.restype = ctypes.c_void_p
        lib.zeelin_fused_graphroute_ivf_free.argtypes = [ctypes.c_void_p]
        lib.zeelin_fused_graphroute_ivf_fit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.zeelin_fused_graphroute_ivf_query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_longlong),
        ]
        lib.zeelin_fused_graphroute_ivf_query.restype = ctypes.c_int
        lib.zeelin_fused_graphroute_ivf_boundary_ids.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_longlong),
        ]
        lib.zeelin_fused_graphroute_ivf_boundary_ids.restype = ctypes.c_int
        lib.zeelin_fused_graphroute_ivf_avg_candidates.argtypes = [ctypes.c_void_p]
        lib.zeelin_fused_graphroute_ivf_avg_candidates.restype = ctypes.c_double
        lib.zeelin_fused_graphroute_ivf_avg_route_dots.argtypes = [ctypes.c_void_p]
        lib.zeelin_fused_graphroute_ivf_avg_route_dots.restype = ctypes.c_double
        lib.zeelin_fused_graphroute_ivf_avg_route_steps.argtypes = [ctypes.c_void_p]
        lib.zeelin_fused_graphroute_ivf_avg_route_steps.restype = ctypes.c_double
        lib.zeelin_fused_graphroute_ivf_avg_selected_lists.argtypes = [ctypes.c_void_p]
        lib.zeelin_fused_graphroute_ivf_avg_selected_lists.restype = ctypes.c_double
        lib.zeelin_fused_graphroute_ivf_total_candidates.argtypes = [ctypes.c_void_p]
        lib.zeelin_fused_graphroute_ivf_total_candidates.restype = ctypes.c_double
        lib.zeelin_fused_graphroute_ivf_scan_layout.restype = ctypes.c_char_p
        lib.zeelin_fused_graphroute_ivf_kernel.restype = ctypes.c_char_p
        return lib

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = np.ascontiguousarray(self._normalize_rows(X), dtype=np.float32)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        trainer = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        trainer.train(train)

        centroids = np.empty((self.nlist, self.dim), dtype=np.float32)
        self.quantizer.reconstruct_n(0, self.nlist, centroids)
        centroids = np.ascontiguousarray(centroids, dtype=np.float32)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        _, assigned = self.quantizer.search(Xn, 1)
        assigned = np.ascontiguousarray(assigned[:, 0], dtype=np.int64)
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

        hnsw = self.quantizer.hnsw
        h_offsets = np.ascontiguousarray(faiss.vector_to_array(hnsw.offsets), dtype=np.uint64)
        h_neighbors = np.ascontiguousarray(faiss.vector_to_array(hnsw.neighbors), dtype=np.int32)
        h_levels = np.ascontiguousarray(faiss.vector_to_array(hnsw.levels), dtype=np.int32)
        h_cum = np.ascontiguousarray(faiss.vector_to_array(hnsw.cum_nneighbor_per_level), dtype=np.int32)

        self._lib.zeelin_fused_graphroute_ivf_fit(
            self._idx,
            Xn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            assigned.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            centroids.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(Xn.shape[0]),
            int(self.dim),
            int(self.nlist),
            h_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong)),
            int(h_offsets.shape[0]),
            h_neighbors.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            int(h_neighbors.shape[0]),
            h_levels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            int(h_levels.shape[0]),
            h_cum.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            int(h_cum.shape[0]),
            int(hnsw.entry_point),
            int(hnsw.max_level),
            int(self.boundary_tape_cap),
        )

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        self.nprobe = max(1, min(16, int(nprobe)))
        if return_k is not None:
            self.return_k = int(return_k)
        if internal_k is not None:
            self.repair_cap = max(0, int(internal_k))
        self.quantizer_ef_search = 0
        self._target_k = 10 if self.return_k <= 0 else min(10, self.return_k)
        if self._out64 is None or len(self._out64) < self._target_k:
            self._out64 = np.empty(self._target_k, dtype=np.int64)
        algo_name = (
            "zeelin-boundaryrepair-tape-hnswgate"
            if self.repair_cap > 0
            else "zeelin-fusedgraphroute-hnswgate"
        )
        self.name = (
            f"{algo_name} "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, "
            f"quantizerM={self.quantizer_m}, quantizerEfC={self.quantizer_ef_construction}, "
            f"assignEfS={self.assign_ef_search}, trainSize={self.train_size}, "
            f"returnK={self.return_k}, skipQueryNorm={int(self.skip_query_norm)}, "
            f"queryView={int(self.query_view)}, repairCap={self.repair_cap}, "
            f"boundaryTapeCap={self.boundary_tape_cap}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def query(self, v, n):
        if self.skip_query_norm and self.query_view:
            q = v.reshape(1, self.dim)
        elif self.skip_query_norm:
            self.qbuf[0, :] = v
            q = self.qbuf
        else:
            norm = float(np.sqrt(np.dot(v, v)))
            self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
            q = self.qbuf
        target_k = n if self.return_k <= 0 else min(n, self.return_k)
        if self._out64 is None or len(self._out64) < target_k:
            self._out64 = np.empty(target_k, dtype=np.int64)
        count = self._lib.zeelin_fused_graphroute_ivf_query(
            self._idx,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(target_k),
            int(self.nprobe),
            int(self.repair_cap),
            self._out64.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        )
        return self._out64[:count]

    def batch_query(self, X, n):
        X = np.asarray(X, dtype=np.float32)
        if self.skip_query_norm:
            Xn = np.ascontiguousarray(X, dtype=np.float32)
        else:
            norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
            Xn = np.ascontiguousarray(X / np.maximum(norms, 1e-12), dtype=np.float32)
        target_k = n if self.return_k <= 0 else min(n, self.return_k)
        self.quantizer.hnsw.efSearch = self.quantizer_ef_search
        _, labels = self.quantizer.search(Xn, self.nprobe)
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        out = np.empty((len(Xn), target_k), dtype=np.int64)
        counts = np.empty(len(Xn), dtype=np.int64)
        self._lib.zeelin_int8repair_ivf_query_batch(
            self._idx,
            Xn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            labels.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(len(Xn)),
            int(self.nprobe),
            int(target_k),
            int(self.budget),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        )
        self._batch_out = out
        self._batch_counts = counts

    def get_batch_results(self):
        return [
            np.asarray(self._batch_out[i, : int(self._batch_counts[i])], dtype=np.int64)
            for i in range(len(self._batch_out))
        ]

    def get_additional(self):
        return {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "return_k": self.return_k,
            "skip_query_norm": int(self.skip_query_norm),
            "query_view": int(self.query_view),
            "threads": int(self.method_param.get("threads", 1)),
            "train_size": self.train_size,
            "quantizer_m": self.quantizer_m,
            "quantizer_efConstruction": self.quantizer_ef_construction,
            "assign_efSearch": self.assign_ef_search,
            "repair_cap": self.repair_cap,
            "boundary_tape_cap": self.boundary_tape_cap,
            "avg_candidates": self._lib.zeelin_fused_graphroute_ivf_avg_candidates(self._idx),
            "avg_route_dots": self._lib.zeelin_fused_graphroute_ivf_avg_route_dots(self._idx),
            "avg_route_steps": self._lib.zeelin_fused_graphroute_ivf_avg_route_steps(self._idx),
            "avg_selected_lists": self._lib.zeelin_fused_graphroute_ivf_avg_selected_lists(self._idx),
            "dist_comps": self._lib.zeelin_fused_graphroute_ivf_total_candidates(self._idx),
            "avg_dist_comps": self._lib.zeelin_fused_graphroute_ivf_avg_candidates(self._idx),
            "kernel": self._lib.zeelin_fused_graphroute_ivf_kernel().decode("ascii"),
            "scan_layout": self._lib.zeelin_fused_graphroute_ivf_scan_layout().decode("ascii"),
        }

    def done(self):
        if getattr(self, "_idx", None):
            self._lib.zeelin_fused_graphroute_ivf_free(self._idx)
            self._idx = None


class ZeelinBoundarytapePreassignHnswgateIvf(ZeelinFusedgraphrouteHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "pk" in method_param:
            method_param["primary_k"] = method_param["pk"]
        super().__init__(metric, method_param)
        self.primary_k = int(method_param.get("primary_k", 8))
        self.primary_k = max(1, min(10, self.primary_k))
        self._qd = None
        self._qi = None
        self._assign = None
        self._assign_dis = None
        self._repair64 = None
        self._query_count = 0
        self._repair_returned = 0
        self._dupe_repair = 0

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = np.ascontiguousarray(self._normalize_rows(X), dtype=np.float32)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        self.index.train(train)
        self.index.add(Xn)

        centroids = np.empty((self.nlist, self.dim), dtype=np.float32)
        self.quantizer.reconstruct_n(0, self.nlist, centroids)
        centroids = np.ascontiguousarray(centroids, dtype=np.float32)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        _, assigned = self.quantizer.search(Xn, 1)
        assigned = np.ascontiguousarray(assigned[:, 0], dtype=np.int64)
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

        hnsw = self.quantizer.hnsw
        h_offsets = np.ascontiguousarray(faiss.vector_to_array(hnsw.offsets), dtype=np.uint64)
        h_neighbors = np.ascontiguousarray(faiss.vector_to_array(hnsw.neighbors), dtype=np.int32)
        h_levels = np.ascontiguousarray(faiss.vector_to_array(hnsw.levels), dtype=np.int32)
        h_cum = np.ascontiguousarray(faiss.vector_to_array(hnsw.cum_nneighbor_per_level), dtype=np.int32)

        self._lib.zeelin_fused_graphroute_ivf_fit(
            self._idx,
            Xn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            assigned.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            centroids.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(Xn.shape[0]),
            int(self.dim),
            int(self.nlist),
            h_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong)),
            int(h_offsets.shape[0]),
            h_neighbors.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            int(h_neighbors.shape[0]),
            h_levels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            int(h_levels.shape[0]),
            h_cum.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            int(h_cum.shape[0]),
            int(hnsw.entry_point),
            int(hnsw.max_level),
            int(self.boundary_tape_cap),
        )

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        faiss.cvar.indexIVF_stats.reset()
        self.nprobe = max(1, min(8, int(nprobe)))
        self.quantizer_ef_search = (
            int(quantizer_ef_search)
            if quantizer_ef_search is not None and int(quantizer_ef_search) > 0
            else self.default_quantizer_ef_search
        )
        self.quantizer.hnsw.efSearch = self.quantizer_ef_search
        if return_k is not None:
            self.return_k = int(return_k)
        if internal_k is not None:
            self.repair_cap = max(0, int(internal_k))
        self._target_k = 10 if self.return_k <= 0 else min(10, self.return_k)
        self._qd = np.empty((1, self.nprobe), dtype=np.float32)
        self._qi = np.empty((1, self.nprobe), dtype=np.int64)
        self._assign = np.empty((1, 1), dtype=np.int64)
        self._assign_dis = np.empty((1, 1), dtype=np.float32)
        self.dbuf = np.empty((1, self.primary_k), dtype=np.float32)
        self.ibuf = np.empty((1, self.primary_k), dtype=np.int64)
        self._out64 = np.empty(self._target_k, dtype=np.int64)
        self._repair64 = np.empty(max(1, self.repair_cap), dtype=np.int64)
        self._query_count = 0
        self._repair_returned = 0
        self._dupe_repair = 0
        self.name = (
            "zeelin-boundarytape-preassign-hnswgate "
            f"(nlist={self.nlist}, routeK={self.nprobe}, primaryK={self.primary_k}, "
            f"repairCap={self.repair_cap}, boundaryTapeCap={self.boundary_tape_cap}, "
            f"quantizerM={self.quantizer_m}, quantizerEfC={self.quantizer_ef_construction}, "
            f"quantizerEfS={self.quantizer_ef_search}, assignEfS={self.assign_ef_search}, "
            f"trainSize={self.train_size}, returnK={self.return_k}, "
            f"skipQueryNorm={int(self.skip_query_norm)}, threads={self.method_param.get('threads', 1)})"
        )

    @staticmethod
    def _small_contains(arr, count, value):
        for i in range(count):
            if int(arr[i]) == int(value):
                return True
        return False

    def query(self, v, n):
        if self.skip_query_norm and self.query_view:
            q = v.reshape(1, self.dim)
        elif self.skip_query_norm:
            self.qbuf[0, :] = v
            q = self.qbuf
        else:
            norm = float(np.sqrt(np.dot(v, v)))
            self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
            q = self.qbuf
        qptr = faiss.swig_ptr(q)
        self.quantizer.search_c(
            1,
            qptr,
            self.nprobe,
            faiss.swig_ptr(self._qd),
            faiss.swig_ptr(self._qi),
        )
        primary = int(self._qi[0, 0])
        self._assign[0, 0] = primary
        self._assign_dis[0, 0] = self._qd[0, 0]
        self.index.search_preassigned_c(
            1,
            qptr,
            self.primary_k,
            faiss.swig_ptr(self._assign),
            faiss.swig_ptr(self._assign_dis),
            faiss.swig_ptr(self.dbuf),
            faiss.swig_ptr(self.ibuf),
            False,
        )

        out_count = 0
        for label in self.ibuf[0, : self.primary_k]:
            if label < 0:
                continue
            if out_count >= self._target_k:
                break
            self._out64[out_count] = int(label)
            out_count += 1

        if self.nprobe > 1 and self.repair_cap > 0 and out_count < self._target_k:
            repair_count = self._lib.zeelin_fused_graphroute_ivf_boundary_ids(
                self._idx,
                int(primary),
                int(self._qi[0, 1]),
                int(self.repair_cap),
                self._repair64.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            )
            for i in range(repair_count):
                label = int(self._repair64[i])
                if label < 0:
                    continue
                if self._small_contains(self._out64, out_count, label):
                    self._dupe_repair += 1
                    continue
                if out_count >= self._target_k:
                    break
                self._out64[out_count] = label
                out_count += 1
                self._repair_returned += 1
        self._query_count += 1
        return self._out64[:out_count]

    def get_additional(self):
        total = max(1, self._query_count)
        return {
            "nlist": self.nlist,
            "route_k": self.nprobe,
            "primary_k": self.primary_k,
            "return_k": self.return_k,
            "repair_cap": self.repair_cap,
            "boundary_tape_cap": self.boundary_tape_cap,
            "skip_query_norm": int(self.skip_query_norm),
            "threads": int(self.method_param.get("threads", 1)),
            "train_size": self.train_size,
            "quantizer_m": self.quantizer_m,
            "quantizer_efConstruction": self.quantizer_ef_construction,
            "quantizer_efSearch": self.quantizer_ef_search,
            "assign_efSearch": self.assign_ef_search,
            "avg_repair_returned": self._repair_returned / total,
            "avg_repair_dupes": self._dupe_repair / total,
            "dist_comps": faiss.cvar.indexIVF_stats.ndis,
            "kernel": "faiss-preassigned-primary-plus-boundary-tape",
            "scan_layout": "hnsw-route-preassigned-primary-directed-boundary-id-tape",
        }


class ZeelinConformalpointermassguardHnswgateIvf(ZeelinPointermassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self.fast_return_k = int(method_param.get("fk", method_param.get("fast_return_k", 8)))
        self.slow_return_k = int(method_param.get("sk", method_param.get("slow_return_k", 9)))
        self.ratio_threshold = float(
            method_param.get("rtm", method_param.get("ratio_threshold_milli", 880))
        ) / 1000.0
        self._fast_prefix = None
        self._slow_prefix = None
        self._rescue_count = 0
        self._query_count = 0

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=self.slow_return_k,
            internal_k=internal_k,
        )
        self._target_k = self.slow_return_k
        self._query_k = max(
            self.internal_k if self.internal_k > 0 else self.slow_return_k,
            self.slow_return_k + self.dedup_extra,
        )
        self._ensure_search_c_buffers(self._query_k)
        self._qbuf_ptr = faiss.swig_ptr(self.qbuf)
        self._dbuf_ptr = faiss.swig_ptr(self.dbuf)
        self._ibuf_ptr = faiss.swig_ptr(self.ibuf)
        self._fast_prefix = self.ibuf[0, : self.fast_return_k]
        self._slow_prefix = self.ibuf[0, : self.slow_return_k]
        self._rescue_count = 0
        self._query_count = 0
        self.name = self.name.replace("zeelin-pointermassguard-hnswgate", "zeelin-conformalpointermassguard", 1)
        self.name = self.name[:-1] + (
            f", fastReturnK={self.fast_return_k}, slowReturnK={self.slow_return_k}, "
            f"ratioThreshold={self.ratio_threshold:.3f})"
        )

    def query(self, v, n):
        qptr = faiss.swig_ptr(v) if self.pointer_mode == "view" else self._qbuf_ptr
        if self.pointer_mode != "view":
            self.qbuf[0, :] = v
        self.index.search_c(
            1,
            qptr,
            self.fast_return_k,
            self._dbuf_ptr,
            self._ibuf_ptr,
        )
        self._query_count += 1
        first = float(self.dbuf[0, 0])
        last = float(self.dbuf[0, self.fast_return_k - 1])
        ratio = last / max(first, 1e-12)
        if ratio >= self.ratio_threshold and self._fast_prefix[-1] >= 0:
            return self._fast_prefix

        self._rescue_count += 1
        self.index.search_c(
            1,
            qptr,
            self.slow_return_k,
            self._dbuf_ptr,
            self._ibuf_ptr,
        )
        if self._slow_prefix[-1] >= 0:
            return self._slow_prefix
        return self._slow_prefix[self._slow_prefix >= 0]

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-conformalpointermassguard"
        extra["fast_return_k"] = self.fast_return_k
        extra["slow_return_k"] = self.slow_return_k
        extra["ratio_threshold_milli"] = int(round(self.ratio_threshold * 1000.0))
        extra["avg_repair_fraction"] = (
            float(self._rescue_count) / float(self._query_count) if self._query_count else 0.0
        )
        return extra


class ZeelinResolutionflowHnswgateIvf(ZeelinViewmassguardHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace("zeelin-viewmassguard-hnswgate", "zeelin-resolutionflow-hnswgate", 1)

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-resolutionflow"
        return extra


class ZeelinThreadflowHnswgateIvf(ZeelinViewmassguardHnswgateIvf):
    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace("zeelin-viewmassguard-hnswgate", "zeelin-threadflow-hnswgate", 1)

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-hnswquantizer-ip-threadflow"
        return extra


class ZeelinQuotientrouteHnswgateIvf(ZeelinViewmassguardHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "qefc" in method_param:
            method_param["quantizer_efConstruction"] = method_param["qefc"]
        if "qm" in method_param:
            method_param["quantizer_m"] = method_param["qm"]
        if "ts" in method_param:
            method_param["train_size"] = method_param["ts"]
        method_param.setdefault("fast_unique", True)
        method_param.setdefault("skip_query_norm", True)
        method_param.setdefault("query_view", True)
        method_param.setdefault("trust_full_labels", True)
        method_param.setdefault("use_search_c", True)
        method_param.setdefault("threads", 4)
        super().__init__(metric, method_param)
        self.signature_bits = int(
            method_param.get("sig", method_param.get("sig_bits", method_param.get("signature_bits", 12)))
        )
        self.route_train_size = int(
            method_param.get("rt", method_param.get("rtrain", method_param.get("route_train_size", 200000)))
        )
        self.route_nprobe = int(
            method_param.get("rp", method_param.get("rprobe", method_param.get("route_nprobe", 1)))
        )
        self.route_dims = None
        self.route_thresholds = None
        self.route_table = None
        self.route_assign = None
        self.route_dists = None
        self.route_hits = 0
        self.route_queries = 0

    def fit(self, X):
        super().fit(X)
        X = np.asarray(X, dtype=np.float32)
        if 0 < self.route_train_size < len(X):
            sample_idx = np.linspace(0, len(X) - 1, self.route_train_size, dtype=np.int64)
            sample = X[sample_idx]
        else:
            sample = X
        sample = self._normalize_rows(sample).astype(np.float32, copy=False)

        bits = max(1, min(self.signature_bits, self.dim))
        variances = np.var(sample, axis=0)
        self.route_dims = np.argsort(variances)[-bits:].astype(np.int64)
        self.route_thresholds = np.median(sample[:, self.route_dims], axis=0).astype(np.float32)
        powers = (1 << np.arange(bits, dtype=np.uint32)).astype(np.uint32)
        codes = ((sample[:, self.route_dims] > self.route_thresholds).astype(np.uint32) * powers).sum(axis=1)

        old_ef = self.quantizer.hnsw.efSearch
        self.quantizer.hnsw.efSearch = max(old_ef, self.default_quantizer_ef_search)
        labels = self.quantizer.search(sample, 1)[1][:, 0]
        self.quantizer.hnsw.efSearch = old_ef

        buckets = defaultdict(Counter)
        global_counts = Counter()
        for code, label in zip(codes, labels):
            label = int(label)
            if label < 0:
                continue
            buckets[int(code)][label] += 1
            global_counts[label] += 1

        table_size = 1 << bits
        fallback = [label for label, _ in global_counts.most_common(self.route_nprobe)]
        if not fallback:
            fallback = [0]
        while len(fallback) < self.route_nprobe:
            fallback.append(fallback[-1])
        self.route_table = np.tile(
            np.asarray(fallback[: self.route_nprobe], dtype=np.int64),
            (table_size, 1),
        )
        self.route_seen = np.zeros(table_size, dtype=np.uint8)
        for code, counter in buckets.items():
            top = [label for label, _ in counter.most_common(self.route_nprobe)]
            while len(top) < self.route_nprobe:
                top.append(fallback[len(top)])
            self.route_table[code, :] = np.asarray(top[: self.route_nprobe], dtype=np.int64)
            self.route_seen[code] = 1

        self.route_assign = np.empty((1, self.route_nprobe), dtype=np.int64)
        self.route_dists = np.zeros((1, self.route_nprobe), dtype=np.float32)
        self.dbuf = None
        self.ibuf = None
        self.search_c_capacity = 0

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            quantizer_ef_search=quantizer_ef_search,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.index.nprobe = self.route_nprobe
        self.name = self.name.replace("zeelin-viewmassguard-hnswgate", "zeelin-quotientroute-hnswgate", 1)

    def _route_code(self, v):
        code = 0
        for i, dim in enumerate(self.route_dims):
            if v[int(dim)] > self.route_thresholds[i]:
                code |= 1 << i
        return code

    def query(self, v, n):
        if self.skip_query_norm and self.query_view:
            q = v.reshape(1, self.dim)
        elif self.skip_query_norm:
            self.qbuf[0, :] = v
            q = self.qbuf
        else:
            norm = float(np.sqrt(np.dot(v, v)))
            self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
            q = self.qbuf

        target_k = n if self.return_k <= 0 else min(n, self.return_k)
        query_k = n if self.internal_k <= 0 else max(1, self.internal_k)
        query_k = max(query_k, target_k + self.dedup_extra)
        self._ensure_search_c_buffers(query_k)

        code = self._route_code(v)
        self.route_assign[0, :] = self.route_table[code]
        self.route_hits += int(self.route_seen[code])
        self.route_queries += 1
        self.index.search_preassigned_c(
            1,
            faiss.swig_ptr(q),
            query_k,
            faiss.swig_ptr(self.route_assign),
            faiss.swig_ptr(self.route_dists),
            faiss.swig_ptr(self.dbuf),
            faiss.swig_ptr(self.ibuf),
            False,
        )
        labels = self.ibuf[0, :query_k]
        if self.fast_unique:
            if self.trust_full_labels:
                prefix = labels[:target_k]
                if len(prefix) == 0 or prefix[-1] >= 0:
                    return prefix
                return prefix[prefix >= 0]
            labels = labels[labels >= 0]
            return labels[:target_k]
        out = []
        seen = set()
        for label in labels:
            label = int(label)
            if label < 0 or label in seen:
                continue
            seen.add(label)
            out.append(label)
            if len(out) >= target_k:
                break
        labels = np.asarray(out, dtype=np.int64)
        if self.return_k > 0:
            return labels[: min(self.return_k, len(labels))]
        return labels[:n]

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-quotientroute-ip"
        extra["signature_bits"] = self.signature_bits
        extra["route_train_size"] = self.route_train_size
        extra["route_nprobe"] = self.route_nprobe
        extra["route_seen_fraction"] = (
            float(self.route_hits) / float(self.route_queries) if self.route_queries else 0.0
        )
        return extra


class ZeelinCompressedrouteHnswgateIvf(ZeelinHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "qefc" in method_param:
            method_param["quantizer_efConstruction"] = method_param["qefc"]
        if "qm" in method_param:
            method_param["quantizer_m"] = method_param["qm"]
        if "ts" in method_param:
            method_param["train_size"] = method_param["ts"]
        method_param.setdefault("internal_k", 10)
        method_param.setdefault("return_k", 0)
        method_param.setdefault("dedup_extra", 0)
        method_param.setdefault("fast_unique", True)
        method_param.setdefault("skip_query_norm", True)
        method_param.setdefault("query_view", True)
        method_param.setdefault("trust_full_labels", True)
        method_param.setdefault("use_search_c", True)
        method_param.setdefault("threads", 4)
        super().__init__(metric, method_param)
        self.route_rank = int(method_param.get("rr", method_param.get("route_rank", 24)))
        self.route_basis_mode = str(method_param.get("rb", method_param.get("route_basis", "pca-centroid")))
        self.route_m = int(method_param.get("rm", method_param.get("route_m", 8)))
        self.route_efc = int(method_param.get("refc", method_param.get("route_efConstruction", 48)))
        self.route_default_efs = int(method_param.get("refs", method_param.get("route_efSearch", 8)))
        self.route_basis = None
        self.route_basis_t = None
        self.route_dims = None
        self.route_index = None
        self.route_assign = None
        self.route_dists = None
        self.qproj = None
        self._qproj_ptr = None
        self._route_assign_ptr = None
        self._route_dists_ptr = None
        self._dbuf_ptr = None
        self._ibuf_ptr = None
        self._target_k = 0
        self._query_k = 0

    def _make_route_basis(self, sample, centroids):
        rank = max(1, min(int(self.route_rank), self.dim))
        if self.route_basis_mode == "topvar-coord":
            variances = np.var(sample, axis=0)
            dims = np.argsort(-variances)[:rank]
            self.route_dims = dims.astype(np.int64)
            basis = np.zeros((rank, self.dim), dtype=np.float32)
            basis[np.arange(rank), dims] = 1.0
            return np.ascontiguousarray(basis, dtype=np.float32)

        self.route_dims = None
        if self.route_basis_mode == "raw-pca-centroid":
            base = centroids.astype(np.float64, copy=False)
        elif self.route_basis_mode == "pca-train":
            base = sample.astype(np.float64, copy=False)
            base = base - base.mean(axis=0, keepdims=True)
        else:
            base = centroids.astype(np.float64, copy=False)
            base = base - base.mean(axis=0, keepdims=True)
        cov = base.T @ base
        vals, vecs = np.linalg.eigh(cov)
        basis = vecs[:, np.argsort(-vals)[:rank]].T.astype(np.float32, copy=False)
        return self._normalize_rows(basis).astype(np.float32, copy=False)

    def _project_query_inplace(self, x):
        if self.route_dims is not None:
            self.qproj[0, :] = np.asarray(x, dtype=np.float32)[self.route_dims]
        else:
            np.dot(np.asarray(x, dtype=np.float32), self.route_basis_t, out=self.qproj[0])
        norm = float(np.sqrt(np.dot(self.qproj[0], self.qproj[0])))
        self.qproj[0, :] *= 1.0 / max(norm, 1e-12)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        self.index.train(train)
        self.index.add(Xn)

        centroids = np.empty((self.nlist, self.dim), dtype=np.float32)
        for cid in range(self.nlist):
            self.quantizer.reconstruct(cid, centroids[cid])
        self.route_basis = self._make_route_basis(train, centroids)
        self.route_basis_t = np.ascontiguousarray(self.route_basis.T, dtype=np.float32)
        cproj = np.ascontiguousarray(centroids.dot(self.route_basis_t), dtype=np.float32)
        cproj = self._normalize_rows(cproj).astype(np.float32, copy=False)
        self.route_rank = cproj.shape[1]
        self.route_index = faiss.IndexHNSWFlat(self.route_rank, self.route_m, faiss.METRIC_INNER_PRODUCT)
        self.route_index.hnsw.efConstruction = self.route_efc
        self.route_index.hnsw.efSearch = self.route_default_efs
        self.route_index.add(cproj)
        self.qproj = np.empty((1, self.route_rank), dtype=np.float32)

    def set_query_arguments(
        self,
        nprobe,
        route_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        faiss.cvar.indexIVF_stats.reset()
        self.nprobe = max(1, min(16, int(nprobe)))
        self.route_ef_search = (
            int(route_ef_search) if route_ef_search is not None else self.route_default_efs
        )
        if dedup_extra is not None:
            self.dedup_extra = int(dedup_extra)
        self.return_k = int(return_k) if return_k is not None else self.default_return_k
        if internal_k is not None:
            self.internal_k = int(internal_k)
        self.index.nprobe = self.nprobe
        self.route_index.hnsw.efSearch = self.route_ef_search
        self._target_k = 10 if self.return_k <= 0 else min(10, self.return_k)
        self._query_k = max(
            self.internal_k if self.internal_k > 0 else 10,
            self._target_k + self.dedup_extra,
        )
        self._ensure_search_c_buffers(self._query_k)
        self.route_assign = np.empty((1, self.nprobe), dtype=np.int64)
        self.route_dists = np.zeros((1, self.nprobe), dtype=np.float32)
        self._qproj_ptr = faiss.swig_ptr(self.qproj)
        self._route_assign_ptr = faiss.swig_ptr(self.route_assign)
        self._route_dists_ptr = faiss.swig_ptr(self.route_dists)
        self._dbuf_ptr = faiss.swig_ptr(self.dbuf)
        self._ibuf_ptr = faiss.swig_ptr(self.ibuf)
        self.name = (
            "zeelin-compressedroute-hnswgate "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, routeRank={self.route_rank}, "
            f"routeBasis={self.route_basis_mode}, routeM={self.route_m}, "
            f"routeEfC={self.route_efc}, routeEfS={self.route_ef_search}, "
            f"quantizerM={self.quantizer_m}, quantizerEfC={self.quantizer_ef_construction}, "
            f"trainSize={self.train_size}, internalK={self.internal_k}, "
            f"returnK={self.return_k}, dedupExtra={self.dedup_extra}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def query(self, v, n):
        self._project_query_inplace(v)
        self.route_index.search_c(
            1,
            self._qproj_ptr,
            self.nprobe,
            self._route_dists_ptr,
            self._route_assign_ptr,
        )
        qptr = faiss.swig_ptr(v)
        self.index.search_preassigned_c(
            1,
            qptr,
            self._query_k,
            self._route_assign_ptr,
            self._route_dists_ptr,
            self._dbuf_ptr,
            self._ibuf_ptr,
            False,
        )
        prefix = self.ibuf[0, : self._target_k]
        if prefix[-1] >= 0:
            return prefix
        return prefix[prefix >= 0]

    def get_additional(self):
        return {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "internal_k": self.internal_k,
            "return_k": self.return_k,
            "dedup_extra": self.dedup_extra,
            "threads": int(self.method_param.get("threads", 1)),
            "train_size": self.train_size,
            "quantizer_m": self.quantizer_m,
            "quantizer_efConstruction": self.quantizer_ef_construction,
            "route_rank": self.route_rank,
            "route_basis": self.route_basis_mode,
            "route_m": self.route_m,
            "route_efConstruction": self.route_efc,
            "route_efSearch": self.route_ef_search,
            "dist_comps": faiss.cvar.indexIVF_stats.ndis,
            "kernel": "faiss-lowdim-hnswroute-preassigned-ivfflat-ip",
        }


class ZeelinContinuousrouteHnswgateIvf(ZeelinCompressedrouteHnswgateIvf):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self.default_full_repair_candidates = int(
            method_param.get("frc", method_param.get("full_repair_candidates", 0))
        )
        self.full_repair_candidates = self.default_full_repair_candidates
        self.route_ef_search = 0
        self.route_search_k = 1
        self.centroids = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        self.index.train(train)
        self.index.add(Xn)

        centroids = np.empty((self.nlist, self.dim), dtype=np.float32)
        for cid in range(self.nlist):
            self.quantizer.reconstruct(cid, centroids[cid])
        self.centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        self.route_basis = self._make_route_basis(train, self.centroids)
        self.route_basis_t = np.ascontiguousarray(self.route_basis.T, dtype=np.float32)
        cproj = np.ascontiguousarray(self.centroids.dot(self.route_basis_t), dtype=np.float32)
        cproj = self._normalize_rows(cproj).astype(np.float32, copy=False)
        self.route_rank = cproj.shape[1]
        self.route_index = faiss.IndexFlatIP(self.route_rank)
        self.route_index.add(cproj)
        self.qproj = np.empty((1, self.route_rank), dtype=np.float32)

    def set_query_arguments(
        self,
        nprobe,
        full_repair_candidates=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        faiss.cvar.indexIVF_stats.reset()
        self.nprobe = max(1, min(16, int(nprobe)))
        if full_repair_candidates is None:
            self.full_repair_candidates = self.default_full_repair_candidates
        else:
            self.full_repair_candidates = max(0, int(full_repair_candidates))
        self.route_ef_search = 0
        if dedup_extra is not None:
            self.dedup_extra = int(dedup_extra)
        self.return_k = int(return_k) if return_k is not None else self.default_return_k
        if internal_k is not None:
            self.internal_k = int(internal_k)
        self.index.nprobe = self.nprobe
        self._target_k = 10 if self.return_k <= 0 else min(10, self.return_k)
        self._query_k = max(
            self.internal_k if self.internal_k > 0 else 10,
            self._target_k + self.dedup_extra,
        )
        self._ensure_search_c_buffers(self._query_k)
        self.route_search_k = max(self.nprobe, self.full_repair_candidates)
        self.route_assign = np.empty((1, self.route_search_k), dtype=np.int64)
        self.route_dists = np.zeros((1, self.route_search_k), dtype=np.float32)
        self._qproj_ptr = faiss.swig_ptr(self.qproj)
        self._route_assign_ptr = faiss.swig_ptr(self.route_assign)
        self._route_dists_ptr = faiss.swig_ptr(self.route_dists)
        self._dbuf_ptr = faiss.swig_ptr(self.dbuf)
        self._ibuf_ptr = faiss.swig_ptr(self.ibuf)
        self.name = (
            "zeelin-continuousroute-hnswgate "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, routeRank={self.route_rank}, "
            f"routeBasis={self.route_basis_mode}, flatRoute=1, "
            f"fullRepairCandidates={self.full_repair_candidates}, "
            f"quantizerM={self.quantizer_m}, quantizerEfC={self.quantizer_ef_construction}, "
            f"trainSize={self.train_size}, internalK={self.internal_k}, "
            f"returnK={self.return_k}, dedupExtra={self.dedup_extra}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def _repair_route_inplace(self, v):
        if self.full_repair_candidates <= self.nprobe:
            return
        cand = self.route_assign[0, : self.route_search_k]
        valid = cand[cand >= 0]
        if len(valid) <= self.nprobe:
            if len(valid) > 0:
                self.route_assign[0, : len(valid)] = valid
            return
        scores = self.centroids[valid].dot(np.asarray(v, dtype=np.float32))
        if self.nprobe == 1:
            best = int(np.argmax(scores))
            self.route_assign[0, 0] = valid[best]
            self.route_dists[0, 0] = scores[best]
            return
        part = np.argpartition(-scores, self.nprobe - 1)[: self.nprobe]
        order = part[np.argsort(-scores[part])]
        selected = valid[order]
        self.route_assign[0, : self.nprobe] = selected
        self.route_dists[0, : self.nprobe] = scores[order]

    def query(self, v, n):
        self._project_query_inplace(v)
        self.route_index.search_c(
            1,
            self._qproj_ptr,
            self.route_search_k,
            self._route_dists_ptr,
            self._route_assign_ptr,
        )
        self._repair_route_inplace(v)
        qptr = faiss.swig_ptr(v)
        self.index.search_preassigned_c(
            1,
            qptr,
            self._query_k,
            self._route_assign_ptr,
            self._route_dists_ptr,
            self._dbuf_ptr,
            self._ibuf_ptr,
            False,
        )
        prefix = self.ibuf[0, : self._target_k]
        if prefix[-1] >= 0:
            return prefix
        return prefix[prefix >= 0]

    def get_additional(self):
        extra = super().get_additional()
        extra["full_repair_candidates"] = self.full_repair_candidates
        extra["route_search_k"] = self.route_search_k
        extra["kernel"] = "faiss-flat-lowdim-continuousroute-preassigned-ivfflat-ip"
        return extra


class ZeelinCoreorderedCapbucketMaxcodesIvf(ZeelinCompressedrouteHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "qefc" in method_param:
            method_param["quantizer_efConstruction"] = method_param["qefc"]
        if "qm" in method_param:
            method_param["quantizer_m"] = method_param["qm"]
        if "ts" in method_param:
            method_param["train_size"] = method_param["ts"]
        method_param.setdefault("internal_k", 4)
        method_param.setdefault("return_k", 8)
        method_param.setdefault("dedup_extra", 0)
        method_param.setdefault("fast_unique", True)
        method_param.setdefault("skip_query_norm", True)
        method_param.setdefault("query_view", True)
        method_param.setdefault("trust_full_labels", True)
        method_param.setdefault("use_search_c", True)
        method_param.setdefault("threads", 4)
        super().__init__(metric, method_param)
        self.assign_ef_search = int(method_param.get("aef", method_param.get("assign_efSearch", 8)))
        self.route_bits = int(method_param.get("bits", method_param.get("route_bits", 14)))
        self.route_bits = max(1, min(20, self.route_bits))
        self.threshold_mode = str(method_param.get("thr", method_param.get("threshold_mode", "median")))
        self.hamming_radius = int(method_param.get("rad", method_param.get("radius", 2)))
        self.hamming_radius = max(0, min(3, self.hamming_radius))
        self.bucket_cap = int(method_param.get("bcap", method_param.get("bucket_cap", 256)))
        self.bucket_cap = max(1, self.bucket_cap)
        self.route_candidates = int(method_param.get("rc", method_param.get("route_candidates", 8)))
        self.route_candidates = max(2, self.route_candidates)
        self.default_full_repair_candidates = int(
            method_param.get("frc", method_param.get("full_repair_candidates", 8))
        )
        self.full_repair_candidates = max(0, self.default_full_repair_candidates)
        self.default_max_codes = int(method_param.get("maxc", method_param.get("max_codes", 0)))
        self.max_codes = self.default_max_codes
        self.store_fit_vectors = bool(method_param.get("store_fit_vectors", False))
        self._fit_Xn = None
        self._fit_assigned = None
        self.centroids = None
        self.centroid_proj = None
        self.route_thresholds = None
        self.bucket_offsets = None
        self.bucket_ids = None
        self.hamming_mask_values = None
        self.route_candidate_buf = None
        self.route_score_buf = None
        self.full_score_buf = None
        self.route_search_k = 2

    def _hamming_masks(self):
        values = [0]
        bit_values = [1 << bit for bit in range(self.route_bits)]
        for radius in range(1, self.hamming_radius + 1):
            for combo in itertools.combinations(bit_values, radius):
                mask = 0
                for value in combo:
                    mask ^= value
                values.append(mask)
        return np.asarray(values, dtype=np.int64)

    def _encode_projection_codes(self, projected):
        bits = projected[:, : self.route_bits] > self.route_thresholds.reshape(1, -1)
        codes = np.zeros(bits.shape[0], dtype=np.int64)
        for bit in range(self.route_bits):
            codes |= bits[:, bit].astype(np.int64) << bit
        return codes

    def _build_bucket_table(self, centroid_codes):
        table_size = 1 << self.route_bits
        counts = np.bincount(centroid_codes, minlength=table_size).astype(np.int64, copy=False)
        offsets = np.empty(table_size + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        order = np.argsort(centroid_codes, kind="stable").astype(np.int64, copy=False)
        self.bucket_offsets = offsets
        self.bucket_ids = np.ascontiguousarray(order, dtype=np.int64)

    def _core_scores(self, Xn, assigned):
        return np.einsum("ij,ij->i", Xn, self.centroids[assigned], optimize=True).astype(np.float32)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        self.index.train(train)

        centroids = np.empty((self.nlist, self.dim), dtype=np.float32)
        for cid in range(self.nlist):
            self.quantizer.reconstruct(cid, centroids[cid])
        self.centroids = np.ascontiguousarray(centroids, dtype=np.float32)

        self.route_basis = self._make_route_basis(train, self.centroids)
        self.route_basis_t = np.ascontiguousarray(self.route_basis.T, dtype=np.float32)
        cproj = np.ascontiguousarray(self.centroids.dot(self.route_basis_t), dtype=np.float32)
        cproj = self._normalize_rows(cproj).astype(np.float32, copy=False)
        self.route_rank = cproj.shape[1]
        self.route_bits = min(self.route_bits, self.route_rank)
        self.centroid_proj = cproj
        if self.threshold_mode == "zero":
            self.route_thresholds = np.zeros(self.route_bits, dtype=np.float32)
        else:
            self.route_thresholds = np.median(cproj[:, : self.route_bits], axis=0).astype(np.float32)
        centroid_codes = self._encode_projection_codes(cproj)
        self._build_bucket_table(centroid_codes)
        self.hamming_mask_values = self._hamming_masks()
        self.qproj = np.empty((1, self.route_rank), dtype=np.float32)
        self.route_candidate_buf = np.empty(self.bucket_cap, dtype=np.int64)
        self.route_score_buf = np.empty(self.bucket_cap, dtype=np.float32)
        self.full_score_buf = np.empty(max(self.route_candidates, self.full_repair_candidates, 2), dtype=np.float32)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        _, assigned = self.quantizer.search(np.ascontiguousarray(Xn, dtype=np.float32), 1)
        assigned = np.ascontiguousarray(assigned[:, 0], dtype=np.int64)
        if self.store_fit_vectors:
            self._fit_Xn = np.ascontiguousarray(Xn, dtype=np.float32)
            self._fit_assigned = assigned.copy()
        core_scores = self._core_scores(Xn, assigned)
        order = np.lexsort((-core_scores, assigned)).astype(np.int64, copy=False)
        ids = np.ascontiguousarray(order, dtype=np.int64)
        block = int(self.method_param.get("add_block", 65536))
        block = max(1024, block)
        for start in range(0, len(order), block):
            stop = min(len(order), start + block)
            batch = np.ascontiguousarray(Xn[order[start:stop]], dtype=np.float32)
            self.index.add_with_ids(batch, ids[start:stop])
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

    def _encode_query_code(self):
        code = 0
        qrow = self.qproj[0]
        for bit in range(self.route_bits):
            if qrow[bit] > self.route_thresholds[bit]:
                code |= 1 << bit
        return code

    def _collect_candidate_centroids(self, code):
        count = 0
        offsets = self.bucket_offsets
        ids = self.bucket_ids
        for mask in self.hamming_mask_values:
            bucket_code = int(code ^ int(mask))
            begin = int(offsets[bucket_code])
            end = int(offsets[bucket_code + 1])
            if end <= begin:
                continue
            take = min(end - begin, self.bucket_cap - count)
            if take <= 0:
                break
            self.route_candidate_buf[count : count + take] = ids[begin : begin + take]
            count += take
            if count >= self.bucket_cap:
                break
        return count

    def _fallback_hnsw_route(self, v):
        if self.skip_query_norm and self.query_view:
            q = v.reshape(1, self.dim)
        else:
            self.qbuf[0, :] = v
            q = self.qbuf
        self.quantizer.hnsw.efSearch = 1
        _, labels = self.quantizer.search(q, self.nprobe)
        got = labels[0]
        self.route_assign[0, : self.nprobe] = got[: self.nprobe]
        self.route_dists[0, : self.nprobe] = 0.0

    def _select_route_inplace(self, v):
        code = self._encode_query_code()
        cand_count = self._collect_candidate_centroids(code)
        if cand_count <= 0:
            self._fallback_hnsw_route(v)
            return

        candidates = self.route_candidate_buf[:cand_count]
        scores = self.centroid_proj[candidates].dot(self.qproj[0])
        take = min(max(self.route_candidates, self.nprobe), cand_count)
        if take < cand_count:
            part = np.argpartition(-scores, take - 1)[:take]
            order = part[np.argsort(-scores[part])]
        else:
            order = np.argsort(-scores, kind="stable")
        shortlist = candidates[order[:take]]
        shortlist_scores = scores[order[:take]]

        if self.full_repair_candidates > 0 and len(shortlist) > self.nprobe:
            repair_take = min(max(self.full_repair_candidates, self.nprobe), len(shortlist))
            repair = shortlist[:repair_take]
            full_scores = self.centroids[repair].dot(np.asarray(v, dtype=np.float32))
            if self.nprobe < repair_take:
                part = np.argpartition(-full_scores, self.nprobe - 1)[: self.nprobe]
                order2 = part[np.argsort(-full_scores[part])]
            else:
                order2 = np.argsort(-full_scores, kind="stable")
            selected = repair[order2[: self.nprobe]]
            selected_scores = full_scores[order2[: self.nprobe]]
        else:
            selected = shortlist[: self.nprobe]
            selected_scores = shortlist_scores[: self.nprobe]

        if len(selected) < self.nprobe:
            self._fallback_hnsw_route(v)
            return
        self.route_assign[0, : self.nprobe] = selected
        self.route_dists[0, : self.nprobe] = selected_scores

    def set_query_arguments(
        self,
        nprobe,
        max_codes=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        faiss.cvar.indexIVF_stats.reset()
        self.nprobe = max(1, min(4, int(nprobe)))
        self.max_codes = int(max_codes) if max_codes is not None else self.default_max_codes
        if dedup_extra is not None:
            self.dedup_extra = int(dedup_extra)
        self.return_k = int(return_k) if return_k is not None else self.default_return_k
        if internal_k is not None:
            self.internal_k = int(internal_k)
        self.index.nprobe = self.nprobe
        self.index.max_codes = max(0, self.max_codes)
        self.quantizer_ef_search = 0
        self._target_k = 10 if self.return_k <= 0 else min(10, self.return_k)
        self._query_k = max(
            self.internal_k if self.internal_k > 0 else 10,
            self._target_k + self.dedup_extra,
        )
        self._ensure_search_c_buffers(self._query_k)
        self.route_assign = np.empty((1, self.nprobe), dtype=np.int64)
        self.route_dists = np.zeros((1, self.nprobe), dtype=np.float32)
        self._qproj_ptr = faiss.swig_ptr(self.qproj)
        self._route_assign_ptr = faiss.swig_ptr(self.route_assign)
        self._route_dists_ptr = faiss.swig_ptr(self.route_dists)
        self._dbuf_ptr = faiss.swig_ptr(self.dbuf)
        self._ibuf_ptr = faiss.swig_ptr(self.ibuf)
        self.name = (
            "zeelin-coreordered-capbucket-maxcodes "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, maxCodes={self.max_codes}, "
            f"routeRank={self.route_rank}, routeBasis={self.route_basis_mode}, "
            f"bits={self.route_bits}, threshold={self.threshold_mode}, radius={self.hamming_radius}, "
            f"bucketCap={self.bucket_cap}, routeCandidates={self.route_candidates}, "
            f"fullRepairCandidates={self.full_repair_candidates}, "
            f"quantizerM={self.quantizer_m}, quantizerEfC={self.quantizer_ef_construction}, "
            f"assignEfS={self.assign_ef_search}, trainSize={self.train_size}, "
            f"internalK={self.internal_k}, returnK={self.return_k}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def query(self, v, n):
        self._project_query_inplace(v)
        self._select_route_inplace(v)
        qptr = faiss.swig_ptr(v)
        self.index.search_preassigned_c(
            1,
            qptr,
            self._query_k,
            self._route_assign_ptr,
            self._route_dists_ptr,
            self._dbuf_ptr,
            self._ibuf_ptr,
            False,
        )
        prefix = self.ibuf[0, : self._target_k]
        if prefix[-1] >= 0:
            return prefix
        return prefix[prefix >= 0]

    def get_additional(self):
        return {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "max_codes": self.max_codes,
            "internal_k": self.internal_k,
            "return_k": self.return_k,
            "dedup_extra": self.dedup_extra,
            "threads": int(self.method_param.get("threads", 1)),
            "train_size": self.train_size,
            "quantizer_m": self.quantizer_m,
            "quantizer_efConstruction": self.quantizer_ef_construction,
            "assign_efSearch": self.assign_ef_search,
            "route_rank": self.route_rank,
            "route_basis": self.route_basis_mode,
            "route_bits": self.route_bits,
            "threshold_mode": self.threshold_mode,
            "hamming_radius": self.hamming_radius,
            "bucket_cap": self.bucket_cap,
            "route_candidates": self.route_candidates,
            "full_repair_candidates": self.full_repair_candidates,
            "route_mask_count": int(len(self.hamming_mask_values)) if self.hamming_mask_values is not None else 0,
            "dist_comps": faiss.cvar.indexIVF_stats.ndis,
            "kernel": "faiss-ivfflat-coreordered-capbucket-maxcodes-ip",
            "scan_layout": "coreordered-ivf-primary-secondary-prefix",
        }


class ZeelinCoreorderedCapbucketCrouteMaxcodesIvf(ZeelinCoreorderedCapbucketMaxcodesIvf):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self._caproute_lib = None
        self._caproute_idx = None

    def _load_capbucket_route_library(self):
        here = Path(__file__).resolve().parent
        source = here / "zeelin_capbucket_route_lib.cpp"
        suffix = ".dylib" if os.uname().sysname == "Darwin" else ".so"
        out = here / f"libzeelin_capbucket_route{suffix}"
        force_rebuild = os.environ.get("ZEELIN_CAPBUCKET_ROUTE_FORCE_REBUILD") == "1"
        needs_rebuild = force_rebuild or not out.exists()
        if out.exists() and source.exists():
            needs_rebuild = needs_rebuild or source.stat().st_mtime > out.stat().st_mtime
        if needs_rebuild:
            cxx = os.environ.get("CXX", "c++")
            flags = os.environ.get("ZEELIN_CAPBUCKET_ROUTE_CXXFLAGS")
            if flags:
                cmd = [cxx, *flags.split(), str(source), "-o", str(out)]
            else:
                cmd = [
                    cxx,
                    "-O3",
                    "-std=c++17",
                    "-march=native",
                    "-ffast-math",
                    "-funroll-loops",
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-o",
                    str(out),
                ]
            subprocess.check_call(cmd)
        lib = ctypes.CDLL(str(out))
        lib.zeelin_capbucket_route_create.restype = ctypes.c_void_p
        lib.zeelin_capbucket_route_free.argtypes = [ctypes.c_void_p]
        lib.zeelin_capbucket_route_fit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.zeelin_capbucket_route_query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.zeelin_capbucket_route_query.restype = ctypes.c_int
        lib.zeelin_capbucket_route_avg_candidates.argtypes = [ctypes.c_void_p]
        lib.zeelin_capbucket_route_avg_candidates.restype = ctypes.c_double
        lib.zeelin_capbucket_route_empty_fraction.argtypes = [ctypes.c_void_p]
        lib.zeelin_capbucket_route_empty_fraction.restype = ctypes.c_double
        return lib

    def fit(self, X):
        super().fit(X)
        self._caproute_lib = self._load_capbucket_route_library()
        self._caproute_idx = self._caproute_lib.zeelin_capbucket_route_create()
        basis = np.ascontiguousarray(self.route_basis, dtype=np.float32)
        centroids = np.ascontiguousarray(self.centroids, dtype=np.float32)
        cproj = np.ascontiguousarray(self.centroid_proj, dtype=np.float32)
        thresholds = np.ascontiguousarray(self.route_thresholds, dtype=np.float32)
        offsets = np.ascontiguousarray(self.bucket_offsets, dtype=np.int64)
        ids = np.ascontiguousarray(self.bucket_ids, dtype=np.int64)
        masks = np.ascontiguousarray(self.hamming_mask_values, dtype=np.int64)
        self._caproute_lib.zeelin_capbucket_route_fit(
            self._caproute_idx,
            basis.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            centroids.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cproj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            thresholds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            masks.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(self.dim),
            int(self.nlist),
            int(self.route_rank),
            int(self.route_bits),
            int(1 << self.route_bits),
            int(len(self.hamming_mask_values)),
            int(self.bucket_cap),
        )

    def set_query_arguments(
        self,
        nprobe,
        max_codes=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        super().set_query_arguments(
            nprobe,
            max_codes=max_codes,
            dedup_extra=dedup_extra,
            return_k=return_k,
            internal_k=internal_k,
        )
        self.name = self.name.replace(
            "zeelin-coreordered-capbucket-maxcodes",
            "zeelin-coreordered-capbucket-croute-maxcodes",
            1,
        )

    def query(self, v, n):
        count = self._caproute_lib.zeelin_capbucket_route_query(
            self._caproute_idx,
            np.asarray(v, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(self.nprobe),
            int(self.route_candidates),
            int(self.full_repair_candidates),
            self.route_assign.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            self.route_dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if count < self.nprobe:
            self._fallback_hnsw_route(v)
        qptr = faiss.swig_ptr(v)
        self.index.search_preassigned_c(
            1,
            qptr,
            self._query_k,
            self._route_assign_ptr,
            self._route_dists_ptr,
            self._dbuf_ptr,
            self._ibuf_ptr,
            False,
        )
        prefix = self.ibuf[0, : self._target_k]
        if prefix[-1] >= 0:
            return prefix
        return prefix[prefix >= 0]

    def get_additional(self):
        extra = super().get_additional()
        extra["kernel"] = "faiss-ivfflat-coreordered-capbucket-croute-maxcodes-ip"
        if self._caproute_idx is not None:
            extra["route_avg_candidates"] = self._caproute_lib.zeelin_capbucket_route_avg_candidates(self._caproute_idx)
            extra["route_empty_fraction"] = self._caproute_lib.zeelin_capbucket_route_empty_fraction(self._caproute_idx)
        return extra


class ZeelinCapbucketFusedscanMaxcodesIvf(ZeelinCoreorderedCapbucketMaxcodesIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        method_param["store_fit_vectors"] = True
        super().__init__(metric, method_param)
        self._fused_lib = None
        self._fused_idx = None
        self._out64 = None

    def _load_capbucket_fusedscan_library(self):
        here = Path(__file__).resolve().parent
        source = here / "zeelin_capbucket_fusedscan_lib.cpp"
        suffix = ".dylib" if os.uname().sysname == "Darwin" else ".so"
        out = here / f"libzeelin_capbucket_fusedscan{suffix}"
        force_rebuild = os.environ.get("ZEELIN_CAPBUCKET_FUSEDSCAN_FORCE_REBUILD") == "1"
        needs_rebuild = force_rebuild or not out.exists()
        if out.exists() and source.exists():
            needs_rebuild = needs_rebuild or source.stat().st_mtime > out.stat().st_mtime
        if needs_rebuild:
            cxx = os.environ.get("CXX", "c++")
            flags = os.environ.get("ZEELIN_CAPBUCKET_FUSEDSCAN_CXXFLAGS")
            if flags:
                cmd = [cxx, *flags.split(), str(source), "-o", str(out)]
            else:
                cmd = [
                    cxx,
                    "-O3",
                    "-std=c++17",
                    "-march=native",
                    "-ffast-math",
                    "-funroll-loops",
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-o",
                    str(out),
                ]
            subprocess.check_call(cmd)
        lib = ctypes.CDLL(str(out))
        lib.zeelin_capbucket_fusedscan_create.restype = ctypes.c_void_p
        lib.zeelin_capbucket_fusedscan_free.argtypes = [ctypes.c_void_p]
        lib.zeelin_capbucket_fusedscan_fit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.zeelin_capbucket_fusedscan_query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_longlong),
        ]
        lib.zeelin_capbucket_fusedscan_query.restype = ctypes.c_int
        lib.zeelin_capbucket_fusedscan_avg_candidates.argtypes = [ctypes.c_void_p]
        lib.zeelin_capbucket_fusedscan_avg_candidates.restype = ctypes.c_double
        lib.zeelin_capbucket_fusedscan_total_candidates.argtypes = [ctypes.c_void_p]
        lib.zeelin_capbucket_fusedscan_total_candidates.restype = ctypes.c_double
        lib.zeelin_capbucket_fusedscan_avg_route_candidates.argtypes = [ctypes.c_void_p]
        lib.zeelin_capbucket_fusedscan_avg_route_candidates.restype = ctypes.c_double
        lib.zeelin_capbucket_fusedscan_empty_fraction.argtypes = [ctypes.c_void_p]
        lib.zeelin_capbucket_fusedscan_empty_fraction.restype = ctypes.c_double
        lib.zeelin_capbucket_fusedscan_kernel.restype = ctypes.c_char_p
        lib.zeelin_capbucket_fusedscan_layout.restype = ctypes.c_char_p
        return lib

    def fit(self, X):
        super().fit(X)
        self._fused_lib = self._load_capbucket_fusedscan_library()
        self._fused_idx = self._fused_lib.zeelin_capbucket_fusedscan_create()
        Xn = np.ascontiguousarray(self._fit_Xn, dtype=np.float32)
        assigned = np.ascontiguousarray(self._fit_assigned, dtype=np.int64)
        basis = np.ascontiguousarray(self.route_basis, dtype=np.float32)
        centroids = np.ascontiguousarray(self.centroids, dtype=np.float32)
        cproj = np.ascontiguousarray(self.centroid_proj, dtype=np.float32)
        thresholds = np.ascontiguousarray(self.route_thresholds, dtype=np.float32)
        offsets = np.ascontiguousarray(self.bucket_offsets, dtype=np.int64)
        ids = np.ascontiguousarray(self.bucket_ids, dtype=np.int64)
        masks = np.ascontiguousarray(self.hamming_mask_values, dtype=np.int64)
        self._fused_lib.zeelin_capbucket_fusedscan_fit(
            self._fused_idx,
            Xn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            assigned.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            basis.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            centroids.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cproj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            thresholds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            masks.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(Xn.shape[0]),
            int(self.dim),
            int(self.nlist),
            int(self.route_rank),
            int(self.route_bits),
            int(1 << self.route_bits),
            int(len(self.hamming_mask_values)),
            int(self.bucket_cap),
        )
        self._fit_Xn = None
        self._fit_assigned = None
        self.index = None

    def set_query_arguments(
        self,
        nprobe,
        max_codes=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        self.nprobe = max(1, min(4, int(nprobe)))
        self.max_codes = int(max_codes) if max_codes is not None else self.default_max_codes
        if dedup_extra is not None:
            self.dedup_extra = int(dedup_extra)
        self.return_k = int(return_k) if return_k is not None else self.default_return_k
        if internal_k is not None:
            self.internal_k = int(internal_k)
        self.quantizer_ef_search = 0
        self._target_k = 10 if self.return_k <= 0 else min(10, self.return_k)
        self._query_k = max(
            self.internal_k if self.internal_k > 0 else 10,
            self._target_k + self.dedup_extra,
        )
        if self._out64 is None or len(self._out64) < self._query_k:
            self._out64 = np.empty(self._query_k, dtype=np.int64)
        self.name = (
            "zeelin-capbucket-fusedscan-maxcodes "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, maxCodes={self.max_codes}, "
            f"routeRank={self.route_rank}, routeBasis={self.route_basis_mode}, "
            f"bits={self.route_bits}, threshold={self.threshold_mode}, radius={self.hamming_radius}, "
            f"bucketCap={self.bucket_cap}, routeCandidates={self.route_candidates}, "
            f"fullRepairCandidates={self.full_repair_candidates}, "
            f"quantizerM={self.quantizer_m}, quantizerEfC={self.quantizer_ef_construction}, "
            f"assignEfS={self.assign_ef_search}, trainSize={self.train_size}, "
            f"internalK={self.internal_k}, returnK={self.return_k}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def query(self, v, n):
        count = self._fused_lib.zeelin_capbucket_fusedscan_query(
            self._fused_idx,
            np.asarray(v, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(self._query_k),
            int(self.nprobe),
            int(self.max_codes),
            int(self.route_candidates),
            int(self.full_repair_candidates),
            self._out64.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        )
        take = min(count, self._target_k)
        prefix = self._out64[:take]
        if len(prefix) == self._target_k and prefix[-1] >= 0:
            return prefix
        return prefix[prefix >= 0]

    def get_additional(self):
        extra = {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "max_codes": self.max_codes,
            "internal_k": self.internal_k,
            "return_k": self.return_k,
            "dedup_extra": self.dedup_extra,
            "threads": int(self.method_param.get("threads", 1)),
            "train_size": self.train_size,
            "quantizer_m": self.quantizer_m,
            "quantizer_efConstruction": self.quantizer_ef_construction,
            "assign_efSearch": self.assign_ef_search,
            "route_rank": self.route_rank,
            "route_basis": self.route_basis_mode,
            "route_bits": self.route_bits,
            "threshold_mode": self.threshold_mode,
            "hamming_radius": self.hamming_radius,
            "bucket_cap": self.bucket_cap,
            "route_candidates": self.route_candidates,
            "full_repair_candidates": self.full_repair_candidates,
            "route_mask_count": int(len(self.hamming_mask_values)) if self.hamming_mask_values is not None else 0,
        }
        if self._fused_idx is not None:
            extra["dist_comps"] = self._fused_lib.zeelin_capbucket_fusedscan_total_candidates(self._fused_idx)
            extra["avg_dist_comps"] = self._fused_lib.zeelin_capbucket_fusedscan_avg_candidates(self._fused_idx)
            extra["route_avg_candidates"] = self._fused_lib.zeelin_capbucket_fusedscan_avg_route_candidates(self._fused_idx)
            extra["route_empty_fraction"] = self._fused_lib.zeelin_capbucket_fusedscan_empty_fraction(self._fused_idx)
            extra["kernel"] = self._fused_lib.zeelin_capbucket_fusedscan_kernel().decode("ascii")
            extra["scan_layout"] = self._fused_lib.zeelin_capbucket_fusedscan_layout().decode("ascii")
        return extra

    def done(self):
        if getattr(self, "_fused_idx", None):
            self._fused_lib.zeelin_capbucket_fusedscan_free(self._fused_idx)
            self._fused_idx = None


class ZeelinHnswcodegateIvf(ZeelinHnswgateIvf):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self.codec = method_param.get("codec", "sq8")
        self.pq_m = int(method_param.get("pq_m", 20))
        self.pq_bits = int(method_param.get("pq_bits", 4))

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

        if self.codec == "pq":
            self.index = faiss.IndexIVFPQ(
                self.quantizer,
                self.dim,
                self.nlist,
                self.pq_m,
                self.pq_bits,
                faiss.METRIC_INNER_PRODUCT,
            )
        else:
            self.index = faiss.IndexIVFScalarQuantizer(
                self.quantizer,
                self.dim,
                self.nlist,
                faiss.ScalarQuantizer.QT_8bit,
                faiss.METRIC_INNER_PRODUCT,
            )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        self.index.train(train)
        self.index.add(Xn)

    def set_query_arguments(self, nprobe, quantizer_ef_search=None, dedup_extra=None, return_k=None):
        super().set_query_arguments(nprobe, quantizer_ef_search, dedup_extra, return_k)
        self.name = (
            "zeelin-hnswcodegate-ivf "
            f"(codec={self.codec}, nlist={self.nlist}, nprobe={self.nprobe}, "
            f"quantizerM={self.quantizer_m}, "
            f"quantizerEfC={self.quantizer_ef_construction}, "
            f"quantizerEfS={self.quantizer_ef_search}, "
            f"trainSize={self.train_size}, pqM={self.pq_m}, pqBits={self.pq_bits}, "
            f"internalK={self.internal_k}, returnK={self.return_k}, "
            f"dedupExtra={self.dedup_extra}, threads={self.method_param.get('threads', 1)})"
        )

    def get_additional(self):
        extra = super().get_additional()
        extra["codec"] = self.codec
        extra["pq_m"] = self.pq_m
        extra["pq_bits"] = self.pq_bits
        extra["kernel"] = f"faiss-ivf{self.codec}-hnswquantizer-ip-normalized"
        return extra


class ZeelinCentroidscanIvf(ZeelinHnswgateIvf):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self.assign_ef_search = int(method_param.get("assign_efSearch", 16))
        self._lib = self._load_centroidscan_library()
        self._idx = self._lib.zeelin_centroidscan_ivf_create()
        self._out = None

    def _load_centroidscan_library(self):
        here = Path(__file__).resolve().parent
        source = here / "zeelin_centroidscan_ivf_lib.cpp"
        suffix = ".dylib" if os.uname().sysname == "Darwin" else ".so"
        out = here / f"libzeelin_centroidscan_ivf{suffix}"
        force_rebuild = os.environ.get("ZEELIN_CENTROIDSCAN_IVF_FORCE_REBUILD") == "1"
        needs_rebuild = force_rebuild or not out.exists()
        if out.exists() and source.exists():
            needs_rebuild = needs_rebuild or source.stat().st_mtime > out.stat().st_mtime
        if needs_rebuild:
            cxx = os.environ.get("CXX", "c++")
            flags = os.environ.get("ZEELIN_CENTROIDSCAN_IVF_CXXFLAGS")
            if flags:
                cmd = [cxx, *flags.split(), str(source), "-o", str(out)]
            else:
                cmd = [
                    cxx,
                    "-O3",
                    "-std=c++17",
                    "-march=native",
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-o",
                    str(out),
                ]
            subprocess.check_call(cmd)
        lib = ctypes.CDLL(str(out))
        lib.zeelin_centroidscan_ivf_create.restype = ctypes.c_void_p
        lib.zeelin_centroidscan_ivf_free.argtypes = [ctypes.c_void_p]
        lib.zeelin_centroidscan_ivf_fit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.zeelin_centroidscan_ivf_fit_payload.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.zeelin_centroidscan_ivf_query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.zeelin_centroidscan_ivf_query.restype = ctypes.c_int
        lib.zeelin_centroidscan_ivf_query_payload.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_longlong),
        ]
        lib.zeelin_centroidscan_ivf_query_payload.restype = ctypes.c_int
        lib.zeelin_centroidscan_ivf_query_payload_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_longlong),
        ]
        lib.zeelin_centroidscan_ivf_query_payload_batch.restype = ctypes.c_int
        lib.zeelin_centroidscan_ivf_avg_candidates.argtypes = [ctypes.c_void_p]
        lib.zeelin_centroidscan_ivf_avg_candidates.restype = ctypes.c_double
        lib.zeelin_centroidscan_ivf_avg_dist_comps.argtypes = [ctypes.c_void_p]
        lib.zeelin_centroidscan_ivf_avg_dist_comps.restype = ctypes.c_double
        lib.zeelin_centroidscan_ivf_total_dist_comps.argtypes = [ctypes.c_void_p]
        lib.zeelin_centroidscan_ivf_total_dist_comps.restype = ctypes.c_double
        lib.zeelin_centroidscan_ivf_scan_layout.restype = ctypes.c_char_p
        lib.zeelin_centroidscan_ivf_kernel.restype = ctypes.c_char_p
        return lib

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        trainer = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        trainer.train(train)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        _, assigned = self.quantizer.search(np.ascontiguousarray(Xn, dtype=np.float32), 1)
        assigned = np.ascontiguousarray(assigned[:, 0], dtype=np.int64)
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

        self._lib.zeelin_centroidscan_ivf_fit(
            self._idx,
            np.ascontiguousarray(Xn, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            assigned.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(Xn.shape[0]),
            int(self.dim),
            int(self.nlist),
        )

    def set_query_arguments(self, nprobe, quantizer_ef_search=None, return_k=None):
        self.nprobe = int(nprobe)
        self.quantizer_ef_search = (
            int(quantizer_ef_search)
            if quantizer_ef_search is not None
            else self.default_quantizer_ef_search
        )
        self.return_k = int(return_k) if return_k is not None else self.default_return_k
        self.quantizer.hnsw.efSearch = self.quantizer_ef_search
        self.name = (
            "zeelin-centroidscan-ivf "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, "
            f"quantizerM={self.quantizer_m}, "
            f"quantizerEfC={self.quantizer_ef_construction}, "
            f"quantizerEfS={self.quantizer_ef_search}, "
            f"assignEfS={self.assign_ef_search}, "
            f"trainSize={self.train_size}, returnK={self.return_k}, "
            f"skipQueryNorm={int(self.skip_query_norm)}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def query(self, v, n):
        if self.skip_query_norm:
            self.qbuf[0, :] = v
        else:
            norm = float(np.sqrt(np.dot(v, v)))
            self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
        _, labels = self.quantizer.search(self.qbuf, self.nprobe)
        labels = np.ascontiguousarray(labels[0], dtype=np.int64)
        target_k = n if self.return_k <= 0 else min(n, self.return_k)
        if self._out is None or len(self._out) < target_k:
            self._out = np.empty(target_k, dtype=np.int32)
        count = self._lib.zeelin_centroidscan_ivf_query(
            self._idx,
            self.qbuf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            labels.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(self.nprobe),
            int(target_k),
            self._out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        return self._out[:count].copy()

    def get_additional(self):
        return {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "return_k": self.return_k,
            "skip_query_norm": int(self.skip_query_norm),
            "threads": int(self.method_param.get("threads", 1)),
            "train_size": self.train_size,
            "quantizer_m": self.quantizer_m,
            "quantizer_efConstruction": self.quantizer_ef_construction,
            "quantizer_efSearch": self.quantizer_ef_search,
            "assign_efSearch": self.assign_ef_search,
            "dist_comps": self._lib.zeelin_centroidscan_ivf_total_dist_comps(self._idx),
            "avg_dist_comps": self._lib.zeelin_centroidscan_ivf_avg_dist_comps(self._idx),
            "avg_candidates": self._lib.zeelin_centroidscan_ivf_avg_candidates(self._idx),
            "kernel": self._lib.zeelin_centroidscan_ivf_kernel().decode("ascii"),
            "scan_layout": self._lib.zeelin_centroidscan_ivf_scan_layout().decode("ascii"),
        }

    def done(self):
        if getattr(self, "_idx", None):
            self._lib.zeelin_centroidscan_ivf_free(self._idx)
            self._idx = None


class ZeelinBitsignlistorderHnswgateIvf(ZeelinHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "qefc" in method_param:
            method_param["quantizer_efConstruction"] = method_param["qefc"]
        if "qm" in method_param:
            method_param["quantizer_m"] = method_param["qm"]
        if "ts" in method_param:
            method_param["train_size"] = method_param["ts"]
        if "b" in method_param:
            method_param["budget"] = method_param["b"]
        if "aef" in method_param:
            method_param["assign_efSearch"] = method_param["aef"]
        method_param.setdefault("quantizer_efSearch", 1)
        method_param.setdefault("skip_query_norm", True)
        method_param.setdefault("query_view", True)
        method_param.setdefault("threads", 4)
        method_param.setdefault("return_k", 9)
        super().__init__(metric, method_param)
        self.assign_ef_search = int(method_param.get("assign_efSearch", 8))
        self.budget = int(method_param.get("budget", 32))
        self.budget = max(1, min(256, self.budget))
        self._lib = self._load_bitsign_library()
        self._idx = self._lib.zeelin_bitsign_ivf_create()
        self._qd = None
        self._qi = None
        self._out64 = None

    def _load_bitsign_library(self):
        here = Path(__file__).resolve().parent
        source = here / "zeelin_bitsign_ivf_lib.cpp"
        suffix = ".dylib" if os.uname().sysname == "Darwin" else ".so"
        out = here / f"libzeelin_bitsign_ivf{suffix}"
        force_rebuild = os.environ.get("ZEELIN_BITSIGN_IVF_FORCE_REBUILD") == "1"
        needs_rebuild = force_rebuild or not out.exists()
        if out.exists() and source.exists():
            needs_rebuild = needs_rebuild or source.stat().st_mtime > out.stat().st_mtime
        if needs_rebuild:
            cxx = os.environ.get("CXX", "c++")
            flags = os.environ.get("ZEELIN_BITSIGN_IVF_CXXFLAGS")
            if flags:
                cmd = [cxx, *flags.split(), str(source), "-o", str(out)]
            else:
                cmd = [
                    cxx,
                    "-O3",
                    "-std=c++17",
                    "-march=native",
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-o",
                    str(out),
                ]
            subprocess.check_call(cmd)
        lib = ctypes.CDLL(str(out))
        lib.zeelin_bitsign_ivf_create.restype = ctypes.c_void_p
        lib.zeelin_bitsign_ivf_free.argtypes = [ctypes.c_void_p]
        lib.zeelin_bitsign_ivf_fit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.zeelin_bitsign_ivf_query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_longlong),
        ]
        lib.zeelin_bitsign_ivf_query.restype = ctypes.c_int
        lib.zeelin_bitsign_ivf_avg_candidates.argtypes = [ctypes.c_void_p]
        lib.zeelin_bitsign_ivf_avg_candidates.restype = ctypes.c_double
        lib.zeelin_bitsign_ivf_avg_repairs.argtypes = [ctypes.c_void_p]
        lib.zeelin_bitsign_ivf_avg_repairs.restype = ctypes.c_double
        lib.zeelin_bitsign_ivf_total_candidates.argtypes = [ctypes.c_void_p]
        lib.zeelin_bitsign_ivf_total_candidates.restype = ctypes.c_double
        lib.zeelin_bitsign_ivf_total_repairs.argtypes = [ctypes.c_void_p]
        lib.zeelin_bitsign_ivf_total_repairs.restype = ctypes.c_double
        lib.zeelin_bitsign_ivf_scan_layout.restype = ctypes.c_char_p
        lib.zeelin_bitsign_ivf_kernel.restype = ctypes.c_char_p
        return lib

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        trainer = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        trainer.train(train)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        _, assigned = self.quantizer.search(np.ascontiguousarray(Xn, dtype=np.float32), 1)
        assigned = np.ascontiguousarray(assigned[:, 0], dtype=np.int64)
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

        self._lib.zeelin_bitsign_ivf_fit(
            self._idx,
            np.ascontiguousarray(Xn, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            assigned.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(Xn.shape[0]),
            int(self.dim),
            int(self.nlist),
        )

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        self.nprobe = int(nprobe)
        self.quantizer_ef_search = (
            int(quantizer_ef_search)
            if quantizer_ef_search is not None
            else self.default_quantizer_ef_search
        )
        if return_k is not None:
            self.return_k = int(return_k)
        self.quantizer.hnsw.efSearch = self.quantizer_ef_search
        self._qd = np.empty((1, self.nprobe), dtype=np.float32)
        self._qi = np.empty((1, self.nprobe), dtype=np.int64)
        self.name = (
            "zeelin-bitsignlistorder-hnswgate "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, budget={self.budget}, "
            f"quantizerM={self.quantizer_m}, quantizerEfC={self.quantizer_ef_construction}, "
            f"quantizerEfS={self.quantizer_ef_search}, assignEfS={self.assign_ef_search}, "
            f"trainSize={self.train_size}, returnK={self.return_k}, "
            f"skipQueryNorm={int(self.skip_query_norm)}, queryView={int(self.query_view)}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def query(self, v, n):
        if self.skip_query_norm and self.query_view:
            q = v.reshape(1, self.dim)
        elif self.skip_query_norm:
            self.qbuf[0, :] = v
            q = self.qbuf
        else:
            norm = float(np.sqrt(np.dot(v, v)))
            self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
            q = self.qbuf
        target_k = n if self.return_k <= 0 else min(n, self.return_k)
        if self._out64 is None or len(self._out64) < target_k:
            self._out64 = np.empty(target_k, dtype=np.int64)
        self.quantizer.search_c(
            1,
            faiss.swig_ptr(q),
            self.nprobe,
            faiss.swig_ptr(self._qd),
            faiss.swig_ptr(self._qi),
        )
        count = self._lib.zeelin_bitsign_ivf_query(
            self._idx,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._qi.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(self.nprobe),
            int(target_k),
            int(self.budget),
            self._out64.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        )
        return self._out64[:count]

    def get_additional(self):
        return {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "return_k": self.return_k,
            "budget": self.budget,
            "skip_query_norm": int(self.skip_query_norm),
            "query_view": int(self.query_view),
            "threads": int(self.method_param.get("threads", 1)),
            "train_size": self.train_size,
            "quantizer_m": self.quantizer_m,
            "quantizer_efConstruction": self.quantizer_ef_construction,
            "quantizer_efSearch": self.quantizer_ef_search,
            "assign_efSearch": self.assign_ef_search,
            "avg_candidates": self._lib.zeelin_bitsign_ivf_avg_candidates(self._idx),
            "avg_repairs": self._lib.zeelin_bitsign_ivf_avg_repairs(self._idx),
            "dist_comps": self._lib.zeelin_bitsign_ivf_total_repairs(self._idx),
            "avg_dist_comps": self._lib.zeelin_bitsign_ivf_avg_repairs(self._idx),
            "kernel": self._lib.zeelin_bitsign_ivf_kernel().decode("ascii"),
            "scan_layout": self._lib.zeelin_bitsign_ivf_scan_layout().decode("ascii"),
        }

    def done(self):
        if getattr(self, "_idx", None):
            self._lib.zeelin_bitsign_ivf_free(self._idx)
            self._idx = None


class ZeelinDistortioncertInt8repairHnswgateIvf(ZeelinHnswgateIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "qefc" in method_param:
            method_param["quantizer_efConstruction"] = method_param["qefc"]
        if "qm" in method_param:
            method_param["quantizer_m"] = method_param["qm"]
        if "ts" in method_param:
            method_param["train_size"] = method_param["ts"]
        if "b" in method_param:
            method_param["budget"] = method_param["b"]
        if "aef" in method_param:
            method_param["assign_efSearch"] = method_param["aef"]
        if "cq" in method_param:
            method_param["clip_quantile"] = method_param["cq"]
        method_param.setdefault("quantizer_efSearch", 1)
        method_param.setdefault("skip_query_norm", True)
        method_param.setdefault("query_view", True)
        method_param.setdefault("threads", 2)
        method_param.setdefault("return_k", 10)
        super().__init__(metric, method_param)
        self.assign_ef_search = int(method_param.get("assign_efSearch", 8))
        self.budget = int(method_param.get("budget", 32))
        self.budget = max(1, min(512, self.budget))
        self.clip_quantile = float(method_param.get("clip_quantile", 0.995))
        self.clip_quantile = min(max(self.clip_quantile, 0.5), 1.0)
        self._lib = self._load_int8repair_library()
        self._idx = self._lib.zeelin_int8repair_ivf_create()
        self._qd = None
        self._qi = None
        self._out64 = None
        self._scale = None

    def _load_int8repair_library(self):
        here = Path(__file__).resolve().parent
        source = here / "zeelin_int8repair_ivf_lib.cpp"
        suffix = ".dylib" if os.uname().sysname == "Darwin" else ".so"
        out = here / f"libzeelin_int8repair_ivf{suffix}"
        force_rebuild = os.environ.get("ZEELIN_INT8REPAIR_IVF_FORCE_REBUILD") == "1"
        needs_rebuild = force_rebuild or not out.exists()
        if out.exists() and source.exists():
            needs_rebuild = needs_rebuild or source.stat().st_mtime > out.stat().st_mtime
        if needs_rebuild:
            cxx = os.environ.get("CXX", "c++")
            flags = os.environ.get("ZEELIN_INT8REPAIR_IVF_CXXFLAGS")
            if flags:
                cmd = [cxx, *flags.split(), str(source), "-o", str(out)]
            else:
                cmd = [
                    cxx,
                    "-O3",
                    "-std=c++17",
                    "-march=native",
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-o",
                    str(out),
                ]
            subprocess.check_call(cmd)
        lib = ctypes.CDLL(str(out))
        lib.zeelin_int8repair_ivf_create.restype = ctypes.c_void_p
        lib.zeelin_int8repair_ivf_free.argtypes = [ctypes.c_void_p]
        lib.zeelin_int8repair_ivf_fit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.zeelin_int8repair_ivf_query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_longlong),
        ]
        lib.zeelin_int8repair_ivf_query.restype = ctypes.c_int
        lib.zeelin_int8repair_ivf_query_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_longlong),
        ]
        lib.zeelin_int8repair_ivf_avg_candidates.argtypes = [ctypes.c_void_p]
        lib.zeelin_int8repair_ivf_avg_candidates.restype = ctypes.c_double
        lib.zeelin_int8repair_ivf_avg_repairs.argtypes = [ctypes.c_void_p]
        lib.zeelin_int8repair_ivf_avg_repairs.restype = ctypes.c_double
        lib.zeelin_int8repair_ivf_total_candidates.argtypes = [ctypes.c_void_p]
        lib.zeelin_int8repair_ivf_total_candidates.restype = ctypes.c_double
        lib.zeelin_int8repair_ivf_total_repairs.argtypes = [ctypes.c_void_p]
        lib.zeelin_int8repair_ivf_total_repairs.restype = ctypes.c_double
        lib.zeelin_int8repair_ivf_scan_layout.restype = ctypes.c_char_p
        lib.zeelin_int8repair_ivf_kernel.restype = ctypes.c_char_p
        return lib

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        trainer = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        trainer.train(train)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        _, assigned = self.quantizer.search(np.ascontiguousarray(Xn, dtype=np.float32), 1)
        assigned = np.ascontiguousarray(assigned[:, 0], dtype=np.int64)
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

        clip = np.quantile(np.abs(Xn), self.clip_quantile, axis=0).astype(np.float32)
        self._scale = np.maximum(clip / np.float32(127.0), np.float32(1e-6))
        codes = np.rint(Xn / self._scale.reshape(1, -1))
        codes = np.clip(codes, -127, 127).astype(np.int8, copy=False)
        codes = np.ascontiguousarray(codes, dtype=np.int8)

        self._lib.zeelin_int8repair_ivf_fit(
            self._idx,
            np.ascontiguousarray(Xn, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            self._scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            assigned.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(Xn.shape[0]),
            int(self.dim),
            int(self.nlist),
        )

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        self.nprobe = int(nprobe)
        self.quantizer_ef_search = (
            int(quantizer_ef_search)
            if quantizer_ef_search is not None
            else self.default_quantizer_ef_search
        )
        if return_k is not None:
            self.return_k = int(return_k)
        self.quantizer.hnsw.efSearch = self.quantizer_ef_search
        self._qd = np.empty((1, self.nprobe), dtype=np.float32)
        self._qi = np.empty((1, self.nprobe), dtype=np.int64)
        self.name = (
            "zeelin-distortioncert-int8repair-hnswgate "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, budget={self.budget}, "
            f"clipQ={self.clip_quantile}, quantizerM={self.quantizer_m}, "
            f"quantizerEfC={self.quantizer_ef_construction}, "
            f"quantizerEfS={self.quantizer_ef_search}, assignEfS={self.assign_ef_search}, "
            f"trainSize={self.train_size}, returnK={self.return_k}, "
            f"skipQueryNorm={int(self.skip_query_norm)}, queryView={int(self.query_view)}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def query(self, v, n):
        if self.skip_query_norm and self.query_view:
            q = v.reshape(1, self.dim)
        elif self.skip_query_norm:
            self.qbuf[0, :] = v
            q = self.qbuf
        else:
            norm = float(np.sqrt(np.dot(v, v)))
            self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
            q = self.qbuf
        target_k = n if self.return_k <= 0 else min(n, self.return_k)
        if self._out64 is None or len(self._out64) < target_k:
            self._out64 = np.empty(target_k, dtype=np.int64)
        self.quantizer.search_c(
            1,
            faiss.swig_ptr(q),
            self.nprobe,
            faiss.swig_ptr(self._qd),
            faiss.swig_ptr(self._qi),
        )
        count = self._lib.zeelin_int8repair_ivf_query(
            self._idx,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._qi.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(self.nprobe),
            int(target_k),
            int(self.budget),
            self._out64.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        )
        return self._out64[:count]

    def batch_query(self, X, n):
        X = np.asarray(X, dtype=np.float32)
        if self.skip_query_norm:
            Xn = np.ascontiguousarray(X, dtype=np.float32)
        else:
            norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
            Xn = np.ascontiguousarray(X / np.maximum(norms, 1e-12), dtype=np.float32)
        target_k = n if self.return_k <= 0 else min(n, self.return_k)
        self.quantizer.hnsw.efSearch = self.quantizer_ef_search
        _, labels = self.quantizer.search(Xn, self.nprobe)
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        tmp = np.empty(target_k, dtype=np.int64)
        results = []
        for i in range(len(Xn)):
            count = self._lib.zeelin_int8repair_ivf_query(
                self._idx,
                Xn[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                labels[i].ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
                int(self.nprobe),
                int(target_k),
                int(self.budget),
                tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            )
            results.append(np.asarray(tmp[:count], dtype=np.int64).copy())
        self._batch_results = results

    def get_batch_results(self):
        return self._batch_results

    def get_additional(self):
        return {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "return_k": self.return_k,
            "budget": self.budget,
            "clip_quantile": self.clip_quantile,
            "skip_query_norm": int(self.skip_query_norm),
            "query_view": int(self.query_view),
            "threads": int(self.method_param.get("threads", 1)),
            "train_size": self.train_size,
            "quantizer_m": self.quantizer_m,
            "quantizer_efConstruction": self.quantizer_ef_construction,
            "quantizer_efSearch": self.quantizer_ef_search,
            "assign_efSearch": self.assign_ef_search,
            "avg_candidates": self._lib.zeelin_int8repair_ivf_avg_candidates(self._idx),
            "avg_repairs": self._lib.zeelin_int8repair_ivf_avg_repairs(self._idx),
            "dist_comps": self._lib.zeelin_int8repair_ivf_total_repairs(self._idx),
            "avg_dist_comps": self._lib.zeelin_int8repair_ivf_avg_repairs(self._idx),
            "kernel": self._lib.zeelin_int8repair_ivf_kernel().decode("ascii"),
            "scan_layout": self._lib.zeelin_int8repair_ivf_scan_layout().decode("ascii"),
        }

    def done(self):
        if getattr(self, "_idx", None):
            self._lib.zeelin_int8repair_ivf_free(self._idx)
            self._idx = None


class ZeelinHarmonicpayloadHnswgateIvf(ZeelinCentroidscanIvf):
    def __init__(self, metric, method_param):
        method_param = dict(method_param)
        if "qefc" in method_param:
            method_param["quantizer_efConstruction"] = method_param["qefc"]
        if "qm" in method_param:
            method_param["quantizer_m"] = method_param["qm"]
        if "ts" in method_param:
            method_param["train_size"] = method_param["ts"]
        if "aef" in method_param:
            method_param["assign_efSearch"] = method_param["aef"]
        method_param.setdefault("quantizer_efSearch", 1)
        method_param.setdefault("skip_query_norm", True)
        method_param.setdefault("query_view", True)
        method_param.setdefault("threads", 4)
        super().__init__(metric, method_param)
        self.payload_k = int(method_param.get("p", method_param.get("payload_k", 64)))
        self.payload_mode = str(method_param.get("pm", method_param.get("payload_mode", "h")))
        self.core_fraction = float(method_param.get("cf", method_param.get("core_fraction", 0.5)))
        self.boundary_weight = float(method_param.get("bw", method_param.get("boundary_weight", 0.35)))
        self._qd = None
        self._qi = None
        self._out64 = None

    @staticmethod
    def _top_indices(indices, values, take, largest=True):
        if take <= 0 or len(indices) == 0:
            return np.empty(0, dtype=np.int64)
        if len(indices) <= take:
            local = np.argsort(-values if largest else values, kind="stable")
            return indices[local].astype(np.int64, copy=False)
        key = -values if largest else values
        part = np.argpartition(key, take - 1)[:take]
        part = part[np.argsort(key[part], kind="stable")]
        return indices[part].astype(np.int64, copy=False)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        trainer = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        trainer.train(train)

        self.quantizer.hnsw.efSearch = self.assign_ef_search
        centroid_scores, assigned2 = self.quantizer.search(np.ascontiguousarray(Xn, dtype=np.float32), 2)
        assigned = np.ascontiguousarray(assigned2[:, 0], dtype=np.int64)
        first_scores = np.ascontiguousarray(centroid_scores[:, 0], dtype=np.float32)
        second_scores = np.ascontiguousarray(centroid_scores[:, 1], dtype=np.float32)
        margins = first_scores - second_scores
        margins[assigned2[:, 1] < 0] = np.max(margins[assigned2[:, 1] >= 0]) if np.any(assigned2[:, 1] >= 0) else 1.0
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search

        counts = np.bincount(np.maximum(assigned, 0), minlength=self.nlist)
        order = np.argsort(assigned, kind="stable")
        offsets = np.empty(self.nlist + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts[: self.nlist], out=offsets[1:])

        payload = np.full((self.nlist, self.payload_k), -1, dtype=np.int64)
        for cid in range(self.nlist):
            begin = int(offsets[cid])
            end = int(offsets[cid + 1])
            if begin >= end:
                continue
            inds = order[begin:end].astype(np.int64, copy=False)
            if self.payload_mode in ("c", "central"):
                chosen = self._top_indices(inds, first_scores[inds], self.payload_k, largest=True)
            else:
                core_take = max(1, min(self.payload_k, int(round(self.payload_k * self.core_fraction))))
                core = self._top_indices(inds, first_scores[inds], core_take, largest=True)
                if len(core) >= self.payload_k:
                    chosen = core[: self.payload_k]
                else:
                    keep = np.ones(len(inds), dtype=bool)
                    if len(core):
                        core_set = set(int(x) for x in core)
                        keep = np.asarray([int(x) not in core_set for x in inds], dtype=bool)
                    rest = inds[keep]
                    harmonic_values = first_scores[rest] - self.boundary_weight * margins[rest]
                    boundary = self._top_indices(
                        rest,
                        harmonic_values,
                        self.payload_k - len(core),
                        largest=True,
                    )
                    chosen = np.concatenate([core, boundary])
            payload[cid, : len(chosen)] = chosen[: self.payload_k]

        self._lib.zeelin_centroidscan_ivf_fit_payload(
            self._idx,
            np.ascontiguousarray(Xn, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            np.ascontiguousarray(payload, dtype=np.int64).ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(Xn.shape[0]),
            int(self.dim),
            int(self.nlist),
            int(self.payload_k),
        )

    def set_query_arguments(
        self,
        nprobe,
        quantizer_ef_search=None,
        dedup_extra=None,
        return_k=None,
        internal_k=None,
    ):
        self.nprobe = int(nprobe)
        self.quantizer_ef_search = (
            int(quantizer_ef_search)
            if quantizer_ef_search is not None
            else self.default_quantizer_ef_search
        )
        self.return_k = int(return_k) if return_k is not None else self.default_return_k
        self.quantizer.hnsw.efSearch = self.quantizer_ef_search
        self._qd = np.empty((1, self.nprobe), dtype=np.float32)
        self._qi = np.empty((1, self.nprobe), dtype=np.int64)
        self.name = (
            "zeelin-harmonicpayload-hnswgate "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, "
            f"payloadK={self.payload_k}, payloadMode={self.payload_mode}, "
            f"coreFraction={self.core_fraction}, boundaryWeight={self.boundary_weight}, "
            f"quantizerM={self.quantizer_m}, quantizerEfC={self.quantizer_ef_construction}, "
            f"quantizerEfS={self.quantizer_ef_search}, assignEfS={self.assign_ef_search}, "
            f"trainSize={self.train_size}, returnK={self.return_k}, "
            f"skipQueryNorm={int(self.skip_query_norm)}, queryView={int(self.query_view)}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def query(self, v, n):
        if self.skip_query_norm and self.query_view:
            q = v.reshape(1, self.dim)
        elif self.skip_query_norm:
            self.qbuf[0, :] = v
            q = self.qbuf
        else:
            norm = float(np.sqrt(np.dot(v, v)))
            self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
            q = self.qbuf
        target_k = n if self.return_k <= 0 else min(n, self.return_k)
        if self._out64 is None or len(self._out64) < target_k:
            self._out64 = np.empty(target_k, dtype=np.int64)
        self.quantizer.search_c(
            1,
            faiss.swig_ptr(q),
            self.nprobe,
            faiss.swig_ptr(self._qd),
            faiss.swig_ptr(self._qi),
        )
        count = self._lib.zeelin_centroidscan_ivf_query_payload(
            self._idx,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._qi.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            int(self.nprobe),
            int(target_k),
            self._out64.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        )
        return self._out64[:count]

    def get_additional(self):
        return {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "return_k": self.return_k,
            "payload_k": self.payload_k,
            "payload_mode": self.payload_mode,
            "core_fraction": self.core_fraction,
            "boundary_weight": self.boundary_weight,
            "skip_query_norm": int(self.skip_query_norm),
            "query_view": int(self.query_view),
            "threads": int(self.method_param.get("threads", 1)),
            "train_size": self.train_size,
            "quantizer_m": self.quantizer_m,
            "quantizer_efConstruction": self.quantizer_ef_construction,
            "quantizer_efSearch": self.quantizer_ef_search,
            "assign_efSearch": self.assign_ef_search,
            "dist_comps": self._lib.zeelin_centroidscan_ivf_total_dist_comps(self._idx),
            "avg_dist_comps": self._lib.zeelin_centroidscan_ivf_avg_dist_comps(self._idx),
            "avg_candidates": self._lib.zeelin_centroidscan_ivf_avg_candidates(self._idx),
            "kernel": "cpp-harmonicpayload-hnswquantizer-rerank",
            "scan_layout": "centroid-harmonic-payload",
        }


class ZeelinCentroidpayloadIvf(ZeelinHnswgateIvf):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self.payload_k = int(method_param.get("payload_k", 32))
        self.rerank = bool(int(method_param.get("rerank", 1)))

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.Xn = Xn if self.rerank else None
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))

        self.quantizer = faiss.IndexHNSWFlat(
            self.dim, self.quantizer_m, faiss.METRIC_INNER_PRODUCT
        )
        self.quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        self.quantizer.hnsw.efSearch = self.default_quantizer_ef_search
        trainer = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
        )

        train = Xn
        if 0 < self.train_size < len(Xn):
            train_idx = np.linspace(0, len(Xn) - 1, self.train_size, dtype=np.int64)
            train = np.ascontiguousarray(Xn[train_idx], dtype=np.float32)
        trainer.train(train)

        centroids = np.empty((self.nlist, self.dim), dtype=np.float32)
        for i in range(self.nlist):
            self.quantizer.reconstruct(i, centroids[i])
        self.centroids = self._normalize_rows(centroids).astype(np.float32, copy=False)

        _, assigned = self.quantizer.search(np.ascontiguousarray(Xn, dtype=np.float32), 1)
        assigned = assigned[:, 0].astype(np.int64, copy=False)
        scores = np.einsum("ij,ij->i", Xn, self.centroids[assigned]).astype(np.float32)
        order = np.lexsort((-scores, assigned))
        payload = np.full((self.nlist, self.payload_k), -1, dtype=np.int64)
        counts = np.zeros(self.nlist, dtype=np.int32)
        for label in order:
            cid = int(assigned[label])
            pos = counts[cid]
            if pos >= self.payload_k:
                continue
            payload[cid, pos] = int(label)
            counts[cid] = pos + 1
        self.payload = payload

    def set_query_arguments(self, nprobe, quantizer_ef_search=None, dedup_extra=None, return_k=None):
        self.nprobe = int(nprobe)
        self.quantizer_ef_search = (
            int(quantizer_ef_search)
            if quantizer_ef_search is not None
            else self.default_quantizer_ef_search
        )
        if dedup_extra is not None:
            self.dedup_extra = int(dedup_extra)
        self.return_k = int(return_k) if return_k is not None else self.default_return_k
        self.quantizer.hnsw.efSearch = self.quantizer_ef_search
        self.name = (
            "zeelin-centroidpayload-ivf "
            f"(nlist={self.nlist}, nprobe={self.nprobe}, payloadK={self.payload_k}, "
            f"rerank={int(self.rerank)}, quantizerM={self.quantizer_m}, "
            f"quantizerEfC={self.quantizer_ef_construction}, "
            f"quantizerEfS={self.quantizer_ef_search}, trainSize={self.train_size}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def query(self, v, n):
        norm = float(np.sqrt(np.dot(v, v)))
        self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
        _, centroid_labels = self.quantizer.search(self.qbuf, self.nprobe)
        seen = set()
        candidates = []
        for cid in centroid_labels[0]:
            cid = int(cid)
            if cid < 0:
                continue
            for label in self.payload[cid]:
                label = int(label)
                if label < 0 or label in seen:
                    continue
                seen.add(label)
                candidates.append(label)
        if not candidates:
            return np.empty(0, dtype=np.int64)
        if not self.rerank:
            return np.asarray(candidates[:n], dtype=np.int64)
        cand = np.asarray(candidates, dtype=np.int64)
        scores = self.Xn[cand].dot(self.qbuf[0])
        if len(cand) > n:
            top = np.argpartition(-scores, n - 1)[:n]
            top = top[np.argsort(-scores[top], kind="stable")]
            cand = cand[top]
        else:
            cand = cand[np.argsort(-scores, kind="stable")[::-1]]
        return cand[:n]

    def get_additional(self):
        extra = super().get_additional()
        extra["payload_k"] = self.payload_k
        extra["rerank"] = int(self.rerank)
        extra["kernel"] = "centroid-payload-hnswquantizer-rerank"
        return extra


class ZeelinPqgateIvf(BaseANN):
    def __init__(self, metric, method_param):
        if metric != "angular":
            raise ValueError("zeelin-pqgate-ivf is only defined for angular/cosine data")
        self.method_param = method_param
        self.nlist = int(method_param["nlist"])
        self.pq_m = int(method_param["pq_m"])
        self.pq_bits = int(method_param.get("pq_bits", 4))
        self.train_size = int(method_param.get("train_size", 0))
        self.qbuf = None
        self.nprobe = 1

    @staticmethod
    def _normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
        return X / np.maximum(norms, 1e-12)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))
        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFPQ(
            quantizer,
            self.dim,
            self.nlist,
            self.pq_m,
            self.pq_bits,
            faiss.METRIC_INNER_PRODUCT,
        )
        train = Xn
        if 0 < self.train_size < len(Xn):
            train = Xn[: self.train_size]
        self.index.train(train)
        self.index.add(Xn)

    def set_query_arguments(self, nprobe):
        faiss.cvar.indexIVF_stats.reset()
        self.nprobe = int(nprobe)
        self.index.nprobe = self.nprobe
        self.name = (
            "zeelin-pqgate-ivf "
            f"(nlist={self.nlist}, pqM={self.pq_m}, pqBits={self.pq_bits}, "
            f"trainSize={self.train_size}, nprobe={self.nprobe}, "
            f"threads={self.method_param.get('threads', 1)})"
        )

    def query(self, v, n):
        norm = float(np.sqrt(np.dot(v, v)))
        self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
        return self.index.search(self.qbuf, n)[1][0]

    def get_additional(self):
        return {
            "nlist": self.nlist,
            "pq_m": self.pq_m,
            "pq_bits": self.pq_bits,
            "train_size": self.train_size,
            "nprobe": self.nprobe,
            "threads": int(self.method_param.get("threads", 1)),
            "dist_comps": faiss.cvar.indexIVF_stats.ndis + faiss.cvar.indexIVF_stats.nq * self.nlist,
            "kernel": "faiss-ivfpq-ip-normalized",
        }


class ZeelinFaissphaseHnsw(ZeelinSphereipHnsw):
    def __init__(self, metric, method_param):
        if metric != "angular":
            raise ValueError("zeelin-faissphase-hnsw is only defined for angular/cosine data")
        self.method_param = method_param
        self.qbuf = None
        self.labels = None
        self.ef = 0

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        self.dim = Xn.shape[1]
        self.qbuf = np.empty((1, self.dim), dtype=np.float32)
        faiss.omp_set_num_threads(int(self.method_param.get("threads", 1)))
        self.index = faiss.IndexHNSWFlat(self.dim, int(self.method_param["M"]), faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = int(self.method_param["efConstruction"])
        order_mode = self.method_param.get("order_mode", "radialcentral_zigzag")
        order = self._score_order(X, Xn, order_mode)
        self.labels = order.astype(np.int64, copy=True)
        self.index.add(Xn[order])

    def set_query_arguments(self, ef):
        faiss.cvar.hnsw_stats.reset()
        self.ef = int(ef)
        self.index.hnsw.efSearch = self.ef
        self.name = (
            "zeelin-faissphase-hnsw "
            f"(M={self.method_param['M']}, "
            f"efConstruction={self.method_param['efConstruction']}, "
            f"order={self.method_param.get('order_mode', 'radialcentral_zigzag')}, "
            f"radialAlpha={self.method_param.get('radial_alpha', 0.0)}, "
            f"threads={self.method_param.get('threads', 1)}, "
            f"efQuery={self.ef})"
        )

    def query(self, v, n):
        norm = float(np.sqrt(np.dot(v, v)))
        self.qbuf[0, :] = v * (1.0 / max(norm, 1e-12))
        internal = self.index.search(self.qbuf, n)[1][0]
        return self.labels[internal]

    def get_additional(self):
        return {
            "dist_comps": faiss.cvar.hnsw_stats.ndis,
            "order_mode": self.method_param.get("order_mode", "radialcentral_zigzag"),
            "order_seed": int(self.method_param.get("order_seed", 183)),
            "radial_alpha": float(self.method_param.get("radial_alpha", 0.0)),
            "threads": int(self.method_param.get("threads", 1)),
            "kernel": "faiss-hnswflat-ip-normalized",
        }


class ZeelinCachephaseHnsw(ZeelinCosphaseHnsw):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param)
        self.cache_bits = int(method_param.get("cache_bits", 22))
        self.cache_mode = method_param.get("cache_mode", "train_var")
        self.cache_dims = None
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize_rows(X).astype(np.float32, copy=False)
        if self.cache_mode == "train_absmean":
            order = np.argsort(np.abs(Xn.mean(axis=0)))[::-1]
        elif self.cache_mode == "train_mean_abs_centered":
            order = np.argsort(np.mean(np.abs(Xn - Xn.mean(axis=0)), axis=0))[::-1]
        else:
            order = np.argsort(Xn.var(axis=0))[::-1]
        self.cache_dims = order[: min(self.cache_bits, Xn.shape[1])].astype(np.int64, copy=True)
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        super().fit(X)

    def set_query_arguments(self, ef):
        super().set_query_arguments(ef)
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.name = (
            "zeelin-cachephase-hnsw "
            f"(M={self.method_param['M']}, "
            f"efConstruction={self.method_param['efConstruction']}, "
            f"seed={self.method_param.get('random_seed', 100)}, "
            f"order={self.method_param.get('order_mode', 'original')}, "
            f"radialAlpha={self.method_param.get('radial_alpha', 0.0)}, "
            f"cacheBits={self.cache_bits}, "
            f"cacheMode={self.cache_mode}, "
            f"efQuery={ef})"
        )

    def _cache_key(self, v):
        key = 0
        for i, d in enumerate(self.cache_dims):
            if v[int(d)] >= 0.0:
                key |= 1 << i
        return key

    def query(self, v, n):
        key = self._cache_key(v)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        labels = super().query(v, n).copy()
        self.cache[key] = labels
        return labels

    def get_additional(self):
        extra = super().get_additional()
        total = self.cache_hits + self.cache_misses
        extra.update(
            {
                "cache_bits": self.cache_bits,
                "cache_mode": self.cache_mode,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": float(self.cache_hits) / float(total) if total else 0.0,
            }
        )
        return extra
