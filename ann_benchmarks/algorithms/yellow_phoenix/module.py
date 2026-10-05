"""Yellow Phoenix for ann-benchmarks (AGPL-3.0-or-later engine).

Two-stage hybrid: 512-bit binary HNSW proposes top-K candidates, exact
float32 L2 verifies. ITQ (Gong et al. 2011, corrected Procrustes) trained
in fit(). Engine binaries are built in the algorithm Dockerfile from the
public yellow-phoenix-release source with a designated public build phrase.

Measured (SIFT1M, M3 Pro, single thread, prior to packaging):
  hybrid m12: 0.851@6.7k ... 0.991@886 QPS | hybrid m24: 0.874@4.7k ... 0.994@681
  binary m12: 0.358@29k QPS (ef16), 64 MB index
"""
import os  # noqa
import select
import shutil
import struct
import subprocess
import tempfile

import numpy as np
import psutil

class BaseANN: pass

ENGINE = os.environ.get("YP_ENGINE", "/usr/local/yp/bin")


def _train_itq(X, bits=512, iters=15, seed=31337):
    """Mean-center + PCA rotation (NO whitening: it rescales and distorts
    L2 neighborhoods) + semi-orthogonal expansion + corrected Procrustes
    (R = V U^T; the naive transpose walks the anti-rotation direction)."""
    n, d = X.shape
    mu = X.mean(0, keepdims=True)
    Xc = (X - mu).astype(np.float32)
    _, evec = np.linalg.eigh((Xc.T @ Xc) / n)
    V = evec[:, ::-1].astype(np.float32)
    Xc = Xc @ V
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, bits)).astype(np.float32)
    Q, _ = np.linalg.qr(W.T)  # wide-matrix QR: transpose trick for orthonormal rows
    P = Q.T.astype(np.float32)
    Xc = Xc @ P
    R = np.eye(bits, dtype=np.float32)
    for _ in range(iters):
        XB = Xc @ R
        B = np.where(XB >= 0, 1.0, -1.0).astype(np.float32)
        U, _, Vt = np.linalg.svd(Xc.T @ B, full_matrices=False)
        R = (Vt.T @ U.T).astype(np.float32)
    return mu.astype(np.float32), (V @ P @ R).astype(np.float32)


def _write_ism(path, codes):
    n = codes.shape[0]
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        f.write(np.ascontiguousarray(codes).tobytes())
        f.write(np.arange(n, dtype="<u8").tobytes())


class _YPBase(BaseANN):
    def __init__(self, metric, method_param):
        assert metric == "euclidean", "yellow-phoenix entries are euclidean-only"
        self.mp = dict(method_param)
        self.dir = tempfile.mkdtemp(prefix="yp_")
        self.server = None
        self._pending = None  # (ef, K) awaiting server spawn at first query

    def _prepare(self, X):
        """Shared fit: ITQ + graph + payload + fused encode matrix."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.mu, self.Wf = _train_itq(X)
        self.mu.tofile(os.path.join(self.dir, "mu.f32"))
        self.Wf.tofile(os.path.join(self.dir, "W.f32"))
        codes = np.packbits((X - self.mu) @ self.Wf >= 0, axis=1)
        ism = os.path.join(self.dir, "train.ism")
        _write_ism(ism, codes)
        subprocess.run(
            [f"{ENGINE}/build_hnsw_from_flat_ism", ism, os.path.join(self.dir, "graph.bin"),
             str(self.mp.get("m", 12)), str(self.mp.get("efC", 200))],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _respawn(self, ef, K):
        if self.server is not None:
            self.server.stdin.close()
            self.server.wait()
            self.server = None
        self.server = subprocess.Popen(
            [f"{ENGINE}/yp_ann_server", os.path.join(self.dir, "graph.bin"),
             os.path.join(self.dir, "payload.f32"),
             os.path.join(self.dir, "W.f32"), os.path.join(self.dir, "mu.f32"),
             str(ef), str(K)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

    def set_query_arguments(self, ef=128, K=200):
        self._pending = (int(ef), int(K))
        if self.server is not None:
            self._respawn(*self._pending)
        self.name = f"yellow-phoenix ({self.mp}, 'efQuery': {ef}, 'K': {K})"

    def query(self, v, n):
        if self.server is None:
            self._respawn(*self._pending)
        self.server.stdin.write(np.ascontiguousarray(v, dtype="<f4").tobytes())
        self.server.stdin.flush()
        # read exactly 80B. select-first: raw blocking reads on this pipe can
        # deadlock at scale (observed 1M, both ARM and x86 — server and client
        # both blocked on read with correct fd wiring); select-based reads ran
        # 10k queries clean. On timeout/EOF, fail loud with the server's state.
        buf = b""
        while len(buf) < 80:
            ready, _, _ = select.select([self.server.stdout], [], [], 60)
            if not ready:
                raise RuntimeError(
                    f"yp_ann_server: no response within 60s mid-query "
                    f"(rc={self.server.poll()}), {len(buf)}/80 bytes")
            chunk = self.server.stdout.read(80 - len(buf))
            if not chunk:
                raise RuntimeError(
                    f"yp_ann_server died mid-query (rc={self.server.poll()}) "
                    f"after {len(buf)}/80 bytes")
            buf += chunk
        ids = struct.unpack("<10Q", buf)
        return np.array(ids[:int(n)])

    def batch_query(self, X, n):
        # sequential on purpose: single-threaded single-query benchmark semantics
        self.res = [self.query(q, n) for q in X]

    def get_memory_usage(self):
        """RSS of this process + the engine server child (the index lives there)."""
        total = psutil.Process().memory_info().rss / 1024
        if self.server is not None:
            try:
                total += psutil.Process(self.server.pid).memory_info().rss / 1024
            except psutil.NoSuchProcess:
                pass
        return total

    def done(self):
        if self.server is not None:
            self.server.stdin.close()
            self.server.wait()
            self.server = None
        shutil.rmtree(self.dir, ignore_errors=True)


class YellowPhoenixHybrid(_YPBase):
    """Two-stage: binary HNSW propose + exact float verify."""

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        X.tofile(os.path.join(self.dir, "payload.f32"))
        self._prepare(X)
        return self


class YellowPhoenixBinary(_YPBase):
    """Pure 512-bit binary HNSW (speed corner; payload doubles as verify set)."""

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        X.tofile(os.path.join(self.dir, "payload.f32"))
        self._prepare(X)
        return self

    def set_query_arguments(self, ef):
        super().set_query_arguments(ef, 10)
        self.name = f"yellow-phoenix-binary ({self.mp}, 'efQuery': {ef})"
