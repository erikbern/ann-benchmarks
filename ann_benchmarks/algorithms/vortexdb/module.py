import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import psutil
from vortexdb import VortexDB
from vortexdb.models import DenseVector, Payload, Similarity

from ..base.module import BaseANN


METRICS = {
    "angular": ("cosine", Similarity.COSINE),
    "euclidean": ("euclidean", Similarity.EUCLIDEAN),
}


class VortexDBANN(BaseANN):
    def __init__(self, metric, method_param):
        if metric not in METRICS:
            raise RuntimeError(f"VortexDB does not support metric {metric}")

        self._metric_name, self._similarity = METRICS[metric]
        self._method_param = method_param
        self._m = method_param["M"]
        self._m0 = method_param.get("M0", 2 * self._m)
        self._ef_construction = method_param["efConstruction"]
        self._ef = method_param.get("ef", 100)
        self._max_layer = method_param.get("maxLayer", 16)
        self._batch_size = method_param.get("batchSize", 1000)
        self._server = None
        self._data_path = None
        self._client = None
        self._point_to_label = {}
        self.name = str(self)

    def fit(self, X):
        self._start_server(X.shape[1])
        self._client = VortexDB(
            grpc_url="127.0.0.1:50051",
            api_key="ann",
            timeout=60.0,
        )

        for start in range(0, len(X), self._batch_size):
            stop = min(start + self._batch_size, len(X))
            points = [
                (
                    DenseVector(np.asarray(X[i], dtype=np.float32).tolist()),
                    Payload.text(str(i)),
                )
                for i in range(start, stop)
            ]
            point_ids = self._client.insert_batch(points=points)
            self._point_to_label.update(
                (point_id, label) for point_id, label in zip(point_ids, range(start, stop))
            )

    def query(self, v, n):
        point_ids = self._client.search(
            vector=DenseVector(np.asarray(v, dtype=np.float32).tolist()),
            similarity=self._similarity,
            limit=n,
            ef=self._ef,
        )
        return [self._point_to_label[point_id] for point_id in point_ids]

    def batch_query(self, X, n):
        queries = [
            (
                DenseVector(np.asarray(v, dtype=np.float32).tolist()),
                self._similarity,
                n,
            )
            for v in X
        ]
        results = self._client.search_batch(queries=queries, ef=self._ef)
        self.res = [
            [self._point_to_label[point_id] for point_id in result]
            for result in results
        ]

    def get_batch_results(self):
        return self.res

    def get_memory_usage(self):
        if self._server is None:
            return 0
        return psutil.Process(self._server.pid).memory_info().rss / 1024

    def done(self):
        if self._client is not None:
            self._client.close()
            self._client = None

        if self._server is not None:
            self._server.terminate()
            try:
                self._server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server.kill()
                self._server.wait(timeout=10)
            self._server = None

        if self._data_path is not None:
            shutil.rmtree(self._data_path, ignore_errors=True)
            self._data_path = None

    def _start_server(self, dimension):
        self._data_path = tempfile.mkdtemp(prefix="vortexdb-ann-")
        env = os.environ.copy()
        env.update(
            {
                "GRPC_ROOT_PASSWORD": "ann",
                "GRPC_HOST": "127.0.0.1",
                "GRPC_PORT": "50051",
                "DISABLE_HTTP": "true",
                "LOGGING": "false",
                "DATA_PATH": self._data_path,
                "DIMENSION": str(dimension),
                "STORAGE_TYPE": "inmemory",
                "INDEX_TYPE": "hnsw",
                "SIMILARITY": self._metric_name,
                "HNSW_M": str(self._m),
                "HNSW_M0": str(self._m0),
                "HNSW_MAX_LAYER": str(self._max_layer),
                "HNSW_EF_CONSTRUCTION": str(self._ef_construction),
            }
        )

        self._server = subprocess.Popen(
            ["/opt/vortexdb/target/release/server"],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_grpc()

    def _wait_for_grpc(self):
        deadline = time.time() + 30
        while time.time() < deadline:
            if self._server.poll() is not None:
                raise RuntimeError(f"VortexDB exited with code {self._server.returncode}")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                if sock.connect_ex(("127.0.0.1", 50051)) == 0:
                    return

            time.sleep(0.2)

        raise RuntimeError("Timed out waiting for VortexDB gRPC server")

    def set_query_arguments(self, ef):
        self._ef = ef
        self.name = str(self)

    def __str__(self):
        return (
            "VortexDB("
            f"M={self._m}, "
            f"M0={self._m0}, "
            f"efConstruction={self._ef_construction}, "
            f"ef={self._ef}"
            ")"
        )
