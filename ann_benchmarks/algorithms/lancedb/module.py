import os
import shutil

import lancedb
import numpy as np
import pyarrow as pa

from ..base.module import BaseANN

# ann-benchmarks metric -> LanceDB metric
METRIC_MAPPING = {"angular": "cosine", "euclidean": "l2"}


class LanceDB(BaseANN):
    """Base wrapper for LanceDB vector database benchmarks.

    LanceDB is an embedded vector database built on the Lance columnar
    format. It requires no separate server process — data is stored
    on local disk and accessed through the Python SDK.
    """

    def __init__(self, metric, index_type, num_partitions,
                 num_sub_vectors=None, m=None, ef_construction=None):
        self._metric = metric
        self._lance_metric = METRIC_MAPPING.get(metric, metric)
        self._index_type = index_type
        self._num_partitions = num_partitions
        self._num_sub_vectors = num_sub_vectors
        self._m = m
        self._ef_construction = ef_construction
        # Use a process- and instance-unique path so that concurrent
        # benchmark tasks on the same node do not share or delete each
        # other's Lance data files.
        self._db_uri = f"/tmp/lancedb_benchmark_{os.getpid()}_{id(self)}"
        self._table_name = "ann_benchmarks"
        self._nprobes = None
        self._refine_factor = None
        self._ef = None
        self._table = None

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if os.path.exists(self._db_uri):
            shutil.rmtree(self._db_uri)

        db = lancedb.connect(self._db_uri)

        dim = X.shape[1]
        ids = pa.array(np.arange(len(X)), type=pa.int64())
        vectors = pa.FixedSizeListArray.from_arrays(
            pa.array(X.ravel(), type=pa.float32()),
            dim,
        )
        data = pa.Table.from_arrays([ids, vectors], names=["id", "vector"])

        self._table = db.create_table(
            self._table_name, data=data, mode="overwrite",
        )

        index_params = {
            "metric": self._lance_metric,
            "num_partitions": self._num_partitions,
            "vector_column_name": "vector",
            "index_type": self._index_type,
        }
        if self._num_sub_vectors is not None:
            index_params["num_sub_vectors"] = self._num_sub_vectors
        if self._m is not None:
            index_params["m"] = self._m
        if self._ef_construction is not None:
            index_params["ef_construction"] = self._ef_construction

        self._table.create_index(**index_params)

    def set_query_arguments(self, nprobes, refine_factor, ef=None):
        self._nprobes = nprobes
        self._refine_factor = refine_factor
        self._ef = ef

    def query(self, v, n):
        search = self._table.search(v.tolist()).limit(n)
        if self._nprobes is not None:
            search = search.nprobes(self._nprobes)
        if self._refine_factor is not None:
            search = search.refine_factor(self._refine_factor)
        if self._ef is not None:
            # Lance HNSW requires ef >= k. With a refine_factor the effective
            # candidate count is n * refine_factor, so ef must cover it.
            min_ef = n * (self._refine_factor or 1)
            search = search.ef(max(self._ef, min_ef))
        results = search.to_pandas()
        return results["id"].tolist()

    def done(self):
        if os.path.exists(self._db_uri):
            shutil.rmtree(self._db_uri)


class LanceDBIVFFlat(LanceDB):
    """LanceDB with IVF_FLAT index (no quantization, full precision)."""

    def __init__(self, metric, index_param):
        num_partitions = index_param["num_partitions"]
        super().__init__(metric, "IVF_FLAT", num_partitions)
        self.name = f"LanceDB-IVFFlat(metric={metric}, np={num_partitions})"

    def set_query_arguments(self, nprobes, refine_factor):
        super().set_query_arguments(nprobes, refine_factor)
        self.name = (
            f"LanceDB-IVFFlat(metric={self._metric}, np={self._num_partitions}, "
            f"nprobes={nprobes}, refine={refine_factor})"
        )


class LanceDBIVFPQ(LanceDB):
    """LanceDB with IVF_PQ index (product quantization for compression)."""

    def __init__(self, metric, index_param):
        num_partitions = index_param["num_partitions"]
        num_sub_vectors = index_param["num_sub_vectors"]
        super().__init__(metric, "IVF_PQ", num_partitions, num_sub_vectors)
        self.name = (
            f"LanceDB-IVFPQ(metric={metric}, np={num_partitions}, "
            f"nsv={num_sub_vectors})"
        )

    def set_query_arguments(self, nprobes, refine_factor):
        super().set_query_arguments(nprobes, refine_factor)
        self.name = (
            f"LanceDB-IVFPQ(metric={self._metric}, np={self._num_partitions}, "
            f"nsv={self._num_sub_vectors}, nprobes={nprobes}, refine={refine_factor})"
        )


class LanceDBIVFHNSWSQ(LanceDB):
    """LanceDB with IVF_HNSW_SQ index (HNSW graph + scalar quantization)."""

    def __init__(self, metric, index_param):
        num_partitions = index_param["num_partitions"]
        m = index_param["m"]
        ef_construction = index_param["efConstruction"]
        super().__init__(
            metric, "IVF_HNSW_SQ", num_partitions,
            m=m, ef_construction=ef_construction,
        )
        self.name = (
            f"LanceDB-IVFHNSWSQ(metric={metric}, np={num_partitions}, "
            f"M={m}, efC={ef_construction})"
        )

    def set_query_arguments(self, nprobes, ef, refine_factor):
        super().set_query_arguments(nprobes, refine_factor, ef)
        self.name = (
            f"LanceDB-IVFHNSWSQ(metric={self._metric}, np={self._num_partitions}, "
            f"M={self._m}, efC={self._ef_construction}, "
            f"nprobes={nprobes}, ef={ef}, refine={refine_factor})"
        )
