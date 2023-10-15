from time import sleep, time
from typing import Iterable, List, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import grpc
from qdrant_client.http.models import (
    CollectionStatus,
    Distance,
    VectorParams,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    BinaryQuantization,
    BinaryQuantizationConfig,
    ScalarType,
    HnswConfigDiff,
)

from ..base.module import BaseANN

TIMEOUT = 30
BATCH_SIZE = 128


class Qdrant(BaseANN):
    _distances_mapping = {"dot": Distance.DOT, "angular": Distance.COSINE, "euclidean": Distance.EUCLID}

    def __init__(self, metric, quantization, m, ef_construct):
        self._ef_construct = ef_construct
        self._m = m
        self._metric = metric
        self._collection_name = "ann_benchmarks_test"
        self._quantization_mode = quantization
        self._grpc = True
        self._search_params = {"hnsw_ef": None, "rescore": True}
        self.batch_results = []
        self.batch_latencies = []

        qdrant_client_params = {
            "host": "localhost",
            "port": 6333,
            "grpc_port": 6334,
            "prefer_grpc": self._grpc,
            "https": False,
        }
        self._client = QdrantClient(**qdrant_client_params)

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        quantization_config = None
        if self._quantization_mode == "scalar":
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    always_ram=True,
                    quantile=0.99,
                    type=ScalarType.INT8,
                )
            )
        elif self._quantization_mode == "binary":
            quantization_config = BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=True)
            )

        # Disabling indexing during bulk upload
        # https://qdrant.tech/documentation/tutorials/bulk-upload/#disable-indexing-during-upload
        # Uploading to multiple shards
        # https://qdrant.tech/documentation/tutorials/bulk-upload/#parallel-upload-into-multiple-shards
        self._client.recreate_collection(
            collection_name=self._collection_name,
            shard_number=2,
            vectors_config=VectorParams(size=X.shape[1], distance=self._distances_mapping[self._metric]),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2,
                memmap_threshold=20000,
                indexing_threshold=0,
            ),
            quantization_config=quantization_config,
            # TODO: benchmark this as well
            hnsw_config=HnswConfigDiff(
                ef_construct=self._ef_construct,
                m=self._m,
            ),
            timeout=TIMEOUT,
        )

        self._client.upload_collection(
            collection_name=self._collection_name,
            vectors=X,
            ids=list(range(X.shape[0])),
            batch_size=BATCH_SIZE,
            parallel=1,
        )

        # Re-enabling indexing
        self._client.update_collection(
            collection_name=self._collection_name,
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
            ),
            timeout=TIMEOUT,
        )

        # wait for vectors to be fully indexed
        SECONDS_WAITING_FOR_INDEXING_API_CALL = 5

        while True:
            sleep(SECONDS_WAITING_FOR_INDEXING_API_CALL)
            collection_info = self._client.get_collection(self._collection_name)
            if collection_info.status != CollectionStatus.GREEN:
                continue
            sleep(SECONDS_WAITING_FOR_INDEXING_API_CALL)  # the flag is sometimes flacky, better double check
            collection_info = self._client.get_collection(self._collection_name)
            if collection_info.status == CollectionStatus.GREEN:
                print(f"Stored vectors: {collection_info.vectors_count}")
                print(f"Indexed vectors: {collection_info.indexed_vectors_count}")
                print(f"Collection status: {collection_info.indexed_vectors_count}")
                break

    def set_query_arguments(self, hnsw_ef, rescore):
        self._search_params["hnsw_ef"] = hnsw_ef
        self._search_params["rescore"] = rescore

    def query(self, q, n):
        search_request = grpc.SearchPoints(
            collection_name=self._collection_name,
            vector=q.tolist(),
            limit=n,
            with_payload=grpc.WithPayloadSelector(enable=False),
            with_vectors=grpc.WithVectorsSelector(enable=False),
            params=grpc.SearchParams(
                hnsw_ef=self._search_params["hnsw_ef"],
                quantization=grpc.QuantizationSearchParams(
                    ignore=False,
                    rescore=self._search_params["rescore"],
                ),
            ),
        )

        search_result = self._client.grpc_points.Search(search_request, timeout=TIMEOUT)
        result_ids = [point.id.num for point in search_result.result]
        return result_ids

    def batch_query(self, X, n):
        def iter_batches(iterable, batch_size) -> Iterable[List[Any]]:
            """Iterate over `iterable` in batches of size `batch_size`."""
            batch = []
            for item in iterable:
                batch.append(item)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        quantization_search_params = grpc.QuantizationSearchParams(
            ignore=False,
            rescore=self._search_params["rescore"],
        )

        search_queries = [
            grpc.SearchPoints(
                collection_name=self._collection_name,
                vector=q.tolist(),
                limit=n,
                with_payload=grpc.WithPayloadSelector(enable=False),
                with_vectors=grpc.WithVectorsSelector(enable=False),
                params=grpc.SearchParams(
                    hnsw_ef=self._search_params["hnsw_ef"],
                    quantization=quantization_search_params,
                ),
            )
            for q in X
        ]

        self.batch_results = []

        for request_batch in iter_batches(search_queries, BATCH_SIZE):
            start = time()
            grpc_res: grpc.SearchBatchResponse = self._client.grpc_points.SearchBatch(
                grpc.SearchBatchPoints(
                    collection_name=self._collection_name,
                    search_points=request_batch,
                    read_consistency=None,
                ),
                timeout=TIMEOUT,
            )
            self.batch_latencies.extend([time() - start] * len(request_batch))

            for r in grpc_res.result:
                self.batch_results.append([hit.id.num for hit in r.result])

    def get_batch_results(self):
        return self.batch_results

    def get_batch_latencies(self):
        return self.batch_latencies

    def __str__(self):
        hnsw_ef = self._search_params["hnsw_ef"]
        return f"Qdrant(quantization={self._quantization_mode}, hnsw_ef={hnsw_ef})"
