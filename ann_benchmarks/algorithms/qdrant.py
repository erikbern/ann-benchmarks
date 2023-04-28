from time import sleep

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (CollectionStatus, Distance,
                                       SearchParams, SearchRequest,
                                       VectorParams, OptimizersConfigDiff, QuantizationConfig, ScalarQuantization,
                                       ScalarQuantizationConfig, ScalarType)

from .base import BaseANN


class Qdrant(BaseANN):
    _distances_mapping = {"dot": Distance.DOT, "angular": Distance.COSINE, "euclidean": Distance.EUCLID}

    def __init__(self, metric, quantization):
        self._metric = metric
        self._collection_name = "ann_benchmarks_test"
        self._quantization = quantization
        self._grpc = True
        self._search_params = {"hnsw_ef": None}

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
        if self._quantization:
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    always_ram=True,
                    quantile=0.99,
                    type=ScalarType.INT8,
                )
            )

        self._client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=X.shape[1], distance=self._distances_mapping[self._metric]),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2,
                max_segment_size=100000000,
            ),
            quantization_config=quantization_config,
            # TODO: benchmark this as well
            # hnsw_config=qdrant_models.HnswConfigDiff(
            #     ef_construct=100, #100 is qdrant default
            #     m=16 #16 is qdrant default
            # ),
            timeout=30,
        )

        self._client.upload_collection(
            collection_name=self._collection_name, vectors=X, ids=list(range(X.shape[0])), parallel=1
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

    def set_query_arguments(self, hnsw_ef):
        self._search_params["hnsw_ef"] = hnsw_ef

    def query(self, q, n):
        search_params = SearchParams.construct(hnsw_ef=self._search_params["hnsw_ef"])

        search_result = self._client.search(
            collection_name=self._collection_name,
            query_vector=q,
            search_params=search_params,
            with_payload=False,  # just in case
            limit=n,
        )

        result_ids = [point.id for point in search_result]
        return result_ids

    def batch_query(self, X, n):
        search_queries = [
            SearchRequest.construct(vector=q.tolist(), limit=n,
                                    params=SearchParams(hnsw_ef=self._search_params["hnsw_ef"]))
            for q in X
        ]

        batch_search_results = self._client.search_batch(collection_name=self._collection_name, requests=search_queries)

        self.batch_results = []
        for search_result in batch_search_results:
            self.batch_results.append([point.id for point in search_result])

    def get_batch_results(self):
        return self.batch_results

    def __str__(self):
        hnsw_ef = self._search_params["hnsw_ef"]
        return f"Qdrant(quantization={self._quantization}, hnsw_ef={hnsw_ef})"
