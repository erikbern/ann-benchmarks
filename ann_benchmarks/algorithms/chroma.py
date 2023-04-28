import uuid

import chromadb
from qdrant_client import ChromaClient
from qdrant_client.http.models import (CollectionStatus, Distance,
                                       SearchParams, SearchRequest,
                                       VectorParams)

from .base import BaseANN


class Chroma(BaseANN):

    def __init__(self, metric, grpc):
        self._metric = metric
        self._collection_name = "ann_benchmarks_test"
        self._collection = None

        self._client = chromadb.Client()

    def fit(self, X):
        collection = self._client.create_collection(
           self.collection_name,
        )

        self._collection = collection

        documents = []
        metadatas = []
        ids = []
        for i, x in enumerate(X):
            documents.append(x)
            properties = {"i": i}
            metadatas.append(properties)
            ids.append(uuid.UUID(int=i))

        self._collection.add(documents, metadatas, ids)

    def set_query_arguments(self):
        # we do nothing

    def query(self, v, n):
        results = self._collection.query(
            query_texts=[v],
            n_results=n,
            # where={"metadata_field": "is_equal_to_this"}, # optional filter
            # where_document={"$contains":"search_string"}  # optional filter
        )
        # results: {'ids': [['id7', 'id2', 'id8']], 'embeddings': None, 'documents': [['A document that discusses international affairs', 'A document that discusses international affairs', 'A document that discusses global affairs']], 'metadatas': [[{'status': 'read'}, {'status': 'unread'}, {'status': 'unread'}]], 'distances': [[16.740001678466797, 87.22000122070312, 87.22000122070312]]}

        return results.ids

    def __str__(self):
        return f"Chroma()"
