from ann_benchmarks.algorithms.base import BaseANN
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SearchRequest, SearchParams, CollectionStatus
import numpy as np
from time import sleep

class Qdrant(BaseANN):
    
    _distances_mapping = {
        'dot': Distance.DOT,
        'angular': Distance.COSINE,
        'euclidean': Distance.EUCLID
    }

    def __init__(self, metric, grpc):
        self._metric = metric
        self._collection_name = 'ann_benchmarks_test'
        self._grpc = grpc
        self._search_params = {
            'hnsw_ef': None
        }

        qdrant_client_params = {
            'host': 'localhost',
            'port': 6333,
            'grpc_port': 6334,
            'prefer_grpc': self._grpc,
            'https': False,
        }        
        self._client = QdrantClient(**qdrant_client_params)
        

    def fit(self, X):
        if X.dtype != np.float32: X = X.astype(np.float32)

        self._client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=X.shape[1], distance=self._distances_mapping[self._metric]),
            # TODO: benchmark this as well
            # hnsw_config=qdrant_models.HnswConfigDiff(
            #     ef_construct=100, #100 is qdrant default
            #     m=16 #16 is qdrant default
            # ),      
            timeout=30            
        )

        self._client.upload_collection(
            collection_name=self._collection_name,
            vectors=X,
            ids=list(range(X.shape[0])),
            parallel=1
        )

        #wait for vectors to be fully indexed
        SECONDS_WAITING_FOR_INDEXING_API_CALL = 5
        while True:
            collection_info = self._client.http.collections_api.get_collection(self._collection_name).dict()['result']

            vectors_count = collection_info['vectors_count']
            indexed_vectors_count = collection_info['indexed_vectors_count']
            status = collection_info['status']

            print('Stored vectors: ' + str(vectors_count))
            print('Indexed vectors: ' + str(indexed_vectors_count))
            print('Collection status: ' + str(status))
            
            print(type(status), status)
            if status == CollectionStatus.GREEN:
                print('Vectors indexing finished.')
                break
            else:
                print('Waiting ' + str(SECONDS_WAITING_FOR_INDEXING_API_CALL) + ' seconds to query collection info again...')
                sleep(SECONDS_WAITING_FOR_INDEXING_API_CALL)


    def set_query_arguments(self, hnsw_ef):
        self._search_params['hnsw_ef'] = hnsw_ef

    def query(self, q, n):
        search_params = SearchParams(hnsw_ef=self._search_params['hnsw_ef'])

        search_result = self._client.search(
            collection_name=self._collection_name,
            query_vector=q,
            search_params=search_params,
            with_payload=False, #just in case
            limit=n
        )

        result_ids = [point.id for point in search_result]
        return result_ids

    def batch_query(self, X, n):
        search_queries = [SearchRequest(vector=q.tolist(), limit=n, params=SearchParams(hnsw_ef=self._search_params['hnsw_ef'])) for q in X]

        batch_search_results = self._client.search_batch(
            collection_name=self._collection_name,
            requests=search_queries
        )

        self.batch_results = []
        for search_result in batch_search_results:
            self.batch_results.append([point.id for point in search_result])

    def get_batch_results(self):
        return self.batch_results

    def __str__(self):
        return "Qdrant(grpc=%s, hnsw_ef=%s)" % (self._grpc, self._search_params['hnsw_ef'])
