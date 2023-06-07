from time import sleep

from elasticsearch import ConnectionError, Elasticsearch
from elasticsearch.helpers import bulk

from ..base.module import BaseANN


class ElasticsearchKNN(BaseANN):
    """Elasticsearch KNN search.

    See https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html for more details.
    """

    def __init__(self, metric: str, dimension: int, index_options: dict):
        self.metric = metric
        self.dimension = dimension
        self.index_options = index_options
        self.num_candidates = 100

        index_options_str = "-".join(sorted(f"{k}-{v}" for k, v in self.index_options.items()))
        self.index_name = f"{metric}-{dimension}-{index_options_str}"
        self.similarity_metric = self._vector_similarity_metric(metric)

        self.client = Elasticsearch(["http://localhost:9200"])
        self.batch_res = []
        self._wait_for_health_status()

    def _vector_similarity_metric(self, metric: str):
        # `dot_product` is more efficient than `cosine`, but requires all vectors to be normalized
        # to unit length. We opt for adaptability, some datasets might not be normalized.
        supported_metrics = {
            "angular": "cosine",
            "euclidean": "l2_norm",
        }
        if metric not in supported_metrics:
            raise NotImplementedError(f"{metric} is not implemented")
        return supported_metrics[metric]

    def _wait_for_health_status(self, wait_seconds=30, status="yellow"):
        print("Waiting for Elasticsearch ...")
        for _ in range(wait_seconds):
            try:
                health = self.client.cluster.health(wait_for_status=status, request_timeout=1)
                print(f'Elasticsearch is ready: status={health["status"]}')
                return
            except ConnectionError:
                pass
            sleep(1)
        raise RuntimeError("Failed to connect to Elasticsearch")

    def fit(self, X):
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": -1,
        }
        mappings = {
            "properties": {
                "id": {"type": "keyword", "store": True},
                "vec": {
                    "type": "dense_vector",
                    "element_type": "float",
                    "dims": self.dimension,
                    "index": True,
                    "similarity": self.similarity_metric,
                    "index_options": {
                        "type": self.index_options.get("type", "hnsw"),
                        "m": self.index_options["m"],
                        "ef_construction": self.index_options["ef_construction"],
                    },
                },
            },
        }
        self.client.indices.create(index=self.index_name, settings=settings, mappings=mappings)

        def gen():
            for i, vec in enumerate(X):
                yield {"_op_type": "index", "_index": self.index_name, "id": str(i), "vec": vec.tolist()}

        print("Indexing ...")
        (_, errors) = bulk(self.client, gen(), chunk_size=500, request_timeout=90)
        if len(errors) != 0:
            raise RuntimeError("Failed to index documents")

        print("Force merge index ...")
        self.client.indices.forcemerge(index=self.index_name, max_num_segments=1, request_timeout=900)

        print("Refreshing index ...")
        self.client.indices.refresh(index=self.index_name, request_timeout=900)

    def set_query_arguments(self, num_candidates):
        self.num_candidates = num_candidates

    def query(self, q, n):
        if n > self.num_candidates:
            raise ValueError("n must be smaller than num_candidates")

        body = {
            "knn": {
                "field": "vec",
                "query_vector": q.tolist(),
                "k": n,
                "num_candidates": self.num_candidates,
            }
        }
        res = self.client.search(
            index=self.index_name,
            body=body,
            size=n,
            _source=False,
            docvalue_fields=["id"],
            stored_fields="_none_",
            filter_path=["hits.hits.fields.id"],
            request_timeout=10,
        )
        return [int(h["fields"]["id"][0]) for h in res["hits"]["hits"]]

    def batch_query(self, X, n):
        self.batch_res = [self.query(q, n) for q in X]

    def get_batch_results(self):
        return self.batch_res

    def __str__(self):
        return f"Elasticsearch(index_options: {self.index_options}, num_canditates: {self.num_candidates})"