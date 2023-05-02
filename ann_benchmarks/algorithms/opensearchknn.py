from time import sleep
from urllib.request import Request, urlopen

from opensearchpy import ConnectionError, OpenSearch
from opensearchpy.helpers import bulk
from tqdm import tqdm

from .base import BaseANN


class OpenSearchKNN(BaseANN):
    def __init__(self, metric, dimension, method_param):
        self.metric = {"angular": "cosinesimil", "euclidean": "l2"}[metric]
        self.dimension = dimension
        self.method_param = method_param
        self.param_string = "-".join(k + "-" + str(v) for k, v in self.method_param.items()).lower()
        self.index_name = f"os-{self.param_string}"
        self.client = OpenSearch(["http://localhost:9200"])
        self.ef_search = None
        self._wait_for_health_status()

    def _wait_for_health_status(self, wait_seconds=30, status="yellow"):
        for _ in range(wait_seconds):
            try:
                self.client.cluster.health(wait_for_status=status)
                return
            except ConnectionError as e:
                pass
            sleep(1)

        raise RuntimeError("Failed to connect to OpenSearch")

    def fit(self, X):
        body = {
            "settings": {"index": {"knn": True}, "number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}
        }

        mapping = {
            "properties": {
                "id": {"type": "keyword", "store": True},
                "vec": {
                    "type": "knn_vector",
                    "dimension": self.dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": self.metric,
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": self.method_param["efConstruction"],
                            "m": self.method_param["M"],
                        },
                    },
                },
            }
        }

        self.client.indices.create(self.index_name, body=body)
        self.client.indices.put_mapping(mapping, self.index_name)

        print("Uploading data to the Index:", self.index_name)

        def gen():
            for i, vec in enumerate(tqdm(X)):
                yield {"_op_type": "index", "_index": self.index_name, "vec": vec.tolist(), "id": str(i + 1)}

        (_, errors) = bulk(self.client, gen(), chunk_size=500, max_retries=2, request_timeout=10)
        assert len(errors) == 0, errors

        print("Force Merge...")
        self.client.indices.forcemerge(self.index_name, max_num_segments=1, request_timeout=1000)

        print("Refreshing the Index...")
        self.client.indices.refresh(self.index_name, request_timeout=1000)

        print("Running Warmup API...")
        res = urlopen(Request("http://localhost:9200/_plugins/_knn/warmup/" + self.index_name + "?pretty"))
        print(res.read().decode("utf-8"))

    def set_query_arguments(self, ef):
        self.ef_search = ef
        body = {"settings": {"index": {"knn.algo_param.ef_search": ef}}}
        self.client.indices.put_settings(body=body)

    def query(self, q, n):
        body = {"query": {"knn": {"vec": {"vector": q.tolist(), "k": n}}}}

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

        return [int(h["fields"]["id"][0]) - 1 for h in res["hits"]["hits"]]

    def batch_query(self, X, n):
        self.batch_res = [self.query(q, n) for q in X]

    def get_batch_results(self):
        return self.batch_res

    def freeIndex(self):
        self.client.indices.delete(index=self.index_name)

    def __str__(self):
        return f"OpenSearch(index_options: {self.method_param}, ef_search: {self.ef_search})"
