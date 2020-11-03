"""
ann-benchmarks interfaces for Elasticsearch.
Note that this requires X-Pack, which is not included in the OSS version of Elasticsearch.
"""
import logging
from time import sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from ann_benchmarks.algorithms.base import BaseANN

# Configure the elasticsearch logger.
# By default, it writes an INFO statement for every request.
logging.getLogger("elasticsearch").setLevel(logging.WARN)

# Uncomment these lines if you want to see timing for every HTTP request and its duration.
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("elasticsearch").setLevel(logging.INFO)

def es_wait():
    print("Waiting for elasticsearch health endpoint...")
    req = Request("http://localhost:9200/_cluster/health?wait_for_status=yellow&timeout=1s")
    for i in range(30):
        try:
            res = urlopen(req)
            if res.getcode() == 200:
                print("Elasticsearch is ready")
                return
        except URLError:
            pass
        sleep(1)
    raise RuntimeError("Failed to connect to local elasticsearch")


class ElasticsearchScriptScoreQuery(BaseANN):
    """
    KNN using the Elasticsearch dense_vector datatype and script score functions.
    - Dense vector field type: https://www.elastic.co/guide/en/elasticsearch/reference/master/dense-vector.html
    - Dense vector queries: https://www.elastic.co/guide/en/elasticsearch/reference/master/query-dsl-script-score-query.html
    """

    def __init__(self, metric: str, dimension: int):
        self.name = f"elasticsearch-script-score-query_metric={metric}_dimension={dimension}"
        self.metric = metric
        self.dimension = dimension
        self.index = f"es-ssq-{metric}-{dimension}"
        self.es = Elasticsearch(["http://localhost:9200"])
        self.batch_res = []
        if self.metric == "euclidean":
            self.script = "1 / (1 + l2norm(params.query_vec, \"vec\"))"
        elif self.metric == "angular":
            self.script = "1.0 + cosineSimilarity(params.query_vec, \"vec\")"
        else:
            raise NotImplementedError(f"Not implemented for metric {self.metric}")
        es_wait()

    def fit(self, X):
        body = dict(settings=dict(number_of_shards=1, number_of_replicas=0))
        mapping = dict(
            properties=dict(
                id=dict(type="keyword", store=True),
                vec=dict(type="dense_vector", dims=self.dimension)
            )
        )
        self.es.indices.create(self.index, body=body)
        self.es.indices.put_mapping(mapping, self.index)

        def gen():
            for i, vec in enumerate(X):
                yield { "_op_type": "index", "_index": self.index, "vec": vec.tolist(), 'id': str(i + 1) }

        (_, errors) = bulk(self.es, gen(), chunk_size=500, max_retries=9)
        assert len(errors) == 0, errors

        self.es.indices.refresh(self.index)
        self.es.indices.forcemerge(self.index, max_num_segments=1)

    def query(self, q, n):
        body = dict(
            query=dict(
                script_score=dict(
                    query=dict(match_all=dict()),
                    script=dict(
                        source=self.script,
                        params=dict(query_vec=q.tolist())
                    )
                )
            )
        )
        res = self.es.search(index=self.index, body=body, size=n, _source=False, docvalue_fields=['id'],
                             stored_fields="_none_", filter_path=["hits.hits.fields.id"])
        return [int(h['fields']['id'][0]) - 1 for h in res['hits']['hits']]

    def batch_query(self, X, n):
        self.batch_res = [self.query(q, n) for q in X]

    def get_batch_results(self):
        return self.batch_res

