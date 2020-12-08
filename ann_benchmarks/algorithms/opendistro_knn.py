import logging
from time import sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from ann_benchmarks.algorithms.base import BaseANN

# Leo additional
import sys
import requests
import json

# Configure the logger.
logging.getLogger("elasticsearch").setLevel(logging.WARN)

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

class OpenDistroKNN(BaseANN):
    def __init__(self, metric, dimension, method_param):
        self.metric = {"angular": "cosinesimil", "euclidean": "l2"}[metric]
        self.dimension = dimension
        self.method_param = method_param
        self.param_string = "-".join(k+"-"+str(v) for k,v in self.method_param.items()).lower()
        self.name = f"od-{self.param_string}"
        self.es = Elasticsearch(["http://localhost:9200"])
        self.url = "http://localhost:9200/"+self.name
        es_wait()

    def fit(self, X):
        body = {
            "settings": {
                "index": {
                    "knn": True, 
                    "knn.space_type": self.metric, 
                    "knn.algo_param.ef_construction": self.method_param["efConstruction"], 
                    "knn.algo_param.m": self.method_param["M"]
                },
                "number_of_shards": 1, 
                "number_of_replicas": 0, 
            }
        }

        mapping = {
            "properties": {
                "id": {"type": "keyword", "store": True},
                "vec": {"type": "knn_vector", "dimension": self.dimension}
            }
        }
            
        self.es.indices.create(self.name, body=body)
        self.es.indices.put_mapping(mapping, self.name)

        print("Uploading data to the Index:", self.name)
        def gen():
            for i, vec in enumerate(X):
                yield { "_op_type": "index", "_index": self.name, "vec": vec.tolist(), 'id': str(i + 1) }

        (_, errors) = bulk(self.es, gen(), chunk_size=500, max_retries=9)
        assert len(errors) == 0, errors
   
        print("Running Warmup API...")
        res = urlopen(Request("http://localhost:9200/_opendistro/_knn/warmup/"+self.name+"?pretty"))
        print(res.read().decode("utf-8"))

        # Read once to load into memory    
        search_url = self.url + "/_msearch" # Query Multiple products at a time
        step = 10

        for n in range(0, 500, step):
            subset_X = X[n:n+step, :]
            data_payload = ''
            for row in subset_X:
                prod_payload = {"size": 5, "query": {"knn": {"vec": {"vector": row.tolist(), "k": 5}}}}
                data_payload += '{}\n' + json.dumps(prod_payload) + '\n'

            r = requests.get(search_url, data=data_payload, headers={'content-type':'application/json'})

        # self.es.indices.refresh(self.name, request_timeout=1000)
        # self.es.indices.forcemerge(self.name, max_num_segments=1, request_timeout=1000)

    def set_query_arguments(self, ef):
        body = {
            "settings": {
                "index": {"knn.algo_param.ef_search": ef}
            }
        }
        self.es.indices.put_settings(body=body)

    def query(self, q, n):
        body = {
            "size": n, 
            "query": {
                "knn": {
                    "vec": {"vector": q.tolist(), "k": n}
                }
            }
        }

        res = self.es.search(index=self.name, body=body, size=n, _source=False, docvalue_fields=['id'],
                             stored_fields="_none_", filter_path=["hits.hits.fields.id"], request_timeout=10)
        
        return [int(h['fields']['id'][0]) - 1 for h in res['hits']['hits']]

    def batch_query(self, X, n):
        self.batch_res = [self.query(q, n) for q in X]

    def get_batch_results(self):
        return self.batch_res
    
    def freeIndex(self):
        self.es.indices.delete(index=self.name)