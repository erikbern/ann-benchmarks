import logging
from time import sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from ann_benchmarks.algorithms.base import BaseANN

from .elasticsearch import es_wait

from tqdm import tqdm

# Configure the logger.
logging.getLogger("elasticsearch").setLevel(logging.WARN)

class OpenSearchKNN(BaseANN):
    def __init__(self, metric, dimension, method_param):
        self.metric = {"angular": "cosinesimil", "euclidean": "l2"}[metric]
        self.dimension = dimension
        self.method_param = method_param
        self.param_string = "-".join(k+"-"+str(v) for k,v in self.method_param.items()).lower()
        self.name = f"os-{self.param_string}"
        self.es = Elasticsearch(["http://localhost:9200"])
        es_wait()

    def fit(self, X):
        body = {
            "settings": {
                "index": {"knn": True},
                "number_of_shards": 1, 
                "number_of_replicas": 0,
                "refresh_interval": -1
            }
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
                            "m": self.method_param["M"]
                        }
                    }
                }
            }
        }
            
        self.es.indices.create(self.name, body=body)
        self.es.indices.put_mapping(mapping, self.name)

        print("Uploading data to the Index:", self.name)
        def gen():
            for i, vec in enumerate(tqdm(X)):
                yield { "_op_type": "index", "_index": self.name, "vec": vec.tolist(), 'id': str(i + 1) }

        (_, errors) = bulk(self.es, gen(), chunk_size=500, max_retries=2, request_timeout=10)
        assert len(errors) == 0, errors
          
        print("Force Merge...")
        self.es.indices.forcemerge(self.name, max_num_segments=1, request_timeout=1000)
               
        print("Refreshing the Index...")
        self.es.indices.refresh(self.name, request_timeout=1000)
       
        print("Running Warmup API...")
        res = urlopen(Request("http://localhost:9200/_plugins/_knn/warmup/"+self.name+"?pretty"))
        print(res.read().decode("utf-8"))

    def set_query_arguments(self, ef):
        body = {
            "settings": {
                "index": {"knn.algo_param.ef_search": ef}
            }
        }
        self.es.indices.put_settings(body=body)

    def query(self, q, n):
        body = {
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