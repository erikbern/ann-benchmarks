from __future__ import absolute_import
import os
import base64
from redis import Redis
from redisearch import Client, Query 
import numpy as np
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.base import BaseANN


class RediSearch(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {'angular': 'cosine', 'euclidean': 'l2'}[metric]
        self.method_param = method_param
        # print(self.method_param,save_index,query_param)
        # self.ef=query_param['ef']
        self.name = 'redisearch (%s)' % (self.method_param)
        self.index_name = "ann_benchmark"
        self.client = Client(self.index_name, conn = Redis(decode_responses=False))

    def fit(self, X):
        # Only l2 is supported currently
        self.client.redis.execute_command('FT.CREATE', self.index_name, 'SCHEMA', 'vector',  'VECTOR', 'FLOAT32', len(X[0]), 'L2', 'HNSW', 'INITIAL_CAP', len(X), 'M', self.method_param['M'] , 'EF', self.method_param["efConstruction"])
        for i, x in enumerate(X):
            self.client.redis.execute_command('HSET', f'ann_{i}', 'vector', x.tobytes())


    def set_query_arguments(self, ef):
        pass

    def query(self, v, k):
        base64_vector = base64.b64encode(v).decode('ascii')
        base64_vector_escaped = base64_vector.translate(str.maketrans({"=":  r"\=",
                                              "/":  r"\/",
                                              "+":  r"\+"}))
        q = Query('@vector:[' + base64_vector_escaped + ' TOPK ' +str(k)+'] => {$BASE64:TRUE}')
        return [int(doc.id.replace('ann_','')) for doc in self.client.search(q).docs]

    def freeIndex(self):
        self.client.redis.execute_command("FLUSHALL")

