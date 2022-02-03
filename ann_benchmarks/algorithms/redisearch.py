from __future__ import absolute_import
from redis import Redis
from redisearch import Client, Query
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.base import BaseANN


class RediSearch(BaseANN):
    def __init__(self, algo, metric, conn_params, method_param):
        self.metric = {'angular': 'cosine', 'euclidean': 'l2'}[metric]
        self.method_param = method_param
        self.algo = algo
        # print(self.method_param,save_index,query_param)
        # self.ef=query_param['ef']
        self.name = 'redisearch-%s (%s)' % (self.algo, self.method_param)
        self.index_name = "ann_benchmark"
        conn = Redis(host=conn_params["host"], port=conn_params["port"],
                     password=conn_params["auth"], username=conn_params["user"],
                     decode_responses=False)
        self.client = Client(self.index_name, conn = conn)

    def fit(self, X, offset=0, cap=None):
        try:
            self.client.redis.execute_command('FT.CREATE', self.index_name, 'SCHEMA', 'vector', 'VECTOR', self.algo, '12', 'TYPE', 'FLOAT32', 'DIM', len(X[0]), 'DISTANCE_METRIC', self.metric, 'INITIAL_CAP', cap if cap else len(X), 'M', self.method_param['M'] , 'EF_CONSTRUCTION', self.method_param["efConstruction"])
        except Exception:
            pass

        for i, x in enumerate(X):
            self.client.redis.execute_command('HSET', f'ann_{i+offset}', 'vector', x.tobytes())

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, k):
        q = Query('*=>[TOP_K ' + str(k) + ' @vector $BLOB EF_RUNTIME ' + str(self.ef) + ']').sort_by('__vector_score', asc=True).no_content()
        return [int(doc.id.replace('ann_','')) for doc in self.client.search(q, query_params = {'BLOB': v.tobytes()}).docs]

    def freeIndex(self):
        self.client.redis.execute_command("FLUSHALL")

