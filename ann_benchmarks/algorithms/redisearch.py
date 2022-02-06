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
        self.name = 'redisearch-%s (%s)' % (self.algo, self.method_param)
        self.index_name = "ann_benchmark"
        conn = Redis(host=conn_params["host"], port=conn_params["port"],
                     password=conn_params["auth"], username=conn_params["user"],
                     decode_responses=False)
        self.client = Client(self.index_name, conn = conn)

    def fit(self, X, offset=0, limit=None):
        limit = limit if limit else len(X)
        # print('inserting %d out of %d vectors' % (limit-offset, len(X)))
        try:
            if self.algo == "HNSW":
                self.client.redis.execute_command('FT.CREATE', self.index_name, 'SCHEMA', 'vector', 'VECTOR', self.algo, '12', 'TYPE', 'FLOAT32', 'DIM', len(X[0]), 'DISTANCE_METRIC', self.metric, 'INITIAL_CAP', len(X), 'M', self.method_param['M'] , 'EF_CONSTRUCTION', self.method_param["efConstruction"])
            elif self.algo == "FLAT":
                self.client.redis.execute_command('FT.CREATE', self.index_name, 'SCHEMA', 'vector', 'VECTOR', self.algo, '10', 'TYPE', 'FLOAT32', 'DIM', len(X[0]), 'DISTANCE_METRIC', self.metric, 'INITIAL_CAP', len(X), 'BLOCK_SIZE', self.method_param['BLOCK_SIZE'])

        except Exception as e:
            if 'Index already exists' not in str(e):
                raise

        for i in range(offset, limit):
            self.client.redis.execute_command('HSET', f'ann_{i}', 'vector', X[i].tobytes())

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, k):
        vq = '*=>[TOP_K ' + str(k) + ' @vector $BLOB'
        vq += ((' EF_RUNTIME ' + str(self.ef)) if self.algo == 'HNSW' else '')
        q = Query(vq + ']').sort_by('__vector_score', asc=True).paging(0, k).no_content()
        return [int(doc.id.replace('ann_','')) for doc in self.client.search(q, query_params = {'BLOB': v.tobytes()}).docs]

    def freeIndex(self):
        self.client.redis.execute_command("FLUSHALL")

