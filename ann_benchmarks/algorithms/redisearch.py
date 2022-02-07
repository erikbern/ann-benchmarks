from __future__ import absolute_import
from redis import Redis
from redis.cluster import RedisCluster
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.base import BaseANN


class RediSearch(BaseANN):
    def __init__(self, algo, metric, conn_params, method_param):
        self.metric = {'angular': 'cosine', 'euclidean': 'l2'}[metric]
        self.method_param = method_param
        self.algo = algo
        self.name = 'redisearch-%s (%s)' % (self.algo, self.method_param)
        self.index_name = "ann_benchmark"
        
        redis = RedisCluster if conn_params['cluster'] else Redis
        self.redis = redis(host=conn_params["host"], port=conn_params["port"],
                           password=conn_params["auth"], username=conn_params["user"],
                           decode_responses=False)

    def fit(self, X, offset=0, limit=None):
        limit = limit if limit else len(X)
        # print('inserting %d out of %d vectors' % (limit-offset, len(X)))
        try:
            if self.algo == "HNSW":
                self.redis.execute_command('FT.CREATE', self.index_name, 'SCHEMA', 'vector', 'VECTOR', self.algo, '12', 'TYPE', 'FLOAT32', 'DIM', len(X[0]), 'DISTANCE_METRIC', self.metric, 'INITIAL_CAP', len(X), 'M', self.method_param['M'] , 'EF_CONSTRUCTION', self.method_param["efConstruction"], target_nodes='random')
            elif self.algo == "FLAT":
                self.redis.execute_command('FT.CREATE', self.index_name, 'SCHEMA', 'vector', 'VECTOR', self.algo, '10', 'TYPE', 'FLOAT32', 'DIM', len(X[0]), 'DISTANCE_METRIC', self.metric, 'INITIAL_CAP', len(X), 'BLOCK_SIZE', self.method_param['BLOCK_SIZE'], target_nodes='random')
        except Exception as e:
            if 'Index already exists' not in str(e):
                raise

        for i in range(offset, limit):
            self.redis.execute_command('HSET', f'ann_{i}', 'vector', X[i].tobytes())

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, k):
        qparams = (' EF_RUNTIME ' + str(self.ef)) if self.algo == 'HNSW' else ''
        vq = '*=>[TOP_K ' + str(k) + ' @vector $BLOB' + qparams + ']'
        q = ['FT.SEARCH', self.index_name, vq, 'NOCONTENT', 'SORTBY', '__vector_score', 'LIMIT', '0', str(k), 'PARAMS', '2', 'BLOB', v.tobytes()]
        return [int(doc.replace(b'ann_',b'')) for doc in self.redis.execute_command(*q, target_nodes='random')[1:]]

    def freeIndex(self):
        self.redis.execute_command("FLUSHALL")

