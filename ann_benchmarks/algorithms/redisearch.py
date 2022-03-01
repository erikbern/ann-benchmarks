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
        host = conn_params["host"] if conn_params["host"] else 'localhost'
        port = conn_params["port"] if conn_params["port"] else 6379
        self.redis = redis(host=host, port=port, decode_responses=False,
                           password=conn_params["auth"], username=conn_params["user"])

    def fit(self, X, offset=0, limit=None):
        limit = limit if limit else len(X)
        try:
            # https://oss.redis.com/redisearch/master/Commands/#ftcreate
            if self.algo == "HNSW":
                self.redis.execute_command('FT.CREATE', self.index_name, 'SCHEMA', 'vector', 'VECTOR', self.algo, '12', 'TYPE', 'FLOAT32', 'DIM', len(X[0]), 'DISTANCE_METRIC', self.metric, 'INITIAL_CAP', len(X), 'M', self.method_param['M'], 'EF_CONSTRUCTION', self.method_param["efConstruction"], target_nodes='random')
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
        # https://oss.redis.com/redisearch/master/Commands/#ftsearch
        qparams = f' EF_RUNTIME {self.ef}' if self.algo == 'HNSW' else ''
        vq = f'*=>[KNN {k} @vector $BLOB {qparams}]'
        q = ['FT.SEARCH', self.index_name, vq, 'NOCONTENT', 'SORTBY', '__vector_score', 'LIMIT', '0', str(k), 'PARAMS', '2', 'BLOB', v.tobytes()]
        return [int(doc.replace(b'ann_',b'')) for doc in self.redis.execute_command(*q, target_nodes='random')[1:]]

    def freeIndex(self):
        self.redis.execute_command("FLUSHALL")

