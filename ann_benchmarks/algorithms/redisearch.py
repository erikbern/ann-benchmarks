from __future__ import absolute_import
from optparse import Values
from redis import Redis
from redis.cluster import RedisCluster
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.base import BaseANN
import math


class RediSearch(BaseANN):
    def __init__(self, algo, metric, conn_params, method_param):
        self.metric = {'angular': 'cosine', 'euclidean': 'l2'}[metric]
        self.method_param = method_param
        self.algo = algo
        self.name = 'redisearch-%s (%s)' % (self.algo, self.method_param)
        self.index_name = "ann_benchmark"
        self.text = None
        
        redis = RedisCluster if conn_params['cluster'] else Redis
        host = conn_params["host"] if conn_params["host"] else 'localhost'
        port = conn_params["port"] if conn_params["port"] else 6379
        self.redis = redis(host=host, port=port, decode_responses=False,
                           password=conn_params["auth"], username=conn_params["user"])
        self.shards = int(conn_params["shards"])
        if conn_params['cluster']:
            self.shards = len(self.redis.get_primaries())

    def fit(self, X, offset=0, limit=None, hybrid_buckets = None):
        limit = limit if limit else len(X)
        try:
            args = [self.index_name, 'SCHEMA']
            if hybrid_buckets:
                args.extend(['n', 'NUMERIC', 't', 'TEXT'])
            # https://oss.redis.com/redisearch/master/Commands/#ftcreate
            if self.algo == "HNSW":
                args.extend(['vector', 'VECTOR', self.algo, '10', 'TYPE', 'FLOAT32', 'DIM', len(X[0]), 'DISTANCE_METRIC', self.metric, 'M', self.method_param['M'], 'EF_CONSTRUCTION', self.method_param["efConstruction"]])
            elif self.algo == "FLAT":
                args.extend(['vector', 'VECTOR', self.algo, '6', 'TYPE', 'FLOAT32', 'DIM', len(X[0]), 'DISTANCE_METRIC', self.metric])
            print("Calling FT.CREATE", *args)
            self.redis.execute_command('FT.CREATE', *args,  target_nodes='random')
        except Exception as e:
            if 'Index already exists' not in str(e):
                raise
        p = self.redis.pipeline(transaction=False)
        count = 0
        if hybrid_buckets:
            print('running hybrid')
            for bucket in hybrid_buckets.values():
                ids = bucket['ids']
                text = bucket['text'].decode()
                number = bucket['number']
                print('calling HSET', f'<id>', 'vector', '<vector blob>', 't', text, 'n', number)
                for id in ids:
                    if id >= offset and id < limit:
                        p.execute_command('HSET', int(id), 'vector', X[id].tobytes(), 't', text, 'n', int(number))
                        count+=1
                        if count % 1000 == 0:
                            p.execute()
                            p.reset()
            p.execute()
        else:
            for i in range(offset, limit):
                p.execute_command('HSET', i, 'vector', X[i].tobytes())
                count+=1
                if count % 1000 == 0:
                    p.execute()
                    p.reset()
            p.execute()

    def set_query_arguments(self, ef):
        self.ef = ef

    def set_hybrid_query(self, text):
        self.text = text

    def query(self, v, k):
        # https://oss.redis.com/redisearch/master/Commands/#ftsearch
        qparams = f' EF_RUNTIME {self.ef}' if self.algo == 'HNSW' else ''
        if self.text:
            vq = f'(@t:{self.text})=>[KNN {k} @vector $BLOB {qparams}]'
        else:
            vq = f'*=>[KNN {k} @vector $BLOB {qparams}]'
        q = ['FT.SEARCH', self.index_name, vq, 'NOCONTENT', 'SORTBY', '__vector_score', 'LIMIT', '0', str(k), 'PARAMS', '2', 'BLOB', v.tobytes(), 'DIALECT', '2']
        return [int(doc) for doc in self.redis.execute_command(*q, target_nodes='random')[1:]]

    def freeIndex(self):
        self.redis.execute_command("FLUSHALL")

    def __str__(self):
        return self.name + f", efRuntime: {self.ef}"
