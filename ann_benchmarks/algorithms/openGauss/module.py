"""
This module supports connecting to a openGauss instance and performing vector
indexing and search using the pgvector extension. The default behavior uses
the "ann" value of openGauss user name, password, and database name, as well
as the default host and port values of the psycopg driver.

If openGaussis managed externally, e.g. in a cloud DBaaS environment, the
environment variable overrides listed below are available for setting openGauss
connection parameters:

ANN_BENCHMARKS_OG_USER
ANN_BENCHMARKS_OG_PASSWORD
ANN_BENCHMARKS_OG_DBNAME
ANN_BENCHMARKS_OG_HOST
ANN_BENCHMARKS_OG_PORT

This module starts the openGauss service automatically using the "service"
command. The environment variable ANN_BENCHMARKS_OG_START_SERVICE could be set
to "false" (or e.g. "0" or "no") in order to disable this behavior.

This module will also attempt to create the pgvector extension inside the
target database, if it has not been already created.
"""
import subprocess
import sys
import os

import pgvector.psycopg
import psycopg

from typing import Dict, Any, Optional

from ..base.module import BaseANN
from ...util import get_bool_env_var

import multiprocessing
from multiprocessing import Pool
import numpy

def get_pg_param_env_var_name(pg_param_name: str) -> str:
    return f'ANN_BENCHMARKS_OG_{pg_param_name.upper()}'


def get_pg_conn_param(
        pg_param_name: str,
        default_value: Optional[str] = None) -> Optional[str]:
    env_var_name = get_pg_param_env_var_name(pg_param_name)
    env_var_value = os.getenv(env_var_name, default_value)
    if env_var_value is None or len(env_var_value.strip()) == 0:
        return default_value
    return env_var_value


class openGaussHNSW(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None
        self.curs = []
        self.connections = []
        self.init_conn_flags = False
        self.search_pool = None
        self.con = method_param['concurrents']

        # default value
        self.user = 'ann'
        self.port = 5432
        self.dbname = 'ann'
        self.host = 'localhost'
        self.password = 'ann'

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        psycopg_connect_kwargs: Dict[str, Any] = dict(
            autocommit=True,
        )
        for arg_name in ['user', 'password', 'dbname']:
            # The default value is "ann" for all of these parameters.
            psycopg_connect_kwargs[arg_name] = get_pg_conn_param(
                arg_name, 'ann')
            if arg_name == 'user':
                self.user = psycopg_connect_kwargs['user']
            if arg_name == 'password':
                self.password = psycopg_connect_kwargs['password']
            if arg_name == 'dbname':
                self.dbname = psycopg_connect_kwargs['dbname']

        # If host/port are not specified, leave the default choice to the
        # psycopg driver.
        og_host: Optional[str] = get_pg_conn_param('host')
        if og_host is not None:
            psycopg_connect_kwargs['host'] = og_host
            self.host = og_host

        og_port_str: Optional[str] = get_pg_conn_param('port')
        if og_port_str is not None:
            psycopg_connect_kwargs['port'] = int(og_port_str)
            self.port = int(og_port_str)

        conn = psycopg.connect(**psycopg_connect_kwargs)

        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        cur.execute("ALTER TABLE items SET (parallel_workers = 32);")
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        print("creating index...")
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops) WITH (m = %d, ef_construction = %d)" % (self._m, self._ef_construction)
            )
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = %d, ef_construction = %d)" % (self._m, self._ef_construction))
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        self._cur = cur


    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET hnsw.ef_search = %d" % ef_search)
        self._cur.execute("set enable_seqscan = off")
        conc = self.con
        self.search_pool = Pool(conc, initializer=self.init_connection)

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024
    
    # batch search
    def init_connection(self):
        conn = psycopg.connect(user=self.user, password=self.password, dbname=self.dbname, host=self.host, autocommit=True, port=self.port)
        pgvector.psycopg.register_vector(conn)
        global cur
        cur = conn.cursor()
        cur.execute("SET hnsw_ef_search = %d" % self._ef_search)
        cur.execute("set enable_seqscan = off")
        global base_query
        base_query = self._query

    def close_connections(self):
        for cur in self.curs:
            if cur is not None:
                cur.close()
                cur = None
        for conn in self.connections:
            if conn is not None:
                conn.close()
                conn = None
        self.curs = []
        self.connections = []
        self.init_conn_flags = False
    
    @staticmethod
    def sub_query(chunk, n):
        ids = []
        for item in chunk:
            cur.execute(base_query, (item, n), binary=True, prepare=True)
            res = cur.fetchall()
            ids.append([i[0] for i in res])
        return ids
    
    def batch_query(self, X: numpy.array, n: int):
        conc = self.con
        chunk_size = len(X) // conc + 1
        chunks = [X[i:i + chunk_size] for i in range(0, len(X), chunk_size)]
        res = self.search_pool.starmap(self.sub_query, [(chunk, n) for chunk in chunks])
        self.res = []
        for item in res:
            for ids in item:
                self.res.append(ids)

    def __str__(self):
        return f"openGaussHNSW(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"

    def __del__(self):
        self.close_connections()

class openGaussHNSWPQ(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None
        self.curs = []
        self.connections = []
        self.init_conn_flags = False
        self.search_pool = None
        self.con = method_param['concurrents']
        self.hnsw_earlystop_threshold = method_param['hnswEarlystopThreshold']
        self.pq_m = method_param['pqM']

        # default value
        self.user = 'ann'
        self.port = 5432
        self.dbname = 'ann'
        self.host = 'localhost'
        self.password = 'ann'

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        psycopg_connect_kwargs: Dict[str, Any] = dict(
            autocommit=True,
        )
        for arg_name in ['user', 'password', 'dbname']:
            # The default value is "ann" for all of these parameters.
            psycopg_connect_kwargs[arg_name] = get_pg_conn_param(
                arg_name, 'ann')
            if arg_name == 'user':
                self.user = psycopg_connect_kwargs['user']
            if arg_name == 'password':
                self.password = psycopg_connect_kwargs['password']
            if arg_name == 'dbname':
                self.dbname = psycopg_connect_kwargs['dbname']

        # If host/port are not specified, leave the default choice to the
        # psycopg driver.
        og_host: Optional[str] = get_pg_conn_param('host')
        if og_host is not None:
            psycopg_connect_kwargs['host'] = og_host
            self.host = og_host

        og_port_str: Optional[str] = get_pg_conn_param('port')
        if og_port_str is not None:
            psycopg_connect_kwargs['port'] = int(og_port_str)
            self.port = int(og_port_str)

        conn = psycopg.connect(**psycopg_connect_kwargs)

        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        cur.execute("ALTER TABLE items SET (parallel_workers = 32);")
        cur.execute("set hnsw_earlystop_threshold = %d" % self.hnsw_earlystop_threshold)
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        print(f"creating index [hnsw_earlystop_threshold={self.hnsw_earlystop_threshold}]...")
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops) WITH (m = %d, ef_construction = %d, enable_pq = on, pq_m = %d)" % (self._m, self._ef_construction, self.pq_m))
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = %d, ef_construction = %d, enable_pq = on, pq_m = %d)" % (self._m, self._ef_construction, self.pq_m))
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        self._cur = cur

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET hnsw_ef_search = %d" % ef_search)
        self._cur.execute("set enable_seqscan = off")
        self._cur.execute("set hnsw_earlystop_threshold = %d" % self.hnsw_earlystop_threshold)
        conc = self.con
        self.search_pool = Pool(conc, initializer=self.init_connection)

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024
    
    def init_connection(self):
        conn = psycopg.connect(user=self.user, password=self.password, dbname=self.dbname, host=self.host, autocommit=True, port=self.port)
        pgvector.psycopg.register_vector(conn)
        global cur
        cur = conn.cursor()
        cur.execute("SET hnsw_ef_search = %d" % self._ef_search)
        cur.execute("set enable_seqscan = off")
        cur.execute("set hnsw_earlystop_threshold = %d" % self.hnsw_earlystop_threshold)
        global base_query
        base_query = self._query

    def close_connections(self):
        for cur in self.curs:
            if cur is not None:
                cur.close()
                cur = None
        for conn in self.connections:
            if conn is not None:
                conn.close()
                conn = None
        self.curs = []
        self.connections = []
        self.init_conn_flags = False
    
    @staticmethod
    def sub_query(chunk, n):
        ids = []
        for item in chunk:
            cur.execute(base_query, (item, n), binary=True, prepare=True)
            res = cur.fetchall()
            ids.append([i[0] for i in res])
        return ids
    
    def batch_query(self, X: numpy.array, n: int):
        conc = self.con
        chunk_size = len(X) // conc + 1
        chunks = [X[i:i + chunk_size] for i in range(0, len(X), chunk_size)]
        res = self.search_pool.starmap(self.sub_query, [(chunk, n) for chunk in chunks])
        print(f"openGauss concuiiency_query process num: {conc}")
        self.res = []
        for item in res:
            for ids in item:
                self.res.append(ids)

    def __str__(self):
        return f"openGaussHNSWPQ(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search}, pq_m={self.pq_m}, hnsw_earlystop_threshold={self.hnsw_earlystop_threshold})"
    
    def __del__(self):
        self.close_connections()


class openGaussIVF(BaseANN):
    def __init__(self, metric, lists):
        self._metric = metric
        self._lists = lists
        self._cur = None

        # default value
        self.user = 'ann'
        self.port = 5432
        self.dbname = 'ann'
        self.host = 'localhost'
        self.password = 'ann'

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        psycopg_connect_kwargs: Dict[str, Any] = dict(
            autocommit=True,
        )
        for arg_name in ['user', 'password', 'dbname']:
            # The default value is "ann" for all of these parameters.
            psycopg_connect_kwargs[arg_name] = get_pg_conn_param(
                arg_name, 'ann')
            if arg_name == 'user':
                self.user = psycopg_connect_kwargs['user']
            if arg_name == 'password':
                self.password = psycopg_connect_kwargs['password']
            if arg_name == 'dbname':
                self.dbname = psycopg_connect_kwargs['dbname']

        # If host/port are not specified, leave the default choice to the
        # psycopg driver.
        og_host: Optional[str] = get_pg_conn_param('host')
        if og_host is not None:
            psycopg_connect_kwargs['host'] = og_host
            self.host = og_host

        og_port_str: Optional[str] = get_pg_conn_param('port')
        if og_port_str is not None:
            psycopg_connect_kwargs['port'] = int(og_port_str)
            self.port = int(og_port_str)

        conn = psycopg.connect(**psycopg_connect_kwargs)

        pgvector.psycopg.register_vector(conn)
        
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        cur.execute("ALTER TABLE items SET (parallel_workers = 32);")
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        print(f"creating index ...")
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = %d)" % (self._lists))
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists= %d)" % (self._lists))
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        self._cur = cur

    def set_query_arguments(self, probes):
        self._probes = probes
        self._cur.execute("SET ivfflat_probes = %d" % probes)
        self._cur.execute("set enable_seqscan = off")

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024

    def __str__(self):
        return f"openGaussIVF(lists={self._lists}, probes={self._probes})"