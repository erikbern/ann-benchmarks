"""
ClickHouse module for ann-benchmarks

ClickHouse Vector Search documentation : https://clickhouse.com/docs/engines/table-engines/mergetree-family/annindexes
"""
import clickhouse_connect
import subprocess
import sys
import os
import time

from typing import Dict, Any, Optional

from ..base.module import BaseANN

class clickhouse(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._chclient = None

    def fit(self, X):
        subprocess.run(
               "sudo sed -i 's/<level>trace<\/level>/<level>warning<\/level>/' /etc/clickhouse-server/config.xml",
               shell=True,
               check=True,
               stdout=sys.stdout,
               stderr=sys.stderr)
        subprocess.run(
               "service clickhouse-server start",
               shell=True,
               check=True,
               stdout=sys.stdout,
               stderr=sys.stderr)
        while True:
            try:
                self._chclient = clickhouse_connect.get_client(compress=False, send_receive_timeout = 1800)
                break
            except:
                time.sleep(1)
                continue
        dim = X.shape[1]
        self._dim = dim
        self._chclient.query('DROP TABLE IF EXISTS items_x')
        print("Fitting ...")
        index_granularity = 512
        create_table = 'CREATE TABLE items_x (id Int32, vector Array(Float32) CODEC(NONE))  ENGINE=MergeTree ORDER BY (id) SETTINGS  index_granularity=' + str(index_granularity)
        print(create_table)
        self._chclient.query(create_table);
        rows = []
        for i, embedding in enumerate(X):
            row = [i, embedding]
            rows.append(row)
        self._chclient.query('SET min_insert_block_size_bytes = 2147483648')
        self._chclient.insert('items_x', rows, column_names=['id', 'vector'])
        print("Optimizing table ...")
        self._chclient.query('OPTIMIZE TABLE items_x FINAL SETTINGS mutations_sync=2')
        print("Adding Index ...")
        if self._metric == "angular":
            distfn = "cosineDistance"
        else:
            distfn = "L2Distance"
        self._chclient.query('SET allow_experimental_vector_similarity_index=1')
        add_index = "ALTER TABLE items_x ADD INDEX vector_index vector TYPE vector_similarity('hnsw','" + distfn + "'," + str(dim) + ", 'bf16', " + str(self._m) + "," + str(self._ef_construction) + ")"
        print(add_index)
        self._chclient.query(add_index)
        print("Building Index ...")
        self._chclient.query('ALTER TABLE items_x MATERIALIZE INDEX vector_index SETTINGS mutations_sync = 0')
        while True:
            time.sleep(5)
            done = False;
            result = self._chclient.query("SELECT COUNT(*) FROM system.mutations WHERE table = 'items_x' AND is_done = 0")
            for f in result.result_rows:
                count = int(f[0])
                if count == 0:
                    done = True;
                break
            if done == True:
                break

        return

    def set_query_arguments(self, ef_search, opt1, opt2):
        self._ef_search = ef_search
        self.rescoring_optimization = opt1
        self.search_vector_in_binary = opt2
        # Setting available only in 25.8+
        try:
            if self.rescoring_optimization == True:
                settings = "SET vector_search_with_rescoring = 0"
            else:
                settings = "SET vector_search_with_rescoring = 1"
            self._chclient.query(settings)
        except Exception as e:
            print("Rescoring optimization error : ", e)

        self._chclient.query("SET log_queries = 0")

    def query(self, v, n):
        ef_search_str = " SETTINGS enable_early_constant_folding=0, hnsw_candidate_list_size_for_search=" + str(self._ef_search)
       
        if self.search_vector_in_binary == True:
            params = {'$v1binary$': v.tobytes() }
            if self._metric == "angular":
                result = self._chclient.query("SELECT id FROM items_x ORDER BY cosineDistance(vector, reinterpret($v1binary$, 'Array(Float32)')) LIMIT " + str(n) + ef_search_str,
                                          parameters=params, settings = {'session_id':'session_annb'})
            else:
                result = self._chclient.query("SELECT id FROM items_x ORDER BY L2Distance(vector, reinterpret($v1binary$, 'Array(Float32)')) LIMIT " + str(n) + ef_search_str,
                                          parameters=params, settings={'session_id':'session_annb'})
        else:
            params = {'v1':list(v)}
            if self._metric == "angular":
                result = self._chclient.query("SELECT id FROM items_x ORDER BY cosineDistance(vector, %(v1)s) LIMIT " + str(n) + ef_search_str,
                                          parameters=params, settings={'session_id':'session_annb'})
            else:
                result = self._chclient.query("SELECT id FROM items_x ORDER BY L2Distance(vector, %(v1)s) LIMIT " + str(n) + ef_search_str,
                                          parameters=params, settings={'session_id':'session_annb'})
        rows = []
        for f in result.result_rows:
            rows.append(f[0])
        return rows

    def batch_query(self, v, n):
        self._results_for_batch = [self.query(q, n) for q in v]

    def get_batch_results(self):
        return self._results_for_batch

    def get_memory_usage(self):
        if self._chclient is not None:
            # Load the index first
            if self._metric == "angular":
                result = self._chclient.query("SELECT id FROM items_x ORDER BY cosineDistance(vector, (SELECT vector FROM items_x WHERE id = 1)) LIMIT 1",
                                          settings={'session_id':'session_annb'})
            else:
                result = self._chclient.query("SELECT id FROM items_x ORDER BY L2Distance(vector, (SELECT vector FROM items_x WHERE id = 1)) LIMIT 1", 
                                          settings={'session_id':'session_annb'})
            result = self._chclient.query("SELECT value from system.metrics where name = 'VectorSimilarityIndexCacheBytes'")
            return int(result.result_rows[0][0])
        return 0

    def __str__(self):
        return f"clickhouse(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search}, rescoring_optimzation={self.rescoring_optimization}, binary_string={self.search_vector_in_binary})"
