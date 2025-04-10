import glob
import os
import subprocess
import sys
import time

from itertools import chain
from multiprocessing.pool import Pool

import mariadb
import numpy as np
import psutil

from ..base.module import BaseANN

def vector_to_hex(v):
    """Convert vector to bytes for database storage"""
    return np.array(v, 'float32').tobytes()

def many_queries(arg):
    conn = mariadb.connect()
    cur = conn.cursor()

    res = []
    for v in arg[2]:
        cur.execute(arg[0], (vector_to_hex(v), arg[1]))
        res.append([id for id, in cur.fetchall()])

    return res

class MariaDB(BaseANN):

    def __init__(self, metric, method_param):
        self._m = method_param['M']
        self._engine = method_param['engine']
        self._cur = None

        self._metric_type = {"angular": "cosine", "euclidean": "euclidean"}.get(metric, None)
        if self._metric_type is None:
            raise Exception(f"[MariaDB] Not support metric type: {metric}!!!")

        self._sql_create_table = f"CREATE TABLE ann.ann (id INT PRIMARY KEY, v VECTOR(%d) NOT NULL) ENGINE={self._engine}"
        self._sql_insert = f"INSERT INTO ann.ann (id, v) VALUES (%s, %s)"
        self._sql_create_index = f"ALTER TABLE ann.ann ADD VECTOR KEY v(v) DISTANCE={self._metric_type} M={self._m}"
        self._sql_search = f"SELECT id FROM ann.ann ORDER by vec_distance_{self._metric_type}(v, %s) LIMIT %d"
        
        self.start_db()

        # Connect to MariaDB using Unix socket
        conn = mariadb.connect()
        self._cur = conn.cursor()

    def start_db(self):
        # Get free memory in MB
        free_memory = psutil.virtual_memory().available
        
        # Set buffer/cache size
        innodb_buffer_size = int(free_memory * 0.4)
        key_buffer_size = int(free_memory * 0.3)
        mhnsw_cache_size = int(free_memory * 0.4)
        
        subprocess.run(
            f"service mariadb start --skip-networking "
            f"--innodb-buffer-pool-size={innodb_buffer_size} "
            f"--key-buffer-size={key_buffer_size} "
            # f"--general_log=1 --general_log_file=/var/lib/mysql/general.log "
            f"--mhnsw-max-cache-size={mhnsw_cache_size}",
            shell=True,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

    def fit(self, X, batch_size=1000):
        """
        Fit the database with vectors
        
        Parameters:
            X: numpy array of vectors to insert
            batch_size: number of records to insert in each batch
        """
        # Prepare database and table
        self._cur.execute("SET GLOBAL max_allowed_packet = 1073741824")
        self._cur.execute("DROP DATABASE IF EXISTS ann")
        self._cur.execute("CREATE DATABASE ann")
        self._cur.execute(self._sql_create_table, (len(X[0]),))
    
        # Insert data in batches
        print("Inserting data in batches...")
        start_time = time.time()
        
        batch = []
        for i, embedding in enumerate(X):
            batch.append((int(i), vector_to_hex(embedding)))
            
            # Insert when batch is full
            if len(batch) >= batch_size:
                self._cur.executemany(self._sql_insert, batch)
                batch.clear()
        
        # Insert remaining records in final batch
        if batch:
            self._cur.executemany(self._sql_insert, batch)
    
        insert_time = time.time() - start_time
        print(f"Insert time for {len(X)} records: {insert_time:.2f}s")

        self._cur.execute("COMMIT")
        self._cur.execute("FLUSH TABLES")

        # Create index
        print("Creating index...")
        start_time = time.time()
        self._cur.execute(self._sql_create_index)
        
        index_time = time.time() - start_time
        print(f"Index creation time: {index_time:.2f}s")

        self._cur.execute("COMMIT")
        self._cur.execute("FLUSH TABLES")

    def set_query_arguments(self, ef_search):
        # Set ef_search
        self._ef_search = ef_search
        self._cur.execute(f"SET GLOBAL mhnsw_ef_search = {ef_search}")
        self._cur.execute("COMMIT")

    def query(self, v, n):
        self._cur.execute(self._sql_search, (vector_to_hex(v), n))

        return [id for id, in self._cur.fetchall()]

    def batch_query(self, X, n):
        XX=[]
        for i in range(os.cpu_count()):
            XX.append((self._sql_search, n, X[int(len(X)/os.cpu_count()*i):int(len(X)/os.cpu_count()*(i+1))]))
        pool = Pool()
        self._res = pool.map(many_queries, XX)

    def get_batch_results(self):
        return np.array(list(chain(*self._res)))

    def get_memory_usage(self):
        stem = '/var/lib/mysql/ann/ann#i#01.'
        return sum(os.stat(f).st_size for f in glob.glob(stem + 'ibd') + glob.glob(stem + 'MY[ID]')) / 1024

    def __str__(self):
        return f"MariaDB(m={self._m}, ef_search={self._ef_search}, engine={self._engine})"
